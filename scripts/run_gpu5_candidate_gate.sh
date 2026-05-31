#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/ubuntu/qwen35a3b_finetune}"
cd "$PROJECT_ROOT"

log() {
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"
}

wait_gpu_free() {
  local gpu_index="$1"
  local gpu_uuid
  local tries=0
  gpu_uuid="$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' -v idx="$gpu_index" '$1==idx{print $2}')"
  if [[ -z "${gpu_uuid:-}" ]]; then
    log "failed to resolve GPU${gpu_index} uuid"
    return 1
  fi
  while true; do
    if ! nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory --format=csv,noheader,nounits 2>/dev/null \
      | awk -F', ' -v gpu="$gpu_uuid" '$1==gpu {found=1} END{exit found ? 0 : 1}'; then
      return 0
    fi
    tries=$((tries + 1))
    if [[ "$tries" -ge "${GPU_FREE_WAIT_TRIES:-120}" ]]; then
      log "GPU${gpu_index} still busy after waiting"
      return 1
    fi
    sleep 5
  done
}

stop_prod_vllm() {
  local service="$1"
  local tries=0
  if [[ -z "$service" ]]; then
    return 0
  fi
  if systemctl --user is-active --quiet "$service"; then
    RESTORE_PROD_VLLM=1
    log "stopping production service for GPU gate: $service"
    systemctl --user stop "$service" || true
    systemctl --user mask --runtime "$service" || true
    while systemctl --user is-active --quiet "$service"; do
      tries=$((tries + 1))
      if [[ "$tries" -ge 60 ]]; then
        log "production service still active after stop: $service"
        return 1
      fi
      systemctl --user stop "$service" || true
      sleep 1
    done
  else
    systemctl --user mask --runtime "$service" || true
  fi
}

cleanup() {
  if [[ -n "${TEMP_PROXY_PID:-}" ]]; then
    kill "$TEMP_PROXY_PID" 2>/dev/null || true
    wait "$TEMP_PROXY_PID" 2>/dev/null || true
  fi
  if [[ -n "${TEMP_VLLM_PID:-}" ]]; then
    kill "$TEMP_VLLM_PID" 2>/dev/null || true
    wait "$TEMP_VLLM_PID" 2>/dev/null || true
  fi
  if [[ "${RESTORE_PROD_VLLM:-0}" = "1" ]]; then
    log "restoring production GPU5 vLLM service: ${PROD_VLLM_SERVICE}"
    systemctl --user unmask "$PROD_VLLM_SERVICE" || true
    systemctl --user start "$PROD_VLLM_SERVICE" || true
  fi
}
trap cleanup EXIT

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-5}"
GPU_INDEX="${GPU_INDEX:-5}"
TS="${TS:-$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_NAME="${RUN_NAME:-gpu5_candidate_gate_${TS}}"
TRACKING_ROOT="${TRACKING_ROOT:-$PROJECT_ROOT/tracking}"
RUN_LOG="${RUN_LOG:-$TRACKING_ROOT/logs/${RUN_NAME}.log}"

MODEL_DIR="${MODEL_DIR:?MODEL_DIR is required}"
EVAL_OUT_DIR="${EVAL_OUT_DIR:-$PROJECT_ROOT/eval_outputs/${RUN_NAME}}"
SERVING_SANITY_OUT_DIR="${SERVING_SANITY_OUT_DIR:-$PROJECT_ROOT/eval_outputs/${RUN_NAME}_serving_sanity}"
OPS_GATE_OUT_DIR="${OPS_GATE_OUT_DIR:-$PROJECT_ROOT/eval_outputs/${RUN_NAME}_ops_gate}"
KNOWLEDGE_GATE_OUT_DIR="${KNOWLEDGE_GATE_OUT_DIR:-$PROJECT_ROOT/eval_outputs/${RUN_NAME}_knowledge_gate}"
SERVING_SANITY_JSONL="${SERVING_SANITY_JSONL:-$PROJECT_ROOT/datasets/eval_serving_sanity_gpu5_20260508.jsonl}"
OPS_HIGH_FREQ_JSONL="${OPS_HIGH_FREQ_JSONL:-$PROJECT_ROOT/datasets/eval_ops_high_freq_20.jsonl}"
TEMP_VLLM_PORT="${TEMP_VLLM_PORT:-8028}"
TEMP_PROXY_PORT="${TEMP_PROXY_PORT:-8038}"
TEMP_MODEL_NAME="${TEMP_MODEL_NAME:-qwen35a3b-candidate}"
TEMP_MAX_MODEL_LEN="${TEMP_MAX_MODEL_LEN:-8192}"
TEMP_GPU_MEMORY_UTILIZATION="${TEMP_GPU_MEMORY_UTILIZATION:-0.84}"
TEMP_PROXY_MAX_OUTPUT_TOKENS="${TEMP_PROXY_MAX_OUTPUT_TOKENS:-512}"
TEMP_PROXY_EVAL_MAX_OUTPUT_TOKENS="${TEMP_PROXY_EVAL_MAX_OUTPUT_TOKENS:-512}"
TOOL_DIALOGUE_TOOL_MAX_TOKENS="${TOOL_DIALOGUE_TOOL_MAX_TOKENS:-$TEMP_PROXY_EVAL_MAX_OUTPUT_TOKENS}"
TOOL_DIALOGUE_KNOWLEDGE_MAX_TOKENS="${TOOL_DIALOGUE_KNOWLEDGE_MAX_TOKENS:-$TEMP_PROXY_EVAL_MAX_OUTPUT_TOKENS}"
PROD_VLLM_SERVICE="${PROD_VLLM_SERVICE:-vllm-qwen35-gpu5-agent-upstream.service}"
VLLM_BIN="${VLLM_BIN:-/home/ubuntu/.local/bin/vllm}"

mkdir -p "$(dirname "$RUN_LOG")" "$EVAL_OUT_DIR"
exec >>"$RUN_LOG" 2>&1

log "candidate gate start"
log "MODEL_DIR=$MODEL_DIR"
log "EVAL_OUT_DIR=$EVAL_OUT_DIR"
log "SERVING_SANITY_OUT_DIR=$SERVING_SANITY_OUT_DIR"
log "OPS_GATE_OUT_DIR=$OPS_GATE_OUT_DIR"
log "KNOWLEDGE_GATE_OUT_DIR=$KNOWLEDGE_GATE_OUT_DIR"

if [[ ! -f "$MODEL_DIR/config.json" ]]; then
  log "invalid MODEL_DIR: missing config.json"
  exit 1
fi

if [[ "${STOP_PROD_VLLM:-1}" = "1" ]]; then
  stop_prod_vllm "$PROD_VLLM_SERVICE"
fi
wait_gpu_free "$GPU_INDEX"
log "GPU${GPU_INDEX} is free"

TEMP_VLLM_LOG="${TEMP_VLLM_LOG:-$TRACKING_ROOT/logs/${RUN_NAME}_temp_vllm_${TS}.log}"
TEMP_PROXY_LOG="${TEMP_PROXY_LOG:-$TRACKING_ROOT/logs/${RUN_NAME}_temp_proxy_${TS}.log}"

log "starting temp vLLM on 127.0.0.1:${TEMP_VLLM_PORT}"
CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" "$VLLM_BIN" serve "$MODEL_DIR" \
  --host 127.0.0.1 \
  --port "$TEMP_VLLM_PORT" \
  --tensor-parallel-size 1 \
  --max-model-len "$TEMP_MAX_MODEL_LEN" \
  --dtype bfloat16 \
  --gpu-memory-utilization "$TEMP_GPU_MEMORY_UTILIZATION" \
  --served-model-name "$TEMP_MODEL_NAME" \
  --enforce-eager \
  --trust-remote-code \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --limit-mm-per-prompt '{"image": 0, "video": 0}' \
  >>"$TEMP_VLLM_LOG" 2>&1 &
TEMP_VLLM_PID=$!

READY=0
for _ in $(seq 1 120); do
  if curl -fsS "http://127.0.0.1:${TEMP_VLLM_PORT}/v1/models" >/dev/null 2>&1; then
    READY=1
    break
  fi
  sleep 2
done
if [[ "$READY" != "1" ]]; then
  log "temp vLLM not ready, see $TEMP_VLLM_LOG"
  exit 1
fi

log "starting temp agent proxy on 127.0.0.1:${TEMP_PROXY_PORT}"
UPSTREAM_BASE_URL="http://127.0.0.1:${TEMP_VLLM_PORT}/v1" \
DEFAULT_MODEL="$TEMP_MODEL_NAME" \
UPSTREAM_MODEL_OVERRIDE="$TEMP_MODEL_NAME" \
MAX_OUTPUT_TOKENS="$TEMP_PROXY_MAX_OUTPUT_TOKENS" \
EVAL_MAX_OUTPUT_TOKENS="$TEMP_PROXY_EVAL_MAX_OUTPUT_TOKENS" \
RAG_ENABLED=0 \
VL_RAG_FALLBACK_ENABLED=0 \
GLOSSARY_ENABLED=0 \
EVAL_RAW_FORMAT_ADAPTER_ENABLED="${EVAL_RAW_FORMAT_ADAPTER_ENABLED:-0}" \
EVAL_RAW_KNOWLEDGE_JSONL="${EVAL_RAW_KNOWLEDGE_JSONL:-$PROJECT_ROOT/datasets/eval_sports_baowang_knowledge_40_20260428.jsonl}" \
SUNCIDI_SWAGGER_AUTO_TOOLS_ENABLED="${SUNCIDI_SWAGGER_AUTO_TOOLS_ENABLED:-0}" \
SKILLS_ENABLED=1 \
SKILLS_MAX_ITER=4 \
AGENT_UPSTREAM_BASE_URL="http://127.0.0.1:${TEMP_VLLM_PORT}/v1" \
AGENT_UPSTREAM_MODEL="$TEMP_MODEL_NAME" \
TOOL_FALLBACK_ENABLED="${GATE_TOOL_FALLBACK_ENABLED:-0}" \
TOOL_FALLBACK_BASE_URL="${GATE_TOOL_FALLBACK_BASE_URL:-http://127.0.0.1:8013/v1}" \
TOOL_FALLBACK_MODEL="${GATE_TOOL_FALLBACK_MODEL:-qwen35a3b-sft-v5-combined}" \
/usr/bin/python3 -m uvicorn scripts.dpo_guardrails_proxy:app \
  --host 127.0.0.1 \
  --port "$TEMP_PROXY_PORT" \
  --workers 1 \
  --app-dir "$PROJECT_ROOT" \
  >>"$TEMP_PROXY_LOG" 2>&1 &
TEMP_PROXY_PID=$!

READY=0
for _ in $(seq 1 60); do
  if curl -fsS "http://127.0.0.1:${TEMP_PROXY_PORT}/v1/agent/health" >/dev/null 2>&1; then
    READY=1
    break
  fi
  sleep 1
done
if [[ "$READY" != "1" ]]; then
  log "temp proxy not ready, see $TEMP_PROXY_LOG"
  exit 1
fi

log "stage 1/3: serving sanity gate"
python3 "$PROJECT_ROOT/scripts/eval_tool_dialogue_gate.py" \
  --base-url "http://127.0.0.1:${TEMP_PROXY_PORT}/v1" \
  --model "$TEMP_MODEL_NAME" \
  --output-dir "$EVAL_OUT_DIR" \
  --workers "${EVAL_WORKERS:-4}" \
  --tool-max-tokens "$TOOL_DIALOGUE_TOOL_MAX_TOKENS" \
  --knowledge-max-tokens "$TOOL_DIALOGUE_KNOWLEDGE_MAX_TOKENS" \
  --min-tool-overall "${MIN_TOOL_OVERALL:-0.98}" \
  --min-tool-called "${MIN_TOOL_CALLED:-0.98}" \
  --min-tool-schema "${MIN_TOOL_SCHEMA:-0.99}" \
  --min-knowledge-overall "${MIN_KNOWLEDGE_OVERALL:-0.92}"

BASE_URL="http://127.0.0.1:${TEMP_PROXY_PORT}" \
MODEL="$TEMP_MODEL_NAME" \
INPUT_JSONL="$SERVING_SANITY_JSONL" \
OUT_DIR="$SERVING_SANITY_OUT_DIR" \
LABEL="gpu5_serving_sanity" \
WORKERS="${SERVING_SANITY_WORKERS:-3}" \
TIMEOUT_SEC="${SERVING_SANITY_TIMEOUT_SEC:-90}" \
MAX_TOKENS="${SERVING_SANITY_MAX_TOKENS:-192}" \
MIN_REQUEST_OK_RATE="${MIN_SERVING_SANITY_REQUEST_OK_RATE:-1.0}" \
MIN_MUST_INCLUDE_AVG="${MIN_SERVING_SANITY_MUST_INCLUDE_AVG:-1.0}" \
MIN_MUST_NOT_AVG="${MIN_SERVING_SANITY_MUST_NOT_AVG:-1.0}" \
MAX_RISK_VIOLATION_RATE="${MAX_SERVING_SANITY_RISK_VIOLATION_RATE:-0.0}" \
MIN_INTENT_ACC="${MIN_SERVING_SANITY_INTENT_ACC:-1.0}" \
MIN_ESCALATION_ACC="${MIN_SERVING_SANITY_ESCALATION_ACC:-1.0}" \
MIN_OVERALL_AVG="${MIN_SERVING_SANITY_OVERALL_AVG:-0.95}" \
"$PROJECT_ROOT/scripts/run_customer_service_eval_gate.sh"

log "stage 2/3: ops high freq gate"
BASE_URL="http://127.0.0.1:${TEMP_PROXY_PORT}" \
MODEL="$TEMP_MODEL_NAME" \
INPUT_JSONL="$OPS_HIGH_FREQ_JSONL" \
OUT_DIR="$OPS_GATE_OUT_DIR" \
LABEL="gpu5_ops_high_freq" \
WORKERS="${OPS_GATE_WORKERS:-4}" \
TIMEOUT_SEC="${OPS_GATE_TIMEOUT_SEC:-90}" \
MAX_TOKENS="${OPS_GATE_MAX_TOKENS:-192}" \
MIN_REQUEST_OK_RATE="${MIN_OPS_REQUEST_OK_RATE:-1.0}" \
MIN_MUST_INCLUDE_AVG="${MIN_OPS_MUST_INCLUDE_AVG:-0.85}" \
MIN_MUST_NOT_AVG="${MIN_OPS_MUST_NOT_AVG:-1.0}" \
MAX_RISK_VIOLATION_RATE="${MAX_OPS_RISK_VIOLATION_RATE:-0.0}" \
MIN_INTENT_ACC="${MIN_OPS_INTENT_ACC:-0.90}" \
MIN_ESCALATION_ACC="${MIN_OPS_ESCALATION_ACC:-0.90}" \
MIN_OVERALL_AVG="${MIN_OPS_OVERALL_AVG:-0.88}" \
"$PROJECT_ROOT/scripts/run_customer_service_eval_gate.sh"

log "stage 3/3: sports knowledge gate"
BASE_URL_GPU5="http://127.0.0.1:${TEMP_PROXY_PORT}" \
MODEL_GPU5="$TEMP_MODEL_NAME" \
BASE_URL_GPU7="http://127.0.0.1:${TEMP_PROXY_PORT}" \
MODEL_GPU7="$TEMP_MODEL_NAME" \
OUT_DIR="$KNOWLEDGE_GATE_OUT_DIR" \
WORKERS="${KNOWLEDGE_GATE_WORKERS:-4}" \
TIMEOUT_SEC="${KNOWLEDGE_GATE_TIMEOUT_SEC:-90}" \
MAX_TOKENS="${KNOWLEDGE_GATE_MAX_TOKENS:-192}" \
"$PROJECT_ROOT/scripts/run_sports_knowledge_gate.sh"

log "gate passed"
log "serving tool summary: $EVAL_OUT_DIR/summary.json"
log "serving sanity summary: $SERVING_SANITY_OUT_DIR/summary.json"
log "ops gate summary: $OPS_GATE_OUT_DIR/summary.json"
log "knowledge gate summary root: $KNOWLEDGE_GATE_OUT_DIR"
