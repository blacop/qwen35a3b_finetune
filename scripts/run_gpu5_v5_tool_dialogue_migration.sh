#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/ubuntu/qwen35a3b_finetune"
cd "$PROJECT_ROOT"

log() {
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"
}

needs_vllm_reexport() {
  local model_dir="$1"
  python3 - "$model_dir" <<'PY'
import json
import sys
from pathlib import Path

model_dir = Path(sys.argv[1])
index_path = model_dir / "model.safetensors.index.json"
if not index_path.exists():
    raise SystemExit(1)
with index_path.open("r", encoding="utf-8") as f:
    weight_map = json.load(f)["weight_map"]
needs = any(k.startswith("model.language_model.visual.") for k in weight_map)
raise SystemExit(0 if needs else 1)
PY
}

latest_ckpt_step() {
  local out_dir="$1"
  find "$out_dir" -maxdepth 1 -type d -name 'checkpoint-*' -printf '%f\n' 2>/dev/null \
    | sed -n 's/^checkpoint-\([0-9]\+\)$/\1/p' | sort -n | tail -n1
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
    if [[ "$tries" -ge "${GPU_FREE_WAIT_TRIES:-90}" ]]; then
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
    log "stopping production service for GPU5 training: $service"
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
RUN_NAME_PREFIX="${RUN_NAME_PREFIX:-swift_sft_v5_1_tool_dialogue_gpu5}"
RUN_NAME="${RUN_NAME:-${RUN_NAME_PREFIX}_${TS}}"

BASE_MODEL="${BASE_MODEL:-/video-storage/ai-customer/qwen35a3b_finetune/outputs/swift_sft_v5_combined_gpu7_20260501T161714Z/merged_vllm}"
DATA_FILE="${DATA_FILE:-$PROJECT_ROOT/datasets/tool_dialogue_migration/sft_v5_1_tool_dialogue_migration.jsonl}"
DATA_SUMMARY_JSON="${DATA_SUMMARY_JSON:-$PROJECT_ROOT/datasets/tool_dialogue_migration/summary.json}"
OUT_DIR="${OUT_DIR:-$PROJECT_ROOT/outputs/${RUN_NAME}}"
MERGED_OUT_DIR="${MERGED_OUT_DIR:-$PROJECT_ROOT/outputs/${RUN_NAME}_merged_${TS}_vllm}"
EVAL_OUT_DIR="${EVAL_OUT_DIR:-$PROJECT_ROOT/eval_outputs/${RUN_NAME}_gate_${TS}}"
TRACKING_ROOT="${TRACKING_ROOT:-$PROJECT_ROOT/tracking}"
RUN_LOG="${RUN_LOG:-$TRACKING_ROOT/logs/${RUN_NAME}.log}"

TEMP_VLLM_PORT="${TEMP_VLLM_PORT:-8028}"
TEMP_PROXY_PORT="${TEMP_PROXY_PORT:-8038}"
TEMP_MODEL_NAME="${TEMP_MODEL_NAME:-qwen35a3b-v5-1-tool-dialogue-candidate}"
TEMP_MAX_MODEL_LEN="${TEMP_MAX_MODEL_LEN:-8192}"
TEMP_GPU_MEMORY_UTILIZATION="${TEMP_GPU_MEMORY_UTILIZATION:-0.84}"
PROD_VLLM_SERVICE="${PROD_VLLM_SERVICE:-vllm-qwen35-gpu5-agent-upstream.service}"

mkdir -p "$(dirname "$RUN_LOG")" "$EVAL_OUT_DIR"
exec >>"$RUN_LOG" 2>&1

log "v5.1 Tool-Dialogue migration start"
log "BASE_MODEL=$BASE_MODEL"
log "DATA_FILE=$DATA_FILE"
log "OUT_DIR=$OUT_DIR"
log "MERGED_OUT_DIR=$MERGED_OUT_DIR"
log "EVAL_OUT_DIR=$EVAL_OUT_DIR"

python3 "$PROJECT_ROOT/scripts/build_tool_dialogue_migration_dataset.py" \
  --output-jsonl "$DATA_FILE" \
  --summary-json "$DATA_SUMMARY_JSON" \
  ${BUILD_DATASET_ARGS:-} \
  ${SHADOW_JSONL_ARGS:-}

if [[ ! -f "$DATA_FILE" ]]; then
  log "missing migration dataset: $DATA_FILE"
  exit 1
fi
if [[ ! -d "$BASE_MODEL" ]]; then
  log "missing BASE_MODEL directory: $BASE_MODEL"
  exit 1
fi

if [[ "${STOP_PROD_VLLM:-1}" = "1" ]]; then
  stop_prod_vllm "$PROD_VLLM_SERVICE"
fi
wait_gpu_free "$GPU_INDEX"
log "GPU${GPU_INDEX} is free"

RUN_NAME="$RUN_NAME" \
BASE_MODEL="$BASE_MODEL" \
DATA_FILE="$DATA_FILE" \
OUT_DIR="$OUT_DIR" \
MAX_LEN="${MAX_LEN:-8192}" \
EPOCHS="${EPOCHS:-1}" \
PER_DEVICE_BATCH="${PER_DEVICE_BATCH:-1}" \
GRAD_ACC="${GRAD_ACC:-8}" \
LR="${LR:-1e-5}" \
LORA_RANK="${LORA_RANK:-16}" \
LORA_ALPHA="${LORA_ALPHA:-32}" \
SAVE_STEPS="${SAVE_STEPS:-100}" \
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-3}" \
AUTO_RESUME="${AUTO_RESUME:-false}" \
SYSTEM_PROMPT="${SYSTEM_PROMPT:-你是体育包网智能客服。需要查询后台时必须调用工具；工具返回后基于结果回答，不要编造后台数据。}" \
USE_GLOBAL_SYSTEM_PROMPT="${USE_GLOBAL_SYSTEM_PROMPT:-1}" \
"$PROJECT_ROOT/run_gpu5_swift_sft.sh"

SFT_STEP="$(latest_ckpt_step "$OUT_DIR" || true)"
if [[ -z "${SFT_STEP:-}" || ! -d "$OUT_DIR/checkpoint-${SFT_STEP}" ]]; then
  log "failed to find SFT checkpoint in $OUT_DIR"
  exit 1
fi
SFT_CKPT="$OUT_DIR/checkpoint-${SFT_STEP}"
log "latest checkpoint=$SFT_CKPT"

UV_BIN="${UV_BIN:-/home/ubuntu/.local/bin/uv}"
SWIFT_VENV="${SWIFT_VENV:-$PROJECT_ROOT/.venv-swift311}"
UV_PYTHON="${UV_PYTHON:-$SWIFT_VENV/bin/python}"
VLLM_BIN="${VLLM_BIN:-/home/ubuntu/.local/bin/vllm}"

"$UV_BIN" run --python "$UV_PYTHON" swift export \
  --use_hf true \
  --model "$BASE_MODEL" \
  --adapters "$SFT_CKPT" \
  --merge_lora true \
  --safe_serialization true \
  --exist_ok true \
  --output_dir "$MERGED_OUT_DIR"

if [[ ! -f "$MERGED_OUT_DIR/config.json" ]]; then
  log "merged model invalid: missing config.json"
  exit 1
fi

if needs_vllm_reexport "$MERGED_OUT_DIR"; then
  log "re-export merged weights for vLLM key compatibility: $MERGED_OUT_DIR"
  "$UV_PYTHON" "$PROJECT_ROOT/scripts/reexport_vllm_compatible_hf.py" \
    --src "$MERGED_OUT_DIR" \
    --dst "$MERGED_OUT_DIR" \
    --base "$BASE_MODEL" \
    --in-place
fi

if [[ "${RUN_GATE:-1}" != "1" ]]; then
  log "RUN_GATE=0, skip temp serving and gate"
  exit 0
fi

TEMP_VLLM_LOG="${TEMP_VLLM_LOG:-$TRACKING_ROOT/logs/${RUN_NAME}_temp_vllm_${TS}.log}"
TEMP_PROXY_LOG="${TEMP_PROXY_LOG:-$TRACKING_ROOT/logs/${RUN_NAME}_temp_proxy_${TS}.log}"

if [[ "${STOP_PROD_VLLM:-1}" = "1" ]]; then
  stop_prod_vllm "$PROD_VLLM_SERVICE"
fi
wait_gpu_free "$GPU_INDEX"
log "GPU${GPU_INDEX} is free for temp vLLM"

log "starting temp vLLM on 127.0.0.1:${TEMP_VLLM_PORT}"
CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" "$VLLM_BIN" serve "$MERGED_OUT_DIR" \
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
MAX_OUTPUT_TOKENS=512 \
EVAL_MAX_OUTPUT_TOKENS=512 \
RAG_ENABLED=0 \
VL_RAG_FALLBACK_ENABLED=0 \
GLOSSARY_ENABLED=0 \
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

python3 "$PROJECT_ROOT/scripts/eval_tool_dialogue_gate.py" \
  --base-url "http://127.0.0.1:${TEMP_PROXY_PORT}/v1" \
  --model "$TEMP_MODEL_NAME" \
  --output-dir "$EVAL_OUT_DIR" \
  --workers "${EVAL_WORKERS:-4}" \
  --min-tool-overall "${MIN_TOOL_OVERALL:-0.98}" \
  --min-tool-called "${MIN_TOOL_CALLED:-0.98}" \
  --min-tool-schema "${MIN_TOOL_SCHEMA:-0.99}" \
  --min-knowledge-overall "${MIN_KNOWLEDGE_OVERALL:-0.92}"

log "gate passed"
log "merged model: $MERGED_OUT_DIR"
log "gate summary: $EVAL_OUT_DIR/summary.json"
