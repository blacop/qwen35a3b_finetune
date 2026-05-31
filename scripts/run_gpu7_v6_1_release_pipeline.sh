#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/ubuntu/qwen35a3b_finetune}"
cd "$PROJECT_ROOT"

log() {
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"
}

latest_ckpt_step() {
  local out_dir="$1"
  find "$out_dir" -maxdepth 1 -type d -name 'checkpoint-*' -printf '%f\n' 2>/dev/null \
    | sed -n 's/^checkpoint-\([0-9]\+\)$/\1/p' | sort -n | tail -n1
}

wait_http_ready() {
  local url="$1"
  local retries="${2:-120}"
  local sleep_s="${3:-2}"
  local i
  for i in $(seq 1 "$retries"); do
    if curl -fsS "$url" >/dev/null 2>&1; then
      return 0
    fi
    sleep "$sleep_s"
  done
  return 1
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
  if [[ "${RESTORE_GPU7_BASELINE:-0}" = "1" ]]; then
    log "restoring production GPU7 baseline service: ${PROD_GPU7_SERVICE}"
    systemctl --user unmask "$PROD_GPU7_SERVICE" || true
    systemctl --user start "$PROD_GPU7_SERVICE" || true
  fi
}
trap cleanup EXIT

stop_prod_gpu7_baseline() {
  local service="$1"
  local tries=0
  if [[ -z "$service" ]]; then
    return 0
  fi
  if systemctl --user is-active --quiet "$service"; then
    RESTORE_GPU7_BASELINE=1
    log "stopping production GPU7 baseline service for training: $service"
    systemctl --user stop "$service" || true
    systemctl --user mask --runtime "$service" || true
    while systemctl --user is-active --quiet "$service"; do
      tries=$((tries + 1))
      if [[ "$tries" -ge 60 ]]; then
        log "service still active after stop: $service"
        return 1
      fi
      systemctl --user stop "$service" || true
      sleep 1
    done
  else
    systemctl --user mask --runtime "$service" || true
  fi
}

kill_gpu_processes() {
  local gpu_index="${1:-7}"
  local pid
  while read -r pid; do
    pid="${pid// /}"
    [[ -z "$pid" ]] && continue
    kill -9 "$pid" 2>/dev/null || true
  done < <(nvidia-smi -i "$gpu_index" --query-compute-apps=pid --format=csv,noheader)
  sleep 2
}

TS="${TS:-$(date -u +%Y%m%dT%H%M%SZ)}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-7}"
RUN_TRAIN="${RUN_TRAIN:-1}"
RUN_MERGE="${RUN_MERGE:-1}"

BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3.5-35B-A3B}"
DATASET_JSONL="${DATASET_JSONL:-$PROJECT_ROOT/datasets/v6_1_combined.jsonl}"
SYSTEM_PROMPT="${SYSTEM_PROMPT:-你是体育包网平台的一线客服助手，只按体育包网平台客服语境回答。回复短句、口语化、像真人客服。涉及后台数据、风控、财务时不要猜测，明确说需要后台核实。}"

OUT_DIR="${OUT_DIR:-$PROJECT_ROOT/outputs/swift_sft_v6_1_gpu7_${TS}}"
TRAIN_LOG="${TRAIN_LOG:-$PROJECT_ROOT/tracking/logs/v6_1_${TS}.log}"
MERGED_OUT_DIR="${MERGED_OUT_DIR:-$PROJECT_ROOT/outputs/swift_sft_v6_1_gpu7_${TS}_merged_vllm}"
EVAL_OUT_DIR="${EVAL_OUT_DIR:-$PROJECT_ROOT/eval_outputs/v6_1_release_gate_${TS}}"

OUT_DIR="$(realpath -m "$OUT_DIR")"
TRAIN_LOG="$(realpath -m "$TRAIN_LOG")"
MERGED_OUT_DIR="$(realpath -m "$MERGED_OUT_DIR")"
EVAL_OUT_DIR="$(realpath -m "$EVAL_OUT_DIR")"

TEMP_VLLM_PORT="${TEMP_VLLM_PORT:-8113}"
TEMP_PROXY_PORT="${TEMP_PROXY_PORT:-8110}"
TEMP_MODEL_NAME="${TEMP_MODEL_NAME:-qwen35a3b-sft-v6-1-candidate}"
TEMP_GPU_MEMORY_UTILIZATION="${TEMP_GPU_MEMORY_UTILIZATION:-0.85}"
TEMP_MAX_MODEL_LEN="${TEMP_MAX_MODEL_LEN:-8192}"
STOP_PROD_GPU7="${STOP_PROD_GPU7:-1}"
PROD_GPU7_SERVICE="${PROD_GPU7_SERVICE:-vllm-qwen35-gpu7-baseline.service}"

FIELD_SUMMARY_JSON="$EVAL_OUT_DIR/field_completeness_summary.json"
TOOL_GATE_OUT_DIR="$EVAL_OUT_DIR/tool_gate"
OPS16_OUT_DIR="$EVAL_OUT_DIR/ops16"
KB40_OUT_DIR="$EVAL_OUT_DIR/kb40"
RELEASE_SUMMARY_JSON="${RELEASE_SUMMARY_JSON:-$EVAL_OUT_DIR/release_gate_summary_strict.json}"
RELEASE_REPORT_MD="${RELEASE_REPORT_MD:-$EVAL_OUT_DIR/release_gate_report_strict.md}"
TOOL_EVAL_JSONL="${TOOL_EVAL_JSONL:-$PROJECT_ROOT/datasets/tool_use/eval_tool_use_suncidi_30_20260514.jsonl}"

mkdir -p "$(dirname "$TRAIN_LOG")" "$EVAL_OUT_DIR"

MIN_FIELD_PASS_RATE="${MIN_FIELD_PASS_RATE:-1.0}"
MIN_TOOL_OVERALL="${MIN_TOOL_OVERALL:-0.98}"
MIN_TOOL_CALLED="${MIN_TOOL_CALLED:-0.98}"
MIN_TOOL_SCHEMA="${MIN_TOOL_SCHEMA:-0.99}"
MIN_KNOWLEDGE_OVERALL="${MIN_KNOWLEDGE_OVERALL:-0.92}"

MIN_OPS16_REQUEST_OK_RATE="${MIN_OPS16_REQUEST_OK_RATE:-1.0}"
MIN_OPS16_MUST_INCLUDE_AVG="${MIN_OPS16_MUST_INCLUDE_AVG:-0.84}"
MIN_OPS16_MUST_NOT_AVG="${MIN_OPS16_MUST_NOT_AVG:-1.0}"
MAX_OPS16_RISK_VIOLATION_RATE="${MAX_OPS16_RISK_VIOLATION_RATE:-0.0}"
MIN_OPS16_INTENT_ACC="${MIN_OPS16_INTENT_ACC:-0.90}"
MIN_OPS16_ESCALATION_ACC="${MIN_OPS16_ESCALATION_ACC:-0.90}"
MIN_OPS16_OVERALL_AVG="${MIN_OPS16_OVERALL_AVG:-0.84}"

MIN_KB40_REQUEST_OK_RATE="${MIN_KB40_REQUEST_OK_RATE:-1.0}"
MIN_KB40_MUST_INCLUDE_AVG="${MIN_KB40_MUST_INCLUDE_AVG:-0.60}"
MIN_KB40_MUST_NOT_AVG="${MIN_KB40_MUST_NOT_AVG:-1.0}"
MAX_KB40_RISK_VIOLATION_RATE="${MAX_KB40_RISK_VIOLATION_RATE:-0.0}"
MIN_KB40_INTENT_ACC="${MIN_KB40_INTENT_ACC:-0.75}"
MIN_KB40_ESCALATION_ACC="${MIN_KB40_ESCALATION_ACC:-0.90}"
MIN_KB40_OVERALL_AVG="${MIN_KB40_OVERALL_AVG:-0.90}"

log "pipeline start TS=$TS RUN_TRAIN=$RUN_TRAIN RUN_MERGE=$RUN_MERGE"
log "OUT_DIR=$OUT_DIR"
log "MERGED_OUT_DIR=$MERGED_OUT_DIR"
log "EVAL_OUT_DIR=$EVAL_OUT_DIR"

if [[ ! -f "$DATASET_JSONL" ]]; then
  log "missing dataset: $DATASET_JSONL"
  exit 1
fi

if [[ "$STOP_PROD_GPU7" == "1" || "$STOP_PROD_GPU7" == "true" || "$STOP_PROD_GPU7" == "yes" ]]; then
  stop_prod_gpu7_baseline "$PROD_GPU7_SERVICE"
fi
kill_gpu_processes 7
log "GPU7 compute processes cleared before training"

if [[ "$RUN_TRAIN" == "1" || "$RUN_TRAIN" == "true" || "$RUN_TRAIN" == "yes" ]]; then
  log "start training"
  CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  /home/ubuntu/qwen35a3b_finetune/.venv-swift311/bin/swift sft \
    --model "$BASE_MODEL" \
    --use_hf true \
    --check_model false \
    --tuner_type lora \
    --torch_dtype bfloat16 \
    --dataset "$DATASET_JSONL" \
    --dataset_num_proc 4 \
    --template qwen \
    --system "$SYSTEM_PROMPT" \
    --max_length 4096 \
    --learning_rate 1e-4 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --lora_rank 16 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --warmup_ratio 0.05 \
    --logging_steps 10 \
    --save_strategy steps \
    --save_steps 200 \
    --eval_strategy no \
    --split_dataset_ratio 0 \
    --save_total_limit 3 \
    --add_version false \
    --ignore_args_error true \
    --output_dir "$OUT_DIR" 2>&1 | tee "$TRAIN_LOG"
  log "training finished"
else
  log "skip training, expecting existing OUT_DIR"
fi

SFT_STEP="$(latest_ckpt_step "$OUT_DIR" || true)"
if [[ -z "${SFT_STEP:-}" ]]; then
  log "no checkpoint found in $OUT_DIR"
  exit 1
fi
SFT_CKPT="$OUT_DIR/checkpoint-$SFT_STEP"
if [[ ! -f "$SFT_CKPT/adapter_config.json" ]]; then
  log "invalid checkpoint: $SFT_CKPT"
  exit 1
fi
log "using checkpoint: $SFT_CKPT"

if [[ "$RUN_MERGE" == "1" || "$RUN_MERGE" == "true" || "$RUN_MERGE" == "yes" ]]; then
  log "start merge lora"
  CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" /home/ubuntu/qwen35a3b_finetune/.venv-swift311/bin/swift export \
    --use_hf true \
    --model "$BASE_MODEL" \
    --adapters "$SFT_CKPT" \
    --merge_lora true \
    --safe_serialization true \
    --exist_ok true \
    --output_dir "$MERGED_OUT_DIR"
  if [[ ! -f "$MERGED_OUT_DIR/config.json" ]]; then
    log "merge output invalid: $MERGED_OUT_DIR"
    exit 1
  fi
  log "merge finished: $MERGED_OUT_DIR"
else
  if [[ ! -f "$MERGED_OUT_DIR/config.json" ]]; then
    log "RUN_MERGE=0 but merged model missing: $MERGED_OUT_DIR"
    exit 1
  fi
  log "skip merge, reuse merged model: $MERGED_OUT_DIR"
fi

TEMP_VLLM_LOG="$EVAL_OUT_DIR/temp_vllm.log"
TEMP_PROXY_LOG="$EVAL_OUT_DIR/temp_proxy.log"

log "start temp vllm :$TEMP_VLLM_PORT"
CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" /home/ubuntu/.local/bin/vllm serve "$MERGED_OUT_DIR" \
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
  >"$TEMP_VLLM_LOG" 2>&1 &
TEMP_VLLM_PID=$!

if ! wait_http_ready "http://127.0.0.1:${TEMP_VLLM_PORT}/v1/models" 180 2; then
  log "temp vllm not ready"
  exit 1
fi

log "start temp proxy :$TEMP_PROXY_PORT"
UPSTREAM_BASE_URL="http://127.0.0.1:${TEMP_VLLM_PORT}/v1" \
DEFAULT_MODEL="$TEMP_MODEL_NAME" \
UPSTREAM_MODEL_OVERRIDE="$TEMP_MODEL_NAME" \
MAX_OUTPUT_TOKENS=512 \
EVAL_MAX_OUTPUT_TOKENS=512 \
RAG_ENABLED=0 \
VL_RAG_FALLBACK_ENABLED=0 \
GLOSSARY_ENABLED=0 \
SKILLS_ENABLED=1 \
SKILLS_MAX_ITER=6 \
AGENT_MAX_OUTPUT_TOKENS=256 \
AGENT_INTENT_ROUTING_ENABLED=1 \
AGENT_TOOL_ROUTE_DEFAULT_GROUP=order \
AGENT_TOOL_ROUTE_GROUPS_JSON='{"order":["get_transfer_log_list","list_bet_orders","get_bet_order_stats","get_order_status_list"]}' \
AGENT_TOOL_ROUTE_RULES_JSON='[{"group":"order","keywords":["订单","转账","投注","记录","注单"]}]' \
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
  >"$TEMP_PROXY_LOG" 2>&1 &
TEMP_PROXY_PID=$!

if ! wait_http_ready "http://127.0.0.1:${TEMP_PROXY_PORT}/v1/agent/health" 90 1; then
  log "temp proxy not ready"
  exit 1
fi

log "gate 1/4: field completeness"
set +e
python3 "$PROJECT_ROOT/scripts/check_agent_trace_fields.py" \
  --base-url "http://127.0.0.1:${TEMP_PROXY_PORT}/v1" \
  --model "$TEMP_MODEL_NAME" \
  --output-json "$FIELD_SUMMARY_JSON" \
  --min-pass-rate "$MIN_FIELD_PASS_RATE"
GATE_FIELD_RC=$?
set -e
if [[ "$GATE_FIELD_RC" -ne 0 ]]; then
  log "gate 1/4 failed rc=$GATE_FIELD_RC"
fi

log "gate 2/4: tool hit rate"
log "tool gate dataset: $TOOL_EVAL_JSONL"
set +e
python3 "$PROJECT_ROOT/scripts/eval_tool_dialogue_gate.py" \
  --base-url "http://127.0.0.1:${TEMP_PROXY_PORT}/v1" \
  --model "$TEMP_MODEL_NAME" \
  --tool-eval-jsonl "$TOOL_EVAL_JSONL" \
  --output-dir "$TOOL_GATE_OUT_DIR" \
  --workers "${EVAL_WORKERS:-4}" \
  --min-tool-overall "$MIN_TOOL_OVERALL" \
  --min-tool-called "$MIN_TOOL_CALLED" \
  --min-tool-schema "$MIN_TOOL_SCHEMA" \
  --min-knowledge-overall "$MIN_KNOWLEDGE_OVERALL"
GATE_TOOL_RC=$?
set -e
if [[ "$GATE_TOOL_RC" -ne 0 ]]; then
  log "gate 2/4 failed rc=$GATE_TOOL_RC"
fi

log "gate 3/4: ops16"
set +e
BASE_URL="http://127.0.0.1:${TEMP_PROXY_PORT}" \
MODEL="$TEMP_MODEL_NAME" \
INPUT_JSONL="$PROJECT_ROOT/datasets/eval_ops_feedback_16_20260513.jsonl" \
OUT_DIR="$OPS16_OUT_DIR" \
LABEL="v6_1_ops16" \
WORKERS="${OPS16_WORKERS:-4}" \
TIMEOUT_SEC="${OPS16_TIMEOUT_SEC:-120}" \
MAX_TOKENS="${OPS16_MAX_TOKENS:-512}" \
MIN_REQUEST_OK_RATE="$MIN_OPS16_REQUEST_OK_RATE" \
MIN_MUST_INCLUDE_AVG="$MIN_OPS16_MUST_INCLUDE_AVG" \
MIN_MUST_NOT_AVG="$MIN_OPS16_MUST_NOT_AVG" \
MAX_RISK_VIOLATION_RATE="$MAX_OPS16_RISK_VIOLATION_RATE" \
MIN_INTENT_ACC="$MIN_OPS16_INTENT_ACC" \
MIN_ESCALATION_ACC="$MIN_OPS16_ESCALATION_ACC" \
MIN_OVERALL_AVG="$MIN_OPS16_OVERALL_AVG" \
"$PROJECT_ROOT/scripts/run_customer_service_eval_gate.sh"
GATE_OPS16_RC=$?
set -e
if [[ "$GATE_OPS16_RC" -ne 0 ]]; then
  log "gate 3/4 failed rc=$GATE_OPS16_RC"
fi

log "gate 4/4: knowledge40"
set +e
BASE_URL="http://127.0.0.1:${TEMP_PROXY_PORT}" \
MODEL="$TEMP_MODEL_NAME" \
INPUT_JSONL="$PROJECT_ROOT/datasets/eval_sports_baowang_knowledge_40_20260428.jsonl" \
OUT_DIR="$KB40_OUT_DIR" \
LABEL="v6_1_kb40" \
WORKERS="${KB40_WORKERS:-4}" \
TIMEOUT_SEC="${KB40_TIMEOUT_SEC:-120}" \
MAX_TOKENS="${KB40_MAX_TOKENS:-512}" \
MIN_REQUEST_OK_RATE="$MIN_KB40_REQUEST_OK_RATE" \
MIN_MUST_INCLUDE_AVG="$MIN_KB40_MUST_INCLUDE_AVG" \
MIN_MUST_NOT_AVG="$MIN_KB40_MUST_NOT_AVG" \
MAX_RISK_VIOLATION_RATE="$MAX_KB40_RISK_VIOLATION_RATE" \
MIN_INTENT_ACC="$MIN_KB40_INTENT_ACC" \
MIN_ESCALATION_ACC="$MIN_KB40_ESCALATION_ACC" \
MIN_OVERALL_AVG="$MIN_KB40_OVERALL_AVG" \
"$PROJECT_ROOT/scripts/run_customer_service_eval_gate.sh"
GATE_KB40_RC=$?
set -e
if [[ "$GATE_KB40_RC" -ne 0 ]]; then
  log "gate 4/4 failed rc=$GATE_KB40_RC"
fi

export OUT_DIR SFT_CKPT MERGED_OUT_DIR FIELD_SUMMARY_JSON TOOL_GATE_OUT_DIR OPS16_OUT_DIR KB40_OUT_DIR RELEASE_SUMMARY_JSON RELEASE_REPORT_MD TS
export MIN_FIELD_PASS_RATE MIN_TOOL_OVERALL MIN_TOOL_CALLED MIN_TOOL_SCHEMA MIN_KNOWLEDGE_OVERALL
export MIN_OPS16_REQUEST_OK_RATE MIN_OPS16_MUST_INCLUDE_AVG MIN_OPS16_MUST_NOT_AVG MAX_OPS16_RISK_VIOLATION_RATE MIN_OPS16_INTENT_ACC MIN_OPS16_ESCALATION_ACC MIN_OPS16_OVERALL_AVG
export MIN_KB40_REQUEST_OK_RATE MIN_KB40_MUST_INCLUDE_AVG MIN_KB40_MUST_NOT_AVG MAX_KB40_RISK_VIOLATION_RATE MIN_KB40_INTENT_ACC MIN_KB40_ESCALATION_ACC MIN_KB40_OVERALL_AVG
export GATE_FIELD_RC GATE_TOOL_RC GATE_OPS16_RC GATE_KB40_RC
python3 - <<'PY'
import json
import os
import sys
from pathlib import Path

def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}

def to_float(value):
    try:
        return float(value)
    except Exception:
        return None

def ge(value, threshold):
    return value is not None and value >= threshold

def le(value, threshold):
    return value is not None and value <= threshold

out_dir = Path(os.environ["OUT_DIR"])
ckpt = Path(os.environ["SFT_CKPT"])
merged = Path(os.environ["MERGED_OUT_DIR"])
field_path = Path(os.environ["FIELD_SUMMARY_JSON"])
tool_path = Path(os.environ["TOOL_GATE_OUT_DIR"]) / "summary.json"
ops16_path = Path(os.environ["OPS16_OUT_DIR"]) / "summary.json"
kb40_path = Path(os.environ["KB40_OUT_DIR"]) / "summary.json"
summary_path = Path(os.environ["RELEASE_SUMMARY_JSON"])
report_path = Path(os.environ["RELEASE_REPORT_MD"])
ts = os.environ["TS"]

field = load_json(field_path)
tool = load_json(tool_path)
ops16 = load_json(ops16_path)
kb40 = load_json(kb40_path)

train_runtime = None
train_loss = None
train_token_acc = None
logging_jsonl = out_dir / "logging.jsonl"
if logging_jsonl.exists():
    for line in logging_jsonl.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except Exception:
            continue
        if "train_runtime" in row:
            train_runtime = row.get("train_runtime")
        if "train_loss" in row:
            train_loss = row.get("train_loss")
        if "train_token_accuracy" in row:
            train_token_acc = row.get("train_token_accuracy")
        elif "train_acc" in row:
            train_token_acc = row.get("train_acc")
        elif "token_acc" in row:
            train_token_acc = row.get("token_acc")

thresholds = {
    "field_min_pass_rate": float(os.environ["MIN_FIELD_PASS_RATE"]),
    "tool_min_overall": float(os.environ["MIN_TOOL_OVERALL"]),
    "tool_min_called": float(os.environ["MIN_TOOL_CALLED"]),
    "tool_min_schema": float(os.environ["MIN_TOOL_SCHEMA"]),
    "tool_min_knowledge_overall": float(os.environ["MIN_KNOWLEDGE_OVERALL"]),
    "ops16_min_request_ok_rate": float(os.environ["MIN_OPS16_REQUEST_OK_RATE"]),
    "ops16_min_must_include_avg": float(os.environ["MIN_OPS16_MUST_INCLUDE_AVG"]),
    "ops16_min_must_not_avg": float(os.environ["MIN_OPS16_MUST_NOT_AVG"]),
    "ops16_max_risk_violation_rate": float(os.environ["MAX_OPS16_RISK_VIOLATION_RATE"]),
    "ops16_min_intent_acc": float(os.environ["MIN_OPS16_INTENT_ACC"]),
    "ops16_min_escalation_acc": float(os.environ["MIN_OPS16_ESCALATION_ACC"]),
    "ops16_min_overall_avg": float(os.environ["MIN_OPS16_OVERALL_AVG"]),
    "kb40_min_request_ok_rate": float(os.environ["MIN_KB40_REQUEST_OK_RATE"]),
    "kb40_min_must_include_avg": float(os.environ["MIN_KB40_MUST_INCLUDE_AVG"]),
    "kb40_min_must_not_avg": float(os.environ["MIN_KB40_MUST_NOT_AVG"]),
    "kb40_max_risk_violation_rate": float(os.environ["MAX_KB40_RISK_VIOLATION_RATE"]),
    "kb40_min_intent_acc": float(os.environ["MIN_KB40_INTENT_ACC"]),
    "kb40_min_escalation_acc": float(os.environ["MIN_KB40_ESCALATION_ACC"]),
    "kb40_min_overall_avg": float(os.environ["MIN_KB40_OVERALL_AVG"]),
}

tool_tool = tool.get("tool", {})
tool_knowledge = tool.get("knowledge", {})
checks = {
    "field_pass": bool(field.get("passed")),
    "tool_overall_pass": ge(to_float(tool_tool.get("overall")), thresholds["tool_min_overall"]),
    "tool_called_pass": ge(to_float(tool_tool.get("tool_called")), thresholds["tool_min_called"]),
    "tool_schema_pass": ge(to_float(tool_tool.get("schema_valid")), thresholds["tool_min_schema"]),
    "tool_knowledge_pass": ge(to_float(tool_knowledge.get("overall")), thresholds["tool_min_knowledge_overall"]),
    "ops16_request_ok_pass": ge(to_float(ops16.get("request_ok_rate")), thresholds["ops16_min_request_ok_rate"]),
    "ops16_must_include_pass": ge(to_float(ops16.get("must_include_avg")), thresholds["ops16_min_must_include_avg"]),
    "ops16_must_not_pass": ge(to_float(ops16.get("must_not_avg")), thresholds["ops16_min_must_not_avg"]),
    "ops16_risk_pass": le(to_float(ops16.get("risk_violation_rate")), thresholds["ops16_max_risk_violation_rate"]),
    "ops16_intent_pass": ge(to_float(ops16.get("intent_acc")), thresholds["ops16_min_intent_acc"]),
    "ops16_escalation_pass": ge(to_float(ops16.get("escalation_acc")), thresholds["ops16_min_escalation_acc"]),
    "ops16_overall_pass": ge(to_float(ops16.get("overall_avg")), thresholds["ops16_min_overall_avg"]),
    "kb40_request_ok_pass": ge(to_float(kb40.get("request_ok_rate")), thresholds["kb40_min_request_ok_rate"]),
    "kb40_must_include_pass": ge(to_float(kb40.get("must_include_avg")), thresholds["kb40_min_must_include_avg"]),
    "kb40_must_not_pass": ge(to_float(kb40.get("must_not_avg")), thresholds["kb40_min_must_not_avg"]),
    "kb40_risk_pass": le(to_float(kb40.get("risk_violation_rate")), thresholds["kb40_max_risk_violation_rate"]),
    "kb40_intent_pass": ge(to_float(kb40.get("intent_acc")), thresholds["kb40_min_intent_acc"]),
    "kb40_escalation_pass": ge(to_float(kb40.get("escalation_acc")), thresholds["kb40_min_escalation_acc"]),
    "kb40_overall_pass": ge(to_float(kb40.get("overall_avg")), thresholds["kb40_min_overall_avg"]),
}

gate_exit_codes = {
    "field_rc": int(os.environ.get("GATE_FIELD_RC", "1")),
    "tool_rc": int(os.environ.get("GATE_TOOL_RC", "1")),
    "ops16_rc": int(os.environ.get("GATE_OPS16_RC", "1")),
    "kb40_rc": int(os.environ.get("GATE_KB40_RC", "1")),
}
checks["gate_commands_pass"] = all(v == 0 for v in gate_exit_codes.values())
passed = all(checks.values())

payload = {
    "ts": ts,
    "train_output_dir": str(out_dir),
    "checkpoint": str(ckpt),
    "merged_model_dir": str(merged),
    "train_runtime_s": train_runtime,
    "train_loss": train_loss,
    "train_token_acc": train_token_acc,
    "metrics": {
        "field": field,
        "tool": tool,
        "ops16": ops16,
        "kb40": kb40,
    },
    "thresholds": thresholds,
    "checks": checks,
    "gate_exit_codes": gate_exit_codes,
    "passed": passed,
}

summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

report = [
    "# v6.1 回归评测报告（严格门槛）",
    "",
    "## 训练产物",
    f"- 训练目录: `{out_dir}`",
    f"- Checkpoint: `{ckpt}`",
    f"- 合并模型: `{merged}`",
    f"- train_runtime: `{train_runtime}` s",
    f"- train_loss: `{train_loss}`",
    f"- train_token_acc: `{train_token_acc}`",
    "",
    "## 评测结果",
    f"- 字段完整性 pass_rate: `{field.get('pass_rate')}`",
    f"- 工具命中 overall/called: `{tool_tool.get('overall')}` / `{tool_tool.get('tool_called')}`",
    f"- 运营16题 overall/intent/escalation/must_include: `{ops16.get('overall_avg')}` / `{ops16.get('intent_acc')}` / `{ops16.get('escalation_acc')}` / `{ops16.get('must_include_avg')}`",
    f"- 知识40题 overall/intent/escalation/must_include: `{kb40.get('overall_avg')}` / `{kb40.get('intent_acc')}` / `{kb40.get('escalation_acc')}` / `{kb40.get('must_include_avg')}`",
    "",
    "## 门槛判定",
]
for key in checks:
    report.append(f"- {key}: `{checks[key]}`")
report.extend(
    [
        "",
        "## Gate 命令返回码",
        f"- field: `{gate_exit_codes['field_rc']}`",
        f"- tool: `{gate_exit_codes['tool_rc']}`",
        f"- ops16: `{gate_exit_codes['ops16_rc']}`",
        f"- kb40: `{gate_exit_codes['kb40_rc']}`",
        "",
        "## 总结",
        f"- Strict Gate Passed: **{passed}**",
    ]
)
report_path.write_text("\n".join(report) + "\n", encoding="utf-8")
print(json.dumps(payload, ensure_ascii=False, indent=2))
print(f"OVERALL_PASSED={'true' if passed else 'false'}")
PY

python3 "$PROJECT_ROOT/scripts/append_release_log_index.py" \
  --index-file "$PROJECT_ROOT/tracking/logs/release_log_index.jsonl" \
  --ts "$TS" \
  --pipeline "run_gpu7_v6_1_release_pipeline" \
  --model-dir "$OUT_DIR" \
  --merged-model-dir "$MERGED_OUT_DIR" \
  --served-model-name "$TEMP_MODEL_NAME" \
  --release-gate-summary-path "$RELEASE_SUMMARY_JSON" \
  --extra-json "{\"field_summary\":\"$FIELD_SUMMARY_JSON\",\"tool_gate_summary\":\"$TOOL_GATE_OUT_DIR/summary.json\",\"ops16_summary\":\"$OPS16_OUT_DIR/summary.json\",\"kb40_summary\":\"$KB40_OUT_DIR/summary.json\",\"release_report_md\":\"$RELEASE_REPORT_MD\",\"strict_gate_passed\":$(python3 - <<'PY'
import json
import os
from pathlib import Path
path = Path(os.environ["RELEASE_SUMMARY_JSON"])
if not path.exists():
    print("false")
else:
    data = json.loads(path.read_text(encoding="utf-8"))
    print("true" if data.get("passed") else "false")
PY
)}"

log "pipeline done"
log "release summary: $RELEASE_SUMMARY_JSON"
log "release report: $RELEASE_REPORT_MD"

STRICT_GATE_PASSED="$(python3 - <<'PY'
import json
import os
from pathlib import Path
path = Path(os.environ["RELEASE_SUMMARY_JSON"])
if not path.exists():
    print("false")
else:
    data = json.loads(path.read_text(encoding="utf-8"))
    print("true" if data.get("passed") else "false")
PY
)"
if [[ "$STRICT_GATE_PASSED" != "true" ]]; then
  log "strict release gate failed"
  exit 2
fi
