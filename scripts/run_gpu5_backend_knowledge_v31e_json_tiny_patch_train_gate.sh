#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/ubuntu/qwen35a3b_finetune"
cd "$PROJECT_ROOT"

log() {
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"
}

TS="${TS:-$(date -u +%Y%m%dT%H%M%SZ)}"
BASE_V31D_MODEL_DIR="${BASE_V31D_MODEL_DIR:-$PROJECT_ROOT/outputs/swift_sft_sports_rule_hotfix_v31d_mixed_gpu5_20260430T035743Z_merged_20260430T054414Z_vllm_v2_fixed}"

if [[ ! -d "$BASE_V31D_MODEL_DIR" ]]; then
  log "missing v31d base model dir: $BASE_V31D_MODEL_DIR"
  exit 1
fi

PACK_DIR="${PACK_DIR:-$PROJECT_ROOT/datasets/backend_knowledge_v31e_json_tiny_patch_${TS}}"
DATA_FILE="${DATA_FILE:-$PACK_DIR/sft_backend_knowledge_v31e_json_tiny_patch.jsonl}"
SUMMARY_JSON="${SUMMARY_JSON:-$PACK_DIR/summary.json}"

BACKEND_EVAL_JSONL="${BACKEND_EVAL_JSONL:-$PROJECT_ROOT/datasets/eval_sports_baowang_knowledge_40_20260428.jsonl}"
RULE_EVAL_JSONL="${RULE_EVAL_JSONL:-$PROJECT_ROOT/datasets/sports_rule_knowledge_regression_gpu5_20260428.jsonl}"

RUN_NAME_PREFIX="${RUN_NAME_PREFIX:-swift_sft_backend_knowledge_v31e_json_tiny_patch_gpu5}"
RUN_NAME="${RUN_NAME:-${RUN_NAME_PREFIX}_${TS}}"
EVAL_OUT_DIR="${EVAL_OUT_DIR:-$PROJECT_ROOT/eval_outputs/${RUN_NAME}_backend_eval_${TS}}"
RULE_EVAL_OUT_DIR="${RULE_EVAL_OUT_DIR:-$PROJECT_ROOT/eval_outputs/${RUN_NAME}_rule18_eval_${TS}}"
RULE_CHECK_OUT_DIR="${RULE_CHECK_OUT_DIR:-$RULE_EVAL_OUT_DIR/rule_check}"
LOG_FILE="${LOG_FILE:-$PROJECT_ROOT/tracking/logs/${RUN_NAME}_train_gate_${TS}.log}"

GATE_MIN_BACKEND_OVERALL="${GATE_MIN_BACKEND_OVERALL:-0.93}"
GATE_MIN_RULE_PASS_RATE="${GATE_MIN_RULE_PASS_RATE:-0.94}"
TEMP_PORT="${TEMP_PORT:-8027}"
PROD_VLLM_SERVICE="${PROD_VLLM_SERVICE:-vllm-qwen35-gpu5-agent-upstream.service}"
EVAL_MODEL_NAME="${EVAL_MODEL_NAME:-qwen35a3b-domain-text-vl-gpu5-v31e-backend-json-candidate}"
DEFAULT_SYSTEM_PROMPT="你是体育包网智能客服。必须遵守合规与风控规则：遇到注单取消/作废/异常、结算争议、赔率异常、限红风控等无法直接确认的情况，必须明确告知需要后台查询或联系平台运营后回复；禁止赌博诱导、代理拉新、洗钱跑分、伪造证件、低龄相关内容。"

mkdir -p "$(dirname "$LOG_FILE")"
exec >>"$LOG_FILE" 2>&1

json_get() {
  local path="$1"
  local expr="$2"
  python3 - "$path" "$expr" <<'PY'
import json
import sys
from pathlib import Path

cur = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
for part in sys.argv[2].split("."):
    if part:
        cur = cur[part]
print(cur)
PY
}

log "backend knowledge v31e JSON tiny patch train+gate start"
log "base v31d model: $BASE_V31D_MODEL_DIR"
log "data file: $DATA_FILE"
log "backend eval: $BACKEND_EVAL_JSONL"
log "rule eval: $RULE_EVAL_JSONL"
log "eval out: $EVAL_OUT_DIR"
log "rule eval out: $RULE_EVAL_OUT_DIR"
log "log file: $LOG_FILE"

python3 "$PROJECT_ROOT/scripts/build_backend_knowledge_v31e_json_tiny_patch.py" \
  --output-jsonl "$DATA_FILE" \
  --summary-json "$SUMMARY_JSON" \
  --repeat "${PATCH_REPEAT:-4}"

BASE_MODEL="$BASE_V31D_MODEL_DIR" \
DATA_FILE="$DATA_FILE" \
EVAL_INPUT_JSONL="$BACKEND_EVAL_JSONL" \
RUN_NAME="$RUN_NAME" \
RUN_NAME_PREFIX="$RUN_NAME_PREFIX" \
EVAL_OUT_DIR="$EVAL_OUT_DIR" \
PROD_VLLM_SERVICE="$PROD_VLLM_SERVICE" \
TEMP_PORT="$TEMP_PORT" \
TEMP_GPU_MEMORY_UTILIZATION="${TEMP_GPU_MEMORY_UTILIZATION:-0.84}" \
EVAL_MODEL_NAME="$EVAL_MODEL_NAME" \
EVAL_WORKERS="${EVAL_WORKERS:-3}" \
EVAL_MAX_TOKENS="${EVAL_MAX_TOKENS:-512}" \
EVAL_DISABLE_THINKING=1 \
EPOCHS="${EPOCHS:-1}" \
GRAD_ACC="${GRAD_ACC:-8}" \
LR="${LR:-1e-5}" \
LORA_RANK="${LORA_RANK:-8}" \
LORA_ALPHA="${LORA_ALPHA:-16}" \
SAVE_STEPS="${SAVE_STEPS:-10}" \
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-2}" \
SYSTEM_PROMPT="${SYSTEM_PROMPT:-$DEFAULT_SYSTEM_PROMPT}" \
"$PROJECT_ROOT/scripts/run_nohallucination_hotfix_sft_eval.sh"

MERGED_OUT_DIR="$PROJECT_ROOT/outputs/${RUN_NAME}_merged_${TS}_vllm"
if [[ ! -d "$MERGED_OUT_DIR" ]]; then
  MERGED_OUT_DIR="$(find "$PROJECT_ROOT/outputs" -maxdepth 1 -type d -name "${RUN_NAME}_merged_*_vllm" | sort | tail -n1)"
fi
if [[ -z "${MERGED_OUT_DIR:-}" || ! -d "$MERGED_OUT_DIR" ]]; then
  log "missing merged output dir for candidate"
  exit 1
fi

backend_overall="$(json_get "$EVAL_OUT_DIR/summary.json" overall_avg)"
log "backend knowledge gate: overall=$backend_overall min=$GATE_MIN_BACKEND_OVERALL"
python3 - "$backend_overall" "$GATE_MIN_BACKEND_OVERALL" <<'PY'
import sys

actual = float(sys.argv[1])
minimum = float(sys.argv[2])
if actual < minimum:
    raise SystemExit(f"backend gate failed: overall={actual:.4f} < {minimum:.4f}")
PY

TEMP_VLLM_LOG_RULE="${TEMP_VLLM_LOG_RULE:-$PROJECT_ROOT/tracking/logs/${RUN_NAME}_rule_temp_vllm_${TS}.log}"
RULE_TEMP_PID=""

cleanup_rule_temp() {
  if [[ -n "${RULE_TEMP_PID:-}" ]]; then
    kill "${RULE_TEMP_PID}" 2>/dev/null || true
    wait "${RULE_TEMP_PID}" 2>/dev/null || true
  fi
  systemctl --user start "$PROD_VLLM_SERVICE" || true
}
trap cleanup_rule_temp EXIT

if systemctl --user is-active --quiet "$PROD_VLLM_SERVICE"; then
  log "stopping production GPU5 vLLM service for rule regression: $PROD_VLLM_SERVICE"
  systemctl --user stop "$PROD_VLLM_SERVICE"
fi

log "starting temp candidate vLLM for rule regression on 127.0.0.1:${TEMP_PORT}"
/home/ubuntu/.local/bin/vllm serve "$MERGED_OUT_DIR" \
  --host 127.0.0.1 \
  --port "$TEMP_PORT" \
  --tensor-parallel-size 1 \
  --max-model-len 4096 \
  --dtype bfloat16 \
  --gpu-memory-utilization "${TEMP_GPU_MEMORY_UTILIZATION:-0.84}" \
  --served-model-name "$EVAL_MODEL_NAME" \
  --enforce-eager \
  --trust-remote-code \
  >>"$TEMP_VLLM_LOG_RULE" 2>&1 &
RULE_TEMP_PID=$!

READY=0
for _ in $(seq 1 120); do
  if curl -fsS "http://127.0.0.1:${TEMP_PORT}/v1/models" >/dev/null 2>&1; then
    READY=1
    break
  fi
  sleep 2
done
if [[ "$READY" != "1" ]]; then
  log "rule temp vLLM not ready, see $TEMP_VLLM_LOG_RULE"
  exit 1
fi

log "starting rule regression eval against candidate on port ${TEMP_PORT}"
python3 "$PROJECT_ROOT/scripts/eval_sports_customer_service.py" \
  --input-jsonl "$RULE_EVAL_JSONL" \
  --output-dir "$RULE_EVAL_OUT_DIR" \
  --base-url "http://127.0.0.1:${TEMP_PORT}" \
  --model "$EVAL_MODEL_NAME" \
  --temperature 0.0 \
  --top-p 0.95 \
  --max-tokens 320 \
  --workers "${RULE_EVAL_WORKERS:-3}" \
  --disable-thinking

python3 "$PROJECT_ROOT/scripts/check_sports_rule_knowledge.py" \
  --eval-jsonl "$RULE_EVAL_JSONL" \
  --predictions "$RULE_EVAL_OUT_DIR/predictions.csv" \
  --output-dir "$RULE_CHECK_OUT_DIR"

rule_pass_rate="$(json_get "$RULE_CHECK_OUT_DIR/summary.json" pass_rate)"
rule_failed_checks="$(json_get "$RULE_CHECK_OUT_DIR/summary.json" failed_checks)"
log "rule gate: pass_rate=$rule_pass_rate failed_checks=$rule_failed_checks min=$GATE_MIN_RULE_PASS_RATE"
python3 - "$rule_pass_rate" "$GATE_MIN_RULE_PASS_RATE" <<'PY'
import sys

actual = float(sys.argv[1])
minimum = float(sys.argv[2])
if actual < minimum:
    raise SystemExit(f"rule gate failed: pass_rate={actual:.4f} < {minimum:.4f}")
PY

log "backend knowledge v31e JSON tiny patch gates passed"
log "merged model: $MERGED_OUT_DIR"
log "backend eval: $EVAL_OUT_DIR"
log "rule eval: $RULE_EVAL_OUT_DIR"
log "rule check: $RULE_CHECK_OUT_DIR"
