#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/ubuntu/qwen35a3b_finetune"
cd "$PROJECT_ROOT"

log() {
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"
}

read_env_value() {
  local file="$1"
  local key="$2"
  awk -F'=' -v k="$key" '$1==k {print substr($0, index($0, "=")+1)}' "$file" | tail -n1
}

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

VLLM_ENV_FILE="${VLLM_ENV_FILE:-$PROJECT_ROOT/services/vllm-gpu5-agent-upstream.env}"
CURRENT_GPU5_MODEL_DIR="${CURRENT_GPU5_MODEL_DIR:-$(read_env_value "$VLLM_ENV_FILE" MODEL_DIR || true)}"
if [[ -z "${CURRENT_GPU5_MODEL_DIR:-}" || ! -d "$CURRENT_GPU5_MODEL_DIR" ]]; then
  log "failed to resolve current GPU5 model dir from $VLLM_ENV_FILE"
  exit 1
fi

TS="${TS:-$(date -u +%Y%m%dT%H%M%SZ)}"
PACK_DIR="${PACK_DIR:-$PROJECT_ROOT/datasets/sports_rule_hotfix_pack_20260428}"
DATA_FILE="${DATA_FILE:-$PACK_DIR/sft_sports_rule_hotfix_merged.jsonl}"
SUMMARY_JSON="${SUMMARY_JSON:-$PACK_DIR/summary.json}"
RULE_EVAL_JSONL="${RULE_EVAL_JSONL:-$PROJECT_ROOT/datasets/sports_rule_knowledge_regression_gpu5_20260428.jsonl}"
RULE_REPEAT="${RULE_REPEAT:-40}"
GATE_MIN_PASS_RATE="${GATE_MIN_PASS_RATE:-0.90}"

RUN_NAME_PREFIX="${RUN_NAME_PREFIX:-swift_sft_sports_rule_hotfix_gpu5}"
RUN_NAME="${RUN_NAME:-${RUN_NAME_PREFIX}_${TS}}"
EVAL_OUT_DIR="${EVAL_OUT_DIR:-$PROJECT_ROOT/eval_outputs/${RUN_NAME}_eval_${TS}}"
RULE_CHECK_OUT_DIR="${RULE_CHECK_OUT_DIR:-$EVAL_OUT_DIR/rule_check}"
LOG_FILE="${LOG_FILE:-$PROJECT_ROOT/tracking/logs/${RUN_NAME}_train_gate_${TS}.log}"

mkdir -p "$(dirname "$LOG_FILE")"
exec >>"$LOG_FILE" 2>&1

log "sports rule hotfix train+gate start"
log "current GPU5 base model: $CURRENT_GPU5_MODEL_DIR"
log "pack data file: $DATA_FILE"
log "rule eval: $RULE_EVAL_JSONL"
log "eval out: $EVAL_OUT_DIR"
log "log file: $LOG_FILE"

python3 "$PROJECT_ROOT/scripts/build_sports_rule_hotfix_sft_pack.py" \
  --rule-repeat "$RULE_REPEAT" \
  --output-jsonl "$DATA_FILE" \
  --summary-json "$SUMMARY_JSON"

BASE_MODEL="$CURRENT_GPU5_MODEL_DIR" \
DATA_FILE="$DATA_FILE" \
EVAL_INPUT_JSONL="$RULE_EVAL_JSONL" \
RUN_NAME="$RUN_NAME" \
RUN_NAME_PREFIX="$RUN_NAME_PREFIX" \
EVAL_OUT_DIR="$EVAL_OUT_DIR" \
PROD_VLLM_SERVICE="${PROD_VLLM_SERVICE:-vllm-qwen35-gpu5-agent-upstream.service}" \
TEMP_PORT="${TEMP_PORT:-8027}" \
TEMP_GPU_MEMORY_UTILIZATION="${TEMP_GPU_MEMORY_UTILIZATION:-0.84}" \
EVAL_MODEL_NAME="${EVAL_MODEL_NAME:-qwen35a3b-domain-text-vl-gpu5-rule-hotfix-candidate}" \
EVAL_WORKERS="${EVAL_WORKERS:-3}" \
EPOCHS="${EPOCHS:-1}" \
GRAD_ACC="${GRAD_ACC:-8}" \
SAVE_STEPS="${SAVE_STEPS:-25}" \
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-2}" \
"$PROJECT_ROOT/scripts/run_nohallucination_hotfix_sft_eval.sh"

python3 "$PROJECT_ROOT/scripts/check_sports_rule_knowledge.py" \
  --eval-jsonl "$RULE_EVAL_JSONL" \
  --predictions "$EVAL_OUT_DIR/predictions.csv" \
  --output-dir "$RULE_CHECK_OUT_DIR"

RULE_SUMMARY="$RULE_CHECK_OUT_DIR/summary.json"
if [[ ! -f "$RULE_SUMMARY" ]]; then
  log "missing rule check summary: $RULE_SUMMARY"
  exit 1
fi

pass_rate="$(json_get "$RULE_SUMMARY" pass_rate)"
failed_checks="$(json_get "$RULE_SUMMARY" failed_checks)"
log "rule gate: pass_rate=$pass_rate failed_checks=$failed_checks min=$GATE_MIN_PASS_RATE"

python3 - "$pass_rate" "$GATE_MIN_PASS_RATE" <<'PY'
import sys
actual = float(sys.argv[1])
minimum = float(sys.argv[2])
if actual < minimum:
    raise SystemExit(f"rule gate failed: pass_rate={actual:.4f} < {minimum:.4f}")
PY

log "rule gate passed"
log "candidate eval: $EVAL_OUT_DIR"
log "rule check: $RULE_CHECK_OUT_DIR"
