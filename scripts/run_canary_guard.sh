#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/ubuntu/qwen35a3b_finetune"
cd "$PROJECT_ROOT"

PYTHON_BIN="${PYTHON_BIN:-/usr/bin/python3}"
INPUT_JSONL="${INPUT_JSONL:-$PROJECT_ROOT/datasets/eval_sports_customer_canary_120.jsonl}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/eval_outputs/canary_guard}"
STATE_DIR="${STATE_DIR:-$PROJECT_ROOT/runtime/canary_guard}"

BASE_URL="${BASE_URL:-http://10.128.203.5}"
MODEL="${MODEL:-qwen35a3b-dpo-latest}"
API_KEY="${API_KEY:-}"
if [[ -z "$API_KEY" && -f "$PROJECT_ROOT/services/vllm-gpu5-dpo.env" ]]; then
  API_KEY="$(sed -n 's/^API_KEY=//p' "$PROJECT_ROOT/services/vllm-gpu5-dpo.env" | head -n1 || true)"
fi
WORKERS="${WORKERS:-6}"
TIMEOUT_SEC="${TIMEOUT_SEC:-60}"
MAX_RETRIES="${MAX_RETRIES:-1}"
MAX_TOKENS="${MAX_TOKENS:-192}"
HTTP_BACKEND="${HTTP_BACKEND:-curl}"
LIMIT="${LIMIT:-0}"

REQUEST_OK_MIN="${REQUEST_OK_MIN:-0.95}"
P95_MAX_MS="${P95_MAX_MS:-15000}"
INTENT_THRESHOLDS="${INTENT_THRESHOLDS:-串关规则=0.5,滚球延迟=0.5,赔率异常=0.5,赛事变更=0.5}"
WINDOW_MIN="${WINDOW_MIN:-10}"
MIN_POINTS="${MIN_POINTS:-3}"
INTERVAL_SEC="${INTERVAL_SEC:-120}"
ROLLBACK_COOLDOWN_MIN="${ROLLBACK_COOLDOWN_MIN:-30}"
ONCE="${ONCE:-0}"

ROLLBACK_MODEL_DIR="${ROLLBACK_MODEL_DIR:-}"
ROLLBACK_COMMAND="${ROLLBACK_COMMAND:-}"
if [[ -z "$ROLLBACK_COMMAND" && -n "$ROLLBACK_MODEL_DIR" ]]; then
  ROLLBACK_COMMAND="bash $PROJECT_ROOT/scripts/rollback_vllm_model.sh --target-model-dir \"$ROLLBACK_MODEL_DIR\" --reason canary_auto_rollback"
fi

mkdir -p "$OUTPUT_ROOT" "$STATE_DIR"

ARGS=(
  --python-bin "$PYTHON_BIN"
  --eval-script "$PROJECT_ROOT/scripts/eval_sports_customer_service.py"
  --input-jsonl "$INPUT_JSONL"
  --output-root "$OUTPUT_ROOT"
  --state-dir "$STATE_DIR"
  --base-url "$BASE_URL"
  --model "$MODEL"
  --workers "$WORKERS"
  --timeout-sec "$TIMEOUT_SEC"
  --max-retries "$MAX_RETRIES"
  --max-tokens "$MAX_TOKENS"
  --http-backend "$HTTP_BACKEND"
  --limit "$LIMIT"
  --request-ok-min "$REQUEST_OK_MIN"
  --p95-max-ms "$P95_MAX_MS"
  --intent-thresholds "$INTENT_THRESHOLDS"
  --window-min "$WINDOW_MIN"
  --min-points "$MIN_POINTS"
  --interval-sec "$INTERVAL_SEC"
  --rollback-cooldown-min "$ROLLBACK_COOLDOWN_MIN"
)

if [[ -n "$API_KEY" ]]; then
  ARGS+=(--api-key "$API_KEY")
fi
if [[ -n "$ROLLBACK_COMMAND" ]]; then
  ARGS+=(--rollback-command "$ROLLBACK_COMMAND")
fi
if [[ "$ONCE" == "1" || "$ONCE" == "true" || "$ONCE" == "yes" ]]; then
  ARGS+=(--once)
fi

exec "$PYTHON_BIN" "$PROJECT_ROOT/scripts/monitor_canary_and_rollback.py" "${ARGS[@]}"
