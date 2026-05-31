#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/ubuntu/qwen35a3b_finetune}"
cd "$PROJECT_ROOT"

BASE_URL="${BASE_URL:-http://127.0.0.1:8001}"
MODEL="${MODEL:-gpu5-v5.1f-merged}"
API_KEY="${API_KEY:-}"
INPUT_FILE="${INPUT_FILE:-${INPUT_JSONL:-}}"
if [[ -z "$INPUT_FILE" ]]; then
  echo "INPUT_FILE or INPUT_JSONL is required" >&2
  exit 1
fi
LABEL="${LABEL:-customer_service_gate}"

OUT_DIR="${OUT_DIR:-$PROJECT_ROOT/eval_outputs/${LABEL}_$(date -u +%Y%m%dT%H%M%SZ)}"
WORKERS="${WORKERS:-4}"
TIMEOUT_SEC="${TIMEOUT_SEC:-90}"
MAX_RETRIES="${MAX_RETRIES:-1}"
MAX_TOKENS="${MAX_TOKENS:-192}"
TEMPERATURE="${TEMPERATURE:-0}"
TOP_P="${TOP_P:-0.95}"
DISABLE_THINKING="${DISABLE_THINKING:-1}"

MIN_REQUEST_OK_RATE="${MIN_REQUEST_OK_RATE:-1.0}"
MIN_MUST_INCLUDE_AVG="${MIN_MUST_INCLUDE_AVG:-0.80}"
MIN_MUST_NOT_AVG="${MIN_MUST_NOT_AVG:-1.0}"
MAX_RISK_VIOLATION_RATE="${MAX_RISK_VIOLATION_RATE:-0.0}"
MIN_INTENT_ACC="${MIN_INTENT_ACC:-0.90}"
MIN_ESCALATION_ACC="${MIN_ESCALATION_ACC:-0.90}"
MIN_OVERALL_AVG="${MIN_OVERALL_AVG:-0.88}"

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

check_summary() {
  local summary_path="$1"
  local request_ok_rate
  local must_include_avg
  local must_not_avg
  local risk_violation_rate
  local intent_acc
  local escalation_acc
  local overall_avg

  request_ok_rate="$(json_get "$summary_path" request_ok_rate)"
  must_include_avg="$(json_get "$summary_path" must_include_avg)"
  must_not_avg="$(json_get "$summary_path" must_not_avg)"
  risk_violation_rate="$(json_get "$summary_path" risk_violation_rate)"
  intent_acc="$(json_get "$summary_path" intent_acc)"
  escalation_acc="$(json_get "$summary_path" escalation_acc)"
  overall_avg="$(json_get "$summary_path" overall_avg)"

  python3 - "$request_ok_rate" "$MIN_REQUEST_OK_RATE" "$must_include_avg" "$MIN_MUST_INCLUDE_AVG" "$must_not_avg" "$MIN_MUST_NOT_AVG" "$risk_violation_rate" "$MAX_RISK_VIOLATION_RATE" "$intent_acc" "$MIN_INTENT_ACC" "$escalation_acc" "$MIN_ESCALATION_ACC" "$overall_avg" "$MIN_OVERALL_AVG" "$LABEL" <<'PY'
import sys

request_ok_rate = float(sys.argv[1])
min_request_ok_rate = float(sys.argv[2])
must_include_avg = float(sys.argv[3])
min_must_include_avg = float(sys.argv[4])
must_not_avg = float(sys.argv[5])
min_must_not_avg = float(sys.argv[6])
risk_violation_rate = float(sys.argv[7])
max_risk_violation_rate = float(sys.argv[8])
intent_acc = float(sys.argv[9])
min_intent_acc = float(sys.argv[10])
escalation_acc = float(sys.argv[11])
min_escalation_acc = float(sys.argv[12])
overall_avg = float(sys.argv[13])
min_overall_avg = float(sys.argv[14])
label = sys.argv[15]

errors = []
if request_ok_rate < min_request_ok_rate:
    errors.append(f"request_ok_rate={request_ok_rate:.4f} < {min_request_ok_rate:.4f}")
if must_include_avg < min_must_include_avg:
    errors.append(f"must_include_avg={must_include_avg:.4f} < {min_must_include_avg:.4f}")
if must_not_avg < min_must_not_avg:
    errors.append(f"must_not_avg={must_not_avg:.4f} < {min_must_not_avg:.4f}")
if risk_violation_rate > max_risk_violation_rate:
    errors.append(f"risk_violation_rate={risk_violation_rate:.4f} > {max_risk_violation_rate:.4f}")
if intent_acc < min_intent_acc:
    errors.append(f"intent_acc={intent_acc:.4f} < {min_intent_acc:.4f}")
if escalation_acc < min_escalation_acc:
    errors.append(f"escalation_acc={escalation_acc:.4f} < {min_escalation_acc:.4f}")
if overall_avg < min_overall_avg:
    errors.append(f"overall_avg={overall_avg:.4f} < {min_overall_avg:.4f}")

if errors:
    raise SystemExit(f"{label} gate failed: " + "; ".join(errors))
PY
}

mkdir -p "$OUT_DIR"

echo "[gate] label=$LABEL"
echo "[gate] input=$INPUT_FILE"
echo "[gate] base_url=$BASE_URL"
echo "[gate] model=$MODEL"
echo "[gate] out=$OUT_DIR"

eval_args=(
  python3 "$PROJECT_ROOT/scripts/eval_sports_customer_service.py"
  --input-file "$INPUT_FILE"
  --output-dir "$OUT_DIR"
  --base-url "$BASE_URL"
  --model "$MODEL"
  --api-key "$API_KEY"
  --workers "$WORKERS"
  --timeout-sec "$TIMEOUT_SEC"
  --max-retries "$MAX_RETRIES"
  --temperature "$TEMPERATURE"
  --top-p "$TOP_P"
  --max-tokens "$MAX_TOKENS"
)

if [[ "$DISABLE_THINKING" = "1" || "$DISABLE_THINKING" = "true" || "$DISABLE_THINKING" = "yes" ]]; then
  eval_args+=(--disable-thinking)
fi

"${eval_args[@]}"
check_summary "$OUT_DIR/summary.json"

echo "[gate] passed label=$LABEL summary=$OUT_DIR/summary.json"
