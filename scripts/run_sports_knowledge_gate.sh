#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/ubuntu/qwen35a3b_finetune"
cd "$PROJECT_ROOT"

BASE_URL_GPU5="${BASE_URL_GPU5:-http://127.0.0.1:8001}"
MODEL_GPU5="${MODEL_GPU5:-gpu5-v5.1f-merged}"
BASE_URL_GPU7="${BASE_URL_GPU7:-http://127.0.0.1:8010}"
MODEL_GPU7="${MODEL_GPU7:-gpu7-v5-combined}"
API_KEY="${API_KEY:-}"

EVAL_20260428_JSONL="${EVAL_20260428_JSONL:-$PROJECT_ROOT/datasets/eval_sports_baowang_knowledge_40_20260428.jsonl}"
EVAL_20260506_JSONL="${EVAL_20260506_JSONL:-$PROJECT_ROOT/datasets/eval_sports_baowang_knowledge_regression_20260506.jsonl}"
EVAL_GPU7_OPS_20260509_JSONL="${EVAL_GPU7_OPS_20260509_JSONL:-$PROJECT_ROOT/datasets/eval_gpu7_ops_feedback_regression_8_20260509.jsonl}"

OUT_DIR="${OUT_DIR:-$PROJECT_ROOT/eval_outputs/sports_knowledge_gate_$(date -u +%Y%m%dT%H%M%SZ)}"
WORKERS="${WORKERS:-4}"
TIMEOUT_SEC="${TIMEOUT_SEC:-90}"
MAX_RETRIES="${MAX_RETRIES:-1}"
MAX_TOKENS="${MAX_TOKENS:-192}"
MIN_REQUEST_OK_RATE_20260506="${MIN_REQUEST_OK_RATE_20260506:-1.0}"
MIN_MUST_INCLUDE_AVG_20260506="${MIN_MUST_INCLUDE_AVG_20260506:-1.0}"
MIN_MUST_NOT_AVG_20260506="${MIN_MUST_NOT_AVG_20260506:-1.0}"
MAX_RISK_VIOLATION_RATE_20260506="${MAX_RISK_VIOLATION_RATE_20260506:-0.0}"
MIN_INTENT_ACC_20260506="${MIN_INTENT_ACC_20260506:-0.90}"
MIN_ESCALATION_ACC_20260506="${MIN_ESCALATION_ACC_20260506:-0.95}"
MIN_OVERALL_AVG_20260506="${MIN_OVERALL_AVG_20260506:-0.95}"

MIN_REQUEST_OK_RATE_20260428="${MIN_REQUEST_OK_RATE_20260428:-1.0}"
MIN_MUST_INCLUDE_AVG_20260428="${MIN_MUST_INCLUDE_AVG_20260428:-0.60}"
MIN_MUST_NOT_AVG_20260428="${MIN_MUST_NOT_AVG_20260428:-1.0}"
MAX_RISK_VIOLATION_RATE_20260428="${MAX_RISK_VIOLATION_RATE_20260428:-0.0}"
MIN_INTENT_ACC_20260428="${MIN_INTENT_ACC_20260428:-0.75}"
MIN_ESCALATION_ACC_20260428="${MIN_ESCALATION_ACC_20260428:-0.90}"
MIN_OVERALL_AVG_20260428="${MIN_OVERALL_AVG_20260428:-0.80}"

MIN_REQUEST_OK_RATE_GPU7_OPS_20260509="${MIN_REQUEST_OK_RATE_GPU7_OPS_20260509:-1.0}"
MIN_MUST_INCLUDE_AVG_GPU7_OPS_20260509="${MIN_MUST_INCLUDE_AVG_GPU7_OPS_20260509:-0.95}"
MIN_MUST_NOT_AVG_GPU7_OPS_20260509="${MIN_MUST_NOT_AVG_GPU7_OPS_20260509:-1.0}"
MAX_RISK_VIOLATION_RATE_GPU7_OPS_20260509="${MAX_RISK_VIOLATION_RATE_GPU7_OPS_20260509:-0.0}"
MIN_INTENT_ACC_GPU7_OPS_20260509="${MIN_INTENT_ACC_GPU7_OPS_20260509:-1.0}"
MIN_ESCALATION_ACC_GPU7_OPS_20260509="${MIN_ESCALATION_ACC_GPU7_OPS_20260509:-1.0}"
MIN_OVERALL_AVG_GPU7_OPS_20260509="${MIN_OVERALL_AVG_GPU7_OPS_20260509:-0.97}"

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

run_eval() {
  local input_jsonl="$1"
  local eval_name="$2"
  local base_url="$3"
  local model="$4"
  local eval_out="$OUT_DIR/$eval_name"
  mkdir -p "$eval_out"
  echo "[gate] eval=$eval_name base_url=$base_url model=$model input=$input_jsonl out=$eval_out" >&2
  python3 "$PROJECT_ROOT/scripts/eval_sports_customer_service.py" \
    --input-jsonl "$input_jsonl" \
    --output-dir "$eval_out" \
    --base-url "$base_url" \
    --model "$model" \
    --api-key "$API_KEY" \
    --workers "$WORKERS" \
    --timeout-sec "$TIMEOUT_SEC" \
    --max-retries "$MAX_RETRIES" \
    --temperature 0 \
    --max-tokens "$MAX_TOKENS" \
    >&2
  echo "$eval_out"
}

check_summary() {
  local summary_path="$1"
  local label="$2"
  local min_request_ok_rate="$3"
  local min_must_include_avg="$4"
  local min_must_not_avg="$5"
  local max_risk_violation_rate="$6"
  local min_intent_acc="$7"
  local min_escalation_acc="$8"
  local min_overall_avg="$9"
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

  python3 - "$request_ok_rate" "$min_request_ok_rate" "$must_include_avg" "$min_must_include_avg" "$must_not_avg" "$min_must_not_avg" "$risk_violation_rate" "$max_risk_violation_rate" "$intent_acc" "$min_intent_acc" "$escalation_acc" "$min_escalation_acc" "$overall_avg" "$min_overall_avg" "$label" <<'PY'
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

echo "[gate] base_url_gpu5=$BASE_URL_GPU5"
echo "[gate] model_gpu5=$MODEL_GPU5"
echo "[gate] base_url_gpu7=$BASE_URL_GPU7"
echo "[gate] model_gpu7=$MODEL_GPU7"
echo "[gate] out=$OUT_DIR"

python3 -m pytest \
  tests/test_dpo_guardrails_knowledge_bypass.py \
  tests/test_dpo_guardrails_proxy_api_governance.py \
  tests/test_external_api_gateway.py \
  tests/test_sync_open_webui_model_system_prompts.py \
  tests/test_sports_baowang_knowledge_regression.py \
  tests/test_run_sports_knowledge_gate.py \
  -q

gpu5_out="$(run_eval "$EVAL_20260506_JSONL" gpu5_20260506 "$BASE_URL_GPU5" "$MODEL_GPU5")"
gpu7_out="$(run_eval "$EVAL_20260506_JSONL" gpu7_20260506 "$BASE_URL_GPU7" "$MODEL_GPU7")"
kb40_out="$(run_eval "$EVAL_20260428_JSONL" kb40_20260428 "$BASE_URL_GPU5" "$MODEL_GPU5")"
gpu7_ops8_out="$(run_eval "$EVAL_GPU7_OPS_20260509_JSONL" gpu7_ops_feedback_8_20260509 "$BASE_URL_GPU7" "$MODEL_GPU7")"

check_summary "$gpu5_out/summary.json" "gpu5_20260506" \
  "$MIN_REQUEST_OK_RATE_20260506" "$MIN_MUST_INCLUDE_AVG_20260506" "$MIN_MUST_NOT_AVG_20260506" \
  "$MAX_RISK_VIOLATION_RATE_20260506" "$MIN_INTENT_ACC_20260506" "$MIN_ESCALATION_ACC_20260506" "$MIN_OVERALL_AVG_20260506"
check_summary "$gpu7_out/summary.json" "gpu7_20260506" \
  "$MIN_REQUEST_OK_RATE_20260506" "$MIN_MUST_INCLUDE_AVG_20260506" "$MIN_MUST_NOT_AVG_20260506" \
  "$MAX_RISK_VIOLATION_RATE_20260506" "$MIN_INTENT_ACC_20260506" "$MIN_ESCALATION_ACC_20260506" "$MIN_OVERALL_AVG_20260506"
check_summary "$kb40_out/summary.json" "kb40_20260428" \
  "$MIN_REQUEST_OK_RATE_20260428" "$MIN_MUST_INCLUDE_AVG_20260428" "$MIN_MUST_NOT_AVG_20260428" \
  "$MAX_RISK_VIOLATION_RATE_20260428" "$MIN_INTENT_ACC_20260428" "$MIN_ESCALATION_ACC_20260428" "$MIN_OVERALL_AVG_20260428"
check_summary "$gpu7_ops8_out/summary.json" "gpu7_ops_feedback_8_20260509" \
  "$MIN_REQUEST_OK_RATE_GPU7_OPS_20260509" "$MIN_MUST_INCLUDE_AVG_GPU7_OPS_20260509" "$MIN_MUST_NOT_AVG_GPU7_OPS_20260509" \
  "$MAX_RISK_VIOLATION_RATE_GPU7_OPS_20260509" "$MIN_INTENT_ACC_GPU7_OPS_20260509" "$MIN_ESCALATION_ACC_GPU7_OPS_20260509" "$MIN_OVERALL_AVG_GPU7_OPS_20260509"

echo "[gate] passed"
