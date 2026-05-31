#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/ubuntu/qwen35a3b_finetune"
cd "$PROJECT_ROOT"

BASE_URL="${BASE_URL:-http://127.0.0.1:8001}"
MODEL="${MODEL:-qwen35a3b-domain-text-vl-gpu5}"
API_KEY="${API_KEY:-}"
RULE_EVAL_JSONL="${RULE_EVAL_JSONL:-$PROJECT_ROOT/datasets/sports_rule_knowledge_regression_gpu5_20260428.jsonl}"
OUT_DIR="${OUT_DIR:-$PROJECT_ROOT/eval_outputs/sports_rule_gate_$(date -u +%Y%m%dT%H%M%SZ)}"
GATE_MIN_PASS_RATE="${GATE_MIN_PASS_RATE:-0.90}"
WORKERS="${WORKERS:-3}"
MAX_TOKENS="${MAX_TOKENS:-256}"
TIMEOUT_SEC="${TIMEOUT_SEC:-90}"

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

mkdir -p "$OUT_DIR"
echo "[gate] base_url=$BASE_URL"
echo "[gate] model=$MODEL"
echo "[gate] eval=$RULE_EVAL_JSONL"
echo "[gate] out=$OUT_DIR"

python3 "$PROJECT_ROOT/scripts/eval_sports_customer_service.py" \
  --input-jsonl "$RULE_EVAL_JSONL" \
  --output-dir "$OUT_DIR" \
  --base-url "$BASE_URL" \
  --model "$MODEL" \
  --api-key "$API_KEY" \
  --workers "$WORKERS" \
  --timeout-sec "$TIMEOUT_SEC" \
  --max-retries 1 \
  --temperature 0 \
  --max-tokens "$MAX_TOKENS"

python3 "$PROJECT_ROOT/scripts/check_sports_rule_knowledge.py" \
  --eval-jsonl "$RULE_EVAL_JSONL" \
  --predictions "$OUT_DIR/predictions.csv" \
  --output-dir "$OUT_DIR/rule_check"

PASS_RATE="$(json_get "$OUT_DIR/rule_check/summary.json" pass_rate)"
FAILED_CHECKS="$(json_get "$OUT_DIR/rule_check/summary.json" failed_checks)"
echo "[gate] pass_rate=$PASS_RATE failed_checks=$FAILED_CHECKS min=$GATE_MIN_PASS_RATE"

python3 - "$PASS_RATE" "$GATE_MIN_PASS_RATE" <<'PY'
import sys
actual = float(sys.argv[1])
minimum = float(sys.argv[2])
if actual < minimum:
    raise SystemExit(f"rule gate failed: pass_rate={actual:.4f} < {minimum:.4f}")
PY

echo "[gate] passed"
