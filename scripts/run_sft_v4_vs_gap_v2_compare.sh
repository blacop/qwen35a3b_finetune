#!/bin/bash
set -euo pipefail

# ---- config (可用环境变量覆盖) ----
EVAL_SET="${EVAL_SET:-/home/ubuntu/qwen35a3b_finetune/datasets/eval_sports_customer_prod_500.jsonl}"
LIMIT="${LIMIT:-0}"                # 0 = 全跑
WORKERS="${WORKERS:-8}"
TS="$(date +%Y%m%dT%H%M%SZ)"
OUT_ROOT="${OUT_ROOT:-/home/ubuntu/qwen35a3b_finetune/eval_outputs/cmp_sft_v4_vs_gap_v2_${TS}}"

A_URL="http://127.0.0.1:8013"; A_MODEL="qwen35a3b-sft-v4-fix"; A_NAME="sft_v4_fix"
B_URL="http://127.0.0.1:8014"; B_MODEL="qwen35a3b-sft-gap-v2"; B_NAME="sft_gap_v2"

SCRIPT=/home/ubuntu/qwen35a3b_finetune/scripts/eval_sports_customer_service.py
COMPARE=/home/ubuntu/qwen35a3b_finetune/scripts/compare_sft_models.py

A_DIR="$OUT_ROOT/$A_NAME"
B_DIR="$OUT_ROOT/$B_NAME"
mkdir -p "$A_DIR" "$B_DIR"

echo "[$(date -u +%FT%TZ)] dataset=$EVAL_SET ($(wc -l < "$EVAL_SET") samples, LIMIT=$LIMIT, WORKERS=$WORKERS)"
echo "[$(date -u +%FT%TZ)] out_root=$OUT_ROOT"

# 预检：两个 endpoint 都活着
for url in "$A_URL" "$B_URL"; do
  code=$(curl -s -o /dev/null -w "%{http_code}" "$url/health" 2>/dev/null || echo 000)
  if [ "$code" != "200" ]; then
    echo "[FATAL] $url/health = $code, endpoint not ready" >&2
    exit 1
  fi
done
echo "[OK] both endpoints healthy"

run_one() {
  local name="$1" url="$2" model="$3" dir="$4"
  "$SCRIPT" \
    --input-jsonl "$EVAL_SET" \
    --output-dir "$dir" \
    --base-url "$url" \
    --model "$model" \
    --workers "$WORKERS" \
    --timeout-sec 120 \
    --max-retries 2 ${USE_GUARDRAILS:+--use-guardrails} \
    --limit "$LIMIT" \
    > "$dir/run.log" 2>&1
}

# 并发跑
run_one "$A_NAME" "$A_URL" "$A_MODEL" "$A_DIR" &
PID_A=$!
run_one "$B_NAME" "$B_URL" "$B_MODEL" "$B_DIR" &
PID_B=$!
echo "[$(date -u +%FT%TZ)] started: $A_NAME pid=$PID_A, $B_NAME pid=$PID_B"

# 监控进度
while kill -0 $PID_A 2>/dev/null || kill -0 $PID_B 2>/dev/null; do
  sleep 30
  a_cnt=$(wc -l < "$A_DIR/predictions.jsonl" 2>/dev/null || echo 0)
  b_cnt=$(wc -l < "$B_DIR/predictions.jsonl" 2>/dev/null || echo 0)
  a_alive=$(kill -0 $PID_A 2>/dev/null && echo "alive" || echo "done")
  b_alive=$(kill -0 $PID_B 2>/dev/null && echo "alive" || echo "done")
  echo "[$(date -u +%FT%TZ)] $A_NAME=$a_cnt ($a_alive)  $B_NAME=$b_cnt ($b_alive)"
done

wait $PID_A || { echo "[ERR] $A_NAME failed, see $A_DIR/run.log"; tail -n 30 "$A_DIR/run.log"; exit 2; }
wait $PID_B || { echo "[ERR] $B_NAME failed, see $B_DIR/run.log"; tail -n 30 "$B_DIR/run.log"; exit 2; }

echo "[$(date -u +%FT%TZ)] both eval runs done, generating report..."

/usr/bin/python3 "$COMPARE" \
  --a-dir "$A_DIR" --a-name "$A_NAME" \
  --b-dir "$B_DIR" --b-name "$B_NAME" \
  --output "$OUT_ROOT/REPORT.md" \
  --bad-case-limit 15

echo ""
echo "=================== DONE ==================="
echo " Report     : $OUT_ROOT/REPORT.md"
echo " Diff CSV   : $OUT_ROOT/diff_cases.csv"
echo " A outputs  : $A_DIR/{summary.json,predictions.jsonl,failure_cases.csv}"
echo " B outputs  : $B_DIR/{summary.json,predictions.jsonl,failure_cases.csv}"
echo "============================================"
