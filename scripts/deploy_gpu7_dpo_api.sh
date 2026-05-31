#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/ubuntu/qwen35a3b_finetune"
MODEL_DIR="${MODEL_DIR:-$PROJECT_ROOT/outputs/swift_dpo_clean_v2_gpu5_merged_20260413_vllm}"
PORT="${PORT:-8010}"
HOST="${HOST:-127.0.0.1}"
GPU_ID="${GPU_ID:-7}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen35a3b-dpo-latest}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.92}"
VLLM_BIN="${VLLM_BIN:-/home/ubuntu/.local/bin/vllm}"
RUNTIME_DIR="${RUNTIME_DIR:-$PROJECT_ROOT/runtime/gpu7_dpo_api}"
API_KEY_FILE="${API_KEY_FILE:-$RUNTIME_DIR/api_key.txt}"
PID_FILE="${PID_FILE:-$RUNTIME_DIR/vllm.pid}"
LOG_FILE="${LOG_FILE:-$RUNTIME_DIR/vllm.log}"

mkdir -p "$RUNTIME_DIR"

if [ ! -d "$MODEL_DIR" ]; then
  echo "[ERROR] MODEL_DIR not found: $MODEL_DIR" >&2
  exit 1
fi

if [ ! -f "$API_KEY_FILE" ]; then
  python3 - <<'PY' > "$API_KEY_FILE"
import secrets
print("sk-" + secrets.token_urlsafe(36))
PY
  chmod 600 "$API_KEY_FILE"
fi

API_KEY="$(cat "$API_KEY_FILE")"
if [ -z "$API_KEY" ]; then
  echo "[ERROR] Empty API key in $API_KEY_FILE" >&2
  exit 1
fi

if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  echo "[INFO] vLLM already running, pid=$(cat "$PID_FILE")"
  echo "[INFO] endpoint: http://$(hostname -I | awk '{print $1}'):$PORT/v1"
  echo "[INFO] api_key_file: $API_KEY_FILE"
  exit 0
fi

if [ -f "$PID_FILE" ]; then
  rm -f "$PID_FILE"
fi

CUDA_VISIBLE_DEVICES="$GPU_ID" nohup "$VLLM_BIN" serve "$MODEL_DIR" \
  --host "$HOST" \
  --port "$PORT" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --gpu-memory-utilization "$GPU_MEM_UTIL" \
  --max-model-len "$MAX_MODEL_LEN" \
  --enforce-eager \
  --dtype bfloat16 \
  --api-key "$API_KEY" \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  >"$LOG_FILE" 2>&1 < /dev/null &

echo $! > "$PID_FILE"
sleep 2

echo "[INFO] started pid=$(cat "$PID_FILE")"
echo "[INFO] endpoint: http://$(hostname -I | awk '{print $1}'):$PORT/v1"
echo "[INFO] api_key_file: $API_KEY_FILE"
echo "[INFO] log_file: $LOG_FILE"
