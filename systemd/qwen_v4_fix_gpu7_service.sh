#!/usr/bin/env bash
set -euo pipefail

: "${MODEL_DIR:?MODEL_DIR is required}"
: "${PORT:?PORT is required}"
: "${GPU:?GPU is required}"
: "${SERVED_MODEL_NAME:?SERVED_MODEL_NAME is required}"
: "${MAX_MODEL_LEN:?MAX_MODEL_LEN is required}"
: "${GPU_MEMORY_UTILIZATION:?GPU_MEMORY_UTILIZATION is required}"
: "${LOG_PATH:?LOG_PATH is required}"
HOST="${HOST:-127.0.0.1}"

mkdir -p "$(dirname "$LOG_PATH")"
exec >>"$LOG_PATH" 2>&1

export CUDA_VISIBLE_DEVICES="$GPU"

exec /home/ubuntu/.local/bin/vllm serve \
  "$MODEL_DIR" \
  --host "$HOST" \
  --port "$PORT" \
  --tensor-parallel-size 1 \
  --max-model-len "$MAX_MODEL_LEN" \
  --dtype bfloat16 \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --enforce-eager \
  --trust-remote-code \
  --limit-mm-per-prompt '{"image": 0, "video": 0}'
