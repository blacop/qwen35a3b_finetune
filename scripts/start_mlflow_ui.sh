#!/usr/bin/env bash
set -euo pipefail

MLFLOW_DIR=${MLFLOW_DIR:-/home/ubuntu/qwen35a3b_finetune/mlruns}
PORT=${PORT:-5000}
HOST=${HOST:-0.0.0.0}
LOGFILE=${LOGFILE:-/home/ubuntu/qwen35a3b_finetune/mlflow_ui.log}

mkdir -p "$MLFLOW_DIR"
nohup mlflow ui --host "$HOST" --port "$PORT" --backend-store-uri "file:$MLFLOW_DIR" > "$LOGFILE" 2>&1 < /dev/null &
echo $!
