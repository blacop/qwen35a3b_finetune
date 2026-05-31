#!/usr/bin/env bash
set -euo pipefail

LOGDIR=${LOGDIR:-/home/ubuntu/qwen35a3b_finetune/outputs}
PORT=${PORT:-6006}
HOST=${HOST:-0.0.0.0}
LOGFILE=${LOGFILE:-/home/ubuntu/qwen35a3b_finetune/tensorboard.log}

nohup tensorboard --logdir "$LOGDIR" --host "$HOST" --port "$PORT" > "$LOGFILE" 2>&1 < /dev/null &
echo $!
