#!/bin/bash
set -euo pipefail

MODEL_DIR=~/qwen35a3b_finetune/outputs/swift_sft_v4_fix_gpu7_merged_vllm
LOG=~/qwen35a3b_finetune/tracking/logs/vllm_v4_fix_8k.log
PORT=8013
GPU=7

mkdir -p "$(dirname "$LOG")"

pkill -9 -f "vllm serve.*swift_sft_v4_fix" 2>/dev/null || true
sleep 3

export CUDA_VISIBLE_DEVICES=$GPU

nohup /home/ubuntu/.local/bin/vllm serve \
  "$MODEL_DIR" \
  --host 0.0.0.0 \
  --port "$PORT" \
  --tensor-parallel-size 1 \
  --max-model-len 8192 \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.85 \
  --served-model-name qwen35a3b-sft-v4-fix-gpu7 \
  --enforce-eager \
  --trust-remote-code \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --limit-mm-per-prompt '{"image": 0, "video": 0}' \
  > "$LOG" 2>&1 &

echo "Started vllm pid=$!, log=$LOG"
