#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=7

exec /home/ubuntu/.local/bin/vllm serve \
  /home/ubuntu/qwen35a3b_finetune/outputs/swift_sft_v4_fix_gpu7_merged_vllm \
  --host 0.0.0.0 \
  --port 8013 \
  --tensor-parallel-size 1 \
  --max-model-len 8192 \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.85 \
  --served-model-name qwen35a3b-sft-v4-fix-gpu7 \
  --enforce-eager \
  --trust-remote-code \
  --limit-mm-per-prompt '{"image": 0, "video": 0}'
