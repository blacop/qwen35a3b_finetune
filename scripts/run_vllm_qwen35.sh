#!/usr/bin/env bash
set -euo pipefail

# Usage:
# MODEL=Qwen/Qwen3.5-35B-A3B ./run_vllm_qwen35.sh

MODEL=${MODEL:-Qwen/Qwen3.5-35B-A3B}
PORT=${PORT:-8000}
TP_SIZE=${TP_SIZE:-1}
MAX_LEN=${MAX_LEN:-262144}
ENABLE_TOOL_USE=${ENABLE_TOOL_USE:-1}
ENABLE_MTP=${ENABLE_MTP:-0}

vllm serve "$MODEL" \
  --host 0.0.0.0 \
  --port "$PORT" \
  --tensor-parallel-size "$TP_SIZE" \
  --max-model-len "$MAX_LEN" \
  --reasoning-parser qwen3 \
  $( [ "$ENABLE_TOOL_USE" = "1" ] && echo "--enable-auto-tool-choice --tool-call-parser qwen3_coder" ) \
  $( [ "$ENABLE_MTP" = "1" ] && echo "--speculative-config '{\"method\":\"qwen3_next_mtp\",\"num_speculative_tokens\":2}'" )

# Set ENABLE_MTP=1 to turn on Qwen3.5 MTP speculative decoding.
