#!/usr/bin/env bash
set -euo pipefail

# Usage:
# MODEL=Qwen/Qwen3.5-35B-A3B ./run_sglang_qwen35.sh

MODEL=${MODEL:-Qwen/Qwen3.5-35B-A3B}
PORT=${PORT:-30000}
TP_SIZE=${TP_SIZE:-1}
ENABLE_TOOL_USE=${ENABLE_TOOL_USE:-1}
ENABLE_MTP=${ENABLE_MTP:-0}
MTP_TOKENS=${MTP_TOKENS:-2}

python -m sglang.launch_server \
  --model-path "$MODEL" \
  --host 0.0.0.0 \
  --port "$PORT" \
  --tp "$TP_SIZE" \
  $( [ "$ENABLE_TOOL_USE" = "1" ] && echo "--tool-call-parser qwen25" ) \
  $( [ "$ENABLE_MTP" = "1" ] && echo "--speculative-algorithm EAGLE3 --speculative-num-steps $MTP_TOKENS" )

# For Qwen3.5 on recent SGLang releases:
# - tool parser typically uses qwen25
# - MTP can use EAGLE3 with small speculative steps
