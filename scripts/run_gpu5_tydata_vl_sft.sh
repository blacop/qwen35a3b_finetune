#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/ubuntu/qwen35a3b_finetune"
ENV_FILE="${ENV_FILE:-$PROJECT_ROOT/services/swift-sft-vl-gpu5-tydata-domain.env}"

if [[ -f "$ENV_FILE" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
fi

exec "$PROJECT_ROOT/scripts/run_swift_sft_vl_gpu5.sh"
