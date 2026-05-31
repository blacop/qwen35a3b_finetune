#!/usr/bin/env bash
set -euo pipefail

TS=${TS:-$(date -u +%Y%m%dT%H%M%SZ)}
GPU_ID=${GPU_ID:-7}
PROJECT_ROOT=/home/ubuntu/qwen35a3b_finetune
TRACKING_ROOT=${TRACKING_ROOT:-$PROJECT_ROOT/tracking}
OUT_DIR=${OUT_DIR:-$PROJECT_ROOT/outputs/swift_sft_clean_v2_${TS}}
RUN_NAME=${RUN_NAME:-$(basename "$OUT_DIR")}
VENV_PATH=${VENV_PATH:-$PROJECT_ROOT/.venv-swift}

mkdir -p "$PROJECT_ROOT/outputs" "$TRACKING_ROOT/logs" "$TRACKING_ROOT/tensorboard" "$TRACKING_ROOT/mlruns"

source "$VENV_PATH/bin/activate"

export CUDA_VISIBLE_DEVICES="$GPU_ID"
export MLFLOW_TRACKING_URI="file:$TRACKING_ROOT/mlruns"
export MLFLOW_EXPERIMENT_NAME="${MLFLOW_EXPERIMENT_NAME:-qwen35a3b_sft_clean_v2}"
export TRACKING_ROOT
export BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3.5-35B-A3B}"
export DATA_FILE="${DATA_FILE:-$PROJECT_ROOT/datasets/sft_openai_messages.cleaned.v3_merged.single75_exact.jsonl}"
export REPORT_TO="${REPORT_TO:-tensorboard mlflow}"
export MAX_LEN="${MAX_LEN:-3072}"
export EPOCHS="${EPOCHS:-1}"
export PER_DEVICE_BATCH="${PER_DEVICE_BATCH:-1}"
export GRAD_ACC="${GRAD_ACC:-16}"
export LORA_RANK="${LORA_RANK:-16}"
export SAVE_STEPS="${SAVE_STEPS:-200}"
export LOGGING_STEPS="${LOGGING_STEPS:-5}"
export OUT_DIR
export RUN_NAME

exec "$PROJECT_ROOT/scripts/run_swift_sft.sh"
