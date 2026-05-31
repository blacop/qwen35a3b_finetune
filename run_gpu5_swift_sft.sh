#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-5}
cd /home/ubuntu/qwen35a3b_finetune

echo "[START] $(date -u +%Y-%m-%dT%H:%M:%SZ) uid=$(id -u) user=$(id -un) gpu=${CUDA_VISIBLE_DEVICES}"

SWIFT_VENV=${SWIFT_VENV:-/home/ubuntu/qwen35a3b_finetune/.venv-swift311}
TRACKING_ROOT=${TRACKING_ROOT:-/home/ubuntu/qwen35a3b_finetune/tracking}
mkdir -p "$TRACKING_ROOT"/{tensorboard,mlruns,logs} /home/ubuntu/qwen35a3b_finetune/outputs

if [ ! -x "$SWIFT_VENV/bin/python" ]; then
  echo "[ERROR] Missing venv at $SWIFT_VENV" >&2
  echo "Create it first: uv venv $SWIFT_VENV && uv pip install --python $SWIFT_VENV/bin/python ms-swift mlflow" >&2
  exit 1
fi

source "$SWIFT_VENV/bin/activate"

export WANDB_DISABLED=true
export WANDB_MODE=disabled
export MLFLOW_TRACKING_URI="file:$TRACKING_ROOT/mlruns"
export MLFLOW_EXPERIMENT_NAME=${MLFLOW_EXPERIMENT_NAME:-qwen35a3b}

TS="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_NAME=${RUN_NAME:-swift_sft_clean_v2_debug_gpu5}
OUT_DIR=${OUT_DIR:-/home/ubuntu/qwen35a3b_finetune/outputs/swift_sft_clean_v2_debug_gpu5}
LOGGING_DIR=${LOGGING_DIR:-$TRACKING_ROOT/tensorboard/${RUN_NAME}}

BASE_MODEL=${BASE_MODEL:-Qwen/Qwen3.5-35B-A3B}
DATA_FILE=${DATA_FILE:-/home/ubuntu/qwen35a3b_finetune/datasets/sft_openai_messages.cleaned.v3_merged.single75_exact.jsonl}
MAX_LEN=${MAX_LEN:-3072}
EPOCHS=${EPOCHS:-1}
PER_DEVICE_BATCH=${PER_DEVICE_BATCH:-1}
GRAD_ACC=${GRAD_ACC:-16}
LR=${LR:-1e-4}
LORA_RANK=${LORA_RANK:-16}
LORA_ALPHA=${LORA_ALPHA:-32}
SAVE_STEPS=${SAVE_STEPS:-200}
SAVE_TOTAL_LIMIT=${SAVE_TOTAL_LIMIT:-3}
SYSTEM_PROMPT=${SYSTEM_PROMPT:-你是体育包网客服，按标准流程处理注单异常，无法确认时提交后台并同步运营联系包网。}
USE_GLOBAL_SYSTEM_PROMPT=${USE_GLOBAL_SYSTEM_PROMPT:-1}
AUTO_RESUME=${AUTO_RESUME:-true}
ADAPTERS=${ADAPTERS:-}

echo "[INFO] run_name=$RUN_NAME"
echo "[INFO] output_dir=$OUT_DIR"
echo "[INFO] logging_dir=$LOGGING_DIR"

# Swift currently writes tensorboard events under "$OUT_DIR/runs".
# Link it into TRACKING_ROOT so tensorboard.service (root watcher) can see all runs.
mkdir -p "$(dirname "$LOGGING_DIR")"
ln -sfn "$OUT_DIR/runs" "$LOGGING_DIR"

RUN_NAME="$RUN_NAME" \
BASE_MODEL="$BASE_MODEL" \
DATA_FILE="$DATA_FILE" \
OUT_DIR="$OUT_DIR" \
TRACKING_ROOT="$TRACKING_ROOT" \
LOGGING_DIR="$LOGGING_DIR" \
REPORT_TO="tensorboard mlflow" \
MAX_LEN="$MAX_LEN" \
EPOCHS="$EPOCHS" \
PER_DEVICE_BATCH="$PER_DEVICE_BATCH" \
GRAD_ACC="$GRAD_ACC" \
LR="$LR" \
LORA_RANK="$LORA_RANK" \
LORA_ALPHA="$LORA_ALPHA" \
SAVE_STEPS="$SAVE_STEPS" \
SAVE_TOTAL_LIMIT="$SAVE_TOTAL_LIMIT" \
SYSTEM_PROMPT="$SYSTEM_PROMPT" \
USE_GLOBAL_SYSTEM_PROMPT="$USE_GLOBAL_SYSTEM_PROMPT" \
AUTO_RESUME="$AUTO_RESUME" \
ADAPTERS="$ADAPTERS" \
/home/ubuntu/qwen35a3b_finetune/scripts/run_swift_sft.sh

echo "[DONE] $(date -u +%Y-%m-%dT%H:%M:%SZ)"
