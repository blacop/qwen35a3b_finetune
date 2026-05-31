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
  exit 1
fi

source "$SWIFT_VENV/bin/activate"

export WANDB_DISABLED=true
export WANDB_MODE=disabled
export MLFLOW_TRACKING_URI="file:$TRACKING_ROOT/mlruns"
export MLFLOW_EXPERIMENT_NAME=${MLFLOW_EXPERIMENT_NAME:-qwen35a3b_dpo}

TS="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_NAME=${RUN_NAME:-swift_dpo_clean_v2_gpu5}
OUT_DIR=${OUT_DIR:-/home/ubuntu/qwen35a3b_finetune/outputs/swift_dpo_clean_v2_gpu5}
LOGGING_DIR=${LOGGING_DIR:-$TRACKING_ROOT/tensorboard/${RUN_NAME}}

BASE_MODEL=${BASE_MODEL:-Qwen/Qwen3.5-35B-A3B}
DATA_FILE=${DATA_FILE:-/home/ubuntu/qwen35a3b_finetune/datasets/dpo_pairs.jsonl}
MAX_LEN=${MAX_LEN:-3072}
EPOCHS=${EPOCHS:-1}
PER_DEVICE_BATCH=${PER_DEVICE_BATCH:-1}
GRAD_ACC=${GRAD_ACC:-16}
AUTO_RESUME=${AUTO_RESUME:-true}

if [ -z "${SFT_CKPT:-}" ]; then
  preferred_ckpt="/home/ubuntu/qwen35a3b_finetune/outputs/swift_sft_clean_v2_debug_gpu5/checkpoint-2000"
  if [ -f "$preferred_ckpt/adapter_config.json" ]; then
    SFT_CKPT="$preferred_ckpt"
  else
    latest_adapter="$(find /home/ubuntu/qwen35a3b_finetune/outputs -maxdepth 3 -type f -name adapter_config.json -path '*/swift_sft_*/*' -printf '%T@ %h\n' 2>/dev/null | sort -nr | head -n1 | awk '{print $2}')"
    SFT_CKPT="${latest_adapter:-}"
  fi
fi

if [ -z "${SFT_CKPT:-}" ] || [ ! -f "$SFT_CKPT/adapter_config.json" ]; then
  echo "[ERROR] SFT_CKPT invalid. Set SFT_CKPT=/path/to/swift_sft_xxx/checkpoint-yyy" >&2
  exit 1
fi

echo "[INFO] sft_ckpt=$SFT_CKPT"
echo "[INFO] run_name=$RUN_NAME"
echo "[INFO] output_dir=$OUT_DIR"
echo "[INFO] logging_dir=$LOGGING_DIR"

# Swift currently writes tensorboard events under "$OUT_DIR/runs".
# Link it into TRACKING_ROOT so tensorboard.service root watcher can see all runs.
mkdir -p "$(dirname "$LOGGING_DIR")"
ln -sfn "$OUT_DIR/runs" "$LOGGING_DIR"

RUN_NAME="$RUN_NAME" \
BASE_MODEL="$BASE_MODEL" \
SFT_CKPT="$SFT_CKPT" \
DATA_FILE="$DATA_FILE" \
OUT_DIR="$OUT_DIR" \
TRACKING_ROOT="$TRACKING_ROOT" \
LOGGING_DIR="$LOGGING_DIR" \
REPORT_TO="tensorboard mlflow" \
MAX_LEN="$MAX_LEN" \
EPOCHS="$EPOCHS" \
PER_DEVICE_BATCH="$PER_DEVICE_BATCH" \
GRAD_ACC="$GRAD_ACC" \
AUTO_RESUME="$AUTO_RESUME" \
/home/ubuntu/qwen35a3b_finetune/scripts/run_swift_dpo.sh

echo "[DONE] $(date -u +%Y-%m-%dT%H:%M:%SZ)"
