#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=5
cd /home/ubuntu/qwen35a3b_finetune
echo "[START] $(date -u +%Y-%m-%dT%H:%M:%SZ) uid=$(id -u) user=$(id -un) cwd=$(pwd)"

LF_VENV=/home/ubuntu/qwen35a3b_finetune/.venv-lf
TRACKING_ROOT=/home/ubuntu/qwen35a3b_finetune/tracking
mkdir -p "$TRACKING_ROOT"/{tensorboard,mlruns,logs} /home/ubuntu/qwen35a3b_finetune/outputs

UV_BIN=${UV_BIN:-/home/ubuntu/.local/bin/uv}
if [ ! -x "$UV_BIN" ]; then
  UV_BIN="$(command -v uv || true)"
fi
if [ -z "${UV_BIN:-}" ] || [ ! -x "${UV_BIN:-}" ]; then
  echo "[ERROR] uv not found in PATH=$PATH" >&2
  exit 1
fi
echo "[INFO] using uv: $UV_BIN"

if [ ! -x "$LF_VENV/bin/python" ]; then
  "$UV_BIN" venv "$LF_VENV"
fi
source "$LF_VENV/bin/activate"

# Pin torch CUDA wheel family to cu128 to match host driver branch.
UV_CACHE_DIR=/tmp/uv-cache "$UV_BIN" pip install --python "$LF_VENV/bin/python" -U \
  --extra-index-url https://download.pytorch.org/whl/cu128 \
  "torch==2.10.0+cu128" \
  "torchvision==0.25.0+cu128" \
  "torchaudio==2.10.0+cu128" \
  "llamafactory==0.9.4" \
  "mlflow>=2.12" \
  "tensorboard>=2.20"

export WANDB_DISABLED=true
export WANDB_MODE=disabled
export MLFLOW_TRACKING_URI="file:$TRACKING_ROOT/mlruns"
export MLFLOW_EXPERIMENT_NAME="qwen35a3b"

TS="$(date -u +%Y%m%dT%H%M%SZ)"

REPORT_TO=all \
RUN_ID="$TS" \
RUN_NAME="lf_sft_gpu5_${TS}" \
BASE_MODEL="Qwen/Qwen3.5-35B-A3B" \
DATA_FILE="/home/ubuntu/qwen35a3b_finetune/datasets/sft_openai_messages.jsonl" \
DATASET_NAME="sft_openai_messages" \
OUT_DIR="/home/ubuntu/qwen35a3b_finetune/outputs/lf_sft_${TS}" \
TRACKING_ROOT="$TRACKING_ROOT" \
LOGGING_DIR="$TRACKING_ROOT/tensorboard/lf_sft_gpu5_${TS}" \
MAX_LEN=4096 \
EPOCHS=1 \
PER_DEVICE_BATCH=1 \
GRAD_ACC=16 \
LORA_RANK=16 \
LORA_ALPHA=32 \
/home/ubuntu/qwen35a3b_finetune/scripts/run_llamafactory_sft.sh

REPORT_TO=all \
RUN_ID="$TS" \
RUN_NAME="lf_dpo_gpu5_${TS}" \
BASE_MODEL="Qwen/Qwen3.5-35B-A3B" \
SFT_CKPT="/home/ubuntu/qwen35a3b_finetune/outputs/lf_sft_${TS}" \
DATA_FILE="/home/ubuntu/qwen35a3b_finetune/datasets/dpo_pairs.jsonl" \
DATASET_NAME="dpo_pairs" \
OUT_DIR="/home/ubuntu/qwen35a3b_finetune/outputs/lf_dpo_${TS}" \
TRACKING_ROOT="$TRACKING_ROOT" \
LOGGING_DIR="$TRACKING_ROOT/tensorboard/lf_dpo_gpu5_${TS}" \
MAX_LEN=4096 \
EPOCHS=1 \
PER_DEVICE_BATCH=1 \
GRAD_ACC=16 \
/home/ubuntu/qwen35a3b_finetune/scripts/run_llamafactory_dpo.sh
