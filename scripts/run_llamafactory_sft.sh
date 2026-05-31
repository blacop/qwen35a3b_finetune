#!/usr/bin/env bash
set -euo pipefail

# Usage:
# BASE_MODEL=Qwen/Qwen3.5-35B-A3B DATA_FILE=./datasets/sft_openai_messages.jsonl OUT_DIR=./outputs/lf_sft ./run_llamafactory_sft.sh

BASE_MODEL=${BASE_MODEL:-Qwen/Qwen3.5-35B-A3B}
DATA_FILE=${DATA_FILE:-/home/ubuntu/qwen35a3b_finetune/datasets/sft_openai_messages.jsonl}
DATASET_NAME=${DATASET_NAME:-sft_openai_messages}
OUT_DIR=${OUT_DIR:-/home/ubuntu/qwen35a3b_finetune/outputs/lf_sft}
MAX_LEN=${MAX_LEN:-4096}
LR=${LR:-1e-4}
EPOCHS=${EPOCHS:-3}
PER_DEVICE_BATCH=${PER_DEVICE_BATCH:-1}
GRAD_ACC=${GRAD_ACC:-16}
LORA_RANK=${LORA_RANK:-16}
LORA_ALPHA=${LORA_ALPHA:-32}
REPORT_TO=${REPORT_TO:-tensorboard}
RUN_NAME=${RUN_NAME:-lf_sft_qwen35}
RUN_ID=${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}
TRACKING_ROOT=${TRACKING_ROOT:-/home/ubuntu/qwen35a3b_finetune/tracking}
LOGGING_DIR=${LOGGING_DIR:-$TRACKING_ROOT/tensorboard/${RUN_NAME}_${RUN_ID}}
PLOT_LOSS=${PLOT_LOSS:-true}
OVERWRITE_CACHE=${OVERWRITE_CACHE:-true}
PREPROCESSING_NUM_WORKERS=${PREPROCESSING_NUM_WORKERS:-1}
DATALOADER_NUM_WORKERS=${DATALOADER_NUM_WORKERS:-0}

mkdir -p "$LOGGING_DIR"
if [[ "$REPORT_TO" == "mlflow" ]]; then
  export MLFLOW_TRACKING_URI=${MLFLOW_TRACKING_URI:-file:$TRACKING_ROOT/mlruns}
  export MLFLOW_EXPERIMENT_NAME=${MLFLOW_EXPERIMENT_NAME:-qwen35a3b}
fi
if [[ "$REPORT_TO" == "wandb" ]]; then
  export WANDB_DIR=${WANDB_DIR:-$TRACKING_ROOT/wandb}
  export WANDB_PROJECT=${WANDB_PROJECT:-qwen35a3b}
  export WANDB_NAME=${WANDB_NAME:-${RUN_NAME}_${RUN_ID}}
  mkdir -p "$WANDB_DIR"
fi

llamafactory-cli train \
  --stage sft \
  --do_train true \
  --model_name_or_path "$BASE_MODEL" \
  --finetuning_type lora \
  --template qwen \
  --dataset_dir "$(dirname "$DATA_FILE")" \
  --dataset "$DATASET_NAME" \
  --cutoff_len "$MAX_LEN" \
  --learning_rate "$LR" \
  --num_train_epochs "$EPOCHS" \
  --per_device_train_batch_size "$PER_DEVICE_BATCH" \
  --gradient_accumulation_steps "$GRAD_ACC" \
  --lora_rank "$LORA_RANK" \
  --lora_alpha "$LORA_ALPHA" \
  --lora_target all \
  --bf16 true \
  --logging_steps 10 \
  --logging_dir "$LOGGING_DIR" \
  --report_to "$REPORT_TO" \
  --run_name "$RUN_NAME" \
  --plot_loss "$PLOT_LOSS" \
  --overwrite_cache "$OVERWRITE_CACHE" \
  --preprocessing_num_workers "$PREPROCESSING_NUM_WORKERS" \
  --dataloader_num_workers "$DATALOADER_NUM_WORKERS" \
  --save_steps 200 \
  --output_dir "$OUT_DIR"
