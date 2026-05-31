#!/usr/bin/env bash
set -euo pipefail

# Usage:
# BASE_MODEL=Qwen/Qwen3.5-35B-A3B \
# SFT_CKPT=./outputs/swift_sft/checkpoint-220 \
# DATA_FILE=./datasets/grpo_prompts.jsonl \
# OUT_DIR=./outputs/swift_grpo \
# REPORT_TO="tensorboard mlflow" \
# AUTO_RESUME=true \
# RESUME_FROM_CHECKPOINT=./outputs/swift_grpo/checkpoint-200 \
# ./run_swift_grpo.sh

BASE_MODEL=${BASE_MODEL:-Qwen/Qwen3.5-35B-A3B}
SFT_CKPT=${SFT_CKPT:-/home/ubuntu/qwen35a3b_finetune/outputs/swift_sft/checkpoint-220}
DATA_FILE=${DATA_FILE:-/home/ubuntu/qwen35a3b_finetune/datasets/grpo_prompts.jsonl}
OUT_DIR=${OUT_DIR:-/home/ubuntu/qwen35a3b_finetune/outputs/swift_grpo}
RUN_NAME=${RUN_NAME:-$(basename "$OUT_DIR")}
TRACKING_ROOT=${TRACKING_ROOT:-/home/ubuntu/qwen35a3b_finetune/tracking}
LOGGING_DIR=${LOGGING_DIR:-$TRACKING_ROOT/tensorboard/$RUN_NAME}
REPORT_TO=${REPORT_TO:-tensorboard mlflow}
TEMPLATE=${TEMPLATE:-qwen}
MAX_LEN=${MAX_LEN:-4096}
LR=${LR:-5e-6}
EPOCHS=${EPOCHS:-1}
PER_DEVICE_BATCH=${PER_DEVICE_BATCH:-1}
GRAD_ACC=${GRAD_ACC:-16}
DATASET_NUM_PROC=${DATASET_NUM_PROC:-8}
LOGGING_STEPS=${LOGGING_STEPS:-5}
SAVE_STEPS=${SAVE_STEPS:-200}
SAVE_TOTAL_LIMIT=${SAVE_TOTAL_LIMIT:-3}
TORCH_DTYPE=${TORCH_DTYPE:-bfloat16}
RESUME_FROM_CHECKPOINT=${RESUME_FROM_CHECKPOINT:-}
AUTO_RESUME=${AUTO_RESUME:-true}

if [ ! -f "$SFT_CKPT/adapter_config.json" ]; then
  echo "[ERROR] Missing adapter checkpoint: $SFT_CKPT/adapter_config.json" >&2
  exit 1
fi

mkdir -p "$OUT_DIR" "$(dirname "$LOGGING_DIR")"
read -r -a REPORT_TO_ARGS <<< "$REPORT_TO"
RESUME_ARGS=()

if [ -z "$RESUME_FROM_CHECKPOINT" ] && [ "${AUTO_RESUME,,}" = "true" ]; then
  latest_step="$(find "$OUT_DIR" -maxdepth 1 -type d -name 'checkpoint-*' -printf '%f\n' 2>/dev/null \
    | sed -n 's/^checkpoint-\([0-9]\+\)$/\1/p' | sort -n | tail -n1 || true)"
  if [ -n "$latest_step" ]; then
    RESUME_FROM_CHECKPOINT="$OUT_DIR/checkpoint-$latest_step"
  fi
fi

if [ -n "$RESUME_FROM_CHECKPOINT" ]; then
  if [ ! -d "$RESUME_FROM_CHECKPOINT" ]; then
    echo "[ERROR] resume checkpoint not found: $RESUME_FROM_CHECKPOINT" >&2
    exit 1
  fi
  RESUME_ARGS=(--resume_from_checkpoint "$RESUME_FROM_CHECKPOINT")
  echo "[INFO] Resuming from checkpoint: $RESUME_FROM_CHECKPOINT"
else
  echo "[INFO] Fresh run (no resume checkpoint)."
fi

swift rlhf \
  --rlhf_type grpo \
  --model "$BASE_MODEL" \
  --use_hf true \
  --check_model false \
  --adapters "$SFT_CKPT" \
  --tuner_type lora \
  --torch_dtype "$TORCH_DTYPE" \
  --dataset "$DATA_FILE" \
  --dataset_num_proc "$DATASET_NUM_PROC" \
  --template "$TEMPLATE" \
  --max_length "$MAX_LEN" \
  --learning_rate "$LR" \
  --num_train_epochs "$EPOCHS" \
  --per_device_train_batch_size "$PER_DEVICE_BATCH" \
  --gradient_accumulation_steps "$GRAD_ACC" \
  --logging_steps "$LOGGING_STEPS" \
  --save_strategy steps \
  --save_steps "$SAVE_STEPS" \
  --eval_strategy no \
  --split_dataset_ratio 0 \
  --save_total_limit "$SAVE_TOTAL_LIMIT" \
  --run_name "$RUN_NAME" \
  --logging_dir "$LOGGING_DIR" \
  --report_to "${REPORT_TO_ARGS[@]}" \
  --add_version false \
  --ignore_args_error true \
  "${RESUME_ARGS[@]}" \
  --output_dir "$OUT_DIR"
