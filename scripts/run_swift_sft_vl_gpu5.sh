#!/usr/bin/env bash
set -euo pipefail

# Usage example:
# BASE_MODEL=Qwen/Qwen3.5-35B-A3B \
# DATA_FILE=/home/ubuntu/tydata_rag/mm_rag/train_datasets/jt_manual_mm_train_docx_web_20260422T075506Z/multimodal_train.swift_vl.jsonl \
# OUT_DIR=/home/ubuntu/qwen35a3b_finetune/outputs/swift_sft_vl_gpu5 \
# CUDA_VISIBLE_DEVICES=5 \
# /home/ubuntu/qwen35a3b_finetune/scripts/run_swift_sft_vl_gpu5.sh

SWIFT_BIN=${SWIFT_BIN:-/home/ubuntu/qwen35a3b_finetune/.venv-swift311/bin/swift}
BASE_MODEL=${BASE_MODEL:-Qwen/Qwen3.5-35B-A3B}
DATA_FILE=${DATA_FILE:-/home/ubuntu/tydata_rag/mm_rag/train_datasets/jt_manual_mm_train_docx_web_20260422T075506Z/multimodal_train.swift_vl.jsonl}
OUT_DIR=${OUT_DIR:-/home/ubuntu/qwen35a3b_finetune/outputs/swift_sft_vl_gpu5}
RUN_NAME=${RUN_NAME:-$(basename "$OUT_DIR")}
TRACKING_ROOT=${TRACKING_ROOT:-/home/ubuntu/qwen35a3b_finetune/tracking}
LOGGING_DIR=${LOGGING_DIR:-$TRACKING_ROOT/tensorboard/$RUN_NAME}
REPORT_TO=${REPORT_TO:-tensorboard mlflow}
TEMPLATE=${TEMPLATE:-}
TARGET_MODULES=${TARGET_MODULES:-all-linear}
MAX_LEN=${MAX_LEN:-3072}
MAX_PIXELS=${MAX_PIXELS:-1048576}
LR=${LR:-5e-5}
EPOCHS=${EPOCHS:-1}
PER_DEVICE_BATCH=${PER_DEVICE_BATCH:-1}
GRAD_ACC=${GRAD_ACC:-16}
DATASET_NUM_PROC=${DATASET_NUM_PROC:-4}
LOGGING_STEPS=${LOGGING_STEPS:-5}
SAVE_STEPS=${SAVE_STEPS:-200}
SAVE_TOTAL_LIMIT=${SAVE_TOTAL_LIMIT:-3}
LORA_RANK=${LORA_RANK:-16}
LORA_ALPHA=${LORA_ALPHA:-32}
TORCH_DTYPE=${TORCH_DTYPE:-bfloat16}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-5}
SYSTEM_PROMPT=${SYSTEM_PROMPT:-}
RESUME_FROM_CHECKPOINT=${RESUME_FROM_CHECKPOINT:-}
AUTO_RESUME=${AUTO_RESUME:-true}

if [ -z "$TEMPLATE" ]; then
  case "$BASE_MODEL" in
    Qwen/Qwen2.5-VL-*|*/Qwen2.5-VL-*)
      TEMPLATE=qwen2_5_vl
      ;;
    Qwen/Qwen3.5-*|*/Qwen3.5-*)
      TEMPLATE=qwen3_5
      ;;
    *)
      echo "[WARN] Unknown multimodal base model: $BASE_MODEL" >&2
      echo "[WARN] Falling back to TEMPLATE=qwen3_5; override TEMPLATE explicitly if needed." >&2
      TEMPLATE=qwen3_5
      ;;
  esac
fi

if [ ! -f "$DATA_FILE" ]; then
  echo "[ERROR] dataset not found: $DATA_FILE" >&2
  exit 1
fi

mkdir -p "$OUT_DIR" "$(dirname "$LOGGING_DIR")"
read -r -a REPORT_TO_ARGS <<< "$REPORT_TO"
RESUME_ARGS=()
SYSTEM_ARGS=()

echo "[INFO] BASE_MODEL=$BASE_MODEL"
echo "[INFO] TEMPLATE=$TEMPLATE"

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

if [ -n "$SYSTEM_PROMPT" ]; then
  SYSTEM_ARGS=(--system "$SYSTEM_PROMPT")
fi

export CUDA_VISIBLE_DEVICES

"$SWIFT_BIN" sft \
  --model "$BASE_MODEL" \
  --use_hf true \
  --check_model false \
  --tuner_type lora \
  --torch_dtype "$TORCH_DTYPE" \
  --dataset "$DATA_FILE" \
  --dataset_num_proc "$DATASET_NUM_PROC" \
  --template "$TEMPLATE" \
  "${SYSTEM_ARGS[@]}" \
  --max_length "$MAX_LEN" \
  --max_pixels "$MAX_PIXELS" \
  --learning_rate "$LR" \
  --num_train_epochs "$EPOCHS" \
  --per_device_train_batch_size "$PER_DEVICE_BATCH" \
  --gradient_accumulation_steps "$GRAD_ACC" \
  --lora_rank "$LORA_RANK" \
  --lora_alpha "$LORA_ALPHA" \
  --target_modules "$TARGET_MODULES" \
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
