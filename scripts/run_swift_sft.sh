#!/usr/bin/env bash
set -euo pipefail

# Usage:
# BASE_MODEL=Qwen/Qwen3.5-35B-A3B \
# DATA_FILE=./datasets/sft_openai_messages.jsonl \
# OUT_DIR=./outputs/swift_sft \
# REPORT_TO="tensorboard mlflow" \
# AUTO_RESUME=true \
# RESUME_FROM_CHECKPOINT=./outputs/swift_sft/checkpoint-200 \
# ./run_swift_sft.sh

BASE_MODEL=${BASE_MODEL:-Qwen/Qwen3.5-35B-A3B}
SWIFT_BIN=${SWIFT_BIN:-/home/ubuntu/qwen35a3b_finetune/.venv-swift311/bin/swift}
DATA_FILE=${DATA_FILE:-/home/ubuntu/qwen35a3b_finetune/datasets/sft_openai_messages.cleaned.v3_merged.single75_exact.jsonl}
OUT_DIR=${OUT_DIR:-/home/ubuntu/qwen35a3b_finetune/outputs/swift_sft}
RUN_NAME=${RUN_NAME:-$(basename "$OUT_DIR")}
TRACKING_ROOT=${TRACKING_ROOT:-/home/ubuntu/qwen35a3b_finetune/tracking}
LOGGING_DIR=${LOGGING_DIR:-$TRACKING_ROOT/tensorboard/$RUN_NAME}
REPORT_TO=${REPORT_TO:-tensorboard mlflow}
SYSTEM_PROMPT=${SYSTEM_PROMPT:-你是体育包网客服，按标准流程处理注单异常，无法确认时提交后台并同步运营联系包网。}
USE_GLOBAL_SYSTEM_PROMPT=${USE_GLOBAL_SYSTEM_PROMPT:-1}
TEMPLATE=${TEMPLATE:-qwen}
TARGET_MODULES=${TARGET_MODULES:-all-linear}
MAX_LEN=${MAX_LEN:-4096}
LR=${LR:-1e-4}
EPOCHS=${EPOCHS:-1}
PER_DEVICE_BATCH=${PER_DEVICE_BATCH:-1}
GRAD_ACC=${GRAD_ACC:-16}
DATASET_NUM_PROC=${DATASET_NUM_PROC:-8}
LOGGING_STEPS=${LOGGING_STEPS:-5}
SAVE_STEPS=${SAVE_STEPS:-200}
SAVE_TOTAL_LIMIT=${SAVE_TOTAL_LIMIT:-3}
LORA_RANK=${LORA_RANK:-16}
LORA_ALPHA=${LORA_ALPHA:-32}
TORCH_DTYPE=${TORCH_DTYPE:-bfloat16}
ADAPTERS=${ADAPTERS:-}
RESUME_FROM_CHECKPOINT=${RESUME_FROM_CHECKPOINT:-}
AUTO_RESUME=${AUTO_RESUME:-true}

mkdir -p "$OUT_DIR" "$(dirname "$LOGGING_DIR")"
read -r -a REPORT_TO_ARGS <<< "$REPORT_TO"
ADAPTER_ARGS=()
RESUME_ARGS=()
SYSTEM_ARGS=()

if [ -n "$ADAPTERS" ]; then
  read -r -a ADAPTER_PATHS <<< "$ADAPTERS"
  for adapter_path in "${ADAPTER_PATHS[@]}"; do
    ADAPTER_ARGS+=(--adapters "$adapter_path")
  done
  echo "[INFO] Loading adapter(s): ${ADAPTER_PATHS[*]}"
fi

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

if [ "${USE_GLOBAL_SYSTEM_PROMPT,,}" != "0" ] && [ -n "$SYSTEM_PROMPT" ]; then
  SYSTEM_ARGS=(--system "$SYSTEM_PROMPT")
fi

"$SWIFT_BIN" sft \
  --model "$BASE_MODEL" \
  --use_hf true \
  --check_model false \
  "${ADAPTER_ARGS[@]}" \
  --tuner_type lora \
  --torch_dtype "$TORCH_DTYPE" \
  --dataset "$DATA_FILE" \
  --dataset_num_proc "$DATASET_NUM_PROC" \
  --template "$TEMPLATE" \
  "${SYSTEM_ARGS[@]}" \
  --max_length "$MAX_LEN" \
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
