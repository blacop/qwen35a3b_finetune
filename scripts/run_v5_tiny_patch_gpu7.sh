#!/usr/bin/env bash
set -euo pipefail

# v5 tiny patch SFT - 基于 v4-fix LoRA adapter 续训
# Usage:
#   bash run_v5_tiny_patch_gpu7.sh                # full training
#   DRY_RUN=5 bash run_v5_tiny_patch_gpu7.sh      # 只用 5 条数据 dry-run

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-7}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-0}

# === 路径配置 ===
SWIFT_BIN=/home/ubuntu/qwen35a3b_finetune/.venv-swift311/bin/swift
BASE_MODEL=${BASE_MODEL:-Qwen/Qwen3.5-35B-A3B}
ADAPTERS=${ADAPTERS:-/video-storage/ai-customer/qwen35a3b_finetune/outputs/swift_sft_v4_fix_gpu7}
DATA_FILE=${DATA_FILE:-/home/ubuntu/qwen35a3b_finetune/datasets/v5_tiny_patch_20260430/v5_tiny_patch_train.jsonl}
TS=$(date +%Y%m%dT%H%M%SZ)
OUT_DIR=${OUT_DIR:-/home/ubuntu/qwen35a3b_finetune/outputs/swift_sft_v5_tiny_patch_gpu7_${TS}}
LOG_DIR=/home/ubuntu/qwen35a3b_finetune/tracking/logs
LOG_FILE=$LOG_DIR/swift_sft_v5_tiny_patch_${TS}.log
mkdir -p "$LOG_DIR" "$OUT_DIR"

# === 训练参数（保守，避免破坏 v4-fix 原能力）===
SYSTEM_PROMPT="你是体育包网智能客服。必须遵守合规与风控规则：遇到注单取消/作废/异常、结算争议、赔率异常、限红风控等无法直接确认的情况，必须明确告知需要后台查询或联系平台运营后回复；禁止赌博诱导、代理拉新、洗钱跑分、伪造证件、低龄相关内容。"

LR=${LR:-5e-6}                        # 小学习率（团队 1e-4 太大，我们 tiny patch 用 5e-6）
EPOCHS=${EPOCHS:-3}                   # 70 条 × 3 epoch = 充分但不过拟合
PER_DEVICE_BATCH=${PER_DEVICE_BATCH:-1}
GRAD_ACC=${GRAD_ACC:-8}               # 等效 batch=8（小数据用小 batch）
LORA_RANK=${LORA_RANK:-16}            # 保持与原 v4-fix 一致
LORA_ALPHA=${LORA_ALPHA:-32}
TARGET_MODULES=${TARGET_MODULES:-all-linear}
MAX_LEN=${MAX_LEN:-4096}
WARMUP_RATIO=${WARMUP_RATIO:-0.1}
SAVE_STEPS=${SAVE_STEPS:-100}
LOGGING_STEPS=${LOGGING_STEPS:-2}

# === Dry-run（前 N 条）===
TRAIN_DATA="$DATA_FILE"
if [ -n "${DRY_RUN:-}" ]; then
  DRY_FILE=$OUT_DIR/dryrun_${DRY_RUN}.jsonl
  head -n "$DRY_RUN" "$DATA_FILE" > "$DRY_FILE"
  TRAIN_DATA="$DRY_FILE"
  EPOCHS=1
  echo "[DRY_RUN] 用前 $DRY_RUN 条 / 1 epoch / 文件: $DRY_FILE"
fi

echo "================== v5 tiny patch SFT =================="
echo "  GPU: $CUDA_VISIBLE_DEVICES"
echo "  BASE: $BASE_MODEL"
echo "  ADAPTERS: $ADAPTERS"
echo "  DATA: $TRAIN_DATA"
echo "  OUT_DIR: $OUT_DIR"
echo "  LR: $LR  EPOCHS: $EPOCHS  BATCH: $PER_DEVICE_BATCH  GRAD_ACC: $GRAD_ACC"
echo "  LOG: $LOG_FILE"
echo "========================================================"

# === 实际训练命令 ===
"$SWIFT_BIN" sft \
  --model "$BASE_MODEL" \
  --use_hf true \
  --check_model false \
  --adapters "$ADAPTERS" \
  --tuner_type lora \
  --torch_dtype bfloat16 \
  --dataset "$TRAIN_DATA" \
  --dataset_num_proc 4 \
  --template qwen \
  --system "$SYSTEM_PROMPT" \
  --max_length "$MAX_LEN" \
  --learning_rate "$LR" \
  --num_train_epochs "$EPOCHS" \
  --per_device_train_batch_size "$PER_DEVICE_BATCH" \
  --gradient_accumulation_steps "$GRAD_ACC" \
  --lora_rank "$LORA_RANK" \
  --lora_alpha "$LORA_ALPHA" \
  --target_modules "$TARGET_MODULES" \
  --warmup_ratio "$WARMUP_RATIO" \
  --logging_steps "$LOGGING_STEPS" \
  --save_strategy steps \
  --save_steps "$SAVE_STEPS" \
  --eval_strategy no \
  --split_dataset_ratio 0 \
  --save_total_limit 3 \
  --add_version false \
  --ignore_args_error true \
  --output_dir "$OUT_DIR" \
  2>&1 | tee "$LOG_FILE"
