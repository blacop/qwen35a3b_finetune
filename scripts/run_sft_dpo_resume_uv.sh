#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/ubuntu/qwen35a3b_finetune"
cd "$PROJECT_ROOT"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-5}"
SWIFT_VENV="${SWIFT_VENV:-$PROJECT_ROOT/.venv-swift311}"
UV_BIN="${UV_BIN:-/home/ubuntu/.local/bin/uv}"
UV_PYTHON="${UV_PYTHON:-$SWIFT_VENV/bin/python}"

BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3.5-35B-A3B}"
SFT_DATA_FILE="${SFT_DATA_FILE:-$PROJECT_ROOT/datasets/sft_openai_messages.cleaned.v3_merged.single75_exact.repair_v3.tydata_pdfcurated.nometa.jsonl}"
DPO_DATA_FILE="${DPO_DATA_FILE:-$PROJECT_ROOT/datasets/dpo_pairs.repair_v3.tydata_pdfcurated.nometa.jsonl}"

SFT_RUN_NAME="${SFT_RUN_NAME:-swift_sft_repair_v3_tydata_pdfcurated_gpu5_20260413T063639Z}"
SFT_OUT_DIR="${SFT_OUT_DIR:-$PROJECT_ROOT/outputs/$SFT_RUN_NAME}"
DPO_RUN_NAME="${DPO_RUN_NAME:-swift_dpo_repair_v3_tydata_pdfcurated_gpu5_resume}"
DPO_OUT_DIR="${DPO_OUT_DIR:-$PROJECT_ROOT/outputs/$DPO_RUN_NAME}"

SFT_RESUME_FROM_CHECKPOINT="${SFT_RESUME_FROM_CHECKPOINT:-}"
SFT_RESUME_FALLBACK="${SFT_RESUME_FALLBACK:-$PROJECT_ROOT/outputs/swift_sft_clean_v2_debug_gpu5/checkpoint-2000}"
DPO_RESUME_FROM_CHECKPOINT="${DPO_RESUME_FROM_CHECKPOINT:-}"

MAX_LEN="${MAX_LEN:-3072}"
EPOCHS="${EPOCHS:-1}"
PER_DEVICE_BATCH="${PER_DEVICE_BATCH:-1}"
GRAD_ACC="${GRAD_ACC:-16}"
LORA_RANK="${LORA_RANK:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
SAVE_STEPS="${SAVE_STEPS:-100}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-5}"

mkdir -p "$SFT_OUT_DIR" "$DPO_OUT_DIR"

echo "[INFO] $(date -u +%Y-%m-%dT%H:%M:%SZ) start sft+dpo resume pipeline"
echo "[INFO] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "[INFO] SFT_OUT_DIR=$SFT_OUT_DIR"
echo "[INFO] DPO_OUT_DIR=$DPO_OUT_DIR"

if [ ! -x "$UV_BIN" ]; then
  echo "[ERROR] uv not found: $UV_BIN" >&2
  exit 1
fi
if [ ! -x "$UV_PYTHON" ]; then
  echo "[ERROR] uv python not found: $UV_PYTHON" >&2
  exit 1
fi

latest_ckpt_step() {
  local out_dir="$1"
  find "$out_dir" -maxdepth 1 -type d -name 'checkpoint-*' -printf '%f\n' 2>/dev/null \
    | sed -n 's/^checkpoint-\([0-9]\+\)$/\1/p' | sort -n | tail -n1
}

pick_sft_resume() {
  local chosen=""
  if [ -n "$SFT_RESUME_FROM_CHECKPOINT" ] && [ -d "$SFT_RESUME_FROM_CHECKPOINT" ]; then
    chosen="$SFT_RESUME_FROM_CHECKPOINT"
  fi
  if [ -z "$chosen" ]; then
    local s
    s="$(latest_ckpt_step "$SFT_OUT_DIR" || true)"
    if [ -n "${s:-}" ] && [ -d "$SFT_OUT_DIR/checkpoint-$s" ]; then
      chosen="$SFT_OUT_DIR/checkpoint-$s"
    fi
  fi
  if [ -z "$chosen" ] && [ -d "$SFT_RESUME_FALLBACK" ]; then
    chosen="$SFT_RESUME_FALLBACK"
  fi
  echo "$chosen"
}

SFT_RESUME_PICKED="$(pick_sft_resume)"
if [ -n "$SFT_RESUME_PICKED" ]; then
  echo "[INFO] SFT resume checkpoint: $SFT_RESUME_PICKED"
else
  echo "[WARN] No SFT checkpoint found. SFT will start fresh."
fi

export CUDA_VISIBLE_DEVICES
export RUN_NAME="$SFT_RUN_NAME"
export OUT_DIR="$SFT_OUT_DIR"
export BASE_MODEL
export DATA_FILE="$SFT_DATA_FILE"
export AUTO_RESUME=true
export MAX_LEN EPOCHS PER_DEVICE_BATCH GRAD_ACC LORA_RANK LORA_ALPHA SAVE_STEPS SAVE_TOTAL_LIMIT
if [ -n "$SFT_RESUME_PICKED" ]; then
  export RESUME_FROM_CHECKPOINT="$SFT_RESUME_PICKED"
else
  unset RESUME_FROM_CHECKPOINT || true
fi

echo "[INFO] launching SFT..."
"$UV_BIN" run --python "$UV_PYTHON" bash "$PROJECT_ROOT/run_gpu5_swift_sft.sh"

sft_step="$(latest_ckpt_step "$SFT_OUT_DIR" || true)"
if [ -z "${sft_step:-}" ] || [ ! -d "$SFT_OUT_DIR/checkpoint-$sft_step" ]; then
  echo "[ERROR] SFT finished without checkpoint in $SFT_OUT_DIR" >&2
  exit 1
fi
SFT_CKPT="$SFT_OUT_DIR/checkpoint-$sft_step"
echo "[INFO] latest SFT checkpoint for DPO: $SFT_CKPT"

dpo_resume=""
if [ -n "$DPO_RESUME_FROM_CHECKPOINT" ] && [ -d "$DPO_RESUME_FROM_CHECKPOINT" ]; then
  dpo_resume="$DPO_RESUME_FROM_CHECKPOINT"
else
  dpo_step="$(latest_ckpt_step "$DPO_OUT_DIR" || true)"
  if [ -n "${dpo_step:-}" ] && [ -d "$DPO_OUT_DIR/checkpoint-$dpo_step" ]; then
    dpo_resume="$DPO_OUT_DIR/checkpoint-$dpo_step"
  fi
fi
if [ -n "$dpo_resume" ]; then
  echo "[INFO] DPO resume checkpoint: $dpo_resume"
fi

export RUN_NAME="$DPO_RUN_NAME"
export OUT_DIR="$DPO_OUT_DIR"
export BASE_MODEL
export DATA_FILE="$DPO_DATA_FILE"
export SFT_CKPT
export AUTO_RESUME=true
export MAX_LEN EPOCHS PER_DEVICE_BATCH GRAD_ACC SAVE_STEPS SAVE_TOTAL_LIMIT
if [ -n "$dpo_resume" ]; then
  export RESUME_FROM_CHECKPOINT="$dpo_resume"
else
  unset RESUME_FROM_CHECKPOINT || true
fi

echo "[INFO] launching DPO..."
"$UV_BIN" run --python "$UV_PYTHON" bash "$PROJECT_ROOT/run_gpu5_swift_dpo.sh"

echo "[DONE] $(date -u +%Y-%m-%dT%H:%M:%SZ) sft+dpo resume pipeline finished"
