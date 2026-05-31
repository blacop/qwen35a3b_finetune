#!/usr/bin/env bash
# v31c: SFT on mixed dataset (v3 base 1616 + v31c patch 420 = 2036 rows)
# Driven by EnvironmentFile= in swift-sft-hotfix-v31c-mixed-gpu5.service.
# All tunables live in services/swift-sft-hotfix-v31c-mixed-gpu5.env.
set -euo pipefail

PROJECT_ROOT="/home/ubuntu/qwen35a3b_finetune"
STORAGE_ROOT="/video-storage/ai-customer/qwen35a3b_finetune"

# These are set by EnvironmentFile= (or fall back when run manually)
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-5}"
GPU5_UPSTREAM_SERVICE="${GPU5_UPSTREAM_SERVICE:-vllm-qwen35-gpu5-agent-upstream.service}"
DATA_FILE="${DATA_FILE:-${STORAGE_ROOT}/datasets/sports_rule_hotfix_v31c_json_mixed_patch_20260429T100809Z/sft_sports_rule_hotfix_v31c_mixed.jsonl}"
BASE_MODEL="${BASE_MODEL:-${STORAGE_ROOT}/outputs/swift_sft_sports_rule_hotfix_v3_json_gpu5_20260429T071807Z_merged_20260429T071807Z_vllm}"
EPOCHS="${EPOCHS:-0.5}"
PER_DEVICE_BATCH="${PER_DEVICE_BATCH:-1}"
GRAD_ACC="${GRAD_ACC:-8}"
LR="${LR:-5e-5}"
LORA_RANK="${LORA_RANK:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
SAVE_STEPS="${SAVE_STEPS:-50}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-3}"
MAX_LEN="${MAX_LEN:-3072}"
AUTO_RESUME="${AUTO_RESUME:-false}"
SWIFT_VENV="${SWIFT_VENV:-${PROJECT_ROOT}/.venv-swift311}"
UV_BIN="${UV_BIN:-/home/ubuntu/.local/bin/uv}"

export CUDA_VISIBLE_DEVICES

TS="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_NAME="${RUN_NAME:-swift_sft_sports_rule_hotfix_v31c_mixed_gpu5_${TS}}"
OUT_DIR="${OUT_DIR:-${STORAGE_ROOT}/outputs/${RUN_NAME}}"
TRACKING_ROOT="${TRACKING_ROOT:-${STORAGE_ROOT}/tracking}"

mkdir -p "$OUT_DIR" "$TRACKING_ROOT"/{tensorboard,mlruns,logs}

echo "[INFO] $(date -u +%Y-%m-%dT%H:%M:%SZ) CUDA=${CUDA_VISIBLE_DEVICES}"
echo "[INFO] base_model=$BASE_MODEL"
echo "[INFO] data_file=$DATA_FILE"
echo "[INFO] out_dir=$OUT_DIR"

if [ ! -x "$SWIFT_VENV/bin/python" ]; then
  echo "[ERROR] Missing venv at $SWIFT_VENV" >&2
  exit 1
fi

# Stop GPU5 upstream to free VRAM for training; restore on exit
echo "[INFO] stopping $GPU5_UPSTREAM_SERVICE"
systemctl --user stop "$GPU5_UPSTREAM_SERVICE" || true
sleep 5

_restore_gpu5() {
  echo "[INFO] restoring $GPU5_UPSTREAM_SERVICE"
  systemctl --user start "$GPU5_UPSTREAM_SERVICE" || true
}
trap _restore_gpu5 EXIT

source "$SWIFT_VENV/bin/activate"

export WANDB_DISABLED=true
export WANDB_MODE=disabled
export MLFLOW_TRACKING_URI="file:$TRACKING_ROOT/mlruns"
export MLFLOW_EXPERIMENT_NAME=qwen35a3b

RUN_NAME="$RUN_NAME" \
OUT_DIR="$OUT_DIR" \
BASE_MODEL="$BASE_MODEL" \
DATA_FILE="$DATA_FILE" \
TRACKING_ROOT="$TRACKING_ROOT" \
LOGGING_DIR="$TRACKING_ROOT/tensorboard/${RUN_NAME}" \
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
AUTO_RESUME="$AUTO_RESUME" \
SYSTEM_PROMPT="你是体育包网客服，按标准流程处理注单异常，无法确认时提交后台并同步运营联系包网。" \
"${PROJECT_ROOT}/scripts/run_swift_sft.sh"

# find latest checkpoint
latest_step="$(
  find "$OUT_DIR" -maxdepth 1 -type d -name 'checkpoint-*' -printf '%f\n' \
    | sed -n 's/^checkpoint-\([0-9]\+\)$/\1/p' | sort -n | tail -n1
)"
if [ -z "${latest_step:-}" ]; then
  echo "[ERROR] no checkpoint found in $OUT_DIR" >&2
  exit 1
fi
SFT_CKPT="${OUT_DIR}/checkpoint-${latest_step}"
echo "[INFO] SFT checkpoint: $SFT_CKPT"

# merge LoRA into base; fresh timestamp so dir is always new even if OUT_DIR was reused
MERGE_TS="$(date -u +%Y%m%dT%H%M%SZ)"
MERGED_OUT="${STORAGE_ROOT}/outputs/${RUN_NAME}_merged_${MERGE_TS}_vllm"
mkdir -p "$(dirname "$MERGED_OUT")"
echo "[INFO] merging -> $MERGED_OUT"

# CUDA_VISIBLE_DEVICES="" forces CPU-only merge, bypassing a peft 0.18.1 bug where
# get_balanced_memory raises TypeError for MoE no_split_module_classes (unhashable set).
CUDA_VISIBLE_DEVICES="" \
"$UV_BIN" run --python "$SWIFT_VENV/bin/python" swift export \
  --use_hf true \
  --model "$BASE_MODEL" \
  --adapters "$SFT_CKPT" \
  --merge_lora true \
  --safe_serialization true \
  --exist_ok true \
  --output_dir "$MERGED_OUT"

echo "[DONE] $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "[DONE] SFT_OUT=$OUT_DIR"
echo "[DONE] MERGED=$MERGED_OUT"
# trap will restore GPU5 upstream on EXIT
