#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/ubuntu/qwen35a3b_finetune"
cd "$PROJECT_ROOT"

log() {
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"
}

latest_ckpt_step() {
  local out_dir="$1"
  find "$out_dir" -maxdepth 1 -type d -name 'checkpoint-*' -printf '%f\n' 2>/dev/null \
    | sed -n 's/^checkpoint-\([0-9]\+\)$/\1/p' | sort -n | tail -n1
}

wait_gpu_free() {
  local gpu_index="$1"
  local gpu_uuid
  local tries=0
  gpu_uuid="$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' -v idx="$gpu_index" '$1==idx{print $2}')"
  if [[ -z "${gpu_uuid:-}" ]]; then
    log "failed to resolve GPU${gpu_index} uuid"
    return 1
  fi
  while true; do
    if ! nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory --format=csv,noheader,nounits 2>/dev/null \
      | awk -F', ' -v gpu="$gpu_uuid" '$1==gpu {found=1} END{exit found ? 0 : 1}'; then
      return 0
    fi
    tries=$((tries + 1))
    if [[ "$tries" -ge "${GPU_FREE_WAIT_TRIES:-120}" ]]; then
      log "GPU${gpu_index} still busy after waiting"
      return 1
    fi
    sleep 5
  done
}

stop_vllm_for_training() {
  local service="$1"
  if [[ -z "$service" ]]; then
    return 0
  fi
  if systemctl --user is-active --quiet "$service"; then
    RESTORE_GPU7_VLLM=1
    log "stopping GPU7 vLLM service for training: $service"
    systemctl --user stop "$service" || true
    systemctl --user mask --runtime "$service" || true
  else
    systemctl --user mask --runtime "$service" || true
  fi
}

cleanup() {
  if [[ -n "${GPU7_VLLM_SERVICE:-}" ]]; then
    systemctl --user unmask "$GPU7_VLLM_SERVICE" || true
  fi
  if [[ "${RESTORE_GPU7_VLLM:-0}" = "1" && "${RESTORE_VLLM_AFTER_TRAIN:-1}" = "1" ]]; then
    log "restoring GPU7 vLLM service: $GPU7_VLLM_SERVICE"
    systemctl --user start "$GPU7_VLLM_SERVICE" || true
  fi
}
trap cleanup EXIT

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-7}"
GPU_INDEX="${GPU_INDEX:-7}"
TS="${TS:-$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_NAME_PREFIX="${RUN_NAME_PREFIX:-swift_sft_gpu7_vl_distill_from_gpu5}"
RUN_NAME="${RUN_NAME:-${RUN_NAME_PREFIX}_${TS}}"

BASE_MODEL="${BASE_MODEL:?BASE_MODEL is required}"
DATA_FILE="${DATA_FILE:?DATA_FILE is required}"
OUT_DIR="${OUT_DIR:-$PROJECT_ROOT/outputs/${RUN_NAME}}"
MERGED_OUT_DIR="${MERGED_OUT_DIR:-$PROJECT_ROOT/outputs/${RUN_NAME}_merged_${TS}_vllm}"
SERVING_KEYFIX_DIR="${SERVING_KEYFIX_DIR:-${MERGED_OUT_DIR}_serving_keyfix}"
TRACKING_ROOT="${TRACKING_ROOT:-$PROJECT_ROOT/tracking}"
RUN_LOG="${RUN_LOG:-$TRACKING_ROOT/logs/${RUN_NAME}.log}"
GPU7_VLLM_SERVICE="${GPU7_VLLM_SERVICE:-vllm-qwen35-gpu7-baseline.service}"

mkdir -p "$(dirname "$RUN_LOG")" "$OUT_DIR"
exec >>"$RUN_LOG" 2>&1

log "GPU5 -> GPU7 VL distillation start"
log "BASE_MODEL=$BASE_MODEL"
log "DATA_FILE=$DATA_FILE"
log "OUT_DIR=$OUT_DIR"
log "MERGED_OUT_DIR=$MERGED_OUT_DIR"
log "SERVING_KEYFIX_DIR=$SERVING_KEYFIX_DIR"

if [[ ! -d "$BASE_MODEL" ]]; then
  log "missing BASE_MODEL directory: $BASE_MODEL"
  exit 1
fi
if [[ ! -f "$DATA_FILE" ]]; then
  log "missing DATA_FILE: $DATA_FILE"
  exit 1
fi

if [[ "${STOP_GPU7_VLLM:-1}" = "1" ]]; then
  stop_vllm_for_training "$GPU7_VLLM_SERVICE"
fi
wait_gpu_free "$GPU_INDEX"
log "GPU${GPU_INDEX} is free"

RUN_NAME="$RUN_NAME" \
BASE_MODEL="$BASE_MODEL" \
DATA_FILE="$DATA_FILE" \
OUT_DIR="$OUT_DIR" \
TRACKING_ROOT="$TRACKING_ROOT" \
LOGGING_DIR="$TRACKING_ROOT/tensorboard/$RUN_NAME" \
REPORT_TO="${REPORT_TO:-tensorboard mlflow}" \
TEMPLATE="${TEMPLATE:-qwen3_5}" \
TARGET_MODULES="${TARGET_MODULES:-all-linear}" \
MAX_LEN="${MAX_LEN:-4096}" \
MAX_PIXELS="${MAX_PIXELS:-1048576}" \
EPOCHS="${EPOCHS:-0.5}" \
PER_DEVICE_BATCH="${PER_DEVICE_BATCH:-1}" \
GRAD_ACC="${GRAD_ACC:-8}" \
LR="${LR:-5e-6}" \
LORA_RANK="${LORA_RANK:-16}" \
LORA_ALPHA="${LORA_ALPHA:-32}" \
SAVE_STEPS="${SAVE_STEPS:-20}" \
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-3}" \
DATASET_NUM_PROC="${DATASET_NUM_PROC:-4}" \
AUTO_RESUME="${AUTO_RESUME:-false}" \
SYSTEM_PROMPT= \
"$PROJECT_ROOT/scripts/run_swift_sft_vl_gpu5.sh"

SFT_STEP="$(latest_ckpt_step "$OUT_DIR" || true)"
if [[ -z "${SFT_STEP:-}" || ! -d "$OUT_DIR/checkpoint-${SFT_STEP}" ]]; then
  log "failed to find SFT checkpoint in $OUT_DIR"
  exit 1
fi
SFT_CKPT="$OUT_DIR/checkpoint-${SFT_STEP}"
log "latest checkpoint=$SFT_CKPT"

UV_BIN="${UV_BIN:-/home/ubuntu/.local/bin/uv}"
SWIFT_VENV="${SWIFT_VENV:-$PROJECT_ROOT/.venv-swift311}"
UV_PYTHON="${UV_PYTHON:-$SWIFT_VENV/bin/python}"

log "merge start: base=$BASE_MODEL adapter=$SFT_CKPT out=$MERGED_OUT_DIR"
"$UV_BIN" run --python "$UV_PYTHON" swift export \
  --use_hf true \
  --model "$BASE_MODEL" \
  --adapters "$SFT_CKPT" \
  --merge_lora true \
  --safe_serialization true \
  --exist_ok true \
  --output_dir "$MERGED_OUT_DIR"

if [[ ! -f "$MERGED_OUT_DIR/config.json" ]]; then
  log "merged model invalid: missing config.json"
  exit 1
fi

log "create vLLM visual keyfix dir: $SERVING_KEYFIX_DIR"
"$UV_PYTHON" "$PROJECT_ROOT/scripts/create_vllm_visual_keyfix_dir.py" \
  --src "$MERGED_OUT_DIR" \
  --dst "$SERVING_KEYFIX_DIR" \
  --force

log "VL distillation candidate merged model: $MERGED_OUT_DIR"
log "VL distillation candidate serving keyfix: $SERVING_KEYFIX_DIR"
log "GPU5 -> GPU7 VL distillation done"
