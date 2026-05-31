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

needs_vllm_reexport() {
  local model_dir="$1"
  python3 - "$model_dir" <<'PY'
import json
import sys
from pathlib import Path

model_dir = Path(sys.argv[1])
index_path = model_dir / "model.safetensors.index.json"
if not index_path.exists():
    raise SystemExit(1)
with index_path.open("r", encoding="utf-8") as f:
    weight_map = json.load(f)["weight_map"]
needs = any(k.startswith("model.language_model.visual.") for k in weight_map)
raise SystemExit(0 if needs else 1)
PY
}

wait_gpu5_free() {
  local gpu_uuid
  local tries=0
  gpu_uuid="$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' '$1==5{print $2}')"
  if [[ -z "${gpu_uuid:-}" ]]; then
    log "failed to resolve GPU5 uuid"
    return 1
  fi
  while true; do
    if ! nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory --format=csv,noheader,nounits 2>/dev/null \
      | awk -F', ' -v gpu="$gpu_uuid" '$1==gpu {found=1} END{exit found ? 0 : 1}'; then
      return 0
    fi
    tries=$((tries + 1))
    if [[ "$tries" -ge 90 ]]; then
      log "GPU5 still busy after waiting"
      return 1
    fi
    sleep 5
  done
}

stop_prod_vllm() {
  local service="$1"
  local tries=0

  while true; do
    if ! systemctl --user is-active --quiet "$service"; then
      return 0
    fi

    tries=$((tries + 1))
    log "stopping production GPU5 vLLM service: ${service} (attempt ${tries})"
    if systemctl --user stop "$service"; then
      return 0
    fi

    if [[ "$tries" -ge 6 ]]; then
      log "failed to stop production GPU5 vLLM service after ${tries} attempts: ${service}"
      return 1
    fi

    sleep 5
  done
}

cleanup() {
  if [[ -n "${TEMP_VLLM_PID:-}" ]]; then
    kill "${TEMP_VLLM_PID}" 2>/dev/null || true
    wait "${TEMP_VLLM_PID}" 2>/dev/null || true
  fi
  if [[ "${RESTORE_PROD_VLLM:-0}" = "1" ]]; then
    log "restoring production GPU5 vLLM service: ${PROD_VLLM_SERVICE}"
    systemctl --user unmask "${PROD_VLLM_SERVICE}" || true
    systemctl --user start "${PROD_VLLM_SERVICE}" || true
  fi
}

trap cleanup EXIT

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-5}"
SWIFT_VENV="${SWIFT_VENV:-$PROJECT_ROOT/.venv-swift311}"
UV_BIN="${UV_BIN:-/home/ubuntu/.local/bin/uv}"
UV_PYTHON="${UV_PYTHON:-$SWIFT_VENV/bin/python}"
VLLM_BIN="${VLLM_BIN:-/home/ubuntu/.local/bin/vllm}"

BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3.5-35B-A3B}"
ADAPTERS="${ADAPTERS:-}"
DATA_FILE="${DATA_FILE:-$PROJECT_ROOT/datasets/nohallucination_hotfix_pack_20260425/sft_openai_messages.cleaned.v3_merged.single75_exact.repair_v3.tydata_pdfcurated.nometa.marketingfix.opsclean.20260425.nohallucination_hotfix_merged.jsonl}"
EVAL_INPUT_JSONL="${EVAL_INPUT_JSONL:-$PROJECT_ROOT/datasets/eval_marketing_repeat_hotfix_20260425.jsonl}"

RUN_NAME_PREFIX="${RUN_NAME_PREFIX:-swift_sft_nohallucination_hotfix_gpu5}"
OUT_ROOT="${OUT_ROOT:-$PROJECT_ROOT/outputs}"
EVAL_ROOT="${EVAL_ROOT:-$PROJECT_ROOT/eval_outputs}"
TRACKING_ROOT="${TRACKING_ROOT:-$PROJECT_ROOT/tracking}"

MAX_LEN="${MAX_LEN:-3072}"
EPOCHS="${EPOCHS:-1}"
PER_DEVICE_BATCH="${PER_DEVICE_BATCH:-1}"
GRAD_ACC="${GRAD_ACC:-16}"
LORA_RANK="${LORA_RANK:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
SAVE_STEPS="${SAVE_STEPS:-100}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-3}"
AUTO_RESUME="${AUTO_RESUME:-false}"

TEMP_PORT="${TEMP_PORT:-8025}"
TEMP_MAX_MODEL_LEN="${TEMP_MAX_MODEL_LEN:-4096}"
TEMP_GPU_MEMORY_UTILIZATION="${TEMP_GPU_MEMORY_UTILIZATION:-0.88}"
TEMP_TIMEOUT_S="${TEMP_TIMEOUT_S:-180}"

EVAL_MODEL_NAME="${EVAL_MODEL_NAME:-qwen35a3b-sft-text-gpu5-hotfix-candidate}"
EVAL_MAX_TOKENS="${EVAL_MAX_TOKENS:-256}"
EVAL_TEMPERATURE="${EVAL_TEMPERATURE:-0.0}"
EVAL_TOP_P="${EVAL_TOP_P:-0.95}"
EVAL_WORKERS="${EVAL_WORKERS:-8}"

PROD_VLLM_SERVICE="${PROD_VLLM_SERVICE:-vllm-qwen35-dpo-gpu5.service}"

TS="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_NAME="${RUN_NAME:-${RUN_NAME_PREFIX}_${TS}}"
OUT_DIR="${OUT_DIR:-${OUT_ROOT}/${RUN_NAME}}"
LOGGING_DIR="${LOGGING_DIR:-${TRACKING_ROOT}/tensorboard/${RUN_NAME}}"
MERGED_OUT_DIR="${MERGED_OUT_DIR:-${OUT_ROOT}/${RUN_NAME}_merged_${TS}_vllm}"
EVAL_OUT_DIR="${EVAL_OUT_DIR:-${EVAL_ROOT}/${RUN_NAME}_eval_${TS}}"
TEMP_VLLM_LOG="${TEMP_VLLM_LOG:-${TRACKING_ROOT}/logs/${RUN_NAME}_temp_vllm_${TS}.log}"
RUN_LOG="${RUN_LOG:-${TRACKING_ROOT}/logs/${RUN_NAME}.log}"

mkdir -p "${EVAL_OUT_DIR}" "$(dirname "$TEMP_VLLM_LOG")" "$(dirname "$RUN_LOG")"
exec >>"$RUN_LOG" 2>&1

log "hotfix SFT+eval pipeline start"
log "DATA_FILE=${DATA_FILE}"
log "EVAL_INPUT_JSONL=${EVAL_INPUT_JSONL}"
log "RUN_NAME=${RUN_NAME}"
log "OUT_DIR=${OUT_DIR}"
log "MERGED_OUT_DIR=${MERGED_OUT_DIR}"
log "EVAL_OUT_DIR=${EVAL_OUT_DIR}"

if [[ ! -f "$DATA_FILE" ]]; then
  log "missing DATA_FILE: $DATA_FILE"
  exit 1
fi
if [[ ! -f "$EVAL_INPUT_JSONL" ]]; then
  log "missing EVAL_INPUT_JSONL: $EVAL_INPUT_JSONL"
  exit 1
fi

if systemctl --user is-active --quiet "${PROD_VLLM_SERVICE}"; then
  RESTORE_PROD_VLLM=1
  systemctl --user mask --runtime "${PROD_VLLM_SERVICE}" || true
  stop_prod_vllm "${PROD_VLLM_SERVICE}"
fi

wait_gpu5_free
log "GPU5 is free"

RUN_NAME="${RUN_NAME}" \
OUT_DIR="${OUT_DIR}" \
LOGGING_DIR="${LOGGING_DIR}" \
BASE_MODEL="${BASE_MODEL}" \
ADAPTERS="${ADAPTERS}" \
DATA_FILE="${DATA_FILE}" \
MAX_LEN="${MAX_LEN}" \
EPOCHS="${EPOCHS}" \
PER_DEVICE_BATCH="${PER_DEVICE_BATCH}" \
GRAD_ACC="${GRAD_ACC}" \
LORA_RANK="${LORA_RANK}" \
LORA_ALPHA="${LORA_ALPHA}" \
SAVE_STEPS="${SAVE_STEPS}" \
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT}" \
AUTO_RESUME="${AUTO_RESUME}" \
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
"${PROJECT_ROOT}/run_gpu5_swift_sft.sh"

SFT_STEP="$(latest_ckpt_step "${OUT_DIR}" || true)"
if [[ -z "${SFT_STEP:-}" || ! -d "${OUT_DIR}/checkpoint-${SFT_STEP}" ]]; then
  log "failed to find SFT checkpoint in ${OUT_DIR}"
  exit 1
fi
SFT_CKPT="${OUT_DIR}/checkpoint-${SFT_STEP}"
log "latest checkpoint=${SFT_CKPT}"

BASE_MODEL_MERGE="${BASE_MODEL}"
if [[ "$BASE_MODEL_MERGE" == "Qwen/Qwen3.5-35B-A3B" ]]; then
  CACHE_ROOT="/home/ubuntu/.cache/huggingface/hub/models--Qwen--Qwen3.5-35B-A3B/snapshots"
  if [[ -d "$CACHE_ROOT" ]]; then
    CACHED_BASE="$(ls -dt "$CACHE_ROOT"/* 2>/dev/null | head -n1 || true)"
    if [[ -n "${CACHED_BASE:-}" && -d "$CACHED_BASE" ]]; then
      BASE_MODEL_MERGE="$CACHED_BASE"
    fi
  fi
fi
log "merge base model=${BASE_MODEL_MERGE}"

"${UV_BIN}" run --python "${UV_PYTHON}" swift export \
  --use_hf true \
  --model "${BASE_MODEL_MERGE}" \
  --adapters "${SFT_CKPT}" \
  --merge_lora true \
  --safe_serialization true \
  --exist_ok true \
  --output_dir "${MERGED_OUT_DIR}"

if [[ ! -f "${MERGED_OUT_DIR}/config.json" ]]; then
  log "merged model invalid: missing ${MERGED_OUT_DIR}/config.json"
  exit 1
fi

if needs_vllm_reexport "${MERGED_OUT_DIR}"; then
  log "re-export merged weights for vLLM key compatibility: ${MERGED_OUT_DIR}"
  "${UV_PYTHON}" "${PROJECT_ROOT}/scripts/reexport_vllm_compatible_hf.py" \
    --src "${MERGED_OUT_DIR}" \
    --dst "${MERGED_OUT_DIR}" \
    --base "${BASE_MODEL_MERGE}" \
    --in-place
fi

log "starting temp vLLM on 127.0.0.1:${TEMP_PORT}"
"${VLLM_BIN}" serve "${MERGED_OUT_DIR}" \
  --host 127.0.0.1 \
  --port "${TEMP_PORT}" \
  --tensor-parallel-size 1 \
  --max-model-len "${TEMP_MAX_MODEL_LEN}" \
  --dtype bfloat16 \
  --gpu-memory-utilization "${TEMP_GPU_MEMORY_UTILIZATION}" \
  --served-model-name "${EVAL_MODEL_NAME}" \
  --enforce-eager \
  --trust-remote-code \
  >>"${TEMP_VLLM_LOG}" 2>&1 &
TEMP_VLLM_PID=$!

READY=0
for _ in $(seq 1 120); do
  if curl -fsS "http://127.0.0.1:${TEMP_PORT}/v1/models" >/dev/null 2>&1; then
    READY=1
    break
  fi
  sleep 2
done
if [[ "$READY" != "1" ]]; then
  log "temp vLLM not ready, see ${TEMP_VLLM_LOG}"
  exit 1
fi
log "temp vLLM ready"

python3 "${PROJECT_ROOT}/scripts/eval_sports_customer_service.py" \
  --input-jsonl "${EVAL_INPUT_JSONL}" \
  --output-dir "${EVAL_OUT_DIR}" \
  --base-url "http://127.0.0.1:${TEMP_PORT}" \
  --model "${EVAL_MODEL_NAME}" \
  --temperature "${EVAL_TEMPERATURE}" \
  --top-p "${EVAL_TOP_P}" \
  --max-tokens "${EVAL_MAX_TOKENS}" \
  --workers "${EVAL_WORKERS}"

SUMMARY_PATH="${EVAL_OUT_DIR}/summary.json"
if [[ -f "$SUMMARY_PATH" ]]; then
  log "eval summary: $(cat "$SUMMARY_PATH")"
fi

log "hotfix SFT+eval pipeline done"
log "train out: ${OUT_DIR}"
log "merged model: ${MERGED_OUT_DIR}"
log "eval out: ${EVAL_OUT_DIR}"
