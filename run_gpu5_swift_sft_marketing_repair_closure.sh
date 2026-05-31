#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/ubuntu/qwen35a3b_finetune"
cd "$PROJECT_ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-5}"
TS="$(date -u +%Y%m%dT%H%M%SZ)"

UV_BIN="${UV_BIN:-/home/ubuntu/.local/bin/uv}"
SWIFT_VENV="${SWIFT_VENV:-$PROJECT_ROOT/.venv-swift311}"
UV_PYTHON="${UV_PYTHON:-$SWIFT_VENV/bin/python}"
VLLM_BIN="${VLLM_BIN:-/home/ubuntu/.local/bin/vllm}"

BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3.5-35B-A3B}"
PACK_DIR="${PACK_DIR:-$PROJECT_ROOT/datasets/marketing_repair_pack_${TS}}"
RUN_NAME="${RUN_NAME:-swift_sft_marketing_repair_gpu5_${TS}}"
SFT_OUT_DIR="${SFT_OUT_DIR:-$PROJECT_ROOT/outputs/${RUN_NAME}}"

SKIP_TRAIN="${SKIP_TRAIN:-false}"
SKIP_MERGE="${SKIP_MERGE:-false}"
AUTO_RESUME="${AUTO_RESUME:-false}"

EVAL_INPUT_JSONL="${EVAL_INPUT_JSONL:-$PROJECT_ROOT/datasets/eval_marketing_repetition_targeted_24.jsonl}"
EVAL_ROOT="${EVAL_ROOT:-$PROJECT_ROOT/eval_outputs/marketing_repair_closure_${TS}}"
CANDIDATE_EVAL_DIR="${CANDIDATE_EVAL_DIR:-$EVAL_ROOT/candidate}"
CURRENT_GPU5_BASELINE_DIR="${CURRENT_GPU5_BASELINE_DIR:-$PROJECT_ROOT/eval_outputs/marketing_eval_gpu5_20260425T0416Z}"
GPU7_REFERENCE_DIR="${GPU7_REFERENCE_DIR:-$PROJECT_ROOT/eval_outputs/marketing_eval_gpu7_20260425T0416Z}"

MERGED_OUT_DIR="${MERGED_OUT_DIR:-$PROJECT_ROOT/outputs/${RUN_NAME}_merged_${TS}_vllm}"
EVAL_PORT="${EVAL_PORT:-8025}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-${RUN_NAME}-eval}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.88}"
EVAL_TIMEOUT="${EVAL_TIMEOUT:-60}"
MAX_TOKENS="${MAX_TOKENS:-220}"
TEMPERATURE="${TEMPERATURE:-0.0}"
BAD_CASE_LIMIT="${BAD_CASE_LIMIT:-12}"
VLLM_LOG_FILE="${VLLM_LOG_FILE:-$PROJECT_ROOT/tracking/logs/${RUN_NAME}_closure_vllm_${TS}.log}"

mkdir -p "$EVAL_ROOT" "$(dirname "$VLLM_LOG_FILE")"

log() {
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"
}

latest_ckpt_step() {
  local out_dir="$1"
  find "$out_dir" -maxdepth 1 -type d -name 'checkpoint-*' -printf '%f\n' 2>/dev/null \
    | sed -n 's/^checkpoint-\([0-9]\+\)$/\1/p' | sort -n | tail -n1
}

resolve_base_model_for_merge() {
  local base_model="$1"
  if [[ "$base_model" == "Qwen/Qwen3.5-35B-A3B" ]]; then
    local cache_root="/home/ubuntu/.cache/huggingface/hub/models--Qwen--Qwen3.5-35B-A3B/snapshots"
    if [[ -d "$cache_root" ]]; then
      local cached
      cached="$(ls -dt "$cache_root"/* 2>/dev/null | head -n1 || true)"
      if [[ -n "${cached:-}" && -d "$cached" ]]; then
        echo "$cached"
        return 0
      fi
    fi
  fi
  echo "$base_model"
}

cleanup() {
  if [[ -n "${VLLM_PID:-}" ]]; then
    kill "$VLLM_PID" 2>/dev/null || true
    wait "$VLLM_PID" 2>/dev/null || true
  fi
}

trap cleanup EXIT

log "closure pipeline start"
log "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
log "RUN_NAME=$RUN_NAME"
log "SFT_OUT_DIR=$SFT_OUT_DIR"
log "EVAL_ROOT=$EVAL_ROOT"
log "SKIP_TRAIN=$SKIP_TRAIN SKIP_MERGE=$SKIP_MERGE"

if [[ "$SKIP_TRAIN" != "true" ]]; then
  log "launch training wrapper"
  RUN_NAME="$RUN_NAME" \
  OUT_DIR="$SFT_OUT_DIR" \
  PACK_DIR="$PACK_DIR" \
  BASE_MODEL="$BASE_MODEL" \
  AUTO_RESUME="$AUTO_RESUME" \
  bash "$PROJECT_ROOT/run_gpu5_swift_sft_marketing_repair.sh"
fi

sft_step="$(latest_ckpt_step "$SFT_OUT_DIR" || true)"
if [[ -z "${sft_step:-}" || ! -d "$SFT_OUT_DIR/checkpoint-$sft_step" ]]; then
  log "missing SFT checkpoint under $SFT_OUT_DIR"
  exit 1
fi
SFT_CKPT="$SFT_OUT_DIR/checkpoint-$sft_step"
if [[ ! -f "$SFT_CKPT/adapter_config.json" ]]; then
  log "invalid checkpoint: $SFT_CKPT"
  exit 1
fi
log "latest checkpoint=$SFT_CKPT"

if [[ "$SKIP_MERGE" != "true" ]]; then
  BASE_MODEL_MERGE="$(resolve_base_model_for_merge "$BASE_MODEL")"
  log "merge start: base=$BASE_MODEL_MERGE adapter=$SFT_CKPT out=$MERGED_OUT_DIR"
  "$UV_BIN" run --python "$UV_PYTHON" swift export \
    --use_hf true \
    --model "$BASE_MODEL_MERGE" \
    --adapters "$SFT_CKPT" \
    --merge_lora true \
    --safe_serialization true \
    --exist_ok true \
    --output_dir "$MERGED_OUT_DIR"
  log "merge done"
fi

if [[ ! -f "$MERGED_OUT_DIR/config.json" ]]; then
  log "merged model invalid: missing $MERGED_OUT_DIR/config.json"
  exit 1
fi

log "start temp vllm for eval on :$EVAL_PORT"
"$VLLM_BIN" serve "$MERGED_OUT_DIR" \
  --host 127.0.0.1 \
  --port "$EVAL_PORT" \
  --tensor-parallel-size 1 \
  --max-model-len "$MAX_MODEL_LEN" \
  --dtype bfloat16 \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --enforce-eager \
  --trust-remote-code \
  >>"$VLLM_LOG_FILE" 2>&1 &
VLLM_PID=$!

ready=0
for _ in $(seq 1 90); do
  if curl -fsS "http://127.0.0.1:${EVAL_PORT}/v1/models" >/dev/null 2>&1; then
    ready=1
    break
  fi
  sleep 2
done
if [[ "$ready" != "1" ]]; then
  log "temp vllm not ready, see $VLLM_LOG_FILE"
  exit 1
fi
log "temp vllm ready"

python3 "$PROJECT_ROOT/scripts/eval_marketing_repetition.py" \
  --input-jsonl "$EVAL_INPUT_JSONL" \
  --base-url "http://127.0.0.1:${EVAL_PORT}" \
  --model "$SERVED_MODEL_NAME" \
  --max-tokens "$MAX_TOKENS" \
  --temperature "$TEMPERATURE" \
  --timeout "$EVAL_TIMEOUT" \
  --output-dir "$CANDIDATE_EVAL_DIR"

if [[ -d "$CURRENT_GPU5_BASELINE_DIR" ]]; then
  mkdir -p "$EVAL_ROOT/compare_vs_current_gpu5"
  python3 "$PROJECT_ROOT/scripts/compare_marketing_eval_runs.py" \
    --a-dir "$CANDIDATE_EVAL_DIR" \
    --a-name "${RUN_NAME}" \
    --b-dir "$CURRENT_GPU5_BASELINE_DIR" \
    --b-name "current_gpu5" \
    --output "$EVAL_ROOT/compare_vs_current_gpu5/REPORT.md" \
    --bad-case-limit "$BAD_CASE_LIMIT"
fi

if [[ -d "$GPU7_REFERENCE_DIR" ]]; then
  mkdir -p "$EVAL_ROOT/compare_vs_gpu7_reference"
  python3 "$PROJECT_ROOT/scripts/compare_marketing_eval_runs.py" \
    --a-dir "$CANDIDATE_EVAL_DIR" \
    --a-name "${RUN_NAME}" \
    --b-dir "$GPU7_REFERENCE_DIR" \
    --b-name "gpu7_reference" \
    --output "$EVAL_ROOT/compare_vs_gpu7_reference/REPORT.md" \
    --bad-case-limit "$BAD_CASE_LIMIT"
fi

log "closure pipeline done"
log "candidate eval: $CANDIDATE_EVAL_DIR"
log "compare vs gpu5: $EVAL_ROOT/compare_vs_current_gpu5/REPORT.md"
log "compare vs gpu7: $EVAL_ROOT/compare_vs_gpu7_reference/REPORT.md"
log "merged model dir: $MERGED_OUT_DIR"
