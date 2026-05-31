#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/ubuntu/qwen35a3b_finetune"
TRAIN_ENV_FILE="${TRAIN_ENV_FILE:-$PROJECT_ROOT/services/swift-sft-text-relation-gpu5.env}"
TRAIN_SERVICE="${TRAIN_SERVICE:-swift-sft-text-relation-gpu5.service}"
VLLM_ENV_FILE="${VLLM_ENV_FILE:-$PROJECT_ROOT/services/vllm-gpu5-dpo.env}"
VLLM_SERVICE="${VLLM_SERVICE:-vllm-qwen35-dpo-gpu5.service}"
LOG_FILE="${LOG_FILE:-$PROJECT_ROOT/tracking/logs/post_train_merge_deploy_text_relation_gpu5.log}"
RELEASE_LOG_INDEX_FILE="${RELEASE_LOG_INDEX_FILE:-$PROJECT_ROOT/tracking/logs/release_log_index.jsonl}"

mkdir -p "$(dirname "$LOG_FILE")"
exec >>"$LOG_FILE" 2>&1

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

upsert_env_kv() {
  local file="$1"
  local key="$2"
  local val="$3"
  if grep -q "^${key}=" "$file"; then
    sed -i "s#^${key}=.*#${key}=${val}#" "$file"
  else
    echo "${key}=${val}" >>"$file"
  fi
}

log "post-train text watcher started, waiting service=${TRAIN_SERVICE}"
while systemctl --user is-active --quiet "$TRAIN_SERVICE"; do
  sleep 20
done

result="$(systemctl --user show -p Result --value "$TRAIN_SERVICE" || true)"
log "training service result: ${result:-unknown}"
if [[ "$result" != "success" ]]; then
  log "training did not finish successfully; skip merge/deploy"
  exit 0
fi

set -a
source "$TRAIN_ENV_FILE"
set +a

UV_BIN="${UV_BIN:-/home/ubuntu/.local/bin/uv}"
UV_PYTHON="${UV_PYTHON:-$PROJECT_ROOT/.venv-swift311/bin/python}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3.5-35B-A3B}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen35a3b-sft-text-gpu5}"

sft_step="$(latest_ckpt_step "$OUT_DIR" || true)"
if [[ -z "${sft_step:-}" || ! -f "$OUT_DIR/checkpoint-$sft_step/adapter_config.json" ]]; then
  log "failed to find adapter checkpoint in OUT_DIR=${OUT_DIR}"
  exit 1
fi

SFT_CKPT="$OUT_DIR/checkpoint-$sft_step"
BASE_MODEL_MERGE="$(resolve_base_model_for_merge "$BASE_MODEL")"
TS="$(date -u +%Y%m%dT%H%M%SZ)"
MERGED_OUT_DIR="$PROJECT_ROOT/outputs/${RUN_NAME}_merged_${TS}_vllm"

log "merge base model: $BASE_MODEL_MERGE"
log "merge adapter: $SFT_CKPT"
log "merge output: $MERGED_OUT_DIR"

"$UV_BIN" run --python "$UV_PYTHON" swift export \
  --use_hf true \
  --model "$BASE_MODEL_MERGE" \
  --adapters "$SFT_CKPT" \
  --merge_lora true \
  --safe_serialization true \
  --exist_ok true \
  --output_dir "$MERGED_OUT_DIR"

upsert_env_kv "$VLLM_ENV_FILE" "MODEL_DIR" "$MERGED_OUT_DIR"
upsert_env_kv "$VLLM_ENV_FILE" "SERVED_MODEL_NAME" "$SERVED_MODEL_NAME"

log "restarting text vLLM service: $VLLM_SERVICE"
systemctl --user restart "$VLLM_SERVICE"

for _ in $(seq 1 60); do
  if curl -fsS "http://127.0.0.1:8001/v1/models" >/dev/null 2>&1; then
    break
  fi
  sleep 2
done

log "post-train text merge+deploy done"
log "merged model dir: $MERGED_OUT_DIR"

python3 "$PROJECT_ROOT/scripts/append_release_log_index.py" \
  --index-file "$RELEASE_LOG_INDEX_FILE" \
  --ts "$TS" \
  --pipeline "post_train_merge_deploy_text_relation_gpu5" \
  --model-dir "$MERGED_OUT_DIR" \
  --merged-model-dir "$MERGED_OUT_DIR" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --extra-json "{\"run_name\":\"${RUN_NAME:-}\"}"
log "release log index: $RELEASE_LOG_INDEX_FILE"
