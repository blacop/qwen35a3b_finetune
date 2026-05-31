#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/ubuntu/qwen35a3b_finetune"
TRAIN_ENV_FILE="${TRAIN_ENV_FILE:-$PROJECT_ROOT/services/swift-sft-vl-relation-gpu4.env}"
TRAIN_SERVICE="${TRAIN_SERVICE:-swift-sft-vl-relation-gpu4.service}"
VLLM_ENV_FILE="${VLLM_ENV_FILE:-$PROJECT_ROOT/services/vllm-gpu4-qwen35vl.env}"
VLLM_SERVICE="${VLLM_SERVICE:-vllm-qwen35vl-gpu4.service}"
LOG_FILE="${LOG_FILE:-$PROJECT_ROOT/tracking/logs/post_train_merge_deploy_vl_relation_gpu4.log}"
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

resolve_adapter_dir() {
  local out_dir="$1"
  if [[ -f "$out_dir/adapter_config.json" ]]; then
    echo "$out_dir"
    return 0
  fi
  local step
  step="$(latest_ckpt_step "$out_dir" || true)"
  if [[ -n "${step:-}" && -f "$out_dir/checkpoint-$step/adapter_config.json" ]]; then
    echo "$out_dir/checkpoint-$step"
    return 0
  fi
  return 1
}

read_env_value() {
  local file="$1"
  local key="$2"
  awk -F'=' -v k="$key" '$1==k {print substr($0, index($0, "=")+1)}' "$file" | tail -n1
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

log "post-train VL watcher started, waiting service=${TRAIN_SERVICE}"
while systemctl --user is-active --quiet "$TRAIN_SERVICE"; do
  sleep 20
done

result="$(systemctl --user show -p Result --value "$TRAIN_SERVICE" || true)"
log "training service result: ${result:-unknown}"
if [[ "$result" != "success" ]]; then
  log "training did not finish successfully; skip merge/deploy"
  exit 0
fi

OUT_DIR="$(read_env_value "$TRAIN_ENV_FILE" "OUT_DIR" || true)"
BASE_MODEL="$(read_env_value "$TRAIN_ENV_FILE" "BASE_MODEL" || true)"
RUN_NAME="$(read_env_value "$TRAIN_ENV_FILE" "RUN_NAME" || true)"
SERVED_MODEL_NAME="$(read_env_value "$VLLM_ENV_FILE" "SERVED_MODEL_NAME" || true)"

if [[ -z "${OUT_DIR:-}" || -z "${BASE_MODEL:-}" || -z "${RUN_NAME:-}" ]]; then
  log "missing required keys in $TRAIN_ENV_FILE"
  exit 1
fi

ADAPTER_DIR="$(resolve_adapter_dir "$OUT_DIR" || true)"
if [[ -z "${ADAPTER_DIR:-}" ]]; then
  log "adapter not found in ${OUT_DIR}"
  exit 1
fi

TS="$(date -u +%Y%m%dT%H%M%SZ)"
MERGED_OUT_DIR="$PROJECT_ROOT/outputs/${RUN_NAME}_merged_${TS}_vllm"
UV_BIN="/home/ubuntu/.local/bin/uv"
UV_PYTHON="$PROJECT_ROOT/.venv-swift311/bin/python"

log "merge start: base=$BASE_MODEL adapter=$ADAPTER_DIR out=$MERGED_OUT_DIR"
"$UV_BIN" run --python "$UV_PYTHON" swift export \
  --use_hf true \
  --model "$BASE_MODEL" \
  --adapters "$ADAPTER_DIR" \
  --merge_lora true \
  --safe_serialization true \
  --exist_ok true \
  --output_dir "$MERGED_OUT_DIR"

upsert_env_kv "$VLLM_ENV_FILE" "MODEL_DIR" "$MERGED_OUT_DIR"
upsert_env_kv "$VLLM_ENV_FILE" "SERVED_MODEL_NAME" "${SERVED_MODEL_NAME:-qwen35a3b-sft-vl-gpu4}"
upsert_env_kv "$VLLM_ENV_FILE" "SERVE_EXTRA_ARGS" "--enforce-eager --trust-remote-code"

log "restarting VL vLLM service: $VLLM_SERVICE"
systemctl --user restart "$VLLM_SERVICE"

API_KEY="$(read_env_value "$VLLM_ENV_FILE" "API_KEY" || true)"
for _ in $(seq 1 60); do
  if [[ -n "${API_KEY:-}" ]]; then
    if curl -fsS -H "Authorization: Bearer ${API_KEY}" "http://127.0.0.1:8015/v1/models" >/dev/null 2>&1; then
      break
    fi
  fi
  sleep 2
done

log "post-train VL merge+deploy done"
log "merged model dir: $MERGED_OUT_DIR"

python3 "$PROJECT_ROOT/scripts/append_release_log_index.py" \
  --index-file "$RELEASE_LOG_INDEX_FILE" \
  --ts "$TS" \
  --pipeline "post_train_merge_deploy_vl_relation_gpu4" \
  --model-dir "$MERGED_OUT_DIR" \
  --merged-model-dir "$MERGED_OUT_DIR" \
  --served-model-name "${SERVED_MODEL_NAME:-qwen35a3b-sft-vl-gpu4}" \
  --extra-json "{\"run_name\":\"${RUN_NAME:-}\"}"
log "release log index: $RELEASE_LOG_INDEX_FILE"
