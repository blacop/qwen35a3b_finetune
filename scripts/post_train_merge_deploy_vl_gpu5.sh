#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/ubuntu/qwen35a3b_finetune"
TRAIN_ENV_FILE="$PROJECT_ROOT/services/swift-sft-vl-gpu5.env"
VLLM_ENV_FILE="${VLLM_ENV_FILE:-$PROJECT_ROOT/services/vllm-gpu4-qwen35vl.env}"
LOG_FILE="$PROJECT_ROOT/tracking/logs/post_train_merge_deploy_vl_gpu5.log"

TRAIN_SERVICE="swift-sft-vl-gpu5.service"
VLLM_SERVICE="${VLLM_SERVICE:-vllm-qwen35vl-gpu4.service}"
DEFAULT_SERVED_MODEL_NAME="${DEFAULT_SERVED_MODEL_NAME:-qwen35a3b-sft-vl-gpu4}"
RELEASE_LOG_INDEX_FILE="${RELEASE_LOG_INDEX_FILE:-$PROJECT_ROOT/tracking/logs/release_log_index.jsonl}"

UV_BIN="/home/ubuntu/.local/bin/uv"
UV_PYTHON="$PROJECT_ROOT/.venv-swift311/bin/python"

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

log "post-train watcher started, waiting service=$TRAIN_SERVICE"
while systemctl --user is-active --quiet "$TRAIN_SERVICE"; do
  sleep 20
done

result="$(systemctl --user show -p Result --value "$TRAIN_SERVICE" || true)"
state="$(systemctl --user show -p ActiveState --value "$TRAIN_SERVICE" || true)"
sub_state="$(systemctl --user show -p SubState --value "$TRAIN_SERVICE" || true)"
log "train service state: active=$state sub=$sub_state result=${result:-unknown}"
if [[ "$result" != "success" ]]; then
  log "train did not finish with success, skip merge/deploy"
  exit 0
fi

OUT_DIR="$(read_env_value "$TRAIN_ENV_FILE" "OUT_DIR" || true)"
BASE_MODEL="$(read_env_value "$TRAIN_ENV_FILE" "BASE_MODEL" || true)"
RUN_NAME="$(read_env_value "$TRAIN_ENV_FILE" "RUN_NAME" || true)"

if [[ -z "${OUT_DIR:-}" || -z "${BASE_MODEL:-}" || -z "${RUN_NAME:-}" ]]; then
  log "missing required keys in $TRAIN_ENV_FILE: OUT_DIR/BASE_MODEL/RUN_NAME"
  exit 1
fi

ADAPTER_DIR="$(resolve_adapter_dir "$OUT_DIR" || true)"
if [[ -z "${ADAPTER_DIR:-}" ]]; then
  log "adapter not found in OUT_DIR/checkpoint dirs: $OUT_DIR"
  exit 1
fi

TS="$(date -u +%Y%m%dT%H%M%SZ)"
MERGED_OUT_DIR="$PROJECT_ROOT/outputs/${RUN_NAME}_merged_${TS}_vllm"

log "merge start: base=$BASE_MODEL adapter=$ADAPTER_DIR out=$MERGED_OUT_DIR"
"$UV_BIN" run --python "$UV_PYTHON" swift export \
  --use_hf true \
  --model "$BASE_MODEL" \
  --adapters "$ADAPTER_DIR" \
  --merge_lora true \
  --safe_serialization true \
  --exist_ok true \
  --output_dir "$MERGED_OUT_DIR"
log "merge done"

if [[ ! -f "$MERGED_OUT_DIR/config.json" ]]; then
  log "merge output invalid (missing config.json): $MERGED_OUT_DIR"
  exit 1
fi

upsert_env_kv "$VLLM_ENV_FILE" "MODEL_DIR" "$MERGED_OUT_DIR"
upsert_env_kv "$VLLM_ENV_FILE" "SERVED_MODEL_NAME" "$DEFAULT_SERVED_MODEL_NAME"

# Keep the VL serving defaults minimal and compatible across merged checkpoints.
upsert_env_kv "$VLLM_ENV_FILE" "SERVE_EXTRA_ARGS" "--enforce-eager --trust-remote-code"
sed -i '/^VLLM_EXTRA_ARGS=/d' "$VLLM_ENV_FILE"

log "updated vl vllm env: MODEL_DIR=$MERGED_OUT_DIR SERVED_MODEL_NAME=$DEFAULT_SERVED_MODEL_NAME"

if systemctl --user is-active --quiet "$VLLM_SERVICE"; then
  systemctl --user restart "$VLLM_SERVICE"
  log "restarted active vl service: $VLLM_SERVICE"
else
  log "vl service not active; skip auto-start"
fi

for _ in $(seq 1 60); do
  port="$(read_env_value "$VLLM_ENV_FILE" "PORT" || true)"
  if [[ -z "${port:-}" ]]; then
    break
  fi
  if curl -fsS "http://127.0.0.1:${port}/v1/models" >/dev/null 2>&1; then
    log "vl vllm is up: 127.0.0.1:${port}"
    break
  fi
  sleep 2
done

log "post-train merge+deploy for vl service finished successfully"

python3 "$PROJECT_ROOT/scripts/append_release_log_index.py" \
  --index-file "$RELEASE_LOG_INDEX_FILE" \
  --ts "$TS" \
  --pipeline "post_train_merge_deploy_vl_gpu5" \
  --model-dir "$MERGED_OUT_DIR" \
  --merged-model-dir "$MERGED_OUT_DIR" \
  --served-model-name "$DEFAULT_SERVED_MODEL_NAME" \
  --extra-json "{\"run_name\":\"${RUN_NAME:-}\"}"
log "release log index: $RELEASE_LOG_INDEX_FILE"
