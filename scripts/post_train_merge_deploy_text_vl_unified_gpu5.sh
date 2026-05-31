#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/ubuntu/qwen35a3b_finetune"
TEXT_TRAIN_ENV_FILE="${TEXT_TRAIN_ENV_FILE:-$PROJECT_ROOT/services/swift-sft-text-relation-gpu5.env}"
VL_TRAIN_ENV_FILE="${VL_TRAIN_ENV_FILE:-$PROJECT_ROOT/services/swift-sft-vl-relation-gpu4.env}"
TEXT_TRAIN_SERVICE="${TEXT_TRAIN_SERVICE:-swift-sft-text-relation-gpu5.service}"
VL_TRAIN_SERVICE="${VL_TRAIN_SERVICE:-swift-sft-vl-relation-gpu4.service}"
VLLM_ENV_FILE="${VLLM_ENV_FILE:-$PROJECT_ROOT/services/vllm-gpu5-qwen35-domain-text-vl.env}"
VLLM_SERVICE="${VLLM_SERVICE:-vllm-qwen35-domain-text-vl-gpu5.service}"
LOG_FILE="${LOG_FILE:-$PROJECT_ROOT/tracking/logs/post_train_merge_deploy_text_vl_unified_gpu5.log}"
DEFAULT_SERVED_MODEL_NAME="${DEFAULT_SERVED_MODEL_NAME:-qwen35a3b-domain-text-vl-gpu5}"
RUN_POST_DEPLOY_KNOWLEDGE_GATE="${RUN_POST_DEPLOY_KNOWLEDGE_GATE:-1}"
RELEASE_LOG_INDEX_FILE="${RELEASE_LOG_INDEX_FILE:-$PROJECT_ROOT/tracking/logs/release_log_index.jsonl}"

mkdir -p "$(dirname "$LOG_FILE")"
exec >>"$LOG_FILE" 2>&1

log() {
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"
}

prepare_clean_output_dir() {
  local dir="$1"
  local allowed_root="$PROJECT_ROOT/outputs/"
  case "$dir" in
    "$allowed_root"*)
      ;;
    *)
      log "refuse to clean non-output path: $dir"
      return 1
      ;;
  esac
  if [[ -e "$dir" ]]; then
    log "remove existing output dir: $dir"
    rm -rf "$dir"
  fi
}

wait_gpu_free() {
  local gpu_index="$1"
  local gpu_uuid
  local tries=0
  gpu_uuid="$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' -v target="$gpu_index" '$1==target{print $2}')"
  if [[ -z "${gpu_uuid:-}" ]]; then
    log "failed to resolve gpu uuid for index=${gpu_index}"
    return 1
  fi
  while true; do
    if ! nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory --format=csv,noheader,nounits 2>/dev/null \
      | awk -F', ' -v gpu="$gpu_uuid" '$1==gpu {found=1} END{exit found ? 0 : 1}'; then
      return 0
    fi
    tries=$((tries + 1))
    if [[ "$tries" -ge 120 ]]; then
      log "gpu ${gpu_index} still busy after waiting"
      return 1
    fi
    sleep 5
  done
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
  local rendered="$val"
  local escaped
  if [[ "$val" == *[[:space:]]* ]]; then
    escaped="${val//\\/\\\\}"
    escaped="${escaped//\"/\\\"}"
    rendered="\"$escaped\""
  fi
  rendered="${rendered//&/\\&}"
  if grep -q "^${key}=" "$file"; then
    sed -i "s#^${key}=.*#${key}=${rendered}#" "$file"
  else
    echo "${key}=${rendered}" >>"$file"
  fi
}

canonicalize_path() {
  local path="$1"
  readlink -f "$path"
}

text_stage_marker_file() {
  local stage_dir="$1"
  echo "$stage_dir/.text_adapter_source"
}

text_stage_matches_adapter() {
  local stage_dir="$1"
  local adapter_dir="$2"
  local marker_file
  marker_file="$(text_stage_marker_file "$stage_dir")"
  [[ -f "$stage_dir/config.json" && -f "$stage_dir/model.safetensors.index.json" && -f "$marker_file" ]] || return 1
  [[ "$(cat "$marker_file")" == "$(canonicalize_path "$adapter_dir")" ]]
}

write_text_stage_marker() {
  local stage_dir="$1"
  local adapter_dir="$2"
  local marker_file
  marker_file="$(text_stage_marker_file "$stage_dir")"
  printf '%s\n' "$(canonicalize_path "$adapter_dir")" >"$marker_file"
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

wait_service_finish() {
  local service="$1"
  log "waiting service=${service}"
  while systemctl --user is-active --quiet "$service"; do
    sleep 20
  done
}

wait_service_finish "$TEXT_TRAIN_SERVICE"
wait_service_finish "$VL_TRAIN_SERVICE"

text_result="$(systemctl --user show -p Result --value "$TEXT_TRAIN_SERVICE" || true)"
vl_result="$(systemctl --user show -p Result --value "$VL_TRAIN_SERVICE" || true)"
log "text training result=${text_result:-unknown}"
log "vl training result=${vl_result:-unknown}"
if [[ "$text_result" != "success" || "$vl_result" != "success" ]]; then
  log "one or both training services did not finish successfully; skip unified merge/deploy"
  exit 0
fi

TEXT_OUT_DIR="$(read_env_value "$TEXT_TRAIN_ENV_FILE" "OUT_DIR" || true)"
VL_OUT_DIR="$(read_env_value "$VL_TRAIN_ENV_FILE" "OUT_DIR" || true)"
TEXT_RUN_NAME="$(read_env_value "$TEXT_TRAIN_ENV_FILE" "RUN_NAME" || true)"
VL_RUN_NAME="$(read_env_value "$VL_TRAIN_ENV_FILE" "RUN_NAME" || true)"
TEXT_BASE_MODEL="$(read_env_value "$TEXT_TRAIN_ENV_FILE" "BASE_MODEL" || true)"
VL_BASE_MODEL="$(read_env_value "$VL_TRAIN_ENV_FILE" "BASE_MODEL" || true)"

if [[ -z "${TEXT_OUT_DIR:-}" || -z "${VL_OUT_DIR:-}" || -z "${TEXT_BASE_MODEL:-}" || -z "${VL_BASE_MODEL:-}" ]]; then
  log "missing required keys in train env files"
  exit 1
fi

if [[ "$TEXT_BASE_MODEL" != "$VL_BASE_MODEL" ]]; then
  log "base model mismatch: text=${TEXT_BASE_MODEL} vl=${VL_BASE_MODEL}"
  exit 1
fi

TEXT_ADAPTER_DIR="$(resolve_adapter_dir "$TEXT_OUT_DIR" || true)"
VL_ADAPTER_DIR="$(resolve_adapter_dir "$VL_OUT_DIR" || true)"
if [[ -z "${TEXT_ADAPTER_DIR:-}" || -z "${VL_ADAPTER_DIR:-}" ]]; then
  log "failed to resolve adapters: text=${TEXT_ADAPTER_DIR:-missing} vl=${VL_ADAPTER_DIR:-missing}"
  exit 1
fi

BASE_MODEL_MERGE="$(resolve_base_model_for_merge "$TEXT_BASE_MODEL")"
UV_BIN="${UV_BIN:-/home/ubuntu/.local/bin/uv}"
UV_PYTHON="${UV_PYTHON:-$PROJECT_ROOT/.venv-swift311/bin/python}"
MERGE_CUDA_VISIBLE_DEVICES="${MERGE_CUDA_VISIBLE_DEVICES:-4,5}"
DIRECT_MULTI_ADAPTER_MERGE="${DIRECT_MULTI_ADAPTER_MERGE:-false}"
TS="$(date -u +%Y%m%dT%H%M%SZ)"
TEXT_STAGE_OUT_DIR="${TEXT_STAGE_OUT_DIR_OVERRIDE:-$PROJECT_ROOT/outputs/qwen35a3b_domain_text_vl_unified_${TS}_text_stage}"
MERGED_OUT_DIR="${MERGED_OUT_DIR_OVERRIDE:-$PROJECT_ROOT/outputs/qwen35a3b_domain_text_vl_unified_${TS}_vllm}"

export CUDA_VISIBLE_DEVICES="$MERGE_CUDA_VISIBLE_DEVICES"

log "merge base model: $BASE_MODEL_MERGE"
log "merge text adapter: $TEXT_ADAPTER_DIR"
log "merge vl adapter: $VL_ADAPTER_DIR"
log "merge CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
log "merge stage1 output: $TEXT_STAGE_OUT_DIR"
log "merge final output: $MERGED_OUT_DIR"
log "direct multi-adapter merge: $DIRECT_MULTI_ADAPTER_MERGE"

if systemctl --user is-active --quiet "$VLLM_SERVICE"; then
  log "stopping unified GPU5 vLLM before merge: $VLLM_SERVICE"
  systemctl --user stop "$VLLM_SERVICE"
fi

IFS=',' read -r -a merge_gpu_list <<< "$MERGE_CUDA_VISIBLE_DEVICES"
for gpu_index in "${merge_gpu_list[@]}"; do
  wait_gpu_free "$gpu_index"
done

if [[ "$DIRECT_MULTI_ADAPTER_MERGE" == "true" ]]; then
  log "skip stage1 export and merge adapters directly from base model"
  prepare_clean_output_dir "$MERGED_OUT_DIR"
  "$UV_BIN" run --python "$UV_PYTHON" swift export \
    --use_hf true \
    --model "$BASE_MODEL_MERGE" \
    --adapters "$TEXT_ADAPTER_DIR" "$VL_ADAPTER_DIR" \
    --merge_lora true \
    --safe_serialization true \
    --output_dir "$MERGED_OUT_DIR"
elif [[ -f "$TEXT_STAGE_OUT_DIR/config.json" && -f "$TEXT_STAGE_OUT_DIR/model.safetensors.index.json" ]]; then
  if text_stage_matches_adapter "$TEXT_STAGE_OUT_DIR" "$TEXT_ADAPTER_DIR"; then
    log "reuse existing text stage output: $TEXT_STAGE_OUT_DIR"
  else
    log "stale text stage output detected, rebuild: $TEXT_STAGE_OUT_DIR"
    prepare_clean_output_dir "$TEXT_STAGE_OUT_DIR"
    "$UV_BIN" run --python "$UV_PYTHON" swift export \
      --use_hf true \
      --model "$BASE_MODEL_MERGE" \
      --adapters "$TEXT_ADAPTER_DIR" \
      --merge_lora true \
      --safe_serialization true \
      --exist_ok true \
      --output_dir "$TEXT_STAGE_OUT_DIR"
    write_text_stage_marker "$TEXT_STAGE_OUT_DIR" "$TEXT_ADAPTER_DIR"
  fi
else
  prepare_clean_output_dir "$TEXT_STAGE_OUT_DIR"
  "$UV_BIN" run --python "$UV_PYTHON" swift export \
    --use_hf true \
    --model "$BASE_MODEL_MERGE" \
    --adapters "$TEXT_ADAPTER_DIR" \
    --merge_lora true \
    --safe_serialization true \
    --exist_ok true \
    --output_dir "$TEXT_STAGE_OUT_DIR"
  write_text_stage_marker "$TEXT_STAGE_OUT_DIR" "$TEXT_ADAPTER_DIR"
  prepare_clean_output_dir "$MERGED_OUT_DIR"

  "$UV_BIN" run --python "$UV_PYTHON" swift export \
    --use_hf true \
    --model "$TEXT_STAGE_OUT_DIR" \
    --adapters "$VL_ADAPTER_DIR" \
    --merge_lora true \
    --safe_serialization true \
    --output_dir "$MERGED_OUT_DIR"
fi

if [[ ! -f "$MERGED_OUT_DIR/config.json" ]]; then
  log "merged output missing config.json: $MERGED_OUT_DIR"
  exit 1
fi

if needs_vllm_reexport "$MERGED_OUT_DIR"; then
  log "re-export merged weights for vLLM key compatibility: $MERGED_OUT_DIR"
  "$UV_PYTHON" "$PROJECT_ROOT/scripts/reexport_vllm_compatible_hf.py" \
    --src "$MERGED_OUT_DIR" \
    --dst "$MERGED_OUT_DIR" \
    --base "$BASE_MODEL_MERGE" \
    --in-place
fi

upsert_env_kv "$VLLM_ENV_FILE" "MODEL_DIR" "$MERGED_OUT_DIR"
upsert_env_kv "$VLLM_ENV_FILE" "SERVED_MODEL_NAME" "$DEFAULT_SERVED_MODEL_NAME"
upsert_env_kv "$VLLM_ENV_FILE" "SERVE_EXTRA_ARGS" "--enforce-eager --trust-remote-code"

log "restart unified GPU5 vLLM service: $VLLM_SERVICE"
systemctl --user restart "$VLLM_SERVICE"

API_KEY="$(read_env_value "$VLLM_ENV_FILE" "API_KEY" || true)"
PORT="$(read_env_value "$VLLM_ENV_FILE" "PORT" || true)"
for _ in $(seq 1 60); do
  if [[ -n "${API_KEY:-}" ]]; then
    if curl -fsS -H "Authorization: Bearer ${API_KEY}" "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
      break
    fi
  else
    if curl -fsS "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
      break
    fi
  fi
  sleep 2
done

if [[ "$RUN_POST_DEPLOY_KNOWLEDGE_GATE" == "1" || "$RUN_POST_DEPLOY_KNOWLEDGE_GATE" == "true" || "$RUN_POST_DEPLOY_KNOWLEDGE_GATE" == "yes" ]]; then
  GATE_TS="$(date -u +%Y%m%dT%H%M%SZ)"
  KNOWLEDGE_GATE_OUT_DIR="${KNOWLEDGE_GATE_OUT_DIR:-$PROJECT_ROOT/eval_outputs/post_deploy_knowledge_gate_text_vl_gpu5_${GATE_TS}}"
  log "running post-deploy sports knowledge gate: $KNOWLEDGE_GATE_OUT_DIR"
  BASE_URL_GPU5="${POST_DEPLOY_GATE_BASE_URL_GPU5:-http://127.0.0.1:8001}" \
  MODEL_GPU5="${POST_DEPLOY_GATE_MODEL_GPU5:-gpu5-v5.1f-merged}" \
  BASE_URL_GPU7="${POST_DEPLOY_GATE_BASE_URL_GPU7:-http://127.0.0.1:8010}" \
  MODEL_GPU7="${POST_DEPLOY_GATE_MODEL_GPU7:-gpu7-v5-combined}" \
  OUT_DIR="$KNOWLEDGE_GATE_OUT_DIR" \
  WORKERS="${POST_DEPLOY_GATE_WORKERS:-4}" \
  TIMEOUT_SEC="${POST_DEPLOY_GATE_TIMEOUT_SEC:-90}" \
  MAX_TOKENS="${POST_DEPLOY_GATE_MAX_TOKENS:-192}" \
  "$PROJECT_ROOT/scripts/run_sports_knowledge_gate.sh"
  log "post-deploy sports knowledge gate passed: $KNOWLEDGE_GATE_OUT_DIR"
fi

log "unified merge+deploy done"
log "text_run=${TEXT_RUN_NAME:-unknown}"
log "vl_run=${VL_RUN_NAME:-unknown}"
log "merged model dir: $MERGED_OUT_DIR"

python3 "$PROJECT_ROOT/scripts/append_release_log_index.py" \
  --index-file "$RELEASE_LOG_INDEX_FILE" \
  --ts "$TS" \
  --pipeline "post_train_merge_deploy_text_vl_unified_gpu5" \
  --model-dir "$MERGED_OUT_DIR" \
  --merged-model-dir "$MERGED_OUT_DIR" \
  --served-model-name "$DEFAULT_SERVED_MODEL_NAME" \
  --knowledge-gate-out-dir "${KNOWLEDGE_GATE_OUT_DIR:-}" \
  --knowledge-gate-gpu5-summary "${KNOWLEDGE_GATE_OUT_DIR:-}/gpu5_20260506/summary.json" \
  --knowledge-gate-gpu7-summary "${KNOWLEDGE_GATE_OUT_DIR:-}/gpu7_20260506/summary.json" \
  --knowledge-gate-kb40-summary "${KNOWLEDGE_GATE_OUT_DIR:-}/kb40_20260428/summary.json" \
  --extra-json "{\"text_run\":\"${TEXT_RUN_NAME:-}\",\"vl_run\":\"${VL_RUN_NAME:-}\",\"knowledge_gate_gpu7_ops8_summary\":\"${KNOWLEDGE_GATE_OUT_DIR:-}/gpu7_ops_feedback_8_20260509/summary.json\"}"
log "release log index: $RELEASE_LOG_INDEX_FILE"
