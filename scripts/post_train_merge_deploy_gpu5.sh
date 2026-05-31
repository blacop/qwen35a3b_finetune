#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/ubuntu/qwen35a3b_finetune"
TRAIN_ENV_FILE="$PROJECT_ROOT/services/swift-sft-dpo-resume.env"
VLLM_ENV_FILE="$PROJECT_ROOT/services/vllm-gpu5-dpo.env"
LOG_FILE="$PROJECT_ROOT/tracking/logs/post_train_merge_deploy_gpu5.log"

TRAIN_SERVICE="swift-sft-dpo-resume.service"
VLLM_SERVICE="vllm-qwen35-dpo-gpu5.service"
PROXY_SERVICE="dpo-guardrails-proxy.service"
RUN_POST_DEPLOY_KNOWLEDGE_GATE="${RUN_POST_DEPLOY_KNOWLEDGE_GATE:-1}"
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

log "post-deploy watcher started"
log "waiting for training service to finish: $TRAIN_SERVICE"

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

sft_step="$(latest_ckpt_step "$SFT_OUT_DIR" || true)"
dpo_step="$(latest_ckpt_step "$DPO_OUT_DIR" || true)"
if [[ -z "${sft_step:-}" || -z "${dpo_step:-}" ]]; then
  log "failed to find checkpoints in SFT_OUT_DIR or DPO_OUT_DIR"
  exit 1
fi

SFT_CKPT="$SFT_OUT_DIR/checkpoint-$sft_step"
DPO_CKPT="$DPO_OUT_DIR/checkpoint-$dpo_step"
if [[ ! -f "$SFT_CKPT/adapter_config.json" || ! -f "$DPO_CKPT/adapter_config.json" ]]; then
  log "adapter files not found in checkpoints"
  exit 1
fi

BASE_MODEL_MERGE="$(resolve_base_model_for_merge "$BASE_MODEL")"
TS="$(date -u +%Y%m%dT%H%M%SZ)"
MERGED_OUT_DIR="$PROJECT_ROOT/outputs/${DPO_RUN_NAME}_merged_${TS}_vllm"
mkdir -p "$MERGED_OUT_DIR"

log "merge base model: $BASE_MODEL_MERGE"
log "merge adapters: $SFT_CKPT + $DPO_CKPT"
log "merge output: $MERGED_OUT_DIR"

"$UV_BIN" run --python "$UV_PYTHON" swift export \
  --use_hf true \
  --model "$BASE_MODEL_MERGE" \
  --adapters "$SFT_CKPT" "$DPO_CKPT" \
  --merge_lora true \
  --safe_serialization true \
  --exist_ok true \
  --output_dir "$MERGED_OUT_DIR"

if grep -q '^MODEL_DIR=' "$VLLM_ENV_FILE"; then
  sed -i "s#^MODEL_DIR=.*#MODEL_DIR=$MERGED_OUT_DIR#" "$VLLM_ENV_FILE"
else
  echo "MODEL_DIR=$MERGED_OUT_DIR" >>"$VLLM_ENV_FILE"
fi
log "updated MODEL_DIR in $VLLM_ENV_FILE"

systemctl --user restart "$VLLM_SERVICE"
systemctl --user restart "$PROXY_SERVICE"
log "restarted services: $VLLM_SERVICE, $PROXY_SERVICE"

for _ in $(seq 1 30); do
  if curl -fsS "http://127.0.0.1:8001/health" >/dev/null 2>&1; then
    break
  fi
  sleep 2
done
for _ in $(seq 1 30); do
  if curl -fsS "http://127.0.0.1:8010/health" >/dev/null 2>&1; then
    break
  fi
  sleep 2
done

if [[ "$RUN_POST_DEPLOY_KNOWLEDGE_GATE" == "1" || "$RUN_POST_DEPLOY_KNOWLEDGE_GATE" == "true" || "$RUN_POST_DEPLOY_KNOWLEDGE_GATE" == "yes" ]]; then
  GATE_TS="$(date -u +%Y%m%dT%H%M%SZ)"
  KNOWLEDGE_GATE_OUT_DIR="${KNOWLEDGE_GATE_OUT_DIR:-$PROJECT_ROOT/eval_outputs/post_deploy_knowledge_gate_gpu5_${GATE_TS}}"
  log "running post-deploy sports knowledge gate: $KNOWLEDGE_GATE_OUT_DIR"
  BASE_URL_GPU5="http://127.0.0.1:8001" \
  MODEL_GPU5="${POST_DEPLOY_GATE_MODEL_GPU5:-gpu5-v5.1f-merged}" \
  BASE_URL_GPU7="http://127.0.0.1:8010" \
  MODEL_GPU7="${POST_DEPLOY_GATE_MODEL_GPU7:-gpu7-v5-combined}" \
  OUT_DIR="$KNOWLEDGE_GATE_OUT_DIR" \
  WORKERS="${POST_DEPLOY_GATE_WORKERS:-4}" \
  TIMEOUT_SEC="${POST_DEPLOY_GATE_TIMEOUT_SEC:-90}" \
  MAX_TOKENS="${POST_DEPLOY_GATE_MAX_TOKENS:-192}" \
  "$PROJECT_ROOT/scripts/run_sports_knowledge_gate.sh"
  log "post-deploy sports knowledge gate passed: $KNOWLEDGE_GATE_OUT_DIR"
fi

log "post-deploy done"
log "merged model dir: $MERGED_OUT_DIR"

python3 "$PROJECT_ROOT/scripts/append_release_log_index.py" \
  --index-file "$RELEASE_LOG_INDEX_FILE" \
  --ts "$TS" \
  --pipeline "post_train_merge_deploy_gpu5" \
  --model-dir "$MERGED_OUT_DIR" \
  --merged-model-dir "$MERGED_OUT_DIR" \
  --knowledge-gate-out-dir "${KNOWLEDGE_GATE_OUT_DIR:-}" \
  --knowledge-gate-gpu5-summary "${KNOWLEDGE_GATE_OUT_DIR:-}/gpu5_20260506/summary.json" \
  --knowledge-gate-gpu7-summary "${KNOWLEDGE_GATE_OUT_DIR:-}/gpu7_20260506/summary.json" \
  --knowledge-gate-kb40-summary "${KNOWLEDGE_GATE_OUT_DIR:-}/kb40_20260428/summary.json" \
  --extra-json "{\"knowledge_gate_gpu7_ops8_summary\":\"${KNOWLEDGE_GATE_OUT_DIR:-}/gpu7_ops_feedback_8_20260509/summary.json\"}"
log "release log index: $RELEASE_LOG_INDEX_FILE"
