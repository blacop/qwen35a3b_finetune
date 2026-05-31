#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/ubuntu/qwen35a3b_finetune"
PROD_VLLM_SERVICE="${PROD_VLLM_SERVICE:-vllm-qwen35-dpo-gpu5.service}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-5}"

log() {
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"
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

if systemctl --user is-active --quiet "${PROD_VLLM_SERVICE}"; then
  log "stopping ${PROD_VLLM_SERVICE} before text SFT"
  systemctl --user stop "${PROD_VLLM_SERVICE}"
fi

wait_gpu_free "${CUDA_VISIBLE_DEVICES}"
log "gpu ${CUDA_VISIBLE_DEVICES} is free, starting text relation patch SFT"

exec "${PROJECT_ROOT}/scripts/run_swift_sft.sh"
