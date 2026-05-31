#!/usr/bin/env bash
set -euo pipefail

cd /home/ubuntu/qwen35a3b_finetune

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-5}
TS="$(date -u +%Y%m%dT%H%M%SZ)"

SFT_DATA_FILE=${SFT_DATA_FILE:-/home/ubuntu/qwen35a3b_finetune/datasets/sft_openai_messages.cleaned.v3_merged.single75_exact.repair_v1.jsonl}
DPO_DATA_FILE=${DPO_DATA_FILE:-/home/ubuntu/qwen35a3b_finetune/datasets/dpo_pairs.repair_v1.jsonl}
BASE_MODEL=${BASE_MODEL:-Qwen/Qwen3.5-35B-A3B}

SFT_RUN_NAME=${SFT_RUN_NAME:-swift_sft_repair_v1_gpu5_${TS}}
SFT_OUT_DIR=${SFT_OUT_DIR:-/home/ubuntu/qwen35a3b_finetune/outputs/${SFT_RUN_NAME}}

DPO_RUN_NAME=${DPO_RUN_NAME:-swift_dpo_repair_v1_gpu5_${TS}}
DPO_OUT_DIR=${DPO_OUT_DIR:-/home/ubuntu/qwen35a3b_finetune/outputs/${DPO_RUN_NAME}}

echo "[INFO] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[INFO] BASE_MODEL=${BASE_MODEL}"
echo "[INFO] SFT_DATA_FILE=${SFT_DATA_FILE}"
echo "[INFO] DPO_DATA_FILE=${DPO_DATA_FILE}"
echo "[INFO] SFT_OUT_DIR=${SFT_OUT_DIR}"
echo "[INFO] DPO_OUT_DIR=${DPO_OUT_DIR}"

RUN_NAME="${SFT_RUN_NAME}" \
OUT_DIR="${SFT_OUT_DIR}" \
BASE_MODEL="${BASE_MODEL}" \
DATA_FILE="${SFT_DATA_FILE}" \
AUTO_RESUME=false \
/home/ubuntu/qwen35a3b_finetune/run_gpu5_swift_sft.sh

latest_step="$(
  find "${SFT_OUT_DIR}" -maxdepth 1 -type d -name 'checkpoint-*' -printf '%f\n' \
    | sed -n 's/^checkpoint-\([0-9]\+\)$/\1/p' \
    | sort -n | tail -n1
)"

if [ -z "${latest_step:-}" ]; then
  echo "[ERROR] no SFT checkpoint found in ${SFT_OUT_DIR}" >&2
  exit 1
fi

SFT_CKPT="${SFT_OUT_DIR}/checkpoint-${latest_step}"
echo "[INFO] use SFT checkpoint for DPO: ${SFT_CKPT}"

RUN_NAME="${DPO_RUN_NAME}" \
OUT_DIR="${DPO_OUT_DIR}" \
BASE_MODEL="${BASE_MODEL}" \
SFT_CKPT="${SFT_CKPT}" \
DATA_FILE="${DPO_DATA_FILE}" \
AUTO_RESUME=false \
/home/ubuntu/qwen35a3b_finetune/run_gpu5_swift_dpo.sh

echo "[DONE] SFT+DPO finished"
echo "[DONE] SFT_OUT_DIR=${SFT_OUT_DIR}"
echo "[DONE] DPO_OUT_DIR=${DPO_OUT_DIR}"

