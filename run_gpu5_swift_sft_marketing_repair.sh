#!/usr/bin/env bash
set -euo pipefail

cd /home/ubuntu/qwen35a3b_finetune

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-5}"
TS="$(date -u +%Y%m%dT%H%M%SZ)"

BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3.5-35B-A3B}"
BASE_SFT_DATA="${BASE_SFT_DATA:-/home/ubuntu/qwen35a3b_finetune/datasets/weak_intent_targeted_12k_20260416T055426Z/sft_openai_messages.cleaned.v3_merged.single75_exact.repair_v3.tydata_pdfcurated.nometa.weak12k_merged.decontam.jsonl}"
REPAIR_TEMPLATE="${REPAIR_TEMPLATE:-/home/ubuntu/qwen35a3b_finetune/templates/repair_pop_marketing_repetition_sft_template.jsonl}"
PACK_DIR="${PACK_DIR:-/home/ubuntu/qwen35a3b_finetune/datasets/marketing_repair_pack_${TS}}"

RUN_NAME="${RUN_NAME:-swift_sft_marketing_repair_gpu5_${TS}}"
OUT_DIR="${OUT_DIR:-/home/ubuntu/qwen35a3b_finetune/outputs/${RUN_NAME}}"
AUTO_RESUME="${AUTO_RESUME:-false}"
PREPARE_ONLY="${PREPARE_ONLY:-false}"

echo "[INFO] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[INFO] BASE_MODEL=${BASE_MODEL}"
echo "[INFO] BASE_SFT_DATA=${BASE_SFT_DATA}"
echo "[INFO] REPAIR_TEMPLATE=${REPAIR_TEMPLATE}"
echo "[INFO] PACK_DIR=${PACK_DIR}"
echo "[INFO] OUT_DIR=${OUT_DIR}"
echo "[INFO] PREPARE_ONLY=${PREPARE_ONLY}"

python3 /home/ubuntu/qwen35a3b_finetune/scripts/build_marketing_repair_sft_dataset.py \
  --base "${BASE_SFT_DATA}" \
  --repair "${REPAIR_TEMPLATE}" \
  --output-dir "${PACK_DIR}"

SFT_DATA_FILE="${PACK_DIR}/$(basename "${BASE_SFT_DATA%.jsonl}").marketing_repair_merged.jsonl"
SUMMARY_FILE="${PACK_DIR}/marketing_repair_merge_summary.json"

if [ ! -f "${SFT_DATA_FILE}" ]; then
  echo "[ERROR] merged SFT dataset missing: ${SFT_DATA_FILE}" >&2
  exit 1
fi

echo "[INFO] SFT_DATA_FILE=${SFT_DATA_FILE}"
echo "[INFO] SUMMARY_FILE=${SUMMARY_FILE}"

if [ "${PREPARE_ONLY,,}" = "true" ]; then
  echo "[DONE] dataset pack prepared only"
  echo "[DONE] PACK_DIR=${PACK_DIR}"
  echo "[DONE] SFT_DATA_FILE=${SFT_DATA_FILE}"
  echo "[DONE] SUMMARY_FILE=${SUMMARY_FILE}"
  exit 0
fi

RUN_NAME="${RUN_NAME}" \
OUT_DIR="${OUT_DIR}" \
BASE_MODEL="${BASE_MODEL}" \
DATA_FILE="${SFT_DATA_FILE}" \
AUTO_RESUME="${AUTO_RESUME}" \
/home/ubuntu/qwen35a3b_finetune/run_gpu5_swift_sft.sh

echo "[DONE] GPU5 marketing repair SFT finished"
echo "[DONE] PACK_DIR=${PACK_DIR}"
echo "[DONE] SFT_DATA_FILE=${SFT_DATA_FILE}"
echo "[DONE] SUMMARY_FILE=${SUMMARY_FILE}"
echo "[DONE] OUT_DIR=${OUT_DIR}"
