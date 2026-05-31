#!/usr/bin/env bash
set -euo pipefail

cd /home/ubuntu/qwen35a3b_finetune

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-5}"
TS="$(date -u +%Y%m%dT%H%M%SZ)"

FAILURE_CSV="${FAILURE_CSV:-/home/ubuntu/qwen35a3b_finetune/eval_outputs/deployed_aibot_proxy8010_prod500_fix4_20260415T132323Z/failure_cases.csv}"
EVAL_JSONL="${EVAL_JSONL:-/home/ubuntu/qwen35a3b_finetune/datasets/eval_sports_customer_prod_500.jsonl}"
TARGET_INTENTS="${TARGET_INTENTS:-串关规则,滚球延迟,赔率异常,赛事变更}"
CONFUSION_INTENT="${CONFUSION_INTENT:-注单异常}"

BASE_SFT="${BASE_SFT:-/home/ubuntu/qwen35a3b_finetune/datasets/sft_openai_messages.cleaned.v3_merged.single75_exact.repair_v3.tydata_pdfcurated.nometa.jsonl}"
BASE_DPO="${BASE_DPO:-/home/ubuntu/qwen35a3b_finetune/datasets/dpo_pairs.repair_v3.tydata_pdfcurated.nometa.jsonl}"

PACK_DIR="${PACK_DIR:-/home/ubuntu/qwen35a3b_finetune/datasets/fix4_confusion_repair_pack_${TS}}"
MAX_SAMPLES_PER_INTENT="${MAX_SAMPLES_PER_INTENT:-250}"
AUG_PER_SAMPLE="${AUG_PER_SAMPLE:-2}"
SEED="${SEED:-42}"

BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3.5-35B-A3B}"
SFT_RUN_NAME="${SFT_RUN_NAME:-swift_sft_fix4_confusion_gpu5_${TS}}"
SFT_OUT_DIR="${SFT_OUT_DIR:-/home/ubuntu/qwen35a3b_finetune/outputs/${SFT_RUN_NAME}}"
DPO_RUN_NAME="${DPO_RUN_NAME:-swift_dpo_fix4_confusion_gpu5_${TS}}"
DPO_OUT_DIR="${DPO_OUT_DIR:-/home/ubuntu/qwen35a3b_finetune/outputs/${DPO_RUN_NAME}}"

echo "[INFO] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[INFO] building confusion repair pack from ${FAILURE_CSV}"

python3 /home/ubuntu/qwen35a3b_finetune/scripts/build_confusion_repair_from_failures.py \
  --failure-csv "${FAILURE_CSV}" \
  --eval-jsonl "${EVAL_JSONL}" \
  --target-intents "${TARGET_INTENTS}" \
  --confusion-intent "${CONFUSION_INTENT}" \
  --max-samples-per-intent "${MAX_SAMPLES_PER_INTENT}" \
  --aug-per-sample "${AUG_PER_SAMPLE}" \
  --seed "${SEED}" \
  --output-dir "${PACK_DIR}" \
  --base-sft "${BASE_SFT}" \
  --base-dpo "${BASE_DPO}" \
  --write-merged

SFT_DATA_FILE="${PACK_DIR}/fix4_confusion_sft_merged.jsonl"
DPO_DATA_FILE="${PACK_DIR}/fix4_confusion_dpo_merged.jsonl"

if [ ! -f "${SFT_DATA_FILE}" ] || [ ! -f "${DPO_DATA_FILE}" ]; then
  echo "[ERROR] merged incremental datasets missing in ${PACK_DIR}" >&2
  exit 1
fi

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

echo "[DONE] fix4 confusion repair SFT+DPO finished"
echo "[DONE] PACK_DIR=${PACK_DIR}"
echo "[DONE] SFT_OUT_DIR=${SFT_OUT_DIR}"
echo "[DONE] DPO_OUT_DIR=${DPO_OUT_DIR}"
