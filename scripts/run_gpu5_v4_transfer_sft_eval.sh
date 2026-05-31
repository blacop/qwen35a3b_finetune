#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/ubuntu/qwen35a3b_finetune"
ENV_FILE="${ENV_FILE:-$PROJECT_ROOT/services/swift-sft-gpu5-v4-transfer.env}"

if [[ -f "$ENV_FILE" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
fi

V4_DATASET="${V4_DATASET:-$PROJECT_ROOT/datasets/sft_merged_v4_with_fix.jsonl}"
HOTFIX_DATASET="${HOTFIX_DATASET:-$PROJECT_ROOT/datasets/nohallucination_hotfix_pack_20260425/sft_openai_messages.cleaned.v3_merged.single75_exact.repair_v3.tydata_pdfcurated.nometa.marketingfix.opsclean.20260425.nohallucination_hotfix_merged.jsonl}"
PARLAY_SEEDS="${PARLAY_SEEDS:-$PROJECT_ROOT/datasets/v5_fix_parlay_regression_seeds.jsonl}"
INTENT_TRANSFER_SOURCES="${INTENT_TRANSFER_SOURCES:-$PROJECT_ROOT/datasets/eval_sports_customer_canary_120.jsonl:$PROJECT_ROOT/datasets/eval_sports_customer_targeted_parlay_v1_120.jsonl:$PROJECT_ROOT/datasets/eval_sports_customer_prod_500.jsonl}"
DOMAIN_GLOSSARY_JSONL="${DOMAIN_GLOSSARY_JSONL:-$PROJECT_ROOT/rag/kb/v2/glossary/core_terms.jsonl}"
DOMAIN_TERMS="${DOMAIN_TERMS:-包网:JT包网:体育包网:白标:包网商:代理:总代:流水:风控:限红:RTP:PNL:负盈利:返水:洗码:抽水:对冲:三方}"
EXTRA_SFT_SOURCES="${EXTRA_SFT_SOURCES:-}"
DATASET_RAW_OUTPUT_JSONL="${DATASET_RAW_OUTPUT_JSONL:-$PROJECT_ROOT/datasets/gpu5_v4_transfer_solution/sft_gpu5_v4_transfer_solution.raw.jsonl}"
DATASET_OUTPUT_JSONL="${DATASET_OUTPUT_JSONL:-$PROJECT_ROOT/datasets/gpu5_v4_transfer_solution/sft_gpu5_v4_transfer_solution.cleaned.jsonl}"
DATASET_SUMMARY_JSON="${DATASET_SUMMARY_JSON:-$PROJECT_ROOT/datasets/gpu5_v4_transfer_solution/summary.json}"
CLEAN_REPORT_DIR="${CLEAN_REPORT_DIR:-$PROJECT_ROOT/datasets/gpu5_v4_transfer_solution/clean_reports}"

BUILD_ARGS=(
  --v4-dataset "$V4_DATASET"
  --hotfix-dataset "$HOTFIX_DATASET"
  --parlay-seeds "$PARLAY_SEEDS"
  --domain-glossary-jsonl "$DOMAIN_GLOSSARY_JSONL"
  --raw-output-jsonl "$DATASET_RAW_OUTPUT_JSONL"
  --output-jsonl "$DATASET_OUTPUT_JSONL"
  --clean-report-dir "$CLEAN_REPORT_DIR"
  --summary-json "$DATASET_SUMMARY_JSON"
)

IFS=':' read -r -a _INTENT_SOURCE_ITEMS <<< "$INTENT_TRANSFER_SOURCES"
for item in "${_INTENT_SOURCE_ITEMS[@]}"; do
  if [[ -n "$item" ]]; then
    BUILD_ARGS+=(--intent-source-jsonl "$item")
  fi
done

IFS=':' read -r -a _DOMAIN_TERM_ITEMS <<< "$DOMAIN_TERMS"
for item in "${_DOMAIN_TERM_ITEMS[@]}"; do
  if [[ -n "$item" ]]; then
    BUILD_ARGS+=(--domain-term "$item")
  fi
done

IFS=':' read -r -a _EXTRA_SFT_ITEMS <<< "$EXTRA_SFT_SOURCES"
for item in "${_EXTRA_SFT_ITEMS[@]}"; do
  if [[ -n "$item" && -f "$item" ]]; then
    BUILD_ARGS+=(--extra-sft-jsonl "$item")
  fi
done

python3 "$PROJECT_ROOT/scripts/build_gpu5_v4_transfer_dataset.py" \
  "${BUILD_ARGS[@]}"

DATA_FILE="$DATASET_OUTPUT_JSONL" \
EVAL_INPUT_JSONL="${EVAL_INPUT_JSONL:-$PROJECT_ROOT/datasets/eval_sports_customer_canary_120.jsonl}" \
RUN_NAME_PREFIX="${RUN_NAME_PREFIX:-swift_sft_gpu5_v4_transfer}" \
EVAL_MODEL_NAME="${EVAL_MODEL_NAME:-qwen35a3b-sft-text-gpu5-v4-transfer-candidate}" \
"$PROJECT_ROOT/scripts/run_nohallucination_hotfix_sft_eval.sh"
