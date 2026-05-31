#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ENV_FILE="${RAG_GUARD_ENV_FILE:-${SCRIPT_DIR}/rag_guard.env}"
if [[ -f "$ENV_FILE" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
fi

exec /usr/bin/python3 "${SCRIPT_DIR}/rag_guard.py" \
  --rag-env "${RAG_ENV:-${SCRIPT_DIR}/rag_api.env}" \
  --state-dir "${STATE_DIR:-${SCRIPT_DIR}/runtime/rag_guard}" \
  --dataset "${DATASET:-/home/ubuntu/generate/qwen35a3b_finetune/datasets/eval_sports_customer_canary_120.jsonl}" \
  --query-field "${QUERY_FIELD:-user_query}" \
  --limit "${LIMIT:-120}" \
  --api-base "${API_BASE:-http://127.0.0.1:18080}" \
  --timeout-s "${TIMEOUT_S:-10}" \
  --check-route "${CHECK_ROUTE:-auto}" \
  --min-request-ok-rate "${MIN_REQUEST_OK_RATE:-0.99}" \
  --max-p95-ms "${MAX_P95_MS:-50}" \
  --sustain-min "${SUSTAIN_MIN:-10}" \
  --cooldown-min "${COOLDOWN_MIN:-30}" \
  --rollback-index-dir "${ROLLBACK_INDEX_DIR:-/home/ubuntu/generate/tydata_rag/index_svd_20260421T065404Z}" \
  --restart-cmd "${RESTART_CMD:-systemctl --user restart tydata-rag-api.service}"
