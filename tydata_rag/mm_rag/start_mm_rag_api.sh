#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ENV_FILE="${MM_RAG_ENV_FILE:-${SCRIPT_DIR}/mm_rag.env}"
if [[ -f "$ENV_FILE" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
fi

export MM_INDEX_DIR="${MM_INDEX_DIR:-/home/ubuntu/generate/tydata_rag/mm_rag/index_improved_360}"
export MM_RAG_API_KEY="${MM_RAG_API_KEY:-}"
export LLM_API_BASE="${LLM_API_BASE:-http://127.0.0.1:8014/v1}"
export LLM_MODEL="${LLM_MODEL:-qwen35a3b-domain-text-vl-gpu5}"
export LLM_API_KEY="${LLM_API_KEY:-}"
export LLM_TIMEOUT_S="${LLM_TIMEOUT_S:-120}"
export LLM_ENABLE_THINKING="${LLM_ENABLE_THINKING:-0}"
export MM_ALLOW_NO_LLM="${MM_ALLOW_NO_LLM:-1}"

HOST="${MM_RAG_API_HOST:-0.0.0.0}"
PORT="${MM_RAG_API_PORT:-18100}"

exec uvicorn mm_rag_api:app --host "$HOST" --port "$PORT" --app-dir "$SCRIPT_DIR"
