#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export RAG_INDEX_DIR="${RAG_INDEX_DIR:-/home/ubuntu/generate/tydata_rag/index_svd_20260421T065404Z}"
export LLM_API_BASE="${LLM_API_BASE:-http://127.0.0.1:8014/v1}"
export LLM_MODEL="${LLM_MODEL:-qwen35a3b-domain-text-vl-gpu5}"
export LLM_API_KEY="${LLM_API_KEY:-}"
export RAG_API_KEY="${RAG_API_KEY:-}"
export LLM_TIMEOUT_S="${LLM_TIMEOUT_S:-120}"
export LLM_ENABLE_THINKING="${LLM_ENABLE_THINKING:-0}"
export RAG_RETRIEVE_CACHE_ENABLED="${RAG_RETRIEVE_CACHE_ENABLED:-1}"
export RAG_RETRIEVE_CACHE_TTL_S="${RAG_RETRIEVE_CACHE_TTL_S:-30}"
export RAG_RETRIEVE_CACHE_MAX_ITEMS="${RAG_RETRIEVE_CACHE_MAX_ITEMS:-2048}"

HOST="${RAG_API_HOST:-0.0.0.0}"
PORT="${RAG_API_PORT:-18080}"
ACCESS_LOG_ARGS=()
if [[ "${RAG_UVICORN_ACCESS_LOG:-0}" != "1" ]]; then
  ACCESS_LOG_ARGS+=(--no-access-log)
fi

exec uvicorn rag_api:app --host "$HOST" --port "$PORT" --app-dir "$SCRIPT_DIR" "${ACCESS_LOG_ARGS[@]}"
