#!/usr/bin/env bash
set -euo pipefail

DOCX_PATH="${1:-/tmp/tydata/JT包网后台操作手册.docx}"
WEB_URL="${2:-}"
WORK_ROOT="${3:-/home/ubuntu/generate/tydata_rag/mm_rag/work}"
CHUNK_SIZE="${4:-360}"
OVERLAP="${5:-80}"

DOCX_OUT="${WORK_ROOT}/docx_assets"
WEB_OUT="${WORK_ROOT}/web_assets"
INDEX_OUT="/home/ubuntu/generate/tydata_rag/mm_rag/index"

echo "[1/3] extract docx assets"
/usr/bin/python3 /home/ubuntu/generate/tydata_rag/mm_rag/extract_docx_assets.py \
  --docx "$DOCX_PATH" \
  --output-dir "$DOCX_OUT" \
  --chunk-size "$CHUNK_SIZE" \
  --overlap "$OVERLAP"

if [[ -n "$WEB_URL" ]]; then
  echo "[2/3] capture web assets: $WEB_URL"
/usr/bin/python3 /home/ubuntu/generate/tydata_rag/mm_rag/capture_site_assets.py \
    --base-url "$WEB_URL" \
    --output-dir "$WEB_OUT"
else
  echo "[2/3] skip web capture (no WEB_URL provided)"
  mkdir -p "$WEB_OUT"
  : > "$WEB_OUT/page_records.jsonl"
fi

echo "[3/3] build multimodal index"
/usr/bin/python3 /home/ubuntu/generate/tydata_rag/mm_rag/build_mm_index.py \
  --docx-dir "$DOCX_OUT" \
  --web-dir "$WEB_OUT" \
  --output-dir "$INDEX_OUT" \
  --chunk-size "$CHUNK_SIZE" \
  --overlap "$OVERLAP"

cat <<'EOF'

Done.
Start API:
  /home/ubuntu/generate/tydata_rag/mm_rag/start_mm_rag_api.sh

Health:
  curl -sS http://127.0.0.1:18100/health

Retrieve:
  curl -sS http://127.0.0.1:18100/retrieve \
    -H 'Content-Type: application/json' \
    -d '{"query":"如何配置后台角色权限","top_k":8}'

EOF
