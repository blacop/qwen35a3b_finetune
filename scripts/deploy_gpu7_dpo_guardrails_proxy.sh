#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/ubuntu/qwen35a3b_finetune"
RUNTIME_DIR="${RUNTIME_DIR:-$PROJECT_ROOT/runtime/gpu7_dpo_guardrails_proxy}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8010}"
UPSTREAM_BASE_URL="${UPSTREAM_BASE_URL:-http://127.0.0.1:8001/v1}"
UPSTREAM_API_KEY_FILE="${UPSTREAM_API_KEY_FILE:-$PROJECT_ROOT/runtime/gpu7_dpo_api/api_key.txt}"
UPSTREAM_API_KEY="${UPSTREAM_API_KEY:-}"
PROXY_API_KEY="${PROXY_API_KEY:-}"
PROXY_API_KEY_FILE="${PROXY_API_KEY_FILE:-$RUNTIME_DIR/proxy_api_key.txt}"
DEFAULT_MODEL="${DEFAULT_MODEL:-qwen35a3b-dpo-latest}"
UPSTREAM_TIMEOUT_S="${UPSTREAM_TIMEOUT_S:-180}"
PID_FILE="${PID_FILE:-$RUNTIME_DIR/proxy.pid}"
LOG_FILE="${LOG_FILE:-$RUNTIME_DIR/proxy.log}"

mkdir -p "$RUNTIME_DIR"

if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  HOST_IP="$( (hostname -I 2>/dev/null || true) | awk '{print $1}' )"
  HOST_IP="${HOST_IP:-127.0.0.1}"
  echo "[INFO] guardrails proxy already running, pid=$(cat "$PID_FILE")"
  echo "[INFO] endpoint: http://$HOST_IP:$PORT/v1"
  echo "[INFO] log_file: $LOG_FILE"
  exit 0
fi

if [ -f "$PID_FILE" ]; then
  rm -f "$PID_FILE"
fi

if [ -z "$UPSTREAM_API_KEY" ] && [ -f "$UPSTREAM_API_KEY_FILE" ]; then
  UPSTREAM_API_KEY="$(cat "$UPSTREAM_API_KEY_FILE")"
fi
if [ -z "$PROXY_API_KEY" ]; then
  if [ ! -f "$PROXY_API_KEY_FILE" ]; then
    python3 - <<'PY' > "$PROXY_API_KEY_FILE"
import secrets
print("sk-proxy-" + secrets.token_urlsafe(36))
PY
    chmod 600 "$PROXY_API_KEY_FILE"
  fi
  PROXY_API_KEY="$(cat "$PROXY_API_KEY_FILE")"
fi

export UPSTREAM_BASE_URL
export UPSTREAM_API_KEY
export PROXY_API_KEY
export PROXY_API_KEY_FILE
export DEFAULT_MODEL
export UPSTREAM_TIMEOUT_S

nohup python3 -m uvicorn \
  scripts.dpo_guardrails_proxy:app \
  --host "$HOST" \
  --port "$PORT" \
  --app-dir "$PROJECT_ROOT" \
  >"$LOG_FILE" 2>&1 < /dev/null &

echo $! > "$PID_FILE"
sleep 1

if ! kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  echo "[ERROR] guardrails proxy failed to start" >&2
  sed -n '1,160p' "$LOG_FILE" >&2 || true
  exit 1
fi

HOST_IP="$( (hostname -I 2>/dev/null || true) | awk '{print $1}' )"
HOST_IP="${HOST_IP:-127.0.0.1}"

echo "[INFO] started pid=$(cat "$PID_FILE")"
echo "[INFO] endpoint: http://$HOST_IP:$PORT/v1"
echo "[INFO] upstream: $UPSTREAM_BASE_URL"
echo "[INFO] proxy_api_key_file: $PROXY_API_KEY_FILE"
echo "[INFO] log_file: $LOG_FILE"
