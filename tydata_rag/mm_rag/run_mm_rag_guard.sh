#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ENV_FILE="${MM_RAG_GUARD_ENV_FILE:-${SCRIPT_DIR}/mm_rag_guard.env}"
if [[ -f "$ENV_FILE" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
fi

HEALTH_URL="${HEALTH_URL:-http://127.0.0.1:18100/health}"
CURL_TIMEOUT_S="${CURL_TIMEOUT_S:-6}"
MAX_RETRIES="${MAX_RETRIES:-2}"
RETRY_SLEEP_S="${RETRY_SLEEP_S:-2}"
RESTART_WAIT_S="${RESTART_WAIT_S:-4}"
EXPECTED_MIN_UNITS="${EXPECTED_MIN_UNITS:-1}"
RESTART_CMD="${RESTART_CMD:-systemctl --user restart mm-rag-api.service}"
ALERT_WEBHOOK="${ALERT_WEBHOOK:-}"
ALERT_TOKEN="${ALERT_TOKEN:-}"
LOG_FILE="${LOG_FILE:-${SCRIPT_DIR}/runtime_mm_rag_guard.log}"

mkdir -p "$(dirname "$LOG_FILE")"

ts() {
  date -u +"%Y-%m-%dT%H:%M:%SZ"
}

log() {
  local level="$1"
  shift
  printf '[%s] [%s] %s\n' "$(ts)" "$level" "$*" >> "$LOG_FILE"
}

send_webhook_alert() {
  local message="$1"
  [[ -n "$ALERT_WEBHOOK" ]] || return 0

  local payload
  payload=$(python3 - <<'PY' "$message" "$ALERT_TOKEN"
import json
import sys
msg = sys.argv[1]
token = sys.argv[2]
body = {"text": f"[mm-rag-guard] {msg}"}
if token:
    body["token"] = token
print(json.dumps(body, ensure_ascii=False))
PY
)

  if ! curl -sS -m 8 -H 'Content-Type: application/json' -d "$payload" "$ALERT_WEBHOOK" >/dev/null; then
    log "WARN" "webhook alert failed"
    return 1
  fi
  return 0
}

health_check_once() {
  local resp
  resp=$(curl -fsS --max-time "$CURL_TIMEOUT_S" "$HEALTH_URL" 2>/dev/null || true)
  [[ -n "$resp" ]] || return 1

  python3 - <<'PY' "$EXPECTED_MIN_UNITS" "$resp"
import json
import sys
expected_min = int(sys.argv[1])
raw = sys.argv[2]
data = json.loads(raw)
if not data.get("ok", False):
    raise SystemExit(1)
meta = data.get("meta") or {}
num_units = int(meta.get("num_units", 0) or 0)
if num_units < expected_min:
    raise SystemExit(2)
print(num_units)
PY
}

health_check_with_retry() {
  local attempt=0
  local out
  while (( attempt <= MAX_RETRIES )); do
    if out=$(health_check_once 2>/dev/null); then
      log "INFO" "health check ok (num_units=${out}, attempt=$((attempt + 1)))"
      return 0
    fi
    attempt=$((attempt + 1))
    if (( attempt <= MAX_RETRIES )); then
      sleep "$RETRY_SLEEP_S"
    fi
  done
  return 1
}

if health_check_with_retry; then
  exit 0
fi

msg="health check failed after retries, restarting mm-rag-api"
log "ERROR" "$msg"
send_webhook_alert "$msg" || true

if ! bash -lc "$RESTART_CMD"; then
  msg="restart command failed: $RESTART_CMD"
  log "ERROR" "$msg"
  send_webhook_alert "$msg" || true
  exit 1
fi

sleep "$RESTART_WAIT_S"
if health_check_with_retry; then
  msg="service recovered after restart"
  log "INFO" "$msg"
  send_webhook_alert "$msg" || true
  exit 0
fi

msg="service still unhealthy after restart"
log "ERROR" "$msg"
send_webhook_alert "$msg" || true
exit 1
