#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/ubuntu/qwen35a3b_finetune"
INSTALL_VLLM="$PROJECT_ROOT/services/install_vllm_gpu5_agent_upstream_service.sh"
INSTALL_PROXY="$PROJECT_ROOT/services/install_dpo_guardrails_proxy_gpu5_service.sh"
CURRENT_GPU5_VLLM_SERVICE="${CURRENT_GPU5_VLLM_SERVICE:-vllm-qwen35-domain-text-vl-gpu5.service}"
CURRENT_GPU5_VLLM_HEALTH_URL="${CURRENT_GPU5_VLLM_HEALTH_URL:-http://127.0.0.1:8001/v1/models}"
VALIDATE_SCRIPT="${VALIDATE_SCRIPT:-$PROJECT_ROOT/scripts/validate_gpu5_agent_proxy.py}"
RUN_VALIDATION="${RUN_VALIDATION:-1}"
VALIDATION_REPORT_JSON="${VALIDATION_REPORT_JSON:-$PROJECT_ROOT/runtime/gpu5_dpo_guardrails_proxy/validation_report.json}"
UPSTREAM_URL="http://127.0.0.1:8014/v1/models"
PROXY_HEALTH_URL="http://127.0.0.1:8001/health"
PROXY_AGENT_HEALTH_URL="http://127.0.0.1:8001/v1/agent/health"
PROXY_GLOSSARY_URL="http://127.0.0.1:8001/glossary_stats"

log() {
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"
}

wait_http() {
  local url="$1"
  local name="$2"
  local auth_header="${3:-}"
  for _ in $(seq 1 60); do
    if [[ -n "$auth_header" ]]; then
      if curl -sS -H "$auth_header" "$url" >/dev/null 2>&1; then
        log "$name ready: $url"
        return 0
      fi
    elif curl -fsS "$url" >/dev/null 2>&1; then
      log "$name ready: $url"
      return 0
    fi
    sleep 2
  done
  log "$name not ready: $url"
  return 1
}

chmod +x "$INSTALL_VLLM" "$INSTALL_PROXY"

if curl -fsS -H "Authorization: Bearer sk-h2byzMZ66c53C8lZcz-9wcvN0Cwhny0AC7gf0qcOeksYCj-k" "$CURRENT_GPU5_VLLM_HEALTH_URL" >/dev/null 2>&1; then
  log "stopping current gpu5 vllm on :8001 before cutover: $CURRENT_GPU5_VLLM_SERVICE"
  systemctl --user stop "$CURRENT_GPU5_VLLM_SERVICE"
fi

log "installing gpu5 upstream vllm service"
bash "$INSTALL_VLLM"
wait_http "$UPSTREAM_URL" "gpu5 upstream vllm" "Authorization: Bearer sk-h2byzMZ66c53C8lZcz-9wcvN0Cwhny0AC7gf0qcOeksYCj-k"

log "installing gpu5 guardrails proxy on :8001"
bash "$INSTALL_PROXY"
wait_http "$PROXY_HEALTH_URL" "gpu5 proxy"
wait_http "$PROXY_AGENT_HEALTH_URL" "gpu5 agent endpoint"
wait_http "$PROXY_GLOSSARY_URL" "gpu5 glossary endpoint"

log "gpu5 rollout completed"
log "proxy base: http://127.0.0.1:8001/v1"
log "agent endpoint: http://127.0.0.1:8001/v1/chat/completions/agent"

if [[ "$RUN_VALIDATION" == "1" && -f "$VALIDATE_SCRIPT" ]]; then
  log "running gpu5 validation: $VALIDATE_SCRIPT"
  python3 "$VALIDATE_SCRIPT" \
    --base-url "http://127.0.0.1:8001" \
    --model "qwen35a3b-domain-text-vl-gpu5" \
    --report-json "$VALIDATION_REPORT_JSON"
  log "validation report: $VALIDATION_REPORT_JSON"
fi
