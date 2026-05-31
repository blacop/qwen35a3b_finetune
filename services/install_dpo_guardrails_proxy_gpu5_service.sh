#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/ubuntu/qwen35a3b_finetune"
SERVICE_NAME="dpo-guardrails-proxy-gpu5.service"
SERVICE_DIR="$HOME/.config/systemd/user"
SERVICE_FILE="$SERVICE_DIR/$SERVICE_NAME"
RUNTIME_DIR="$PROJECT_ROOT/runtime/gpu5_dpo_guardrails_proxy"
ENV_FILE="$PROJECT_ROOT/services/dpo-guardrails-proxy-gpu5.env"
PROXY_API_KEY_FILE="$RUNTIME_DIR/proxy_api_key.txt"

mkdir -p "$SERVICE_DIR" "$RUNTIME_DIR"
if [ ! -f "$PROXY_API_KEY_FILE" ]; then
  python3 - <<'PY' > "$PROXY_API_KEY_FILE"
import secrets
print("sk-proxy-" + secrets.token_urlsafe(36))
PY
  chmod 600 "$PROXY_API_KEY_FILE"
fi

cat >"$SERVICE_FILE" <<EOF
[Unit]
Description=DPO Guardrails Proxy for GPU5 (OpenAI-compatible)
After=network.target

[Service]
Type=simple
WorkingDirectory=$PROJECT_ROOT
EnvironmentFile=$ENV_FILE
Environment=PROXY_API_KEY_FILE=$PROXY_API_KEY_FILE
ExecStart=/usr/bin/python3 -m uvicorn scripts.dpo_guardrails_proxy:app --host 0.0.0.0 --port 8001 --workers 4 --app-dir $PROJECT_ROOT
Restart=always
RestartSec=2
StandardOutput=append:$RUNTIME_DIR/systemd.log
StandardError=append:$RUNTIME_DIR/systemd.log

[Install]
WantedBy=default.target
EOF

systemctl --user daemon-reload
systemctl --user enable --now "$SERVICE_NAME"
systemctl --user status "$SERVICE_NAME" --no-pager -n 20 || true

echo "[INFO] service installed: $SERVICE_FILE"
echo "[INFO] endpoint: http://127.0.0.1:8001/v1"
echo "[INFO] proxy_api_key_file: $PROXY_API_KEY_FILE"
echo "[INFO] logs: journalctl --user -u $SERVICE_NAME -f"
