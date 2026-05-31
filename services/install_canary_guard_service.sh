#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/ubuntu/qwen35a3b_finetune"
SERVICE_NAME="canary-guard-monitor.service"
SERVICE_FILE="$HOME/.config/systemd/user/$SERVICE_NAME"
ENV_FILE="$PROJECT_ROOT/services/canary-guard.env"
RUNTIME_DIR="$PROJECT_ROOT/runtime/canary_guard"
LOG_FILE="$RUNTIME_DIR/canary_guard_service.log"

mkdir -p "$HOME/.config/systemd/user" "$RUNTIME_DIR"
chmod +x "$PROJECT_ROOT/scripts/run_canary_guard.sh"
chmod +x "$PROJECT_ROOT/scripts/rollback_vllm_model.sh"

cat >"$SERVICE_FILE" <<EOF
[Unit]
Description=Canary Guard Monitor with Auto Rollback
After=network.target

[Service]
Type=simple
WorkingDirectory=$PROJECT_ROOT
EnvironmentFile=$ENV_FILE
ExecStart=/bin/bash -lc '$PROJECT_ROOT/scripts/run_canary_guard.sh'
Restart=always
RestartSec=5
StandardOutput=append:$LOG_FILE
StandardError=append:$LOG_FILE

[Install]
WantedBy=default.target
EOF

systemctl --user daemon-reload
systemctl --user enable --now "$SERVICE_NAME"

echo "[DONE] installed and started: $SERVICE_NAME"
systemctl --user --no-pager --full status "$SERVICE_NAME" --lines 20 | sed -n '1,200p'
echo "[INFO] log file: $LOG_FILE"
