#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/ubuntu/qwen35a3b_finetune"
SERVICE_NAME="swift-sft-v5-1f-knowledge-format-gpu5.service"
SERVICE_DIR="$HOME/.config/systemd/user"
SERVICE_FILE="$SERVICE_DIR/$SERVICE_NAME"
ENV_FILE="$PROJECT_ROOT/services/swift-sft-v5-1f-knowledge-format-gpu5.env"
LOG_DIR="$PROJECT_ROOT/tracking/logs"
SERVICE_LOG="$LOG_DIR/swift_sft_v5_1f_knowledge_format_gpu5.service.log"

mkdir -p "$SERVICE_DIR" "$LOG_DIR"
chmod +x "$PROJECT_ROOT/scripts/run_gpu5_v5_tool_dialogue_migration.sh"

cat >"$SERVICE_FILE" <<EOF
[Unit]
Description=Swift GPU5 v5.1f Knowledge Format Patch SFT + Gate
After=network.target

[Service]
Type=simple
WorkingDirectory=$PROJECT_ROOT
EnvironmentFile=$ENV_FILE
ExecStart=/bin/bash -lc '$PROJECT_ROOT/scripts/run_gpu5_v5_tool_dialogue_migration.sh'
Restart=no
TimeoutStartSec=infinity
TimeoutStopSec=300
KillMode=control-group
StandardOutput=append:$SERVICE_LOG
StandardError=append:$SERVICE_LOG

[Install]
WantedBy=default.target
EOF

systemctl --user daemon-reload
if [[ "${ENABLE_ON_BOOT:-0}" = "1" ]]; then
  systemctl --user enable "$SERVICE_NAME"
else
  systemctl --user disable "$SERVICE_NAME" >/dev/null 2>&1 || true
fi
if [[ "${START_NOW:-0}" = "1" ]]; then
  systemctl --user start "$SERVICE_NAME"
fi

echo "[INFO] installed: $SERVICE_FILE"
echo "[INFO] env file: $ENV_FILE"
echo "[INFO] service log: $SERVICE_LOG"
echo "[INFO] start: systemctl --user start $SERVICE_NAME"
echo "[INFO] status: systemctl --user status $SERVICE_NAME --no-pager -n 80"
