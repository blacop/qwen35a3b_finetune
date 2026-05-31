#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/ubuntu/qwen35a3b_finetune"
SERVICE_NAME="swift-sft-dpo-resume.service"
SERVICE_DIR="$HOME/.config/systemd/user"
SERVICE_FILE="$SERVICE_DIR/$SERVICE_NAME"
ENV_FILE="$PROJECT_ROOT/services/swift-sft-dpo-resume.env"
LOG_DIR="$PROJECT_ROOT/tracking/logs"
RUN_LOG="$LOG_DIR/swift_sft_dpo_resume_service.log"

mkdir -p "$SERVICE_DIR" "$LOG_DIR"
chmod +x "$PROJECT_ROOT/scripts/run_sft_dpo_resume_uv.sh"

cat >"$SERVICE_FILE" <<EOF
[Unit]
Description=Swift SFT+DPO Resume Pipeline (systemd + uv)
After=network.target

[Service]
Type=simple
WorkingDirectory=$PROJECT_ROOT
EnvironmentFile=$ENV_FILE
ExecStart=/bin/bash -lc '$PROJECT_ROOT/scripts/run_sft_dpo_resume_uv.sh'
Restart=on-failure
RestartSec=20
StandardOutput=append:$RUN_LOG
StandardError=append:$RUN_LOG

[Install]
WantedBy=default.target
EOF

systemctl --user daemon-reload
systemctl --user enable --now "$SERVICE_NAME"
systemctl --user status "$SERVICE_NAME" --no-pager -n 30 || true

echo "[INFO] installed: $SERVICE_FILE"
echo "[INFO] env file: $ENV_FILE"
echo "[INFO] run log: $RUN_LOG"
echo "[INFO] follow log: journalctl --user -u $SERVICE_NAME -f"
