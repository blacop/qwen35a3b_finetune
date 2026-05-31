#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/ubuntu/qwen35a3b_finetune"
SERVICE_NAME="swift-sft-gpu7-vl-distill-from-gpu5.service"
SERVICE_DIR="$HOME/.config/systemd/user"
SERVICE_FILE="$SERVICE_DIR/$SERVICE_NAME"
ENV_FILE="$PROJECT_ROOT/services/swift-sft-gpu7-vl-distill-from-gpu5.env"
LOG_DIR="$PROJECT_ROOT/tracking/logs"
SERVICE_LOG="$LOG_DIR/swift_sft_gpu7_vl_distill_from_gpu5.service.log"

mkdir -p "$SERVICE_DIR" "$LOG_DIR"
chmod +x "$PROJECT_ROOT/scripts/run_gpu7_vl_distill_from_gpu5.sh"

cat >"$SERVICE_FILE" <<EOF
[Unit]
Description=Swift GPU7 VL Distillation from GPU5 Teacher
After=network.target

[Service]
Type=simple
WorkingDirectory=$PROJECT_ROOT
EnvironmentFile=$ENV_FILE
ExecStart=/bin/bash -lc '$PROJECT_ROOT/scripts/run_gpu7_vl_distill_from_gpu5.sh'
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
