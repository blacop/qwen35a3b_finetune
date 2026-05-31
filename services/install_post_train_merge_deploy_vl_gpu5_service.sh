#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/ubuntu/qwen35a3b_finetune"
SERVICE_NAME="post-train-merge-deploy-vl-gpu5.service"
SERVICE_DIR="$HOME/.config/systemd/user"
SERVICE_FILE="$SERVICE_DIR/$SERVICE_NAME"
SCRIPT_FILE="$PROJECT_ROOT/scripts/post_train_merge_deploy_vl_gpu5.sh"
LOG_DIR="$PROJECT_ROOT/tracking/logs"
RUN_LOG="$LOG_DIR/post_train_merge_deploy_vl_gpu5.service.log"

mkdir -p "$SERVICE_DIR" "$LOG_DIR"
chmod +x "$SCRIPT_FILE"

cat >"$SERVICE_FILE" <<EOF
[Unit]
Description=Post-train merge+deploy watcher for VL SFT on GPU5
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
WorkingDirectory=$PROJECT_ROOT
ExecStart=/bin/bash -lc '$SCRIPT_FILE'
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
echo "[INFO] run log: $RUN_LOG"
echo "[INFO] follow log: journalctl --user -u $SERVICE_NAME -f"
