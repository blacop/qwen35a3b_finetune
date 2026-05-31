#!/usr/bin/env bash
set -euo pipefail

SERVICE_NAME="swift-sft-v5-1b-tool-dialogue-gpu5-gate.service"
PROJECT_ROOT="/home/ubuntu/qwen35a3b_finetune"
USER_SYSTEMD_DIR="${HOME}/.config/systemd/user"

mkdir -p "$USER_SYSTEMD_DIR"

cat >"${USER_SYSTEMD_DIR}/${SERVICE_NAME}" <<UNIT
[Unit]
Description=Swift GPU5 v5.1b Tool-Dialogue Candidate Gate Rerun
After=network.target

[Service]
Type=simple
WorkingDirectory=${PROJECT_ROOT}
EnvironmentFile=${PROJECT_ROOT}/services/swift-sft-v5-1b-tool-dialogue-gpu5-gate.env
ExecStart=/bin/bash -lc '${PROJECT_ROOT}/scripts/run_gpu5_candidate_gate.sh'
Restart=no
TimeoutStartSec=infinity
TimeoutStopSec=300
KillMode=control-group
StandardOutput=append:${PROJECT_ROOT}/tracking/logs/swift_sft_v5_1b_tool_dialogue_gpu5_gate.service.log
StandardError=append:${PROJECT_ROOT}/tracking/logs/swift_sft_v5_1b_tool_dialogue_gpu5_gate.service.log

[Install]
WantedBy=default.target
UNIT

systemctl --user daemon-reload
echo "Installed ${SERVICE_NAME}"
