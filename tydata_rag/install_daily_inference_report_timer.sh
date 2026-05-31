#!/usr/bin/env bash
set -euo pipefail

SYSTEMD_DIR="${HOME}/.config/systemd/user"
SERVICE_PATH="${SYSTEMD_DIR}/daily-inference-report.service"
TIMER_PATH="${SYSTEMD_DIR}/daily-inference-report.timer"
PROJECT_ROOT="/home/ubuntu/generate/qwen35a3b_finetune/tydata_rag"
REPORT_DIR="${PROJECT_ROOT}/reports"
LOG_DIR="${PROJECT_ROOT}/runtime/daily_inference_report"
RUN_SCRIPT="${PROJECT_ROOT}/daily_inference_report.sh"

mkdir -p "$SYSTEMD_DIR" "$REPORT_DIR" "$LOG_DIR"
chmod +x \
  "${PROJECT_ROOT}/daily_inference_report.py" \
  "${PROJECT_ROOT}/daily_inference_report.sh"

cat >"$SERVICE_PATH" <<'EOF'
[Unit]
Description=Generate daily GPU5/GPU7 inference report
After=network.target dpo-guardrails-proxy-gpu5.service dpo-guardrails-proxy.service tydata-rag-api.service tydata-rag-api-gpu7.service mm-rag-api.service
Wants=dpo-guardrails-proxy-gpu5.service dpo-guardrails-proxy.service tydata-rag-api.service tydata-rag-api-gpu7.service mm-rag-api.service

[Service]
Type=oneshot
WorkingDirectory=/home/ubuntu/generate/qwen35a3b_finetune/tydata_rag
Environment=PATH=/home/ubuntu/.local/bin:/usr/local/bin:/usr/bin:/bin
ExecStart=/bin/bash -lc '/home/ubuntu/generate/qwen35a3b_finetune/tydata_rag/daily_inference_report.sh'
StandardOutput=append:/home/ubuntu/generate/qwen35a3b_finetune/tydata_rag/runtime/daily_inference_report/report.log
StandardError=append:/home/ubuntu/generate/qwen35a3b_finetune/tydata_rag/runtime/daily_inference_report/report.log
EOF

cat >"$TIMER_PATH" <<'EOF'
[Unit]
Description=Run daily GPU5/GPU7 inference report at 00:10 UTC

[Timer]
OnCalendar=*-*-* 00:10:00
AccuracySec=1min
Persistent=true
Unit=daily-inference-report.service

[Install]
WantedBy=timers.target
EOF

systemctl --user daemon-reload
systemctl --user enable --now daily-inference-report.timer
systemctl --user start daily-inference-report.service
systemctl --user status daily-inference-report.timer --no-pager
