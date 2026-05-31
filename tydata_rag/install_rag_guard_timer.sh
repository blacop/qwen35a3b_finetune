#!/usr/bin/env bash
set -euo pipefail

SYSTEMD_DIR="${HOME}/.config/systemd/user"
SERVICE_PATH="${SYSTEMD_DIR}/tydata-rag-guard.service"
TIMER_PATH="${SYSTEMD_DIR}/tydata-rag-guard.timer"
PROJECT_ROOT="/home/ubuntu/generate/qwen35a3b_finetune/tydata_rag"

mkdir -p "$SYSTEMD_DIR" "${PROJECT_ROOT}/runtime/rag_guard"
chmod +x "${PROJECT_ROOT}/rag_guard.py" "${PROJECT_ROOT}/run_rag_guard.sh" "${PROJECT_ROOT}/set_rag_gray.py"

cat >"$SERVICE_PATH" <<'EOF'
[Unit]
Description=Tydata RAG Guard (Auto Rollback)
After=network.target tydata-rag-api.service
Wants=tydata-rag-api.service

[Service]
Type=oneshot
WorkingDirectory=/home/ubuntu/generate/qwen35a3b_finetune/tydata_rag
Environment=PATH=/home/ubuntu/.local/bin:/usr/local/bin:/usr/bin:/bin
Environment=RAG_GUARD_ENV_FILE=%h/generate/qwen35a3b_finetune/tydata_rag/rag_guard.env
ExecStart=/bin/bash -lc '/home/ubuntu/generate/qwen35a3b_finetune/tydata_rag/run_rag_guard.sh'
StandardOutput=append:/home/ubuntu/generate/qwen35a3b_finetune/tydata_rag/runtime/rag_guard/guard.log
StandardError=append:/home/ubuntu/generate/qwen35a3b_finetune/tydata_rag/runtime/rag_guard/guard.log
EOF

cat >"$TIMER_PATH" <<'EOF'
[Unit]
Description=Run Tydata RAG Guard every 2 minutes

[Timer]
OnBootSec=2min
OnUnitActiveSec=2min
AccuracySec=30s
Persistent=true
Unit=tydata-rag-guard.service

[Install]
WantedBy=timers.target
EOF

systemctl --user daemon-reload
systemctl --user enable --now tydata-rag-guard.timer
systemctl --user start tydata-rag-guard.service
systemctl --user status tydata-rag-guard.timer --no-pager
