#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/ubuntu/qwen35a3b_finetune"
SERVICE_NAME="vllm-qwen35-gpu5-agent-upstream.service"
SERVICE_DIR="$HOME/.config/systemd/user"
SERVICE_FILE="$SERVICE_DIR/$SERVICE_NAME"
ENV_FILE="$PROJECT_ROOT/services/vllm-gpu5-agent-upstream.env"

mkdir -p "$SERVICE_DIR"

cat >"$SERVICE_FILE" <<EOF
[Unit]
Description=vLLM GPU5 upstream for guardrails proxy
After=network.target

[Service]
Type=simple
WorkingDirectory=$PROJECT_ROOT
Environment=CUDA_VISIBLE_DEVICES=5
Environment=VLLM_HOST_IP=127.0.0.1
Environment=UV_CACHE_DIR=/tmp/uv-cache
EnvironmentFile=$ENV_FILE
ExecStart=/bin/bash -lc "set -euo pipefail; UV_CACHE_DIR=/tmp/uv-cache /home/ubuntu/.local/bin/vllm serve \${MODEL_DIR} --host 127.0.0.1 --port \${PORT} --tensor-parallel-size \${TP_SIZE} --max-model-len \${MAX_LEN} --dtype bfloat16 --gpu-memory-utilization \${GPU_MEM_UTIL} --served-model-name \${SERVED_MODEL_NAME} --api-key \${API_KEY} \${SERVE_EXTRA_ARGS:-}"
Restart=always
RestartSec=10
StandardOutput=append:/home/ubuntu/qwen35a3b_finetune/tracking/logs/vllm_qwen35_gpu5_agent_upstream.log
StandardError=append:/home/ubuntu/qwen35a3b_finetune/tracking/logs/vllm_qwen35_gpu5_agent_upstream.log

[Install]
WantedBy=default.target
EOF

systemctl --user daemon-reload
systemctl --user enable --now "$SERVICE_NAME"
systemctl --user status "$SERVICE_NAME" --no-pager -n 20 || true

echo "[INFO] service installed: $SERVICE_FILE"
echo "[INFO] upstream: http://127.0.0.1:8014/v1"
echo "[INFO] logs: journalctl --user -u $SERVICE_NAME -f"
