#!/usr/bin/env bash
set -euo pipefail

systemctl --user daemon-reload

mkdir -p /home/ubuntu/qwen35a3b_finetune/tracking/tensorboard
mkdir -p /home/ubuntu/qwen35a3b_finetune/tracking/mlruns
mkdir -p /home/ubuntu/qwen35a3b_finetune/tracking/wandb

# Enable and start TensorBoard + MLflow + W&B sync by default.
systemctl --user enable --now tensorboard.service
systemctl --user enable --now mlflow.service
systemctl --user enable --now wandb.service

echo "[ok] enabled services:"
systemctl --user --no-pager --full status tensorboard.service mlflow.service wandb.service --lines 0 | sed -n '1,180p'

echo
echo "Tips:"
echo "- tensorboard: http://<host>:6006"
echo "- mlflow ui : http://<host>:5000"
echo "- wandb sync root: /home/ubuntu/qwen35a3b_finetune/tracking/wandb"
echo "- To persist after reboot without active login: sudo loginctl enable-linger $USER"
