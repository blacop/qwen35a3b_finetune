#!/bin/bash
set -e

echo "正在安装训练服务..."

# 设置执行权限
chmod +x /home/ubuntu/qwen35a3b_finetune/train_v5_gpu5.sh

# 复制到systemd目录
sudo cp /home/ubuntu/qwen35a3b_finetune/qwen35-v5-train-gpu5.service /etc/systemd/system/

# 重载systemd
sudo systemctl daemon-reload

echo "服务安装完成！"
echo ""
echo "启动训练:"
echo "  sudo systemctl start qwen35-v5-train-gpu5"
echo ""
echo "查看训练日志:"
echo "  sudo journalctl -u qwen35-v5-train-gpu5 -f"
echo ""
echo "查看训练状态:"
echo "  sudo systemctl status qwen35-v5-train-gpu5"
