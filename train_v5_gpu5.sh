#!/bin/bash
set -e

# V5 Combined 模型 GPU5 训练脚本
# 训练时间：约1.5-2小时
# 显存占用：~120GB

# 激活虚拟环境
source /home/ubuntu/qwen35a3b_finetune/.venv-swift/bin/activate

TIMESTAMP=$(date +%Y%m%dT%H%M%SZ)
OUTPUT_DIR="/home/ubuntu/qwen35a3b_finetune/outputs/swift_sft_v5_combined_gpu5_${TIMESTAMP}"
DATASET_PATH="/home/ubuntu/qwen35a3b_finetune/datasets/v5_combined_final.jsonl"
MODEL_PATH="/home/ubuntu/.cache/modelscope/hub/qwen3_5-35b-a3b-instruct"

echo "======================================"
echo "V5 Combined 训练启动"
echo "时间: ${TIMESTAMP}"
echo "输出目录: ${OUTPUT_DIR}"
echo "======================================"

mkdir -p ${OUTPUT_DIR}

export CUDA_VISIBLE_DEVICES=5
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

swift sft \
    --model_type qwen3_5-35b-a3b-instruct \
    --model ${MODEL_PATH} \
    --dataset ${DATASET_PATH} \
    --output_dir ${OUTPUT_DIR} \
    --tuner_type lora \
    --lora_rank 64 \
    --lora_alpha 128 \
    --lora_target_modules all \
    --batch_size 1 \
    --gradient_accumulation_steps 16 \
    --num_train_epochs 1 \
    --learning_rate 1e-4 \
    --weight_decay 0.1 \
    --warmup_ratio 0.05 \
    --lr_scheduler_type cosine \
    --gradient_checkpointing true \
    --save_strategy steps \
    --save_steps 200 \
    --save_total_limit 3 \
    --logging_steps 10 \
    --optim adamw_torch_fused \
    --bf16 true \
    --seed 42 \
    --max_seq_len 8192 \
    --dataloader_num_workers 1 \
    --dataloader_prefetch_factor 2 \
    2>&1 | tee ${OUTPUT_DIR}/train.log

echo "======================================"
echo "训练完成，开始合并权重"
echo "======================================"

# 找到最后一个checkpoint
LAST_CHECKPOINT=$(ls -d ${OUTPUT_DIR}/checkpoint-* | sort -t '-' -k 2 -n | tail -1)

swift merge \
    --model ${MODEL_PATH} \
    --ckpt_dir ${LAST_CHECKPOINT} \
    --output_dir ${OUTPUT_DIR}/merged_vllm

echo "======================================"
echo "权重合并完成！"
echo "模型路径: ${OUTPUT_DIR}/merged_vllm"
echo "下一步: 更新 vLLM 启动配置并切换流量"
echo "======================================"
