# Qwen3.5-35B-A3B 微调与部署模板

## 已生成内容
- `scripts/export_from_processed_jsonl.py`
- `scripts/run_swift_sft.sh`
- `scripts/run_swift_dpo.sh`
- `scripts/run_swift_grpo.sh`
- `scripts/run_llamafactory_sft.sh`
- `scripts/run_llamafactory_dpo.sh`
- `scripts/unsloth_sft_qwen35.py`
- `scripts/run_vllm_qwen35.sh`
- `scripts/run_sglang_qwen35.sh`
- `scripts/start_tensorboard.sh`
- `scripts/start_mlflow_ui.sh`
- `datasets/DATA_FORMAT_SPEC.md`
- `datasets/sft_openai_messages.jsonl`
- `datasets/dpo_pairs.jsonl`
- `datasets/grpo_prompts.jsonl`

## 一键导出数据（已执行）
```bash
python3 scripts/export_from_processed_jsonl.py \
  --input /video-storage/ai-customer/dataset/processed_outputs_20260407/train_dataset_augmented_sports_gap_v1.jsonl \
  --out_dir ./datasets
```

## Swift
### SFT
```bash
BASE_MODEL=Qwen/Qwen3.5-35B-A3B DATA_DIR=./datasets OUT_DIR=./outputs/swift_sft ./scripts/run_swift_sft.sh
```

### DPO
```bash
BASE_MODEL=Qwen/Qwen3.5-35B-A3B SFT_CKPT=./outputs/swift_sft DATA_DIR=./datasets OUT_DIR=./outputs/swift_dpo ./scripts/run_swift_dpo.sh
```

### GRPO
```bash
BASE_MODEL=Qwen/Qwen3.5-35B-A3B SFT_CKPT=./outputs/swift_sft DATA_DIR=./datasets OUT_DIR=./outputs/swift_grpo ./scripts/run_swift_grpo.sh
```

## LLaMA-Factory
### SFT
```bash
BASE_MODEL=Qwen/Qwen3.5-35B-A3B DATA_FILE=./datasets/sft_openai_messages.jsonl OUT_DIR=./outputs/lf_sft ./scripts/run_llamafactory_sft.sh
```

### DPO
```bash
BASE_MODEL=Qwen/Qwen3.5-35B-A3B SFT_CKPT=./outputs/lf_sft DATA_FILE=./datasets/dpo_pairs.jsonl OUT_DIR=./outputs/lf_dpo ./scripts/run_llamafactory_dpo.sh
```

## Unsloth（Python脚本）
```bash
python3 scripts/unsloth_sft_qwen35.py \
  --model_name Qwen/Qwen3.5-35B-A3B \
  --data_file ./datasets/sft_openai_messages.jsonl \
  --output_dir ./outputs/unsloth_sft
```

## 推理部署
### vLLM
```bash
MODEL=Qwen/Qwen3.5-35B-A3B PORT=8000 TP_SIZE=1 ./scripts/run_vllm_qwen35.sh
```

### SGLang
```bash
MODEL=Qwen/Qwen3.5-35B-A3B PORT=30000 TP_SIZE=1 ./scripts/run_sglang_qwen35.sh
```

## 训练日志追踪（TensorBoard / MLflow）
LLaMA-Factory脚本已支持：
- `REPORT_TO=tensorboard`（默认）
- `REPORT_TO=mlflow`
- `REPORT_TO=wandb`
- `LOGGING_DIR=/path/to/logs`
- `RUN_NAME=xxx`

默认统一追踪根目录：
- TensorBoard 事件：`/home/ubuntu/qwen35a3b_finetune/tracking/tensorboard`
- MLflow 后端：`/home/ubuntu/qwen35a3b_finetune/tracking/mlruns`
- W&B 本地目录：`/home/ubuntu/qwen35a3b_finetune/tracking/wandb`

### TensorBoard
```bash
REPORT_TO=tensorboard RUN_NAME=lf_sft_tb ./scripts/run_llamafactory_sft.sh
./scripts/start_tensorboard.sh
```

### MLflow
先设置追踪地址（本机文件后端）：
```bash
export MLFLOW_TRACKING_URI=file:/home/ubuntu/qwen35a3b_finetune/tracking/mlruns
REPORT_TO=mlflow RUN_NAME=lf_sft_mlflow ./scripts/run_llamafactory_sft.sh
./scripts/start_mlflow_ui.sh
```

### W&B（云端）
```bash
wandb login
REPORT_TO=wandb RUN_NAME=lf_sft_wandb WANDB_PROJECT=qwen35a3b ./scripts/run_llamafactory_sft.sh
```

## 常驻服务（systemd --user）
已提供用户级常驻服务配置：
- `~/.config/systemd/user/tensorboard.service`
- `~/.config/systemd/user/mlflow.service`
- `~/.config/systemd/user/wandb.service`

一键安装并启动（默认启用 TensorBoard + MLflow + W&B）：
```bash
/home/ubuntu/qwen35a3b_finetune/services/install_user_services.sh
```

统一管理：
```bash
/home/ubuntu/qwen35a3b_finetune/services/service_ctl.sh status
/home/ubuntu/qwen35a3b_finetune/services/service_ctl.sh restart
/home/ubuntu/qwen35a3b_finetune/services/service_ctl.sh logs
```

说明：`wandb.service`会周期同步`tracking/wandb`下的离线/在线运行到W&B（需先`wandb login`）。

## 说明
- Qwen3.5-35B-A3B 体量较大，建议优先 LoRA/QLoRA。
- vLLM脚本已包含 `--reasoning-parser qwen3`、`--tool-call-parser qwen3_coder` 和可选 MTP 参数。
- SGLang脚本已包含可选 Tool Use / MTP 参数（默认关闭MTP）。
- 多模态微调建议先用文本流程打底，再加入图片样本做第二阶段SFT。

## 参数覆盖（可直接按机器资源调）
以下环境变量已在脚本中生效：
- `MAX_LEN`：上下文长度（默认 4096）
- `PER_DEVICE_BATCH`：单卡 batch（默认 1）
- `GRAD_ACC`：梯度累积（默认 16）
- `LORA_RANK` / `LORA_ALPHA`：LoRA 规模（SFT脚本默认 16/32）
- `LR`：学习率
- `EPOCHS`：训练轮数
- `TORCH_DTYPE`：`bfloat16` 或 `float16`（Swift）

示例：
```bash
CUDA_VISIBLE_DEVICES=7 \
MAX_LEN=4096 PER_DEVICE_BATCH=2 GRAD_ACC=16 LORA_RANK=16 \
BASE_MODEL=Qwen/Qwen3.5-35B-A3B DATA_DIR=./datasets OUT_DIR=./outputs/swift_sft \
./scripts/run_swift_sft.sh
```
