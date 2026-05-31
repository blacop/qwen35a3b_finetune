#!/usr/bin/env python3
import argparse
import json
from datasets import Dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments


def load_openai_messages_jsonl(path: str):
    rows = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            o = json.loads(line)
            msgs = o.get('messages', [])
            if not msgs:
                continue
            rows.append({'messages': msgs})
    return Dataset.from_list(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model_name', default='Qwen/Qwen3.5-35B-A3B')
    ap.add_argument('--data_file', required=True)
    ap.add_argument('--output_dir', required=True)
    ap.add_argument('--max_seq_length', type=int, default=4096)
    ap.add_argument('--report_to', default='tensorboard', help='tensorboard | mlflow | none')
    ap.add_argument('--logging_dir', default=None)
    ap.add_argument('--run_name', default='unsloth_sft_qwen35')
    args = ap.parse_args()

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        load_in_4bit=True,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        lora_alpha=32,
        lora_dropout=0,
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
        use_gradient_checkpointing='unsloth',
    )

    ds = load_openai_messages_jsonl(args.data_file)

    def to_text(batch):
        texts = []
        for msgs in batch['messages']:
            texts.append(tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False))
        return {'text': texts}

    ds = ds.map(to_text, batched=True)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=ds,
        dataset_text_field='text',
        max_seq_length=args.max_seq_length,
        args=TrainingArguments(
            output_dir=args.output_dir,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=16,
            learning_rate=1e-4,
            num_train_epochs=3,
            bf16=True,
            logging_steps=10,
            save_steps=200,
            save_total_limit=3,
            logging_dir=args.logging_dir or f"{args.output_dir}/logs",
            run_name=args.run_name,
            report_to=[] if args.report_to == 'none' else [args.report_to],
        ),
    )
    trainer.train()
    trainer.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == '__main__':
    main()
