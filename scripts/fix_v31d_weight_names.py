#!/usr/bin/env python3
"""Fix triple language_model prefix caused by Swift LoRA merge"""
import os
import glob
from pathlib import Path
from safetensors import safe_open
from safetensors.torch import save_file
import torch

MERGED_DIR = Path("/video-storage/ai-customer/qwen35a3b_finetune/outputs/swift_sft_sports_rule_hotfix_v31d_mixed_gpu5_20260430T035743Z_merged_20260430T054414Z_vllm")
FIXED_DIR = MERGED_DIR.with_name(MERGED_DIR.name + "_fixed")
FIXED_DIR.mkdir(parents=True, exist_ok=True)

def fix_key(key):
    # Fix: model.language_model.language_model.language_model -> model.language_model
    key = key.replace("model.language_model.language_model.language_model.", "model.language_model.")
    # Also handle model.visual.visual if needed
    key = key.replace("model.visual.visual.", "model.visual.")
    return key

shard_files = sorted(glob.glob(str(MERGED_DIR / "*.safetensors")))

for shard_path in shard_files:
    shard_name = os.path.basename(shard_path)
    print(f"Processing {shard_name}...")

    with safe_open(shard_path, framework="pt") as f:
        tensor_dict = {}
        for k in f.keys():
            new_k = fix_key(k)
            tensor_dict[new_k] = f.get_tensor(k)

    save_file(tensor_dict, str(FIXED_DIR / shard_name))
    print(f"  Saved {len(tensor_dict)} tensors")

# Copy other files
for f in MERGED_DIR.iterdir():
    if f.suffix not in (".safetensors",):
        import shutil
        if f.is_file():
            shutil.copy2(f, FIXED_DIR / f.name)
        elif f.is_dir():
            shutil.copytree(f, FIXED_DIR / f.name, dirs_exist_ok=True)

print(f"\nDone! Fixed model at: {FIXED_DIR}")
