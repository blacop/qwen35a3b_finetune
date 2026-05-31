#!/usr/bin/env python3
"""Fix v31d weights: fix language_model prefix and add missing visual weights from base model"""
import os
import glob
import json
from pathlib import Path
from safetensors import safe_open
from safetensors.torch import save_file
import torch

BASE_DIR = Path("/home/ubuntu/qwen35a3b_finetune/outputs/qwen35a3b_domain_text_vl_unified_20260427T091024Z_vllm")
MERGED_DIR = Path("/video-storage/ai-customer/qwen35a3b_finetune/outputs/swift_sft_sports_rule_hotfix_v31d_mixed_gpu5_20260430T035743Z_merged_20260430T054414Z_vllm")
FIXED_DIR = MERGED_DIR.with_name(MERGED_DIR.name + "_v2_fixed")
FIXED_DIR.mkdir(parents=True, exist_ok=True)

# Step 1: Collect all weights from base model (visual + language)
base_weights = {}
for shard_path in sorted(glob.glob(str(BASE_DIR / "*.safetensors"))):
    with safe_open(shard_path, framework="pt") as f:
        for k in f.keys():
            base_weights[k] = f.get_tensor(k)

print(f"Base model: {len(base_weights)} total weights")
base_visual = {k: v for k, v in base_weights.items() if 'visual' in k}
base_language = {k: v for k, v in base_weights.items() if 'language' in k}
print(f"  visual: {len(base_visual)} keys")
print(f"  language: {len(base_language)} keys")

# Step 2: Collect LoRA-fine-tuned weights from merged model, fixing prefix
merged_language_fixed = {}
for shard_path in sorted(glob.glob(str(MERGED_DIR / "*.safetensors"))):
    with safe_open(shard_path, framework="pt") as f:
        for k in f.keys():
            # Fix triple language_model prefix
            new_k = k.replace("model.language_model.language_model.language_model.", "model.language_model.")
            if 'language' in new_k:
                merged_language_fixed[new_k] = f.get_tensor(k)

print(f"\nMerged model fixed language keys: {len(merged_language_fixed)}")

# Step 3: Combine: base visual + merged fine-tuned language
combined_weights = {}
# Add base visual weights
combined_weights.update(base_visual)
# Add fine-tuned language weights
combined_weights.update(merged_language_fixed)
# Add lm_head
if 'lm_head.weight' in base_weights:
    combined_weights['lm_head.weight'] = base_weights['lm_head.weight']

print(f"\nCombined total: {len(combined_weights)} weights")
print(f"  visual: {len([k for k in combined_weights if 'visual' in k])}")
print(f"  language: {len([k for k in combined_weights if 'language' in k])}")

# Step 4: Distribute to shards (matching base model's index distribution)
# Load base model index
with open(BASE_DIR / "model.safetensors.index.json") as f:
    base_index = json.load(f)

# Create new index
new_index = {
    "metadata": base_index.get("metadata", {}),
    "weight_map": {}
}

# Distribute weights to shards
for weight_name in sorted(combined_weights.keys()):
    # Use same shard name as base if possible
    if weight_name in base_index["weight_map"]:
        shard_name = base_index["weight_map"][weight_name]
    else:
        # Fallback - use first shard
        shard_name = "model-00001-of-00016.safetensors"
    new_index["weight_map"][weight_name] = shard_name

# Group by shard
shard_groups = {}
for weight_name, shard_name in new_index["weight_map"].items():
    if shard_name not in shard_groups:
        shard_groups[shard_name] = {}
    shard_groups[shard_name][weight_name] = combined_weights[weight_name]

# Save each shard
for shard_name, shard_weights in shard_groups.items():
    save_file(shard_weights, str(FIXED_DIR / shard_name))
    print(f"Saved {shard_name}: {len(shard_weights)} weights")

# Save index
with open(FIXED_DIR / "model.safetensors.index.json", "w") as f:
    json.dump(new_index, f, indent=2)

# Copy other files
for f in BASE_DIR.iterdir():
    if f.is_file() and f.suffix not in (".safetensors",):
        import shutil
        shutil.copy2(f, FIXED_DIR / f.name)

print(f"\nDone! Fixed model at: {FIXED_DIR}")
