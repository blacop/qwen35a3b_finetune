#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file


OLD_VISUAL_PREFIX = "model.language_model.visual."
NEW_VISUAL_PREFIX = "model.visual."


def rename_key(key: str) -> str:
    if key.startswith(OLD_VISUAL_PREFIX):
        return NEW_VISUAL_PREFIX + key[len(OLD_VISUAL_PREFIX) :]
    return key


def symlink_or_copy(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        os.symlink(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create a sparse vLLM-compatible model dir by renaming "
            "model.language_model.visual.* weights to model.visual.*."
        )
    )
    parser.add_argument("--src", required=True, help="Source HF/vLLM model directory")
    parser.add_argument("--dst", required=True, help="Destination keyfix directory")
    parser.add_argument("--force", action="store_true", help="Replace destination if it exists")
    args = parser.parse_args()

    src = Path(args.src).resolve()
    dst = Path(args.dst).resolve()
    index_path = src / "model.safetensors.index.json"
    if not index_path.exists():
        raise FileNotFoundError(f"missing index: {index_path}")

    with index_path.open("r", encoding="utf-8") as f:
        index = json.load(f)
    weight_map = index["weight_map"]
    visual_shards = sorted(
        {shard for key, shard in weight_map.items() if key.startswith(OLD_VISUAL_PREFIX)}
    )
    if not visual_shards:
        raise SystemExit(f"no {OLD_VISUAL_PREFIX} keys found in {index_path}")

    renamed_weight_map = {}
    renamed_count = 0
    for key, shard in weight_map.items():
        new_key = rename_key(key)
        if new_key != key:
            renamed_count += 1
        if new_key in renamed_weight_map:
            raise ValueError(f"duplicate key after rename: {new_key}")
        renamed_weight_map[new_key] = shard

    if dst.exists():
        if not args.force:
            raise FileExistsError(f"destination exists: {dst}")
        shutil.rmtree(dst)
    dst.mkdir(parents=True)

    for item in src.iterdir():
        target = dst / item.name
        if item.name == "model.safetensors.index.json":
            continue
        if item.name in visual_shards:
            continue
        if item.is_file():
            symlink_or_copy(item, target)

    for shard in visual_shards:
        src_shard = src / shard
        dst_shard = dst / shard
        tensors = {}
        with safe_open(str(src_shard), framework="pt", device="cpu") as f:
            metadata = f.metadata()
            for key in f.keys():
                tensors[rename_key(key)] = f.get_tensor(key)
        with tempfile.NamedTemporaryFile(dir=str(dst), prefix=f".tmp.{shard}.", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        save_file(tensors, str(tmp_path), metadata=metadata)
        os.replace(tmp_path, dst_shard)

    fixed_index = dict(index)
    fixed_index["weight_map"] = renamed_weight_map
    with (dst / "model.safetensors.index.json").open("w", encoding="utf-8") as f:
        json.dump(fixed_index, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(
        json.dumps(
            {
                "src": str(src),
                "dst": str(dst),
                "visual_shards_rewritten": visual_shards,
                "visual_keys_renamed": renamed_count,
                "total_keys": len(weight_map),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
