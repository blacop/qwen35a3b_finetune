#!/usr/bin/env python3
import argparse
import gc
import json
import os
import shutil
import time
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file


OLD_PREFIX = "model.language_model.language_model.language_model."
NEW_PREFIX = "model.language_model."
OLD_VISUAL_PREFIX = "model.language_model.visual."
NEW_VISUAL_PREFIX = "model.visual."


def rename_key(key: str) -> str:
    if key.startswith(OLD_PREFIX):
        key = NEW_PREFIX + key[len(OLD_PREFIX):]
    if key.startswith(OLD_VISUAL_PREFIX):
        key = NEW_VISUAL_PREFIX + key[len(OLD_VISUAL_PREFIX):]
    return key


def copy_non_weight_files(src: Path, dst: Path) -> None:
    for p in src.iterdir():
        name = p.name
        if name.startswith("model-") and name.endswith(".safetensors"):
            continue
        if name == "model.safetensors.index.json":
            continue
        target = dst / name
        if p.is_file():
            shutil.copy2(p, target)


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-export HF safetensors with vLLM-compatible key names.")
    parser.add_argument("--src", required=True, help="Source HF model dir")
    parser.add_argument("--dst", required=True, help="Destination HF model dir")
    parser.add_argument("--base", required=True, help="Base HF model dir (used to fill mtp.* keys)")
    parser.add_argument("--force", action="store_true", help="Delete dst if exists")
    parser.add_argument("--in-place", action="store_true", help="Rewrite src shards in place instead of creating a second full copy")
    args = parser.parse_args()

    src = Path(args.src).resolve()
    dst = Path(args.dst).resolve()
    base = Path(args.base).resolve()
    index_path = src / "model.safetensors.index.json"
    base_index_path = base / "model.safetensors.index.json"
    if not index_path.exists():
        raise FileNotFoundError(f"missing index: {index_path}")
    if not base_index_path.exists():
        raise FileNotFoundError(f"missing base index: {base_index_path}")

    if args.in_place and src != dst:
        raise ValueError("--in-place requires --src and --dst to be the same path")

    if dst.exists() and args.force:
        shutil.rmtree(dst)
    dst.mkdir(parents=True, exist_ok=True)

    with index_path.open("r", encoding="utf-8") as f:
        index = json.load(f)
    with base_index_path.open("r", encoding="utf-8") as f:
        base_index = json.load(f)
    weight_map = index["weight_map"]
    base_weight_map = base_index["weight_map"]

    renamed_weight_map = {}
    renamed_count = 0
    for old_key, shard in weight_map.items():
        new_key = rename_key(old_key)
        if new_key != old_key:
            renamed_count += 1
        if new_key in renamed_weight_map:
            raise ValueError(f"duplicate key after rename: {new_key}")
        renamed_weight_map[new_key] = shard

    mtp_added = 0
    for k, shard in base_weight_map.items():
        if not k.startswith("mtp."):
            continue
        if k not in renamed_weight_map:
            renamed_weight_map[k] = shard
            mtp_added += 1

    shard_files = sorted(set(renamed_weight_map.values()))
    print(f"[INFO] src={src}")
    print(f"[INFO] base={base}")
    print(f"[INFO] dst={dst}")
    print(f"[INFO] shards={len(shard_files)} src_keys={len(weight_map)} renamed={renamed_count} mtp_added={mtp_added}")

    if not args.in_place:
        copy_non_weight_files(src, dst)
        print("[INFO] copied non-weight files")

    for i, shard in enumerate(shard_files, start=1):
        t0 = time.time()
        src_shard = src / shard
        dst_shard = dst / shard
        tmp_shard = dst / f".tmp.{shard}"
        tensors = {}
        metadata = None
        if src_shard.exists():
            with safe_open(str(src_shard), framework="pt", device="cpu") as f:
                metadata = f.metadata()
                for old_key in f.keys():
                    tensors[rename_key(old_key)] = f.get_tensor(old_key)
        base_shard = base / shard
        if base_shard.exists():
            with safe_open(str(base_shard), framework="pt", device="cpu") as f:
                for k in f.keys():
                    if k.startswith("mtp.") and k not in tensors:
                        tensors[k] = f.get_tensor(k)
                if metadata is None:
                    metadata = f.metadata()
        if not tensors:
            raise RuntimeError(f"no tensors written for shard: {shard}")
        target_path = tmp_shard if args.in_place else dst_shard
        save_file(tensors, str(target_path), metadata=metadata)
        if args.in_place:
            os.replace(tmp_shard, dst_shard)
        elapsed = time.time() - t0
        print(f"[INFO] ({i}/{len(shard_files)}) wrote {shard} in {elapsed:.1f}s")
        del tensors
        gc.collect()

    new_index = dict(index)
    new_index["weight_map"] = renamed_weight_map
    with (dst / "model.safetensors.index.json").open("w", encoding="utf-8") as f:
        json.dump(new_index, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print("[INFO] wrote updated model.safetensors.index.json")
    print("[INFO] done")


if __name__ == "__main__":
    main()
