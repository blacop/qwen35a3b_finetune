#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Dict, List


def set_env_values(path: Path, updates: Dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        lines = path.read_text(encoding="utf-8").splitlines()
    else:
        lines = []
    out_lines: List[str] = []
    seen = set()
    for line in lines:
        s = line.strip()
        if not s or s.startswith("#") or "=" not in s:
            out_lines.append(line)
            continue
        k = s.split("=", 1)[0].strip()
        if k in updates:
            out_lines.append(f"{k}={updates[k]}")
            seen.add(k)
        else:
            out_lines.append(line)
    for k, v in updates.items():
        if k not in seen:
            out_lines.append(f"{k}={v}")
    path.write_text("\n".join(out_lines).rstrip() + "\n", encoding="utf-8")


def main() -> int:
    p = argparse.ArgumentParser(description="Update RAG gray switch settings")
    p.add_argument("--env-file", default="/home/ubuntu/generate/tydata_rag/rag_api.env")
    p.add_argument("--enable", choices=["0", "1"], default=None)
    p.add_argument("--ratio", type=float, default=None)
    p.add_argument("--canary-index-dir", default=None)
    p.add_argument("--stable-index-dir", default=None)
    p.add_argument("--restart", action="store_true")
    args = p.parse_args()

    updates: Dict[str, str] = {}
    if args.enable is not None:
        updates["RAG_GRAY_ENABLED"] = args.enable
    if args.ratio is not None:
        ratio = min(max(float(args.ratio), 0.0), 1.0)
        updates["RAG_GRAY_RATIO"] = f"{ratio:.6f}".rstrip("0").rstrip(".")
    if args.canary_index_dir is not None:
        updates["RAG_GRAY_CANARY_INDEX_DIR"] = args.canary_index_dir
    if args.stable_index_dir is not None:
        updates["RAG_INDEX_DIR"] = args.stable_index_dir

    if not updates:
        raise SystemExit("No updates specified. Use --enable/--ratio/--canary-index-dir/--stable-index-dir")

    env_path = Path(args.env_file)
    set_env_values(env_path, updates)
    out = {"env_file": str(env_path), "updates": updates}

    if args.restart:
        proc = subprocess.run(
            "systemctl --user restart tydata-rag-api.service",
            shell=True,
            capture_output=True,
            text=True,
        )
        out["restart_rc"] = proc.returncode
        out["restart_stdout"] = proc.stdout[-2000:]
        out["restart_stderr"] = proc.stderr[-2000:]
        if proc.returncode != 0:
            print(json.dumps(out, ensure_ascii=False, indent=2))
            raise SystemExit(proc.returncode)

    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
