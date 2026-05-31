#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict


def process(in_path: Path, out_path: Path) -> Dict[str, Any]:
    total = 0
    removed_meta = 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with in_path.open("r", encoding="utf-8") as fin, out_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            total += 1
            row = json.loads(line)
            if "meta" in row:
                row.pop("meta", None)
                removed_meta += 1
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
    return {
        "input": str(in_path),
        "output": str(out_path),
        "total_rows": total,
        "meta_removed_rows": removed_meta,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Strip top-level meta from JSONL rows for Swift schema compatibility.")
    parser.add_argument("--sft-in", required=True)
    parser.add_argument("--sft-out", required=True)
    parser.add_argument("--dpo-in", required=True)
    parser.add_argument("--dpo-out", required=True)
    parser.add_argument("--summary-out", required=True)
    args = parser.parse_args()

    sft_summary = process(Path(args.sft_in), Path(args.sft_out))
    dpo_summary = process(Path(args.dpo_in), Path(args.dpo_out))
    summary = {"sft": sft_summary, "dpo": dpo_summary}
    Path(args.summary_out).write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
