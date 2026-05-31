#!/usr/bin/env python3
"""
Build a compact canary eval set from full production eval JSONL.

Default behavior:
  - sample up to N rows per intent (stratified)
  - preserve original schema/fields
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build canary eval JSONL by intent stratified sampling.")
    parser.add_argument(
        "--input-jsonl",
        default="/home/ubuntu/qwen35a3b_finetune/datasets/eval_sports_customer_prod_500.jsonl",
    )
    parser.add_argument(
        "--output-jsonl",
        default="/home/ubuntu/qwen35a3b_finetune/datasets/eval_sports_customer_canary_120.jsonl",
    )
    parser.add_argument("--per-intent", type=int, default=12, help="Max rows per intent.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    rows = read_jsonl(Path(args.input_jsonl).resolve())
    by_intent: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        intent = str(row.get("gold_intent", "")).strip() or "UNLABELED"
        by_intent[intent].append(row)

    sampled: List[Dict[str, Any]] = []
    for intent in sorted(by_intent):
        group = by_intent[intent]
        if len(group) > args.per_intent:
            group = rng.sample(group, args.per_intent)
        sampled.extend(group)

    sampled.sort(key=lambda x: str(x.get("id", "")))
    out = Path(args.output_jsonl).resolve()
    write_jsonl(out, sampled)

    summary = {
        "input_jsonl": str(Path(args.input_jsonl).resolve()),
        "output_jsonl": str(out),
        "per_intent": args.per_intent,
        "total_rows": len(sampled),
        "by_intent": {k: min(len(v), args.per_intent) for k, v in sorted(by_intent.items())},
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
