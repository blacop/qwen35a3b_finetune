#!/usr/bin/env python3
"""Rebalance SFT dataset to target single-turn ratio.

Single-turn sample definition:
- exactly one user turn and one assistant turn in messages
or
- extracted first valid user->assistant pair from multi-turn dialogues
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def stable_hash(text: str, length: int = 16) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:length]


def normalize_for_hash(messages: List[Dict[str, str]]) -> str:
    return "\n".join(f"{m['role']}:{m['content'].strip().lower()}" for m in messages)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def all_qa_pairs(messages: List[Dict[str, str]]) -> List[List[Dict[str, str]]]:
    pairs: List[List[Dict[str, str]]] = []
    last_user = None
    for m in messages:
        role = m.get("role")
        txt = str(m.get("content", "")).strip()
        if not txt:
            continue
        if role == "user":
            last_user = txt
            continue
        if role == "assistant" and last_user:
            pairs.append([{"role": "user", "content": last_user}, {"role": "assistant", "content": txt}])
            last_user = None
    return pairs


def is_single_turn(messages: List[Dict[str, str]]) -> bool:
    user_turns = sum(1 for m in messages if m.get("role") == "user")
    assistant_turns = sum(1 for m in messages if m.get("role") == "assistant")
    return user_turns == 1 and assistant_turns == 1


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Rebalance SFT single/multi turn ratio.")
    p.add_argument(
        "--input",
        default="/home/ubuntu/qwen35a3b_finetune/datasets/sft_openai_messages.cleaned.v2.jsonl",
    )
    p.add_argument(
        "--output",
        default="/home/ubuntu/qwen35a3b_finetune/datasets/sft_openai_messages.cleaned.v2.single75.jsonl",
    )
    p.add_argument("--target-total", type=int, default=16000)
    p.add_argument("--single-ratio", type=float, default=0.75)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rnd = random.Random(args.seed)
    rows = load_jsonl(Path(args.input))

    single_pool: List[Dict[str, Any]] = []
    multi_pool: List[Dict[str, Any]] = []
    seen_single = set()

    for row in rows:
        msgs = row.get("messages", [])
        if not isinstance(msgs, list):
            continue

        if is_single_turn(msgs):
            h = stable_hash(normalize_for_hash(msgs), 20)
            if h not in seen_single:
                seen_single.add(h)
                single_pool.append(row)
        else:
            multi_pool.append(row)
            pairs = all_qa_pairs(msgs)
            for idx, pair in enumerate(pairs, 1):
                h = stable_hash(normalize_for_hash(pair), 20)
                if h not in seen_single:
                    seen_single.add(h)
                    row_pair = {
                        "id": f"{row.get('id', 'row')}_single_{idx}",
                        "split": row.get("split", "train"),
                        "messages": pair,
                    }
                    single_pool.append(row_pair)

    target_total = max(100, args.target_total)
    target_single = int(target_total * args.single_ratio)
    target_multi = target_total - target_single

    if len(single_pool) < target_single:
        target_single = len(single_pool)
        target_multi = min(len(multi_pool), target_total - target_single)
        target_total = target_single + target_multi
    if len(multi_pool) < target_multi:
        target_multi = len(multi_pool)
        target_single = min(len(single_pool), target_total - target_multi)
        target_total = target_single + target_multi

    rnd.shuffle(single_pool)
    rnd.shuffle(multi_pool)
    selected = single_pool[:target_single] + multi_pool[:target_multi]
    rnd.shuffle(selected)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for r in selected:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    summary = {
        "input_rows": len(rows),
        "single_pool": len(single_pool),
        "multi_pool": len(multi_pool),
        "output_rows": len(selected),
        "output_single_target": target_single,
        "output_multi_target": target_multi,
        "output_single_ratio": round(target_single / len(selected), 4) if selected else 0,
        "seed": args.seed,
        "output_path": str(out),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
