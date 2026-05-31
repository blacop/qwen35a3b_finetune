#!/usr/bin/env python3
"""
Generate targeted weak-intent SFT data and merge into an existing cleaned SFT dataset.

Default target counts (total 12,000):
  - 注单异常: 3000
  - 串关: 2500
  - 滚球: 2000
  - 赔率: 2000
  - 限红: 2500

Output files:
  - weak_intent_targeted_12000.jsonl
  - <base>.weak12k_merged.jsonl
  - weak_intent_targeted_12000_summary.json
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


DEFAULT_COUNTS = "注单异常=3000,串关=2500,滚球=2000,赔率=2000,限红=2500"

CANONICAL = {
    "注单异常": "注单异常",
    "串关": "串关规则",
    "串关规则": "串关规则",
    "滚球": "滚球延迟",
    "滚球延迟": "滚球延迟",
    "赔率": "赔率异常",
    "赔率异常": "赔率异常",
    "限红": "限红风控",
    "限红风控": "限红风控",
}

QUERY_TEMPLATES: Dict[str, List[str]] = {
    "注单异常": [
        "我的注单{order_id}状态异常，一直显示处理中，麻烦核查。",
        "注单{order_id}被取消了，原因是什么？请尽快处理。",
        "订单{order_id}结算结果不对，帮我查下明细。",
        "我这笔单{order_id}查不到结果，后台能帮忙确认吗？",
        "注单{order_id}显示作废，但比赛已结束，怎么处理？",
    ],
    "串关规则": [
        "串关单{order_id}里有一场延期，整单怎么结算？",
        "3串1里有一场走水，剩余场次怎么算奖金？",
        "我的串关{order_id}命中两场，一场取消，规则怎么判？",
        "请问过关投注里一场腰斩，串关单如何处理？",
        "串关单{order_id}结算和我理解不一致，麻烦按规则复核。",
    ],
    "滚球延迟": [
        "滚球单{order_id}赛后{minute}分钟还没结算，麻烦查一下。",
        "赛中单{order_id}迟迟不派彩，是系统延迟吗？",
        "我这笔滚球{order_id}一直待结算，多久能处理好？",
        "滚球赛事结束了，注单{order_id}还未派彩，帮忙核实。",
        "走地订单{order_id}未结算，后台能加急处理吗？",
    ],
    "赔率异常": [
        "下单前赔率{odds_a}，成交变成{odds_b}，这单按哪个赔率算？",
        "注单{order_id}显示赔率和我下单时不一致，麻烦复核。",
        "盘口跳动后赔率变了，订单{order_id}结算规则是什么？",
        "我确认下注时是{odds_a}，最终按{odds_b}结算，能查明细吗？",
        "这笔单{order_id}赔率争议，麻烦按快照核验。",
    ],
    "限红风控": [
        "账号突然被限红，单笔额度降到{limit_amt}，请帮我核查原因。",
        "为什么我现在下注提示限额？之前还能下{old_amt}。",
        "账户被风控限制投注，什么时候能恢复正常额度？",
        "限红后很多玩法下不了，麻烦后台复核账户状态。",
        "我这边显示高风险限额，能否确认具体规则和恢复时间？",
    ],
}

ANSWER_TEMPLATES: Dict[str, str] = {
    "注单异常": "该问题属于注单异常场景，需要后台核验注单状态与结算链路。预计15-30分钟反馈，高峰期不超过60分钟。受理编号：{ticket}。请提供账号、注单号与异常现象说明。",
    "串关规则": "该问题属于串关规则场景，需要按关数规则、赛事状态与结算规则复核。预计15-30分钟反馈，高峰期不超过60分钟。受理编号：{ticket}。请提供账号、注单号、串关场次与对应赔率信息。",
    "滚球延迟": "该问题属于滚球延迟场景，需要核验赛中数据与结算链路。预计15-30分钟反馈，高峰期不超过60分钟。受理编号：{ticket}。请提供账号、注单号、比赛名称与当前赛中时间信息。",
    "赔率异常": "该问题属于赔率异常场景，需要按下单时间、盘口快照与成交赔率复核。预计15-30分钟反馈，高峰期不超过60分钟。受理编号：{ticket}。请提供账号、注单号、下单时间、盘口与赔率截图。",
    "限红风控": "该问题属于限红风控场景，需要后台核验账户风险策略与限额规则。预计15-30分钟反馈，高峰期不超过60分钟。受理编号：{ticket}。请提供账号、限额提示截图与最近投注时间段信息。",
}


def now_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def parse_counts(raw: str) -> List[Tuple[str, int]]:
    pairs: List[Tuple[str, int]] = []
    for item in re.split(r"[,;；，]\s*", (raw or "").strip()):
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"invalid count item: {item}")
        k, v = item.split("=", 1)
        key = CANONICAL.get(k.strip(), "")
        if not key:
            raise ValueError(f"unsupported intent key: {k}")
        val = int(v.strip())
        if val < 0:
            raise ValueError(f"count must be >=0: {item}")
        pairs.append((key, val))
    if not pairs:
        raise ValueError("empty counts")
    merged: Dict[str, int] = {}
    for k, v in pairs:
        merged[k] = merged.get(k, 0) + v
    return list(merged.items())


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


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_user_query(intent: str, idx: int, rng: random.Random) -> str:
    tmpl = rng.choice(QUERY_TEMPLATES[intent])
    minute = rng.choice([3, 5, 8, 10, 15, 20, 30, 45])
    odds_a = f"{rng.uniform(1.60, 2.80):.2f}"
    odds_b = f"{rng.uniform(1.60, 2.80):.2f}"
    limit_amt = rng.choice(["300", "500", "800", "1000"])
    old_amt = rng.choice(["2000", "3000", "5000"])
    order_id = f"BET{idx:07d}"
    return tmpl.format(
        order_id=order_id,
        minute=minute,
        odds_a=odds_a,
        odds_b=odds_b,
        limit_amt=limit_amt,
        old_amt=old_amt,
    )


def build_assistant(intent: str, idx: int) -> str:
    ticket = f"TK{idx:08d}"
    return ANSWER_TEMPLATES[intent].format(ticket=ticket)


def make_increment_rows(counts: List[Tuple[str, int]], seed: int) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    rng = random.Random(seed)
    rows: List[Dict[str, Any]] = []
    dist: Counter[str] = Counter()
    cur = 1
    for intent, n in counts:
        for i in range(n):
            rid = f"WEAK12K_{intent}_{i+1:05d}"
            user_q = build_user_query(intent, cur, rng)
            ans = build_assistant(intent, cur)
            rows.append(
                {
                    "id": rid,
                    "split": "train",
                    "messages": [
                        {"role": "user", "content": user_q},
                        {"role": "assistant", "content": ans},
                    ],
                }
            )
            dist[intent] += 1
            cur += 1
    rng.shuffle(rows)
    return rows, dict(dist)


def merge_unique_by_id(base: List[Dict[str, Any]], extra: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], int]:
    seen = {str(x.get("id", "")).strip() for x in base}
    merged = list(base)
    added = 0
    for row in extra:
        rid = str(row.get("id", "")).strip()
        if not rid or rid in seen:
            continue
        seen.add(rid)
        merged.append(row)
        added += 1
    return merged, added


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate 12k targeted weak-intent data and merge into cleaned SFT.")
    parser.add_argument(
        "--base-sft",
        default="/home/ubuntu/qwen35a3b_finetune/datasets/sft_openai_messages.cleaned.v3_merged.single75_exact.repair_v3.tydata_pdfcurated.nometa.jsonl",
    )
    parser.add_argument(
        "--counts",
        default=DEFAULT_COUNTS,
        help="intent=count, supports: 注单异常, 串关, 滚球, 赔率, 限红",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        default=f"/home/ubuntu/qwen35a3b_finetune/datasets/weak_intent_targeted_12k_{now_ts()}",
    )
    args = parser.parse_args()

    base_path = Path(args.base_sft).resolve()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    counts = parse_counts(args.counts)
    increment_rows, dist = make_increment_rows(counts, args.seed)
    inc_path = out_dir / "weak_intent_targeted_12000.jsonl"
    write_jsonl(inc_path, increment_rows)

    base_rows = read_jsonl(base_path)
    merged_rows, added = merge_unique_by_id(base_rows, increment_rows)
    merged_path = out_dir / f"{base_path.stem}.weak12k_merged.jsonl"
    write_jsonl(merged_path, merged_rows)

    payload = {
        "base_sft": str(base_path),
        "output_dir": str(out_dir),
        "increment_path": str(inc_path),
        "merged_path": str(merged_path),
        "target_counts": {k: v for k, v in counts},
        "generated_increment_rows": len(increment_rows),
        "generated_distribution": dist,
        "base_rows": len(base_rows),
        "merged_added_rows": added,
        "merged_total_rows": len(merged_rows),
        "seed": args.seed,
    }
    summary_path = out_dir / "weak_intent_targeted_12000_summary.json"
    summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"[INFO] increment: {inc_path}")
    print(f"[INFO] merged: {merged_path}")
    print(f"[INFO] summary: {summary_path}")


if __name__ == "__main__":
    main()
