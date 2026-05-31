#!/usr/bin/env python3
"""
Generate weak-intent boost data (SFT + DPO) from eval JSONL.

Outputs:
  - weak_intent_sft_boost.jsonl
  - weak_intent_dpo_boost.jsonl
  - weak_intent_boost_summary.json
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List


DEFAULT_WEAK_INTENTS = ["赛事变更", "赔率异常", "限红风控", "滚球延迟"]

WRONG_INTENT_HINT = {
    "赛事变更": "注单异常",
    "赔率异常": "充值",
    "限红风控": "充值",
    "滚球延迟": "其他",
}

INTENT_GOOD_OPENING = {
    "赛事变更": "该问题属于赛事变更场景，按平台公告与官方赛程规则执行。",
    "赔率异常": "该问题属于赔率异常争议，需要按下单时点和盘口快照复核。",
    "限红风控": "该问题属于限红风控场景，需要按账户风险规则复核。",
    "滚球延迟": "该问题属于滚球延迟结算场景，需要核查赛中结算链路。",
    "注单异常": "该问题属于注单异常，需要核查注单状态与结算记录。",
    "投诉": "已收到您的投诉，我们会按升级流程记录并处理。",
}


def now_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"invalid json at line {line_no}: {e}") from e
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def to_list(v: Any) -> List[str]:
    if v is None:
        return []
    if isinstance(v, list):
        return [str(x).strip() for x in v if str(x).strip()]
    s = str(v).strip()
    if not s:
        return []
    if s.startswith("[") and s.endswith("]"):
        try:
            arr = json.loads(s)
            if isinstance(arr, list):
                return [str(x).strip() for x in arr if str(x).strip()]
        except Exception:
            pass
    return [x.strip() for x in re.split(r"[,;；，]\s*", s) if x.strip()]


def make_query_variants(query: str, scenario: str, n: int, rng: random.Random) -> List[str]:
    q = query.strip()
    prefixes = ["", "麻烦确认一下，", "请尽快协助，", "我这边比较着急，", "请帮我核实："]
    suffixes = ["", " 请尽快回复。", " 谢谢。", " 麻烦尽快处理。", " 我这边等很久了。"]
    variants = []
    base = q if q else scenario
    for _ in range(max(1, n)):
        v = f"{rng.choice(prefixes)}{base}{rng.choice(suffixes)}".strip()
        if v not in variants:
            variants.append(v)
    if not variants:
        variants = [base]
    return variants


def build_chosen(intent: str, need_escalation: bool, must_include: List[str]) -> str:
    opening = INTENT_GOOD_OPENING.get(intent, "该问题需要按平台规则核实处理。")
    pieces = [opening]
    if need_escalation:
        pieces.append("该问题需要后台查询并联系运营核实。")
        pieces.append("请提供账号与注单号（或订单号），核实后第一时间回复您。")
    else:
        pieces.append("请您以平台最新公告与规则说明为准。")
    if must_include:
        kw = "、".join(must_include[:2])
        pieces.append(f"处理时会重点核对：{kw}。")
    return "".join(pieces)


def build_rejected(intent: str, need_escalation: bool) -> str:
    wrong_hint = WRONG_INTENT_HINT.get(intent, "充值")
    if need_escalation:
        return (
            f"这类问题一般不需要后台处理，先按{wrong_hint}流程操作即可。"
            "请您稍后再试，系统会自动恢复。"
        )
    return (
        f"该问题需要立刻走人工工单并升级运营（按{wrong_hint}处理）。"
        "请先提交账号、姓名、银行卡信息等待人工回访。"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate weak-intent boost SFT/DPO packs.")
    parser.add_argument(
        "--input-jsonl",
        default="/home/ubuntu/qwen35a3b_finetune/datasets/eval_sports_customer_prod_500.jsonl",
        help="Input eval JSONL with fields user_query/gold_intent/gold_need_escalation/must_include.",
    )
    parser.add_argument(
        "--output-dir",
        default=f"/home/ubuntu/qwen35a3b_finetune/datasets/weak_intent_boost_{now_ts()}",
        help="Output directory.",
    )
    parser.add_argument(
        "--weak-intents",
        default=",".join(DEFAULT_WEAK_INTENTS),
        help="Comma-separated intents to boost.",
    )
    parser.add_argument("--aug-per-sample", type=int, default=2, help="How many query variants per sample.")
    parser.add_argument("--max-samples-per-intent", type=int, default=0, help="0 means all.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    weak_intents = [x.strip() for x in args.weak_intents.split(",") if x.strip()]
    rows = read_jsonl(Path(args.input_jsonl).resolve())

    by_intent: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        intent = str(row.get("gold_intent", "")).strip()
        if intent in weak_intents:
            by_intent[intent].append(row)

    selected: List[Dict[str, Any]] = []
    for intent in weak_intents:
        group = by_intent.get(intent, [])
        if args.max_samples_per_intent and len(group) > args.max_samples_per_intent:
            group = rng.sample(group, args.max_samples_per_intent)
        selected.extend(group)

    sft_rows: List[Dict[str, Any]] = []
    dpo_rows: List[Dict[str, Any]] = []
    gen_counter: Counter[str] = Counter()

    for row in selected:
        sample_id = str(row.get("id", "sample"))
        intent = str(row.get("gold_intent", "")).strip()
        need_escalation = bool(row.get("gold_need_escalation", False))
        query = str(row.get("user_query", "")).strip()
        scenario = str(row.get("scenario", "")).strip()
        must_include = to_list(row.get("must_include"))

        chosen = build_chosen(intent=intent, need_escalation=need_escalation, must_include=must_include)
        rejected = build_rejected(intent=intent, need_escalation=need_escalation)
        variants = make_query_variants(query=query, scenario=scenario, n=args.aug_per_sample, rng=rng)

        for idx, q in enumerate(variants, 1):
            rid = f"weak_{sample_id}_{idx}"
            sft_rows.append(
                {
                    "id": rid,
                    "split": "train",
                    "messages": [
                        {"role": "user", "content": q},
                        {"role": "assistant", "content": chosen},
                    ],
                    "meta": {
                        "source_id": sample_id,
                        "intent": intent,
                        "need_escalation": need_escalation,
                        "weak_intent_boost": True,
                    },
                }
            )
            dpo_rows.append(
                {
                    "id": f"dpo_{rid}",
                    "split": "train",
                    "prompt": q,
                    "chosen": chosen,
                    "rejected": rejected,
                    "meta": {
                        "source_id": sample_id,
                        "intent": intent,
                        "need_escalation": need_escalation,
                        "weak_intent_boost": True,
                    },
                }
            )
            gen_counter[intent] += 1

    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    sft_path = out_dir / "weak_intent_sft_boost.jsonl"
    dpo_path = out_dir / "weak_intent_dpo_boost.jsonl"
    summary_path = out_dir / "weak_intent_boost_summary.json"

    write_jsonl(sft_path, sft_rows)
    write_jsonl(dpo_path, dpo_rows)

    summary = {
        "input_jsonl": str(Path(args.input_jsonl).resolve()),
        "output_dir": str(out_dir),
        "weak_intents": weak_intents,
        "selected_source_samples": len(selected),
        "aug_per_sample": args.aug_per_sample,
        "generated_sft_rows": len(sft_rows),
        "generated_dpo_rows": len(dpo_rows),
        "generated_by_intent": dict(gen_counter),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[INFO] sft: {sft_path}")
    print(f"[INFO] dpo: {dpo_path}")
    print(f"[INFO] summary: {summary_path}")


if __name__ == "__main__":
    main()

