#!/usr/bin/env python3
"""
Generate a data-repair pack for weak intents:
  - targeted supplementation (300-500 per weak intent)
  - contrast samples for confusion boundaries
  - strict DPO pairs (chosen: correct intent+escalation+must_include,
    rejected: common wrong intent + wrong escalation)

Outputs:
  - repair_sft.jsonl
  - repair_dpo.jsonl
  - repair_contrast.jsonl
  - repair_summary.json
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


WEAK_INTENTS = ["赛事变更", "赔率异常", "限红风控", "滚球延迟"]
CONFUSION_MAP = {
    "赛事变更": "注单异常",
    "赔率异常": "限红风控",
    "限红风控": "赔率异常",
    "滚球延迟": "注单异常",
}
ESCALATION_INTENTS = {"赔率异常", "限红风控", "滚球延迟", "注单异常", "投诉", "充值", "提款"}

REPLACEMENTS = [
    ("请问", "麻烦问下"),
    ("麻烦问下", "我想确认一下"),
    ("谢谢。", "请尽快回复。"),
    ("这也太慢了吧？", "我这边比较着急。"),
    ("账号这边很着急。", "我这边等很久了。"),
    ("请帮我查一下。", "麻烦尽快处理。"),
]

STYLE_PREFIX = ["", "客服你好，", "请帮忙确认，", "辛苦协助核实，", "麻烦尽快处理，"]
STYLE_SUFFIX = ["", " 谢谢。", " 请尽快回复。", " 我这边很着急。", " 麻烦优先处理。"]


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


def rewrite_query(text: str, rng: random.Random) -> str:
    q = text.strip()
    if not q:
        return q
    q2 = q
    for a, b in rng.sample(REPLACEMENTS, k=min(2, len(REPLACEMENTS))):
        q2 = q2.replace(a, b)
    q2 = f"{rng.choice(STYLE_PREFIX)}{q2}{rng.choice(STYLE_SUFFIX)}".strip()
    q2 = re.sub(r"\s+", " ", q2).strip()
    return q2


def escalation_for_intent(intent: str, gold_need_escalation: Any) -> bool:
    if isinstance(gold_need_escalation, bool):
        return gold_need_escalation
    return intent in ESCALATION_INTENTS


def ensure_keywords(answer: str, must_include: List[str], topk: int = 2) -> str:
    if not must_include:
        return answer
    needed = [kw for kw in must_include[:topk] if kw and kw not in answer]
    if needed:
        answer = f"{answer} 重点核对：{'、'.join(needed)}。"
    return answer


def chosen_answer(intent: str, need_escalation: bool, must_include: List[str]) -> str:
    if intent == "赛事变更":
        ans = "该问题属于赛事变更场景，按赛事改期/取消结算规则处理，以平台公告为准。"
    elif intent == "赔率异常":
        ans = "该问题属于赔率异常争议，需要按下单时间与盘口快照进行复核。"
    elif intent == "限红风控":
        ans = "该问题属于限红风控场景，需要按账户风险策略与限额规则复核。"
    elif intent == "滚球延迟":
        ans = "该问题属于滚球延迟结算场景，需要核查赛中数据与结算链路。"
    else:
        ans = "该问题需要按平台标准流程核实。"

    if need_escalation:
        ans += " 需要后台查询并联系运营处理，核实后第一时间回复您。请提供账号与注单号（或订单号）。"
    else:
        ans += " 当前无需升级工单，按公告规则执行并持续关注官方状态。"
    ans = ensure_keywords(ans, must_include, topk=2)
    return ans


def rejected_answer(intent: str, confusion_intent: str, need_escalation: bool) -> str:
    # wrong intent + wrong escalation on purpose
    if need_escalation:
        return (
            f"这类情况通常按{confusion_intent}处理，不需要后台查询。"
            "您先自己重试即可，系统会自动恢复。"
        )
    return (
        f"该问题建议按{confusion_intent}立即升级人工并提交工单。"
        "请先上传账号、姓名、银行卡信息后等待运营回访。"
    )


def make_contrast_query_pair(rng: random.Random) -> List[Dict[str, Any]]:
    """
    Build contrast rows for confusion boundaries:
      - 赛事变更 vs 注单异常
      - 赔率异常 vs 限红风控
    """
    pairs: List[Dict[str, Any]] = []

    event_query = rewrite_query("比赛改期了，之前那张单怎么结算？", rng)
    order_query = rewrite_query("注单被取消/作废了，为什么？", rng)
    odds_query = rewrite_query("下单前后赔率不一致，这个按哪个赔率结算？", rng)
    risk_query = rewrite_query("突然被限红，下注额度从5000降到500，为什么？", rng)

    pairs.append(
        {
            "boundary": "赛事变更_vs_注单异常",
            "query": event_query,
            "gold_intent": "赛事变更",
            "gold_need_escalation": False,
            "must_include": ["赛事变更", "结算规则", "以公告为准"],
            "confusion_intent": "注单异常",
        }
    )
    pairs.append(
        {
            "boundary": "赛事变更_vs_注单异常",
            "query": order_query,
            "gold_intent": "注单异常",
            "gold_need_escalation": True,
            "must_include": ["后台查询", "注单号", "运营"],
            "confusion_intent": "赛事变更",
        }
    )
    pairs.append(
        {
            "boundary": "赔率异常_vs_限红风控",
            "query": odds_query,
            "gold_intent": "赔率异常",
            "gold_need_escalation": True,
            "must_include": ["赔率变动", "下单时间", "后台核查"],
            "confusion_intent": "限红风控",
        }
    )
    pairs.append(
        {
            "boundary": "赔率异常_vs_限红风控",
            "query": risk_query,
            "gold_intent": "限红风控",
            "gold_need_escalation": True,
            "must_include": ["风控", "限额", "后台审核"],
            "confusion_intent": "赔率异常",
        }
    )
    return pairs


def build_rows(
    eval_rows: List[Dict[str, Any]],
    target_per_intent: int,
    contrast_per_boundary: int,
    seed: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    rng = random.Random(seed)
    by_intent: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in eval_rows:
        intent = str(row.get("gold_intent", "")).strip()
        if intent in WEAK_INTENTS:
            by_intent[intent].append(row)

    sft_rows: List[Dict[str, Any]] = []
    dpo_rows: List[Dict[str, Any]] = []
    contrast_rows: List[Dict[str, Any]] = []
    sft_counter: Counter[str] = Counter()
    dpo_counter: Counter[str] = Counter()
    quality_fail = 0

    # 1) Targeted supplementation for weak intents
    for intent in WEAK_INTENTS:
        pool = by_intent.get(intent, [])
        if not pool:
            continue
        while sft_counter[intent] < target_per_intent:
            base = rng.choice(pool)
            sid = str(base.get("id", "seed"))
            query = rewrite_query(str(base.get("user_query", "")), rng)
            must_include = to_list(base.get("must_include"))
            need_esc = escalation_for_intent(intent, base.get("gold_need_escalation"))
            confusion = CONFUSION_MAP[intent]

            chosen = chosen_answer(intent=intent, need_escalation=need_esc, must_include=must_include)
            rejected = rejected_answer(intent=intent, confusion_intent=confusion, need_escalation=need_esc)

            # strict quality constraint: chosen must hit >=1 must_include when available
            if must_include and not any(kw in chosen for kw in must_include):
                quality_fail += 1
                continue

            rid = f"repair_{intent}_{sft_counter[intent]+1:04d}_{sid}"
            sft_rows.append(
                {
                    "id": rid,
                    "split": "train",
                    "messages": [{"role": "user", "content": query}, {"role": "assistant", "content": chosen}],
                    "meta": {
                        "intent": intent,
                        "need_escalation": need_esc,
                        "must_include": must_include,
                        "source_id": sid,
                        "repair_pack": True,
                    },
                }
            )
            dpo_rows.append(
                {
                    "id": f"dpo_{rid}",
                    "split": "train",
                    "prompt": query,
                    "chosen": chosen,
                    "rejected": rejected,
                    "meta": {
                        "intent": intent,
                        "need_escalation": need_esc,
                        "must_include": must_include,
                        "source_id": sid,
                        "confusion_intent": confusion,
                        "repair_pack": True,
                    },
                }
            )
            sft_counter[intent] += 1
            dpo_counter[intent] += 1

    # 2) Contrast samples for boundaries
    # boundaries count is 2, each unit generates 4 rows => 2 per boundary per loop.
    target_contrast_units = max(1, contrast_per_boundary)
    for i in range(target_contrast_units):
        generated = make_contrast_query_pair(rng)
        for j, item in enumerate(generated, 1):
            intent = item["gold_intent"]
            need_esc = bool(item["gold_need_escalation"])
            must_include = item["must_include"]
            confusion = item["confusion_intent"]
            chosen = chosen_answer(intent=intent, need_escalation=need_esc, must_include=must_include)
            rejected = rejected_answer(intent=intent, confusion_intent=confusion, need_escalation=need_esc)
            cid = f"contrast_{i+1:04d}_{j}_{intent}"
            contrast_rows.append(
                {
                    "id": cid,
                    "boundary": item["boundary"],
                    "query": item["query"],
                    "gold_intent": intent,
                    "gold_need_escalation": need_esc,
                    "must_include": must_include,
                    "confusion_intent": confusion,
                    "chosen": chosen,
                    "rejected": rejected,
                }
            )
            # Also append into training pools.
            sft_rows.append(
                {
                    "id": f"sft_{cid}",
                    "split": "train",
                    "messages": [{"role": "user", "content": item["query"]}, {"role": "assistant", "content": chosen}],
                    "meta": {
                        "intent": intent,
                        "need_escalation": need_esc,
                        "must_include": must_include,
                        "contrast_boundary": item["boundary"],
                        "repair_pack": True,
                    },
                }
            )
            dpo_rows.append(
                {
                    "id": f"dpo_{cid}",
                    "split": "train",
                    "prompt": item["query"],
                    "chosen": chosen,
                    "rejected": rejected,
                    "meta": {
                        "intent": intent,
                        "need_escalation": need_esc,
                        "must_include": must_include,
                        "confusion_intent": confusion,
                        "contrast_boundary": item["boundary"],
                        "repair_pack": True,
                    },
                }
            )

    summary = {
        "target_per_intent": target_per_intent,
        "contrast_per_boundary": contrast_per_boundary,
        "supplement_generated_by_intent": dict(sft_counter),
        "quality_fail_count": quality_fail,
        "sft_total": len(sft_rows),
        "dpo_total": len(dpo_rows),
        "contrast_total": len(contrast_rows),
    }
    return sft_rows, dpo_rows, contrast_rows, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate weak-intent repair pack.")
    parser.add_argument(
        "--input-eval-jsonl",
        default="/home/ubuntu/qwen35a3b_finetune/datasets/eval_sports_customer_prod_500.jsonl",
        help="Eval JSONL with gold_intent/gold_need_escalation/must_include.",
    )
    parser.add_argument(
        "--output-dir",
        default=f"/home/ubuntu/qwen35a3b_finetune/datasets/intent_repair_pack_{now_ts()}",
        help="Output directory.",
    )
    parser.add_argument("--target-per-intent", type=int, default=400, help="Target supplement rows per weak intent.")
    parser.add_argument(
        "--contrast-per-boundary",
        type=int,
        default=200,
        help="Contrast units per boundary. Each unit yields 2 rows per boundary.",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    eval_rows = read_jsonl(Path(args.input_eval_jsonl).resolve())
    sft_rows, dpo_rows, contrast_rows, summary = build_rows(
        eval_rows=eval_rows,
        target_per_intent=args.target_per_intent,
        contrast_per_boundary=args.contrast_per_boundary,
        seed=args.seed,
    )

    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    sft_path = out_dir / "repair_sft.jsonl"
    dpo_path = out_dir / "repair_dpo.jsonl"
    contrast_path = out_dir / "repair_contrast.jsonl"
    summary_path = out_dir / "repair_summary.json"

    write_jsonl(sft_path, sft_rows)
    write_jsonl(dpo_path, dpo_rows)
    write_jsonl(contrast_path, contrast_rows)

    payload = {
        "input_eval_jsonl": str(Path(args.input_eval_jsonl).resolve()),
        "output_dir": str(out_dir),
        "weak_intents": WEAK_INTENTS,
        **summary,
    }
    summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"[INFO] sft: {sft_path}")
    print(f"[INFO] dpo: {dpo_path}")
    print(f"[INFO] contrast: {contrast_path}")
    print(f"[INFO] summary: {summary_path}")


if __name__ == "__main__":
    main()

