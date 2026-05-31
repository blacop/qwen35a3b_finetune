#!/usr/bin/env python3
"""
Build confusion-repair incremental datasets from eval failure cases.

Primary goal:
  - Focus on top confusion pairs from fix4 failure_cases.csv
  - Especially "gold in {串关规则, 滚球延迟, 赔率异常, 赛事变更}" predicted as 注单异常
  - Generate:
      * SFT incremental data (messages format)
      * DPO incremental data (prompt/chosen/rejected)
  - Optional merged datasets with existing base SFT/DPO corpora.

This script is deterministic with --seed and works with existing swift schema.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


DEFAULT_TARGET_INTENTS = ["串关规则", "滚球延迟", "赔率异常", "赛事变更"]
DEFAULT_CONFUSION_INTENT = "注单异常"

STYLE_PREFIX = ["", "客服你好，", "麻烦问下，", "请帮忙确认，", "我想确认一下，"]
STYLE_SUFFIX = ["", " 谢谢。", " 请尽快回复。", " 我这边很着急。", " 麻烦优先处理。"]

INTENT_TEMPLATE_PREFIX = {
    "串关规则": "该问题属于串关规则结算场景，需要按关数规则与赛事状态复核。",
    "滚球延迟": "该问题属于滚球延迟结算场景，需要核查赛中数据与结算链路。",
    "赔率异常": "该问题属于赔率异常争议，需要按下单时点与盘口快照复核。",
    "赛事变更": "该问题属于赛事变更场景，需要按赛程变更公告与结算规则核验。",
    "注单异常": "该问题属于注单异常场景，需要后台核对注单状态与结算记录。",
}

REJECTED_WRONG_BY_INTENT = {
    "串关规则": "这类问题按注单异常处理即可，无需区分串关规则。",
    "滚球延迟": "这类问题按注单异常处理即可，不需要核对赛中结算延迟。",
    "赔率异常": "这类问题按注单异常处理即可，不需要核对赔率变动。",
    "赛事变更": "这类问题按注单异常处理即可，不需要核对赛事变更公告。",
}


def now_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def parse_bool(v: Any) -> Optional[bool]:
    if v is None or v == "":
        return None
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    s = str(v).strip().lower()
    if s in {"true", "1", "yes", "y", "是", "需要", "需升级", "需转人工"}:
        return True
    if s in {"false", "0", "no", "n", "否", "不需要", "无需"}:
        return False
    return None


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid jsonl at line {line_no}: {exc}") from exc
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


def rewrite_query(query: str, rng: random.Random) -> str:
    q = re.sub(r"\s+", " ", (query or "").strip())
    if not q:
        return q
    return f"{rng.choice(STYLE_PREFIX)}{q}{rng.choice(STYLE_SUFFIX)}".strip()


def ensure_keywords(answer: str, must_include: List[str], topk: int = 2) -> str:
    if not must_include:
        return answer
    missing = [kw for kw in must_include[:topk] if kw and kw not in answer]
    if missing:
        answer = f"{answer} 处理时重点核对：{'、'.join(missing)}。"
    return answer


def build_chosen(intent: str, need_escalation: bool, must_include: List[str]) -> str:
    prefix = INTENT_TEMPLATE_PREFIX.get(intent, "该问题需要按平台规则核验。")
    pieces = [prefix]
    if need_escalation:
        pieces.append("需要后台查询并联系运营复核。")
        pieces.append("请提供账号与注单号（或订单号），核实后第一时间回复。")
    else:
        pieces.append("先按规则核验并给您同步处理结果。")
    out = "".join(pieces)
    return ensure_keywords(out, must_include, topk=2)


def build_rejected(intent: str, need_escalation: bool, confusion_intent: str) -> str:
    wrong = REJECTED_WRONG_BY_INTENT.get(intent, f"这类问题按{confusion_intent}处理即可。")
    if need_escalation:
        return f"{wrong} 不需要后台查询，稍后系统会自动恢复。"
    return f"{wrong} 建议直接升级人工并提交工单，不用核对当前场景。"


def dedupe_by_id(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    out: List[Dict[str, Any]] = []
    for row in rows:
        rid = str(row.get("id", "")).strip()
        if not rid or rid in seen:
            continue
        seen.add(rid)
        out.append(row)
    return out


def merge_unique(base_rows: List[Dict[str, Any]], extra_rows: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], int]:
    merged = list(base_rows)
    seen = {str(x.get("id", "")).strip() for x in base_rows}
    added = 0
    for row in extra_rows:
        rid = str(row.get("id", "")).strip()
        if not rid or rid in seen:
            continue
        seen.add(rid)
        merged.append(row)
        added += 1
    return merged, added


def read_failure_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def build_increment_pack(
    failure_rows: List[Dict[str, str]],
    eval_rows: List[Dict[str, Any]],
    target_intents: List[str],
    confusion_intent: str,
    max_samples_per_intent: int,
    aug_per_sample: int,
    seed: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    rng = random.Random(seed)
    eval_by_id = {str(x.get("id", "")).strip(): x for x in eval_rows}

    source_by_intent: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    source_confusions: Counter[Tuple[str, str]] = Counter()
    dropped_missing_source = 0
    dropped_no_query = 0

    for row in failure_rows:
        rid = str(row.get("id", "")).strip()
        gold_intent = str(row.get("gold_intent", "")).strip()
        pred_intent = str(row.get("pred_intent", "")).strip()
        request_ok = str(row.get("request_ok", "")).strip()
        if gold_intent not in target_intents:
            continue
        if pred_intent != confusion_intent:
            continue
        if request_ok != "1":
            continue
        source_confusions[(gold_intent, pred_intent)] += 1

        ev = eval_by_id.get(rid)
        if not ev:
            dropped_missing_source += 1
            continue
        user_query = str(ev.get("user_query", "")).strip()
        if not user_query:
            dropped_no_query += 1
            continue
        source_by_intent[gold_intent].append(
            {
                "id": rid,
                "user_query": user_query,
                "gold_intent": gold_intent,
                "gold_need_escalation": parse_bool(ev.get("gold_need_escalation")),
                "must_include": to_list(ev.get("must_include")),
                "scenario": str(ev.get("scenario", "")).strip(),
            }
        )

    sft_rows: List[Dict[str, Any]] = []
    dpo_rows: List[Dict[str, Any]] = []
    generated_counter: Counter[str] = Counter()
    quality_drop = 0

    for intent in target_intents:
        pool = source_by_intent.get(intent, [])
        if not pool:
            continue
        if max_samples_per_intent > 0 and len(pool) > max_samples_per_intent:
            pool = rng.sample(pool, max_samples_per_intent)

        for item in pool:
            sid = item["id"]
            need_esc = bool(item["gold_need_escalation"]) if item["gold_need_escalation"] is not None else True
            must_include = list(item["must_include"])
            chosen = build_chosen(intent=intent, need_escalation=need_esc, must_include=must_include)
            rejected = build_rejected(intent=intent, need_escalation=need_esc, confusion_intent=confusion_intent)
            if must_include and not any(kw in chosen for kw in must_include):
                quality_drop += 1
                continue

            for k in range(max(1, aug_per_sample)):
                q = rewrite_query(item["user_query"], rng)
                rid = f"fix4_confuse_{intent}_{sid}_{k+1:02d}"
                sft_rows.append(
                    {
                        "id": rid,
                        "split": "train",
                        "messages": [
                            {"role": "user", "content": q},
                            {"role": "assistant", "content": chosen},
                        ],
                        "meta": {
                            "intent": intent,
                            "need_escalation": need_esc,
                            "must_include": must_include,
                            "source_id": sid,
                            "from_fix4_failure": True,
                            "confusion_pred_intent": confusion_intent,
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
                            "intent": intent,
                            "need_escalation": need_esc,
                            "must_include": must_include,
                            "source_id": sid,
                            "from_fix4_failure": True,
                            "confusion_pred_intent": confusion_intent,
                            "rejected_policy": "wrong_route_to_注单异常",
                        },
                    }
                )
                generated_counter[intent] += 1

    sft_rows = dedupe_by_id(sft_rows)
    dpo_rows = dedupe_by_id(dpo_rows)

    summary = {
        "target_intents": target_intents,
        "confusion_intent": confusion_intent,
        "source_confusions": {f"{a}->{b}": c for (a, b), c in source_confusions.items()},
        "source_pool_by_intent": {k: len(v) for k, v in source_by_intent.items()},
        "generated_by_intent": dict(generated_counter),
        "generated_sft_rows": len(sft_rows),
        "generated_dpo_rows": len(dpo_rows),
        "dropped_missing_source": dropped_missing_source,
        "dropped_no_query": dropped_no_query,
        "quality_drop": quality_drop,
    }
    return sft_rows, dpo_rows, summary


def parse_target_intents(raw: str) -> List[str]:
    out = [x.strip() for x in re.split(r"[,;；，]\s*", raw or "") if x.strip()]
    return out or list(DEFAULT_TARGET_INTENTS)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build fix4 confusion repair SFT/DPO incremental pack.")
    parser.add_argument(
        "--failure-csv",
        default="/home/ubuntu/qwen35a3b_finetune/eval_outputs/deployed_aibot_proxy8010_prod500_fix4_20260415T132323Z/failure_cases.csv",
    )
    parser.add_argument(
        "--eval-jsonl",
        default="/home/ubuntu/qwen35a3b_finetune/datasets/eval_sports_customer_prod_500.jsonl",
    )
    parser.add_argument(
        "--target-intents",
        default=",".join(DEFAULT_TARGET_INTENTS),
        help="Comma-separated intents to repair.",
    )
    parser.add_argument("--confusion-intent", default=DEFAULT_CONFUSION_INTENT)
    parser.add_argument("--max-samples-per-intent", type=int, default=250)
    parser.add_argument("--aug-per-sample", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        default=f"/home/ubuntu/qwen35a3b_finetune/datasets/fix4_confusion_repair_pack_{now_ts()}",
    )
    parser.add_argument("--base-sft", default="", help="Optional base SFT dataset path.")
    parser.add_argument("--base-dpo", default="", help="Optional base DPO dataset path.")
    parser.add_argument("--write-merged", action="store_true", help="Write merged SFT/DPO with base datasets.")
    args = parser.parse_args()

    failure_csv = Path(args.failure_csv).resolve()
    eval_jsonl = Path(args.eval_jsonl).resolve()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    failure_rows = read_failure_rows(failure_csv)
    eval_rows = read_jsonl(eval_jsonl)
    target_intents = parse_target_intents(args.target_intents)

    sft_rows, dpo_rows, summary = build_increment_pack(
        failure_rows=failure_rows,
        eval_rows=eval_rows,
        target_intents=target_intents,
        confusion_intent=str(args.confusion_intent).strip() or DEFAULT_CONFUSION_INTENT,
        max_samples_per_intent=max(0, int(args.max_samples_per_intent)),
        aug_per_sample=max(1, int(args.aug_per_sample)),
        seed=int(args.seed),
    )

    sft_path = out_dir / "fix4_confusion_sft_increment.jsonl"
    dpo_path = out_dir / "fix4_confusion_dpo_increment.jsonl"
    write_jsonl(sft_path, sft_rows)
    write_jsonl(dpo_path, dpo_rows)

    payload: Dict[str, Any] = {
        "failure_csv": str(failure_csv),
        "eval_jsonl": str(eval_jsonl),
        "output_dir": str(out_dir),
        **summary,
        "sft_path": str(sft_path),
        "dpo_path": str(dpo_path),
    }

    if args.write_merged:
        if not args.base_sft or not args.base_dpo:
            raise ValueError("--write-merged requires --base-sft and --base-dpo")
        base_sft_path = Path(args.base_sft).resolve()
        base_dpo_path = Path(args.base_dpo).resolve()
        base_sft_rows = read_jsonl(base_sft_path)
        base_dpo_rows = read_jsonl(base_dpo_path)
        merged_sft, add_sft = merge_unique(base_sft_rows, sft_rows)
        merged_dpo, add_dpo = merge_unique(base_dpo_rows, dpo_rows)
        merged_sft_path = out_dir / "fix4_confusion_sft_merged.jsonl"
        merged_dpo_path = out_dir / "fix4_confusion_dpo_merged.jsonl"
        write_jsonl(merged_sft_path, merged_sft)
        write_jsonl(merged_dpo_path, merged_dpo)
        payload.update(
            {
                "base_sft": str(base_sft_path),
                "base_dpo": str(base_dpo_path),
                "merged_sft_path": str(merged_sft_path),
                "merged_dpo_path": str(merged_dpo_path),
                "merged_added": {"sft": add_sft, "dpo": add_dpo},
                "merged_total": {"sft": len(merged_sft), "dpo": len(merged_dpo)},
            }
        )

    summary_path = out_dir / "fix4_confusion_repair_summary.json"
    summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"[INFO] sft: {sft_path}")
    print(f"[INFO] dpo: {dpo_path}")
    if args.write_merged:
        print(f"[INFO] merged_sft: {payload['merged_sft_path']}")
        print(f"[INFO] merged_dpo: {payload['merged_dpo_path']}")
    print(f"[INFO] summary: {summary_path}")


if __name__ == "__main__":
    main()
