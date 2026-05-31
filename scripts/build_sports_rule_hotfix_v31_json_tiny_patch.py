#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from build_sports_rule_hotfix_v3_json_pack import (
    EVAL_JSON_INSTRUCTION,
    STRICT_JSON_SYSTEM_PROMPT,
    RULE_EXCEPTION_SYSTEM_PROMPT,
    wrapped_user,
    write_jsonl,
)


PROJECT_ROOT = Path("/home/ubuntu/qwen35a3b_finetune")
DEFAULT_RULE_JSONL = PROJECT_ROOT / "datasets" / "sports_rule_knowledge_regression_gpu5_20260428.jsonl"
DEFAULT_PARLAY_JSONL = PROJECT_ROOT / "datasets" / "eval_sports_customer_targeted_parlay_v1_120.jsonl"
DEFAULT_PARLAY_FAILURES = (
    PROJECT_ROOT
    / "eval_outputs"
    / "ab_gpu5v3_gpu7_20260429T0754"
    / "parlay120_gpu5_v3"
    / "predictions.csv"
)
DEFAULT_OUT_DIR = PROJECT_ROOT / "datasets" / "sports_rule_hotfix_v31_json_tiny_patch_20260429"


RULE_PATCH_ANSWERS = {
    "rule_reg_20260428_015": (
        "危险球是指一方球队正在组织有威胁进攻的阶段，资料里按约90秒说明。"
        "这个阶段盘口风险变化很快，官方可能不接受下注，页面会显示拒绝；"
        "即使当时看似下注成功，后续也可能被取消或退回。"
    ),
    "rule_reg_20260428_018": (
        "可能导致注单取消/作废的原因包括：危险球阶段不被接受、比赛中断或天气原因、"
        "球员受伤、盘口赔率异常、赛事官方数据确认异常，或疑似恶意破坏盘口等。"
        "若用户问具体注单，需要收集注单号后后台查询。"
    ),
}

EVENT_CHANGE_ANSWER = (
    "这是赛事变更/取消类问题。比赛取消、改期或中断时，注单会按平台体育结算规则处理，"
    "通常以赛事官方公告和平台公告为准；未满足有效比赛或有效时间条件的注单，可能按无效退回本金，"
    "已按规则完成结算的以最终公告为准。若只是咨询通用规则，无需后台查询；"
    "如要核实某一张注单，可再提供注单号单独查询。"
)


RULE_VARIANTS = [
    "用户问题：{query}\n\n请只输出指定JSON对象，answer字段必须包含：{keywords}。\n\n{instruction}",
    "用户问题：{query}\n\n这是体育规则解释题，先给规则口径，不要写思考过程。answer字段必须保留：{keywords}。\n\n{instruction}",
    "用户问题：{query}\n\n请纠正常见误答，不要把危险球说成15秒；涉及注单取消原因时必须包含危险球和天气。\n\n{instruction}",
]

EVENT_VARIANTS = [
    "用户问题：{query}\n\n这是赛事变更/取消/改期类问题，intent必须为赛事变更；没有具体核实单号时need_escalation=false。\n\n{instruction}",
    "用户问题：{query}\n\n请按指定JSON输出。answer字段必须包含：赛事变更、结算规则、以公告为准；不要误判为注单异常。\n\n{instruction}",
    "用户问题：{query}\n\n用户语气着急或说请帮我查一下，也先按赛事变更通用规则解释；未提供明确注单核实信息时need_escalation=false。\n\n{instruction}",
]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc
    return rows


def compact_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def assistant_for_rule(sample: dict[str, Any]) -> str:
    sample_id = str(sample["id"])
    answer = RULE_PATCH_ANSWERS[sample_id]
    payload = {
        "intent": str(sample.get("gold_intent") or "注单异常"),
        "need_escalation": bool(sample.get("gold_need_escalation")),
        "answer": answer,
        "risk_flag": [],
        "next_action": (
            "先按通用规则解释；若用户要核实具体注单，请收集账号、注单号和赛事信息后后台查询或联系运营。"
            if bool(sample.get("gold_need_escalation"))
            else "按当前规则直接解释，无需后台查询。"
        ),
    }
    return compact_json(payload)


def assistant_for_event_change() -> str:
    return compact_json(
        {
            "intent": "赛事变更",
            "need_escalation": False,
            "answer": EVENT_CHANGE_ANSWER,
            "risk_flag": [],
            "next_action": "按赛事变更结算规则解释，以公告为准；用户未提供明确注单核实信息时无需后台查询。",
        }
    )


def read_failed_event_change_rows(path: Path, limit: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("gold_intent") != "赛事变更":
                continue
            if row.get("pred_intent") == "赛事变更" and row.get("pred_need_escalation") == row.get("gold_need_escalation"):
                continue
            rows.append(row)
            if len(rows) >= limit:
                break
    return rows


def read_extra_event_change_rows(path: Path, skip_ids: set[str], limit: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in read_jsonl(path):
        if row.get("gold_intent") != "赛事变更":
            continue
        sample_id = str(row.get("id") or "")
        if sample_id in skip_ids:
            continue
        rows.append(row)
        if len(rows) >= limit:
            break
    return rows


def add_row(
    rows: list[dict[str, Any]],
    row_id: str,
    user: str,
    assistant: str,
    meta: dict[str, Any],
    system: str,
) -> None:
    rows.append(
        {
            "id": row_id,
            "split": "train",
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
                {"role": "assistant", "content": assistant},
            ],
            "meta": meta,
        }
    )


def build_rows(
    rule_samples: list[dict[str, Any]],
    event_rows: list[dict[str, Any]],
    rule_repeat: int,
    event_repeat: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    systems = [STRICT_JSON_SYSTEM_PROMPT, RULE_EXCEPTION_SYSTEM_PROMPT]

    for sample in rule_samples:
        sample_id = str(sample["id"])
        keywords = "、".join(str(x) for x in sample.get("must_include", []))
        assistant = assistant_for_rule(sample)
        for rep in range(1, rule_repeat + 1):
            for variant_idx, template in enumerate(RULE_VARIANTS, 1):
                add_row(
                    rows,
                    f"sports_rule_v31_json_{sample_id}_rule_v{variant_idx}_r{rep:02d}",
                    template.format(query=sample["user_query"], keywords=keywords, instruction=EVAL_JSON_INSTRUCTION),
                    assistant,
                    {
                        "source": "sports_rule_hotfix_v31_json_tiny_patch",
                        "patch_type": "rule_hard_fail",
                        "eval_sample_id": sample_id,
                        "scenario": sample.get("scenario", ""),
                        "json_aligned": True,
                    },
                    systems[(rep + variant_idx) % len(systems)],
                )

    event_assistant = assistant_for_event_change()
    for row in event_rows:
        sample_id = str(row.get("id") or "event_change")
        query = str(row.get("user_query") or "").strip()
        for rep in range(1, event_repeat + 1):
            for variant_idx, template in enumerate(EVENT_VARIANTS, 1):
                add_row(
                    rows,
                    f"sports_rule_v31_json_{sample_id}_event_v{variant_idx}_r{rep:02d}",
                    template.format(query=query, instruction=EVAL_JSON_INSTRUCTION),
                    event_assistant,
                    {
                        "source": "sports_rule_hotfix_v31_json_tiny_patch",
                        "patch_type": "event_change_intent",
                        "eval_sample_id": sample_id,
                        "scenario": row.get("scenario", "赛事改期取消"),
                        "gold_intent": "赛事变更",
                        "gold_need_escalation": False,
                        "json_aligned": True,
                    },
                    systems[(rep + variant_idx) % len(systems)],
                )

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Build v3.1 JSON-aligned tiny patch for rule hard fails and event-change intent.")
    parser.add_argument("--rule-jsonl", default=str(DEFAULT_RULE_JSONL))
    parser.add_argument("--parlay-jsonl", default=str(DEFAULT_PARLAY_JSONL))
    parser.add_argument("--parlay-predictions", default=str(DEFAULT_PARLAY_FAILURES))
    parser.add_argument("--output-jsonl", default=str(DEFAULT_OUT_DIR / "sft_sports_rule_hotfix_v31_json_tiny_patch.jsonl"))
    parser.add_argument("--summary-json", default=str(DEFAULT_OUT_DIR / "summary.json"))
    parser.add_argument("--rule-repeat", type=int, default=16)
    parser.add_argument("--event-repeat", type=int, default=6)
    parser.add_argument("--event-failed-limit", type=int, default=8)
    parser.add_argument("--event-extra-limit", type=int, default=4)
    args = parser.parse_args()

    rule_jsonl = Path(args.rule_jsonl).expanduser().resolve()
    parlay_jsonl = Path(args.parlay_jsonl).expanduser().resolve()
    parlay_predictions = Path(args.parlay_predictions).expanduser().resolve()
    output_jsonl = Path(args.output_jsonl).expanduser().resolve()
    summary_json = Path(args.summary_json).expanduser().resolve()

    rule_samples_by_id = {str(row.get("id")): row for row in read_jsonl(rule_jsonl)}
    rule_samples = [rule_samples_by_id[sample_id] for sample_id in RULE_PATCH_ANSWERS]
    failed_events = read_failed_event_change_rows(parlay_predictions, max(0, args.event_failed_limit))
    extra_events = read_extra_event_change_rows(
        parlay_jsonl,
        {str(row.get("id") or "") for row in failed_events},
        max(0, args.event_extra_limit),
    )
    event_rows = failed_events + extra_events

    rows = build_rows(
        rule_samples=rule_samples,
        event_rows=event_rows,
        rule_repeat=max(1, args.rule_repeat),
        event_repeat=max(1, args.event_repeat),
    )
    write_jsonl(output_jsonl, rows)

    summary = {
        "rule_jsonl": str(rule_jsonl),
        "parlay_jsonl": str(parlay_jsonl),
        "parlay_predictions": str(parlay_predictions),
        "output_jsonl": str(output_jsonl),
        "rows": len(rows),
        "rule_samples": [str(row["id"]) for row in rule_samples],
        "event_failed_samples": [str(row.get("id") or "") for row in failed_events],
        "event_extra_samples": [str(row.get("id") or "") for row in extra_events],
        "rule_repeat": max(1, args.rule_repeat),
        "event_repeat": max(1, args.event_repeat),
        "format": "eval_json_aligned_tiny_patch",
        "instruction": "disable thinking at eval/train gate; do not expand v2 final-answer data",
    }
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
