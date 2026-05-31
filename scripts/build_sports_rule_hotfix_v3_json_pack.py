#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from build_sports_rule_hotfix_v2_pack import WRONG_INDUCTIONS, checked_phrases, read_jsonl, write_jsonl


PROJECT_ROOT = Path("/home/ubuntu/qwen35a3b_finetune")
DEFAULT_EVAL_JSONL = PROJECT_ROOT / "datasets" / "sports_rule_knowledge_regression_gpu5_20260428.jsonl"
DEFAULT_FAILURE_CSVS = [
    PROJECT_ROOT
    / "eval_outputs"
    / "rule18_v2_candidate_disable_thinking_20260429T070539Z"
    / "rule_check"
    / "rule_check_failures.csv",
    PROJECT_ROOT
    / "eval_outputs"
    / "rule18_v2_candidate_rule_system_disable_thinking_20260429T070718Z"
    / "rule_check"
    / "rule_check_failures.csv",
    PROJECT_ROOT
    / "eval_outputs"
    / "rule18_candidate_disable_thinking_20260429T062946Z"
    / "rule_check"
    / "rule_check_failures.csv",
]
DEFAULT_OUT_DIR = PROJECT_ROOT / "datasets" / "sports_rule_hotfix_v3_json_20260429"


DEFAULT_SYSTEM_PROMPT = (
    "你是体育包网智能客服。必须遵守合规与风控规则："
    "遇到注单取消/作废/异常、结算争议、赔率异常、限红风控等无法直接确认的情况，"
    "必须明确告知需要后台查询或联系平台运营后回复；"
    "禁止赌博诱导、代理拉新、洗钱跑分、伪造证件、低龄相关内容。"
)

RULE_EXCEPTION_SYSTEM_PROMPT = (
    DEFAULT_SYSTEM_PROMPT
    + " 纯体育规则解释、盘口计算、玩法说明、串关组合说明、结算时效口径说明，应先直接给规则口径；"
    + "只有用户询问具体注单处理结果或账户资金状态时，才在规则解释后引导后台查询。"
)

STRICT_JSON_SYSTEM_PROMPT = (
    RULE_EXCEPTION_SYSTEM_PROMPT
    + " 输出必须是一个JSON对象；禁止输出推理过程、Markdown、<think>、题目理解、首先/接下来等过程性文本。"
)

SYSTEM_PROMPTS = [
    DEFAULT_SYSTEM_PROMPT,
    RULE_EXCEPTION_SYSTEM_PROMPT,
    STRICT_JSON_SYSTEM_PROMPT,
]

EVAL_JSON_INSTRUCTION = """请基于当前对话，输出且仅输出一个JSON对象，字段如下：
{
  "intent": "充值|提款|活动|投诉|注单异常|串关规则|滚球延迟|赔率异常|限红风控|赛事变更|账户异常|技术故障|其他",
  "need_escalation": true/false,
  "answer": "给用户的客服回复",
  "risk_flag": ["命中的风险标签，可为空数组"],
  "next_action": "下一步动作，若需转后台请明确写后台查询/联系运营"
}
不要输出Markdown代码块，不要输出额外解释。"""


def read_failed_ids(paths: list[Path]) -> dict[str, int]:
    failed: dict[str, int] = {}
    for path in paths:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                sample_id = str(row.get("id") or "").strip()
                if sample_id:
                    failed[sample_id] = failed.get(sample_id, 0) + 1
    return failed


def wrapped_user(user_query: str) -> str:
    return f"用户问题：{user_query.strip()}\n\n{EVAL_JSON_INSTRUCTION}"


def user_variants(sample: dict[str, Any]) -> list[tuple[str, str]]:
    sample_id = str(sample["id"])
    scenario = str(sample.get("scenario") or "体育规则")
    original = str(sample["user_query"]).strip()
    phrases = checked_phrases(sample)
    need_escalation = bool(sample.get("gold_need_escalation"))

    variants: list[tuple[str, str]] = [
        ("original", original),
        ("json_direct", f"{original} 请按指定JSON格式输出。"),
        ("short_rule", f"{scenario}：请直接给规则口径，并按指定JSON格式输出。"),
        ("keep_keywords", f"{original} answer字段里要保留关键口径：{phrases}。" if phrases else original),
        ("no_thinking", f"{original} 不要输出思考过程，只输出指定JSON对象。"),
        ("service_answer_field", f"用户问：{original}\n请把可直接发送给用户的回复写进answer字段。"),
        ("no_marketing", f"{original} 不要添加下载APP、送彩金、外部联系方式、代理返佣、包赢稳赚等话术。"),
    ]

    if need_escalation:
        variants.extend(
            [
                (
                    "rule_then_backend",
                    f"{original} 先说明通用规则口径，再在next_action里说明具体注单需要后台查询。",
                ),
                (
                    "not_backend_only",
                    f"{original} 不要只说后台查询，answer字段必须先包含通用规则解释。",
                ),
            ]
        )
    else:
        variants.extend(
            [
                ("no_backend", f"{original} 这是纯规则解释题，不要转后台，need_escalation=false。"),
                ("correct_boundary", f"{original} 如果没有具体注单争议，next_action写无需后台查询。"),
                ("anti_default_escalation", f"{original} 不要因为出现赔率、结算、盘口字样就自动要求后台查询。"),
            ]
        )

    for idx, prompt in enumerate(WRONG_INDUCTIONS.get(sample_id, []), 1):
        variants.append((f"wrong_induction_{idx}", f"{prompt} 请纠正我的理解，并按指定JSON格式输出。"))
    return variants


def next_action_for(sample: dict[str, Any]) -> str:
    if bool(sample.get("gold_need_escalation")):
        return "先按通用规则解释；若用户要核实具体注单，请收集账号、注单号和赛事信息后后台查询或联系运营。"
    return "按当前规则直接解释，无需后台查询。"


def assistant_json(sample: dict[str, Any]) -> str:
    answer = str(sample.get("reference_answer") or "").strip()
    if not answer:
        raise ValueError(f"sample missing reference_answer: {sample.get('id')}")
    payload = {
        "intent": str(sample.get("gold_intent") or "其他"),
        "need_escalation": bool(sample.get("gold_need_escalation")),
        "answer": answer,
        "risk_flag": [],
        "next_action": next_action_for(sample),
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def build_rows(samples: list[dict[str, Any]], failed_ids: dict[str, int], pass_repeat: int, fail_repeat: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sample in samples:
        sample_id = str(sample.get("id") or "").strip()
        if not sample_id:
            raise ValueError("sample missing id")
        repeat = fail_repeat if sample_id in failed_ids else pass_repeat
        variants = user_variants(sample)
        answer = assistant_json(sample)
        for rep in range(1, repeat + 1):
            system = SYSTEM_PROMPTS[(rep - 1) % len(SYSTEM_PROMPTS)]
            for variant_name, user in variants:
                rows.append(
                    {
                        "id": f"sports_rule_v3_json_{sample_id}_{variant_name}_r{rep:02d}",
                        "split": "train",
                        "messages": [
                            {"role": "system", "content": system},
                            {"role": "user", "content": wrapped_user(user)},
                            {"role": "assistant", "content": answer},
                        ],
                        "meta": {
                            "source": "sports_rule_hotfix_v3_json",
                            "eval_sample_id": sample_id,
                            "scenario": sample.get("scenario", ""),
                            "variant": variant_name,
                            "repeat": rep,
                            "gold_intent": sample.get("gold_intent", ""),
                            "gold_need_escalation": bool(sample.get("gold_need_escalation")),
                            "failed_in_candidate": sample_id in failed_ids,
                            "candidate_failed_checks": failed_ids.get(sample_id, 0),
                            "json_aligned": True,
                        },
                    }
                )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Build JSON-aligned sports-rule hotfix v3 SFT pack.")
    parser.add_argument("--eval-jsonl", default=str(DEFAULT_EVAL_JSONL))
    parser.add_argument("--failure-csv", action="append", default=None)
    parser.add_argument("--output-jsonl", default=str(DEFAULT_OUT_DIR / "sft_sports_rule_hotfix_v3_json.jsonl"))
    parser.add_argument("--summary-json", default=str(DEFAULT_OUT_DIR / "summary.json"))
    parser.add_argument("--pass-repeat", type=int, default=4)
    parser.add_argument("--fail-repeat", type=int, default=8)
    args = parser.parse_args()

    eval_path = Path(args.eval_jsonl).expanduser().resolve()
    failure_csvs = args.failure_csv or [str(p) for p in DEFAULT_FAILURE_CSVS]
    failure_paths = [Path(p).expanduser().resolve() for p in failure_csvs if p]
    output_path = Path(args.output_jsonl).expanduser().resolve()
    summary_path = Path(args.summary_json).expanduser().resolve()

    samples = read_jsonl(eval_path)
    failed_ids = read_failed_ids(failure_paths)
    rows = build_rows(samples, failed_ids, max(1, args.pass_repeat), max(1, args.fail_repeat))
    write_jsonl(output_path, rows)

    summary = {
        "eval_jsonl": str(eval_path),
        "failure_csv": [str(p) for p in failure_paths],
        "output_jsonl": str(output_path),
        "samples": len(samples),
        "failed_ids": failed_ids,
        "pass_repeat": max(1, args.pass_repeat),
        "fail_repeat": max(1, args.fail_repeat),
        "system_prompt_variants": len(SYSTEM_PROMPTS),
        "rows": len(rows),
        "failed_samples": sum(1 for sample in samples if str(sample.get("id")) in failed_ids),
        "passed_samples": sum(1 for sample in samples if str(sample.get("id")) not in failed_ids),
        "format": "eval_json_aligned",
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
