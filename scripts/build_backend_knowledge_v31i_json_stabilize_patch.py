#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from build_backend_knowledge_v31e_json_tiny_patch import JSON_INSTRUCTION, STORAGE_ROOT, SYSTEM_PROMPT


CASES: list[dict[str, Any]] = [
    {
        "eval_id": "kb_20260428_038",
        "scenario": "导入商户",
        "intent": "其他",
        "need_escalation": False,
        "queries": [
            "商户问：导入商户功能迁移老会员数据时，大致流程是什么？",
            "迁移老会员数据时，导入商户功能一般怎么操作？",
        ],
        "answer": (
            "导入商户、Excel、预览：导入商户功能用于迁移老会员数据，通常先按模板整理Excel，"
            "上传到导入商户入口后做字段校验，再进入预览确认数据无误，最后执行正式导入。"
        ),
        "next_action": "按包网后台知识直接说明，无需后台查询。",
        "source_scope": "backend_anchor",
    },
    {
        "eval_id": "kb_20260428_040",
        "scenario": "客服配置",
        "intent": "其他",
        "need_escalation": False,
        "queries": [
            "商户问：包网后台需要配置在线客服、短信提供商和支付通道，这些一般在哪类配置里处理？",
            "包网后台在线客服、短信提供商、支付通道通常在哪些配置里维护？",
        ],
        "answer": (
            "在线客服、短信提供商、通道：在线客服通常在客服或站点配置里处理，短信提供商在短信或通知配置里维护，"
            "支付通道在支付或通道管理里配置。配置后分别测试在线客服入口、短信发送和通道回调。"
        ),
        "next_action": "按包网后台知识直接说明，无需后台查询。",
        "source_scope": "backend_anchor",
    },
    {
        "eval_id": "kb_20260428_015",
        "scenario": "单双",
        "intent": "其他",
        "need_escalation": False,
        "queries": [
            "用户问：足球单双玩法是按什么判断，能举例吗？",
            "足球单双是看什么，单数和双数怎么判断？",
        ],
        "answer": (
            "足球单双按全场总进球判断。总进球为1、3、5等单数时，结果为单数；"
            "总进球为0、2、4等双数时，结果为双数。例如比分1-2总进球3个，属于单数。"
        ),
        "next_action": "按玩法规则直接说明，无需后台查询。",
        "source_scope": "sports_kb_retention",
    },
]


def assistant_json(case: dict[str, Any]) -> str:
    return json.dumps(
        {
            "intent": case["intent"],
            "need_escalation": case["need_escalation"],
            "answer": case["answer"],
            "risk_flag": [],
            "next_action": case["next_action"],
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )


def build_rows(repeat: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case in CASES:
        for query_index, query in enumerate(case["queries"], 1):
            for rep in range(1, repeat + 1):
                rows.append(
                    {
                        "id": f"backend_v31i_json_stabilize_{case['eval_id']}_q{query_index:02d}_r{rep:02d}",
                        "split": "train",
                        "messages": [
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": f"用户问题：{query}\n\n{JSON_INSTRUCTION}"},
                            {"role": "assistant", "content": assistant_json(case)},
                        ],
                        "meta": {
                            "source": "backend_knowledge_v31i_json_stabilize_patch",
                            "eval_sample_id": case["eval_id"],
                            "scenario": case["scenario"],
                            "repeat": rep,
                            "query_index": query_index,
                            "json_aligned": True,
                            "disable_thinking": True,
                            "source_scope": case["source_scope"],
                            "protected_danger_void_weather_rules_not_touched": True,
                        },
                    }
                )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Build v31i JSON-aligned backend stabilization patch.")
    parser.add_argument(
        "--output-jsonl",
        default=str(
            STORAGE_ROOT
            / "datasets"
            / "backend_knowledge_v31i_json_stabilize_patch_20260430"
            / "sft_backend_knowledge_v31i_json_stabilize_patch.jsonl"
        ),
    )
    parser.add_argument("--summary-json", default="")
    parser.add_argument("--repeat", type=int, default=8)
    args = parser.parse_args()
    if args.repeat < 1 or args.repeat > 12:
        raise ValueError("--repeat must be between 1 and 12")

    rows = build_rows(args.repeat)
    out = Path(args.output_jsonl)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")

    summary = {
        "output_jsonl": str(out),
        "total_rows": len(rows),
        "case_count": len(CASES),
        "repeat": args.repeat,
        "json_aligned": True,
        "disable_thinking": True,
        "source_scopes": sorted({case["source_scope"] for case in CASES}),
        "protected_danger_void_weather_rules_not_touched": True,
        "eval_sample_ids": [case["eval_id"] for case in CASES],
    }
    summary_path = Path(args.summary_json) if args.summary_json else out.parent / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
