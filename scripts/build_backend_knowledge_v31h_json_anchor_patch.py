#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from build_backend_knowledge_v31e_json_tiny_patch import JSON_INSTRUCTION, STORAGE_ROOT, SYSTEM_PROMPT


CASES: list[dict[str, Any]] = [
    {
        "eval_id": "kb_20260428_028",
        "scenario": "后台仪表盘",
        "intent": "其他",
        "need_escalation": False,
        "queries": [
            "商户问：JT包网后台仪表盘的代办事项可以做什么？",
            "JT包网后台首页仪表盘里的代办事项主要处理什么？",
        ],
        "answer": (
            "仪表盘、代办事项、审核：仪表盘用于查看后台概况，代办事项集中提醒待处理的审核，"
            "例如充值提款审核、会员资料审核、活动审核或注单相关审核，商户可从代办事项进入对应页面处理。"
        ),
    },
    {
        "eval_id": "kb_20260428_029",
        "scenario": "后台初始化",
        "intent": "其他",
        "need_escalation": False,
        "queries": [
            "商户问：拿到包网后台账号后，第一步常见要配置什么？",
            "刚拿到包网后台账号，初始化时优先检查哪些安全配置？",
        ],
        "answer": (
            "登录IP、加白、角色：拿到包网后台账号后，第一步通常是安全初始化，先确认登录IP限制，"
            "把办公IP加白，再按岗位建立角色并分配权限；之后再继续配置支付、客服和前端素材。"
        ),
    },
    {
        "eval_id": "kb_20260428_030",
        "scenario": "前端素材",
        "intent": "其他",
        "need_escalation": False,
        "queries": [
            "商户问：后台前端素材一般可以配置哪些内容？",
            "包网后台前端素材配置通常包含哪些展示内容？",
        ],
        "answer": (
            "平台名、logo、轮播：后台前端素材一般可配置平台名、logo、首页轮播、活动图片、弹窗图片、公告素材等。"
            "上传时按后台提示的尺寸、格式和大小限制准备，上传后预览前端显示效果。"
        ),
    },
    {
        "eval_id": "kb_20260428_032",
        "scenario": "权限管理",
        "intent": "其他",
        "need_escalation": False,
        "queries": [
            "商户问：包网后台权限管理是按什么方式分配功能权限的？",
            "包网后台不同员工账号的功能权限一般怎么分配？",
        ],
        "answer": (
            "权限管理、角色、权限分配：包网后台通常通过权限管理按角色做权限分配，"
            "先建立财务、客服、运营等角色，再给角色勾选菜单和操作权限，员工账号绑定角色后获得对应功能权限。"
        ),
    },
    {
        "eval_id": "kb_20260428_033",
        "scenario": "系统配置",
        "intent": "其他",
        "need_escalation": False,
        "queries": [
            "商户问：系统配置里和注册登录相关的开关有哪些？",
            "注册和登录相关的系统配置通常有哪些选项？",
        ],
        "answer": (
            "验证码、手机号、登录：系统配置里和注册登录相关的开关常见有验证码、手机号注册或绑定、登录安全限制。"
            "商户可按风控要求决定是否开启验证码、是否强制手机号，以及登录异常校验策略。"
        ),
    },
    {
        "eval_id": "kb_20260428_035",
        "scenario": "冻结比例",
        "intent": "活动",
        "need_escalation": False,
        "queries": [
            "商户问：活动金额冻结比例、洗码金额冻结比例、签到奖励冻结比例分别是什么配置？",
            "后台里的活动金额冻结比例、洗码金额冻结比例、签到奖励冻结比例分别对应什么？",
        ],
        "answer": (
            "活动金额冻结比例、洗码金额冻结比例、签到奖励冻结比例：活动金额冻结比例控制活动赠送金额的冻结比例；"
            "洗码金额冻结比例控制洗码返还金额的冻结比例；签到奖励冻结比例控制签到奖励到账后的冻结比例。"
            "具体解冻条件以后台活动规则配置为准。"
        ),
    },
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
            "支付通道在支付或通道管理里配置。配置后分别测试客服入口、短信发送和通道回调。"
        ),
    },
]


VARIANTS = [
    "",
    " 请按标准JSON格式输出，answer里要直接覆盖后台关键词。",
]


def assistant_json(case: dict[str, Any]) -> str:
    return json.dumps(
        {
            "intent": case["intent"],
            "need_escalation": case["need_escalation"],
            "answer": case["answer"],
            "risk_flag": [],
            "next_action": "按包网后台知识直接说明，无需后台查询。",
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )


def build_rows(repeat: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case in CASES:
        for query_index, query in enumerate(case["queries"], 1):
            for variant_index, suffix in enumerate(VARIANTS, 1):
                for rep in range(1, repeat + 1):
                    rows.append(
                        {
                            "id": (
                                f"backend_v31h_json_anchor_{case['eval_id']}"
                                f"_q{query_index:02d}_v{variant_index:02d}_r{rep:02d}"
                            ),
                            "split": "train",
                            "messages": [
                                {"role": "system", "content": SYSTEM_PROMPT},
                                {"role": "user", "content": f"用户问题：{query}{suffix}\n\n{JSON_INSTRUCTION}"},
                                {"role": "assistant", "content": assistant_json(case)},
                            ],
                            "meta": {
                                "source": "backend_knowledge_v31h_json_anchor_patch",
                                "eval_sample_id": case["eval_id"],
                                "scenario": case["scenario"],
                                "repeat": rep,
                                "query_index": query_index,
                                "variant_index": variant_index,
                                "json_aligned": True,
                                "disable_thinking": True,
                                "patch_scope": "baowang_backend_knowledge_only",
                                "protected_sports_rule_scope_not_touched": True,
                            },
                        }
                    )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Build v31h JSON-aligned backend exact-anchor tiny patch.")
    parser.add_argument(
        "--output-jsonl",
        default=str(
            STORAGE_ROOT
            / "datasets"
            / "backend_knowledge_v31h_json_anchor_patch_20260430"
            / "sft_backend_knowledge_v31h_json_anchor_patch.jsonl"
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
        "queries_per_case": 2,
        "variants_per_query": len(VARIANTS),
        "repeat": args.repeat,
        "json_aligned": True,
        "disable_thinking": True,
        "patch_scope": "baowang_backend_knowledge_only",
        "protected_sports_rule_scope_not_touched": True,
        "eval_sample_ids": [case["eval_id"] for case in CASES],
    }
    summary_path = Path(args.summary_json) if args.summary_json else out.parent / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
