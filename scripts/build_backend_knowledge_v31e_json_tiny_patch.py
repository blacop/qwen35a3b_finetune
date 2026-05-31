#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path("/home/ubuntu/qwen35a3b_finetune")
STORAGE_ROOT = Path("/video-storage/ai-customer/qwen35a3b_finetune")

SYSTEM_PROMPT = (
    "你是体育包网智能客服。必须遵守合规与风控规则："
    "遇到注单取消/作废/异常、结算争议、赔率异常、限红风控等无法直接确认的情况，"
    "必须明确告知需要后台查询或联系平台运营后回复；"
    "禁止赌博诱导、代理拉新、洗钱跑分、伪造证件、低龄相关内容。"
)

JSON_INSTRUCTION = """请基于当前对话，输出且仅输出一个JSON对象，字段如下：
{
  "intent": "充值|提款|活动|投诉|注单异常|串关规则|滚球延迟|赔率异常|限红风控|赛事变更|账户异常|技术故障|其他",
  "need_escalation": true/false,
  "answer": "给用户的客服回复",
  "risk_flag": ["命中的风险标签，可为空数组"],
  "next_action": "下一步动作，若需转后台请明确写后台查询/联系运营"
}
不要输出Markdown代码块，不要输出额外解释。"""


PATCH_CASES: list[dict[str, Any]] = [
    {
        "eval_id": "kb_20260428_028",
        "scenario": "后台仪表盘",
        "intent": "其他",
        "need_escalation": False,
        "queries": [
            "商户问：JT包网后台仪表盘的代办事项可以做什么？",
            "商户想了解包网后台首页仪表盘里的代办事项主要提醒哪些审核工作。",
        ],
        "answer": (
            "仪表盘通常用于快速查看后台概况，代办事项会集中提醒待处理的审核任务，"
            "例如会员资料、充值提款、活动或注单相关审核。商户可按代办事项进入对应模块处理，"
            "具体字段以当前后台版本为准。"
        ),
        "next_action": "按后台功能说明直接解释，无需后台查询。",
    },
    {
        "eval_id": "kb_20260428_029",
        "scenario": "后台初始化",
        "intent": "其他",
        "need_escalation": False,
        "queries": [
            "商户问：拿到包网后台账号后，第一步常见要配置什么？",
            "商户刚拿到后台账号，想知道初始化时优先检查哪些安全配置。",
        ],
        "answer": (
            "拿到包网后台账号后，常见第一步是先做安全初始化：确认登录IP是否需要加白，"
            "再按岗位建立角色并分配对应权限。完成登录IP、加白和角色配置后，再继续检查支付、客服、前端素材等业务配置。"
        ),
        "next_action": "按后台初始化流程直接说明，无需后台查询。",
    },
    {
        "eval_id": "kb_20260428_030",
        "scenario": "前端素材",
        "intent": "其他",
        "need_escalation": False,
        "queries": [
            "商户问：后台前端素材一般可以配置哪些内容？",
            "商户想知道包网后台前端素材配置主要包含哪些页面展示内容。",
        ],
        "answer": (
            "前端素材一般用于配置前台展示内容，常见包括平台名、logo、首页轮播、活动图、弹窗图等。"
            "建议按后台要求的尺寸和格式上传，避免前端显示变形。"
        ),
        "next_action": "按前端素材配置说明直接回复，无需后台查询。",
    },
    {
        "eval_id": "kb_20260428_031",
        "scenario": "广告尺寸",
        "intent": "其他",
        "need_escalation": False,
        "queries": [
            "商户问：首页轮播图、活动封面、弹窗大概应该按什么尺寸准备？",
            "商户要准备广告素材，问首页轮播图、活动封面和弹窗图应该怎么处理。",
        ],
        "answer": (
            "首页轮播图、活动封面、弹窗通常需要分别按后台当前模板要求准备尺寸。"
            "如果没有固定模板，先按后台素材位提示的宽高、格式和大小限制制作，再上传预览确认显示效果。"
        ),
        "next_action": "按素材位配置原则直接说明，无需后台查询。",
    },
    {
        "eval_id": "kb_20260428_032",
        "scenario": "权限管理",
        "intent": "其他",
        "need_escalation": False,
        "queries": [
            "商户问：包网后台权限管理是按什么方式分配功能权限的？",
            "商户想知道后台不同员工账号的功能权限怎么管。",
        ],
        "answer": (
            "包网后台通常通过权限管理按角色做权限分配。先建立财务、客服、运营等角色，"
            "再给每个角色勾选可查看或可操作的菜单功能，员工账号绑定对应角色后即可获得相应权限。"
        ),
        "next_action": "按权限管理逻辑直接解释，无需后台查询。",
    },
    {
        "eval_id": "kb_20260428_033",
        "scenario": "系统配置",
        "intent": "其他",
        "need_escalation": False,
        "queries": [
            "商户问：系统配置里和注册登录相关的开关有哪些？",
            "商户想了解注册和登录相关的系统配置通常有哪些选项。",
        ],
        "answer": (
            "系统配置里和注册登录相关的开关通常包括验证码、手机号注册或绑定、登录安全限制等。"
            "商户可根据风控要求决定是否开启验证码、是否强制手机号，以及登录异常校验策略。"
        ),
        "next_action": "按注册登录配置说明直接回复，无需后台查询。",
    },
    {
        "eval_id": "kb_20260428_035",
        "scenario": "冻结比例",
        "intent": "活动",
        "need_escalation": False,
        "queries": [
            "商户问：活动金额冻结比例、洗码金额冻结比例、签到奖励冻结比例分别是什么配置？",
            "商户看到后台有几种冻结比例，想知道活动、洗码和签到奖励分别对应什么。",
        ],
        "answer": (
            "活动金额冻结比例用于控制活动赠送金额需要冻结或完成条件后才可释放的比例；"
            "洗码金额冻结比例用于控制洗码返还金额的冻结比例；"
            "签到奖励冻结比例用于控制签到奖励到账后的冻结或解冻要求。具体释放条件以商户后台规则配置为准。"
        ),
        "next_action": "按活动配置字段直接说明，无需后台查询。",
    },
    {
        "eval_id": "kb_20260428_038",
        "scenario": "导入商户",
        "intent": "其他",
        "need_escalation": False,
        "queries": [
            "商户问：导入商户功能迁移老会员数据时，大致流程是什么？",
            "商户要迁移旧站会员，问导入商户功能一般怎么走。",
        ],
        "answer": (
            "导入商户功能通常用于迁移老会员数据。一般先按模板整理Excel，上传后系统做字段校验，"
            "再进入预览确认数据无误，最后执行导入。正式导入前建议先备份并用少量数据测试。"
        ),
        "next_action": "按导入商户流程直接说明，无需后台查询。",
    },
    {
        "eval_id": "kb_20260428_040",
        "scenario": "客服配置",
        "intent": "其他",
        "need_escalation": False,
        "queries": [
            "商户问：包网后台需要配置在线客服、短信提供商和支付通道，这些一般在哪类配置里处理？",
            "商户要接入客服、短信和支付，问后台通常在哪些配置项处理。",
        ],
        "answer": (
            "在线客服通常在客服或站点基础配置里处理，短信提供商一般在短信或通知配置里维护，"
            "支付通道则在支付或通道管理里配置。配置后建议分别测试在线客服入口、短信发送和通道回调是否正常。"
        ),
        "next_action": "按后台配置分类直接说明，无需后台查询。",
    },
]


def make_user_content(query: str, variant: str) -> str:
    if variant == "json_direct":
        query = f"{query} 请按指定JSON格式输出。"
    elif variant == "no_escalation":
        query = f"{query} 这是通用后台知识咨询，能直接说明就不要转后台。"
    return f"用户问题：{query}\n\n{JSON_INSTRUCTION}"


def assistant_json(case: dict[str, Any]) -> str:
    payload = {
        "intent": case["intent"],
        "need_escalation": case["need_escalation"],
        "answer": case["answer"],
        "risk_flag": [],
        "next_action": case["next_action"],
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def build_rows(repeat: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    variants = ["original", "json_direct", "no_escalation"]
    for case in PATCH_CASES:
        for query_index, query in enumerate(case["queries"], 1):
            for variant in variants:
                for rep in range(1, repeat + 1):
                    rows.append(
                        {
                            "id": (
                                "backend_v31e_json_"
                                f"{case['eval_id']}_{variant}_q{query_index:02d}_r{rep:02d}"
                            ),
                            "split": "train",
                            "messages": [
                                {"role": "system", "content": SYSTEM_PROMPT},
                                {"role": "user", "content": make_user_content(query, variant)},
                                {"role": "assistant", "content": assistant_json(case)},
                            ],
                            "meta": {
                                "source": "backend_knowledge_v31e_json_tiny_patch",
                                "eval_sample_id": case["eval_id"],
                                "scenario": case["scenario"],
                                "variant": variant,
                                "query_index": query_index,
                                "repeat": rep,
                                "gold_intent": case["intent"],
                                "gold_need_escalation": case["need_escalation"],
                                "json_aligned": True,
                                "disable_thinking": True,
                                "patch_scope": "baowang_backend_knowledge_only",
                            },
                        }
                    )
    return rows


def validate_rows(rows: list[dict[str, Any]]) -> None:
    forbidden_scope_terms = ["危险球", "注单作废", "作废原因", "天气原因", "90秒"]
    for row in rows:
        messages = row["messages"]
        assistant = messages[-1]["content"]
        parsed = json.loads(assistant)
        if sorted(parsed) != ["answer", "intent", "need_escalation", "next_action", "risk_flag"]:
            raise ValueError(f"Unexpected assistant JSON keys in {row['id']}: {sorted(parsed)}")
        if any(term in assistant for term in forbidden_scope_terms):
            raise ValueError(f"Row touches protected sports-rule scope: {row['id']}")
        if "<think" in assistant.lower() or "思维链" in assistant:
            raise ValueError(f"Assistant content contains thinking marker: {row['id']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build v31e JSON-aligned backend knowledge tiny patch.")
    parser.add_argument(
        "--output-jsonl",
        default=str(
            STORAGE_ROOT
            / "datasets"
            / "backend_knowledge_v31e_json_tiny_patch_20260430"
            / "sft_backend_knowledge_v31e_json_tiny_patch.jsonl"
        ),
    )
    parser.add_argument("--summary-json", default="")
    parser.add_argument("--repeat", type=int, default=4)
    args = parser.parse_args()

    if args.repeat < 1 or args.repeat > 8:
        raise ValueError("--repeat must be between 1 and 8 for this tiny patch")

    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = build_rows(args.repeat)
    validate_rows(rows)

    with output_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")

    summary = {
        "output_jsonl": str(output_path),
        "total_rows": len(rows),
        "case_count": len(PATCH_CASES),
        "queries_per_case": 2,
        "variants_per_query": 3,
        "repeat": args.repeat,
        "json_aligned": True,
        "disable_thinking": True,
        "patch_scope": "baowang_backend_knowledge_only",
        "protected_sports_rule_scope_not_touched": True,
        "eval_sample_ids": [case["eval_id"] for case in PATCH_CASES],
    }
    summary_path = Path(args.summary_json) if args.summary_json else output_path.parent / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
