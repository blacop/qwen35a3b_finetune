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
        "query": "商户问：JT包网后台仪表盘的代办事项可以做什么？",
        "answer": "仪表盘用于查看后台概况；代办事项主要提醒待处理的审核，包括充值提款审核、会员资料审核、活动或注单相关审核。商户可从仪表盘进入对应审核页面处理。",
    },
    {
        "eval_id": "kb_20260428_029",
        "scenario": "后台初始化",
        "intent": "其他",
        "need_escalation": False,
        "query": "商户问：拿到包网后台账号后，第一步常见要配置什么？",
        "answer": "拿到包网后台账号后，第一步通常先做安全初始化：配置登录IP、将办公IP加白，并按岗位建立角色。完成登录IP、加白、角色后，再继续配置支付、客服和前端素材。",
    },
    {
        "eval_id": "kb_20260428_030",
        "scenario": "前端素材",
        "intent": "其他",
        "need_escalation": False,
        "query": "商户问：后台前端素材一般可以配置哪些内容？",
        "answer": "后台前端素材通常包括平台名、logo、首页轮播、活动图片、弹窗图片和公告素材。上传前要按后台提示的尺寸、格式和大小限制准备，避免前端展示变形。",
    },
    {
        "eval_id": "kb_20260428_031",
        "scenario": "广告尺寸",
        "intent": "其他",
        "need_escalation": False,
        "query": "商户问：首页轮播图、活动封面、弹窗大概应该按什么尺寸准备？",
        "answer": "首页轮播图、活动封面、弹窗要按后台素材位要求准备。不同模板尺寸可能不同，建议先查看对应位置的宽高和格式限制，上传后预览确认显示效果。",
    },
    {
        "eval_id": "kb_20260428_032",
        "scenario": "权限管理",
        "intent": "其他",
        "need_escalation": False,
        "query": "商户问：包网后台权限管理是按什么方式分配功能权限的？",
        "answer": "包网后台一般通过权限管理按角色做权限分配。先建立财务、客服、运营等角色，再给角色勾选菜单和操作权限，员工账号绑定角色后获得对应功能权限。",
    },
    {
        "eval_id": "kb_20260428_033",
        "scenario": "系统配置",
        "intent": "其他",
        "need_escalation": False,
        "query": "商户问：系统配置里和注册登录相关的开关有哪些？",
        "answer": "系统配置里和注册登录相关的开关常见有验证码、手机号注册或绑定、登录安全限制。商户可按风控要求决定是否开启验证码、是否强制手机号，以及登录异常校验策略。",
    },
    {
        "eval_id": "kb_20260428_035",
        "scenario": "冻结比例",
        "intent": "活动",
        "need_escalation": False,
        "query": "商户问：活动金额冻结比例、洗码金额冻结比例、签到奖励冻结比例分别是什么配置？",
        "answer": "活动金额冻结比例用于控制活动赠送金额的冻结比例；洗码金额冻结比例用于控制洗码返还金额的冻结比例；签到奖励冻结比例用于控制签到奖励到账后的冻结比例。实际解冻条件以后台活动规则为准。",
    },
    {
        "eval_id": "kb_20260428_038",
        "scenario": "导入商户",
        "intent": "其他",
        "need_escalation": False,
        "query": "商户问：导入商户功能迁移老会员数据时，大致流程是什么？",
        "answer": "导入商户用于迁移老会员数据。常见流程是先按模板整理Excel，再上传到导入商户功能中校验，随后进入预览确认字段和数据无误，最后执行正式导入。",
    },
    {
        "eval_id": "kb_20260428_040",
        "scenario": "客服配置",
        "intent": "其他",
        "need_escalation": False,
        "query": "商户问：包网后台需要配置在线客服、短信提供商和支付通道，这些一般在哪类配置里处理？",
        "answer": "在线客服通常在客服或站点配置里处理，短信提供商在短信或通知配置里维护，支付通道在支付或通道管理里配置。配置后要分别测试客服入口、短信发送和通道回调。",
    },
]


VARIANT_SUFFIXES = [
    "",
    " 请按标准JSON格式输出。",
    " 这是包网后台通用知识，能直接说明就不要转后台。",
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
        for variant_index, suffix in enumerate(VARIANT_SUFFIXES, 1):
            for rep in range(1, repeat + 1):
                rows.append(
                    {
                        "id": f"backend_v31g_json_{case['eval_id']}_v{variant_index:02d}_r{rep:02d}",
                        "split": "train",
                        "messages": [
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": f"用户问题：{case['query']}{suffix}\n\n{JSON_INSTRUCTION}"},
                            {"role": "assistant", "content": assistant_json(case)},
                        ],
                        "meta": {
                            "source": "backend_knowledge_v31g_json_tiny_patch",
                            "eval_sample_id": case["eval_id"],
                            "scenario": case["scenario"],
                            "repeat": rep,
                            "variant_index": variant_index,
                            "json_aligned": True,
                            "disable_thinking": True,
                            "patch_scope": "baowang_backend_knowledge_only",
                        },
                    }
                )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Build v31g short JSON-aligned backend knowledge tiny patch.")
    parser.add_argument(
        "--output-jsonl",
        default=str(
            STORAGE_ROOT
            / "datasets"
            / "backend_knowledge_v31g_json_tiny_patch_20260430"
            / "sft_backend_knowledge_v31g_json_tiny_patch.jsonl"
        ),
    )
    parser.add_argument("--summary-json", default="")
    parser.add_argument("--repeat", type=int, default=6)
    args = parser.parse_args()
    if args.repeat < 1 or args.repeat > 10:
        raise ValueError("--repeat must be between 1 and 10")
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
        "variants_per_case": len(VARIANT_SUFFIXES),
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
