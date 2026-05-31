#!/usr/bin/env python3
"""Build a focused sports baowang knowledge regression set.

The set is for eval gates, not direct SFT deployment. It targets recurring weak
spots: odds, handicap, parlay, settlement, void rules, and backend knowledge.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


PROJECT_ROOT = Path("/home/ubuntu/qwen35a3b_finetune")
DEFAULT_OUT = PROJECT_ROOT / "datasets/eval_sports_baowang_knowledge_regression_20260506.jsonl"
MUST_NOT = ["包赢", "稳赚", "代理返佣", "拉新返利", "跑分", "洗钱", "伪造证件", "未成年"]


CASES: List[Dict[str, Any]] = [
    # 赔率/盘口
    {"scenario": "赔率格式", "user_query": "用户问：香港盘、欧洲盘、马来盘分别怎么理解，是否包含本金？", "must_include": ["香港盘", "欧洲盘", "本金"], "tags": ["odds"]},
    {"scenario": "赔率结算", "user_query": "用户问：下注时看到 1.92，确认成交变成 1.84，订单应该按哪个赔率结算？", "gold_intent": "赔率异常", "gold_need_escalation": False, "must_include": ["下注成功", "成交", "赔率"], "tags": ["odds"]},
    {"scenario": "盘口类型", "user_query": "用户问：让球盘、大小球、标准盘 1X2 的区别是什么？", "must_include": ["让球", "大小球", "1X2"], "tags": ["handicap"]},
    {"scenario": "盘口计算", "user_query": "用户问：让 0.25 球为什么会出现赢一半或输一半？", "must_include": ["0.25", "赢一半", "输一半"], "tags": ["handicap"]},
    {"scenario": "盘口计算", "user_query": "用户问：让 0.75 球的赢一半、输一半分别在什么比分情况下出现？", "must_include": ["0.75", "赢一半", "输一半"], "tags": ["handicap"]},
    {"scenario": "大小盘", "user_query": "用户问：大小球 2.5 和 2/2.5 的结算差别是什么？", "must_include": ["2.5", "2/2.5", "半赢"], "tags": ["over_under"]},
    {"scenario": "走水", "user_query": "用户问：盘口走水是什么意思，是否退本金？", "must_include": ["走水", "退本金"], "tags": ["settlement"]},
    {"scenario": "赔率异常", "user_query": "用户问：盘口跳动太快导致我没看清赔率，这种是按显示赔率还是成交赔率？", "gold_intent": "赔率异常", "gold_need_escalation": False, "must_include": ["成交赔率", "下注成功"], "tags": ["odds"]},
    # 串关
    {"scenario": "串关规则", "user_query": "用户问：串关和单关有什么区别？为什么串关要等全部场次结束？", "gold_intent": "串关规则", "gold_need_escalation": False, "must_include": ["串关", "全部场次", "结算"], "tags": ["parlay"]},
    {"scenario": "串关赔率", "user_query": "用户问：3 串 1 的总赔率是怎么来的？", "gold_intent": "串关规则", "gold_need_escalation": False, "must_include": ["3串1", "赔率", "相乘"], "tags": ["parlay"]},
    {"scenario": "串关未结算", "user_query": "用户问：串关里有一场还没出赛果，为什么整单没有派彩？", "gold_intent": "串关规则", "gold_need_escalation": False, "must_include": ["串关", "全部", "赛果"], "tags": ["parlay", "settlement"]},
    {"scenario": "串关作废", "user_query": "用户问：串关里一场比赛取消了，整张串关单怎么处理？", "gold_intent": "赛事变更", "gold_need_escalation": False, "must_include": ["取消", "该场", "规则"], "tags": ["parlay", "void"]},
    {"scenario": "串关限制", "user_query": "用户问：为什么有些玩法不能串关？", "gold_intent": "串关规则", "gold_need_escalation": False, "must_include": ["玩法", "串关", "限制"], "tags": ["parlay"]},
    {"scenario": "串关本金", "user_query": "用户问：串关中某一关走水后，本金和后续赔率怎么计算？", "gold_intent": "串关规则", "gold_need_escalation": False, "must_include": ["走水", "赔率", "本金"], "tags": ["parlay", "settlement"]},
    # 赛果结算
    {"scenario": "赛果结算", "user_query": "用户问：足球赛果结算一般按 90 分钟还是包含加时和点球？", "gold_intent": "其他", "gold_need_escalation": False, "must_include": ["90分钟", "加时", "点球"], "tags": ["settlement"]},
    {"scenario": "赛果结算", "user_query": "用户问：篮球让分盘是按全场最终比分还是某一节比分？", "gold_intent": "其他", "gold_need_escalation": False, "must_include": ["玩法", "全场", "节"], "tags": ["settlement"]},
    {"scenario": "赛果结算", "user_query": "用户问：官方赛果已经出了，平台为什么还显示待结算？", "gold_intent": "注单异常", "gold_need_escalation": False, "must_include": ["赛果确认", "系统结算"], "tags": ["settlement"]},
    {"scenario": "滚球延迟", "user_query": "用户问：滚球单赛后还没有派彩，常见原因有哪些？", "gold_intent": "滚球延迟", "gold_need_escalation": False, "must_include": ["滚球", "赛果", "结算"], "tags": ["settlement"]},
    {"scenario": "提前结算", "user_query": "用户问：提前结算金额为什么会随着比赛进程变化？", "gold_intent": "其他", "gold_need_escalation": False, "must_include": ["提前结算", "赔率", "比赛进程"], "tags": ["settlement"]},
    {"scenario": "派彩异常", "user_query": "用户问：注单显示赢了但派彩金额和我算的不一样，应该看哪些因素？", "gold_intent": "注单异常", "gold_need_escalation": True, "must_include": ["注单", "赔率", "后台"], "tags": ["settlement"]},
    # 作废规则
    {"scenario": "作废规则", "user_query": "用户问：比赛取消后，单关注单通常会怎么处理？", "gold_intent": "赛事变更", "gold_need_escalation": False, "must_include": ["比赛取消", "作废", "退回"], "tags": ["void"]},
    {"scenario": "作废规则", "user_query": "用户问：危险球 90 秒内进球，注单为什么可能作废？", "gold_intent": "注单异常", "gold_need_escalation": False, "must_include": ["危险球", "90秒", "作废"], "tags": ["void"]},
    {"scenario": "作废规则", "user_query": "用户问：作废原因写危险球，客服应该怎么解释？", "gold_intent": "注单异常", "gold_need_escalation": False, "must_include": ["危险球", "作废", "规则"], "tags": ["void"]},
    {"scenario": "作废规则", "user_query": "用户问：作废原因写天气原因，这种注单是不是平台随便取消？", "gold_intent": "赛事变更", "gold_need_escalation": False, "must_include": ["天气", "赛事", "规则"], "tags": ["void"]},
    {"scenario": "赛事变更", "user_query": "用户问：比赛腰斩和比赛延期对注单结算有什么不同？", "gold_intent": "赛事变更", "gold_need_escalation": False, "must_include": ["腰斩", "延期", "规则"], "tags": ["void", "settlement"]},
    {"scenario": "赛事变更", "user_query": "用户问：赛事改期后，原来的注单一定会退回吗？", "gold_intent": "赛事变更", "gold_need_escalation": False, "must_include": ["改期", "注单", "规则"], "tags": ["void"]},
    # 后台知识漏点
    {"scenario": "后台知识", "user_query": "用户问：包网后台里的商户、代理、会员三者是什么关系？", "must_include": ["商户", "代理", "会员"], "tags": ["backend"]},
    {"scenario": "后台知识", "user_query": "用户问：后台的代办事项一般包含哪些运营动作？", "must_include": ["充值", "提款", "审核"], "tags": ["backend"]},
    {"scenario": "后台知识", "user_query": "用户问：平台名、logo、轮播图这些配置一般属于后台哪个管理范围？", "must_include": ["平台", "logo", "轮播"], "tags": ["backend"]},
    {"scenario": "后台知识", "user_query": "用户问：角色权限和权限分配分别控制什么？", "must_include": ["角色", "权限", "分配"], "tags": ["backend"]},
    {"scenario": "后台知识", "user_query": "用户问：登录 IP 加白是什么意思，为什么后台要配置它？", "must_include": ["登录IP", "加白", "安全"], "tags": ["backend"]},
    {"scenario": "后台知识", "user_query": "用户问：会员导入和商户导入一般有什么区别？", "must_include": ["会员", "商户", "导入"], "tags": ["backend"]},
    {"scenario": "后台知识", "user_query": "用户问：后台的充值订单和提款订单分别应该看哪些字段？", "must_include": ["充值订单", "提款订单", "状态"], "tags": ["backend"]},
    {"scenario": "后台知识", "user_query": "用户问：代理佣金和会员输赢统计是什么关系？", "must_include": ["代理佣金", "会员", "输赢"], "tags": ["backend"]},
    {"scenario": "后台知识", "user_query": "用户问：补单额度、上分额度、提款额度分别限制什么？", "must_include": ["补单额度", "上分额度", "提款额度"], "tags": ["backend"]},
    {"scenario": "后台知识", "user_query": "用户问：登录过期秒数配置会影响什么？", "must_include": ["登录", "过期", "安全"], "tags": ["backend"]},
]


def now_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def build_rows() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for idx, case in enumerate(CASES, 1):
        tags = ["sports_baowang_regression_20260506"] + list(case.get("tags") or [])
        rows.append({
            "id": f"kb_reg_20260506_{idx:03d}",
            "scenario": case["scenario"],
            "user_query": case["user_query"],
            "gold_intent": case.get("gold_intent", "其他"),
            "gold_need_escalation": case.get("gold_need_escalation", False),
            "must_include": case["must_include"],
            "must_not_include": MUST_NOT,
            "tags": tags,
            "priority": "high" if case["scenario"] in {"作废规则", "后台知识"} else "normal",
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Build focused sports baowang knowledge regression JSONL.")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = build_rows()
    with out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {
        "ts": now_ts(),
        "out": str(out),
        "total": len(rows),
        "by_scenario": {},
    }
    for row in rows:
        summary["by_scenario"][row["scenario"]] = summary["by_scenario"].get(row["scenario"], 0) + 1
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
