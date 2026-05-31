#!/usr/bin/env python3
import argparse
import json
import random
from collections import Counter
from pathlib import Path


COMMON_MUST_NOT = [
    "包赢",
    "稳赚",
    "代理返佣",
    "拉新返利",
    "跑分",
    "洗钱",
    "伪造证件",
    "未成年",
]

AMOUNTS = [50, 88, 100, 168, 200, 300, 500, 800, 1000, 2000, 3000, 5000, 8000, 10000]
MINUTES = [3, 5, 8, 10, 15, 20, 30, 45, 60]
BANKS = ["工商银行", "建设银行", "农业银行", "中国银行", "招商银行", "交通银行"]
MATCHES = [
    "英超 曼联 vs 阿森纳",
    "西甲 皇马 vs 巴萨",
    "意甲 国米 vs AC米兰",
    "德甲 拜仁 vs 多特",
    "欧冠 巴黎 vs 曼城",
    "NBA 湖人 vs 勇士",
    "CBA 辽宁 vs 广东",
]
ORDER_PREFIX = ["A", "B", "C", "T", "OD", "ZX"]


def rand_bet_id(rng: random.Random) -> str:
    return f"{rng.choice(ORDER_PREFIX)}{rng.randint(100000, 999999)}"


def pick(pool, rng: random.Random):
    return rng.choice(pool)


def render(template: str, rng: random.Random) -> str:
    odds_old = rng.choice([1.95, 2.05, 1.88, 2.15, 1.76, 2.30])
    odds_new = round(max(1.10, odds_old - rng.choice([0.08, 0.12, 0.20, -0.10, -0.18])), 2)
    data = {
        "amount": pick(AMOUNTS, rng),
        "minutes": pick(MINUTES, rng),
        "bank": pick(BANKS, rng),
        "bet_id": rand_bet_id(rng),
        "match": pick(MATCHES, rng),
        "odds_old": odds_old,
        "odds_new": odds_new,
        "limit_old": pick([2000, 3000, 5000, 8000], rng),
        "limit_new": pick([200, 300, 500, 800, 1000], rng),
    }
    s = template.format(**data)
    prefix = rng.choice(
        [
            "",
            "你好，",
            "客服你好，",
            "麻烦问下，",
            "请问，",
            "我想确认一下，",
            "帮我看下，",
        ]
    )
    tone_tail = rng.choice(["", "，麻烦尽快处理。", "，请帮我查一下。", "，这也太慢了吧？", "，谢谢。"])
    suffix2 = rng.choice(["", " 账号这边很着急。", " 我这边等很久了。", " 请尽快回复。"])
    return prefix + s + tone_tail + suffix2


INTENT_CONFIG = {
    "充值": {
        "quota": 55,
        "scenarios": [
            {
                "name": "充值到账延迟",
                "need_escalation": True,
                "must_include": ["到账时间", "订单号", "后台查询"],
                "templates": [
                    "我充值{amount}已经{minutes}分钟了还没到账",
                    "入款{amount}一直没上分，订单是不是卡住了",
                    "充值到{bank}后显示成功，但余额没变",
                ],
            },
            {
                "name": "充值失败",
                "need_escalation": True,
                "must_include": ["失败原因", "截图", "后台查询"],
                "templates": [
                    "我这笔充值{amount}直接失败了，是什么问题",
                    "支付通道提示成功但平台报失败，怎么处理",
                ],
            },
        ],
    },
    "提款": {
        "quota": 55,
        "scenarios": [
            {
                "name": "提款审核中",
                "need_escalation": True,
                "must_include": ["审核", "到账时间", "后台查询"],
                "templates": [
                    "提款显示审核中已经{minutes}分钟了，什么时候到账",
                    "我提了{amount}一直在审核，麻烦看下进度",
                ],
            },
            {
                "name": "提款被驳回",
                "need_escalation": True,
                "must_include": ["驳回原因", "风控", "后台查询"],
                "templates": [
                    "我的提款被驳回了，原因是什么",
                    "连续两次提款失败，麻烦给个明确原因",
                ],
            },
        ],
    },
    "活动": {
        "quota": 35,
        "scenarios": [
            {
                "name": "活动规则咨询",
                "need_escalation": False,
                "must_include": ["活动规则", "流水", "时效"],
                "templates": [
                    "首充活动现在还有吗，流水要求是多少",
                    "这个优惠的打码要求怎么计算",
                    "活动奖励多久到账，有没有截止时间",
                ],
            }
        ],
    },
    "投诉": {
        "quota": 35,
        "scenarios": [
            {
                "name": "服务投诉",
                "need_escalation": True,
                "must_include": ["抱歉", "记录反馈", "升级处理"],
                "templates": [
                    "你们处理太慢了，我要投诉",
                    "客服一直复制粘贴，我要升级投诉",
                    "这个问题反映很多次都没人解决",
                ],
            }
        ],
    },
    "注单异常": {
        "quota": 95,
        "scenarios": [
            {
                "name": "注单取消作废",
                "need_escalation": True,
                "must_include": ["后台查询", "注单号", "运营"],
                "templates": [
                    "注单{bet_id}为什么被取消作废了",
                    "我这张单{bet_id}无效了，给个原因",
                    "注单突然作废，这正常吗",
                ],
            },
            {
                "name": "注单状态异常",
                "need_escalation": True,
                "must_include": ["后台查询", "结算状态", "注单号"],
                "templates": [
                    "注单{bet_id}一直待结算，麻烦查下",
                    "我的订单状态卡住了，查不到结果",
                    "这张注单显示异常，请帮我核实",
                ],
            },
        ],
    },
    "串关规则": {
        "quota": 55,
        "scenarios": [
            {
                "name": "串关延期规则",
                "need_escalation": False,
                "must_include": ["串关规则", "延期", "以平台规则为准"],
                "templates": [
                    "串关里有一场延期，整单怎么结算",
                    "过关单碰到比赛改期，是按1.0还是作废",
                    "串关一场取消后，剩下场次怎么算",
                ],
            }
        ],
    },
    "滚球延迟": {
        "quota": 50,
        "scenarios": [
            {
                "name": "滚球延迟结算",
                "need_escalation": True,
                "must_include": ["延迟结算", "后台核实", "耐心等待"],
                "templates": [
                    "滚球赛后{minutes}分钟还没结算，什么情况",
                    "赛中单迟迟不派彩，麻烦查一下",
                    "滚球结算太慢了，是否系统异常",
                ],
            }
        ],
    },
    "赔率异常": {
        "quota": 45,
        "scenarios": [
            {
                "name": "赔率跳变争议",
                "need_escalation": True,
                "must_include": ["赔率变动", "下单时间", "后台核查"],
                "templates": [
                    "下单前是{odds_old}，成交变成{odds_new}，这怎么算",
                    "我点确认时赔率变了，注单按哪个赔率",
                    "盘口跳动太快导致赔率不同，能复核吗",
                ],
            }
        ],
    },
    "限红风控": {
        "quota": 45,
        "scenarios": [
            {
                "name": "限额下调",
                "need_escalation": True,
                "must_include": ["风控", "限额", "后台审核"],
                "templates": [
                    "我之前能下{limit_old}，现在只能下{limit_new}，为什么",
                    "系统提示限红，麻烦解释下规则",
                    "突然限制下注额度，是账号异常吗",
                ],
            }
        ],
    },
    "赛事变更": {
        "quota": 30,
        "scenarios": [
            {
                "name": "赛事改期取消",
                "need_escalation": False,
                "must_include": ["赛事变更", "结算规则", "以公告为准"],
                "templates": [
                    "{match}临时改期了，我的注单怎么办",
                    "比赛取消后是退单还是按无效处理",
                    "官方改赛程，之前的单会怎么结算",
                ],
            }
        ],
    },
}


def scaled_quotas(target_count: int) -> dict:
    base_total = sum(v["quota"] for v in INTENT_CONFIG.values())
    if target_count == base_total:
        return {k: v["quota"] for k, v in INTENT_CONFIG.items()}
    raw = {k: (v["quota"] / base_total) * target_count for k, v in INTENT_CONFIG.items()}
    floored = {k: int(raw[k]) for k in raw}
    remain = target_count - sum(floored.values())
    frac_order = sorted(raw.keys(), key=lambda x: raw[x] - floored[x], reverse=True)
    i = 0
    while remain > 0:
        floored[frac_order[i % len(frac_order)]] += 1
        remain -= 1
        i += 1
    return floored


def build_sample(intent: str, cfg: dict, idx: int, rng: random.Random) -> dict:
    scenario = rng.choice(cfg["scenarios"])
    query = render(rng.choice(scenario["templates"]), rng)
    need_escalation = bool(scenario["need_escalation"])
    sample = {
        "id": f"prod_eval_{idx:05d}",
        "scenario": scenario["name"],
        "user_query": query,
        "gold_intent": intent,
        "gold_need_escalation": need_escalation,
        "must_include": scenario["must_include"],
        "must_not_include": COMMON_MUST_NOT,
        "reference_answer": (
            "请按平台标准流程回复；涉及注单异常/风控/赔率争议时需明确后台核查与后续反馈。"
        ),
        "tags": ["sports_cs", "production_eval"],
        "priority": rng.choice(["normal", "high"]),
        "meta": {
            "channel": rng.choice(["web", "app", "telegram"]),
            "language": "zh-CN",
            "version": "v1",
        },
    }
    return sample


def main():
    parser = argparse.ArgumentParser(description="Generate production eval JSONL for sports customer-service model.")
    parser.add_argument(
        "--output-jsonl",
        default="/home/ubuntu/qwen35a3b_finetune/datasets/eval_sports_customer_prod_500.jsonl",
        help="output jsonl path",
    )
    parser.add_argument("--count", type=int, default=500, help="number of samples")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--overwrite", action="store_true", help="overwrite output file if exists")
    args = parser.parse_args()

    out_path = Path(args.output_jsonl).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and not args.overwrite:
        raise SystemExit(f"Output exists: {out_path}. Use --overwrite.")

    rng = random.Random(args.seed)
    quotas = scaled_quotas(args.count)

    rows = []
    idx = 1
    seen = set()
    for intent, cfg in INTENT_CONFIG.items():
        q = quotas[intent]
        attempts = 0
        while q > 0:
            attempts += 1
            if attempts > 30000:
                raise RuntimeError(f"Too many attempts while generating intent={intent}")
            row = build_sample(intent, cfg, idx, rng)
            key = (row["scenario"], row["user_query"])
            if key in seen:
                continue
            seen.add(key)
            rows.append(row)
            idx += 1
            q -= 1

    rng.shuffle(rows)
    for i, r in enumerate(rows, 1):
        r["id"] = f"prod_eval_{i:05d}"

    with out_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    counts = Counter([r["gold_intent"] for r in rows])
    summary = {
        "path": str(out_path),
        "count": len(rows),
        "intent_distribution": dict(sorted(counts.items(), key=lambda x: x[0])),
    }
    summary_path = out_path.with_suffix(".summary.json")
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(f"[OK] generated: {out_path}")
    print(f"[OK] summary:   {summary_path}")
    print(f"[OK] count:     {len(rows)}")
    for k, v in sorted(counts.items()):
        print(f"  - {k}: {v}")


if __name__ == "__main__":
    main()
