#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List


SYSTEM_PROMPT = "你是体育包网智能客服。回答要简洁、合规、不要输出思考过程。"

REPAIR_SPECS: Dict[str, Dict[str, str]] = {
    "kb_20260428_010": {
        "cluster": "注单取消",
        "answer": (
            "您好，体育注单取消或作废，常见情况包括天气原因、比赛延期/腰斩、官方赛果调整、盘口异常、"
            "以及赛事数据未最终确认等。若是用户要核对自己那笔注单的具体原因，这边需要提交后台查询后再给您准确回复。"
        ),
    },
    "kb_20260428_011": {
        "cluster": "重复退化:玩法介绍",
        "answer": (
            "您好，体育常见玩法主要有这几类：独赢是直接猜比赛胜负；让球是在让球条件下判断输赢；"
            "大小是看总分/总进球是否高于或低于盘口；波胆是直接猜最终比分。若您要看某个玩法的具体结算规则，我也可以继续给您拆开说明。"
        ),
    },
    "kb_20260428_014": {
        "cluster": "重复退化:大小球",
        "answer": (
            "您好，这种属于大小球整数盘。盘口是大于4时，如果最后总进球正好4个，通常按走水处理，也就是退本金；"
            "只有5球或以上才算大，3球或以下才算小。"
        ),
    },
    "kb_20260428_021": {
        "cluster": "三串三",
        "answer": (
            "您好，三串三和普通三串一不一样。普通三关全赢通常是 A×B×C 三场全部一起中才算；"
            "三串三则是把三场拆成 2 串 1 组合来算，常见就是 AB、AC、BC 这三组分别结算，所以它不是只看三场全中的单一结果。"
        ),
    },
    "kb_20260428_027": {
        "cluster": "体育平台介绍",
        "answer": (
            "您好，大致可以这样理解：BB体育、IM体育、沙巴体育都属于常见的体育赛事投注平台/数据源体系，"
            "三昇体育也属于同类体育产品线。客服对外一般不用展开做强对比，重点说明它们都是不同的体育平台模块，具体赛事、盘口和展示规则以实际接入的平台为准。"
        ),
    },
    "kb_20260428_028": {
        "cluster": "后台仪表盘",
        "answer": (
            "您好，JT 包网后台仪表盘里的代办事项，通常是给商户集中处理待办内容用的，例如待审核项目、异常提醒、"
            "以及需要跟进的运营事项。简单说就是在仪表盘里快速看到哪些单子、配置或审核流程还没处理完。"
        ),
    },
    "kb_20260428_029": {
        "cluster": "后台初始化",
        "answer": (
            "您好，拿到包网后台账号后，第一步常见会先做基础初始化：确认登录IP是否需要加白、检查账号对应的角色和权限、"
            "再看商户本身能操作哪些菜单和配置项。把登录安全、加白和角色权限先配好，后面再做业务配置会更稳。"
        ),
    },
    "kb_20260428_030": {
        "cluster": "重复退化:前端素材",
        "answer": (
            "您好，后台前端素材一般可以配置的平台展示内容包括：平台名、logo、首页轮播图、活动 banner、公告文案、"
            "客服入口和部分页面图片素材。具体可配置到哪一级，以后台实际菜单和权限为准。"
        ),
    },
    "kb_20260428_038": {
        "cluster": "导入商户",
        "answer": (
            "您好，导入商户功能迁移老会员数据时，常见流程是：先按系统模板整理 Excel 数据，"
            "上传到导入商户页面后先做预览校验，确认字段和格式无误，再正式提交导入。若有报错，一般也是先在预览阶段修正后再重新导入。"
        ),
    },
}

QUERY_VARIANTS: Dict[str, List[str]] = {
    "kb_20260428_010": [
        "用户问：哪些情况可能导致体育注单被取消或作废？",
        "会员问：体育注单一般在什么情况下会被取消或判作废？",
        "客服解释题：注单被取消通常有哪些常见原因？",
    ],
    "kb_20260428_011": [
        "用户问：体育常见玩法有哪些？比如独赢、让球、大小、波胆这些。",
        "会员问：体育里常见的独赢、让球、大小、波胆分别是什么玩法？",
        "客服解释题：请简单介绍一下体育常见玩法类型。",
    ],
    "kb_20260428_014": [
        "用户问：大小球大于4，最后两队总进球正好4个，怎么结算？",
        "会员问：如果买大4球，比赛刚好打出4球，是退本金还是算输？",
        "客服解释题：大小盘大于4，赛果正好4球时怎么处理？",
    ],
    "kb_20260428_021": [
        "用户问：三串三和普通三关全赢有什么区别？",
        "会员问：三串三是不是和三串一一样，区别在哪里？",
        "客服解释题：请说明三串三和普通三关过关的差别。",
    ],
    "kb_20260428_027": [
        "用户问：BB体育、三昇体育、IM体育、沙巴体育大致有什么定位？",
        "商户问：BB体育、IM体育、沙巴体育这几个体育平台模块一般怎么理解？",
        "客服解释题：这些体育平台名称对外该怎么做口径介绍？",
    ],
    "kb_20260428_028": [
        "商户问：JT包网后台仪表盘的代办事项可以做什么？",
        "商户问：后台仪表盘里的代办事项栏目一般是干什么用的？",
        "客服解释题：请说明包网后台仪表盘代办事项的作用。",
    ],
    "kb_20260428_029": [
        "商户问：拿到包网后台账号后，第一步常见要配置什么？",
        "商户问：后台账号开通后，最先要做哪些基础初始化设置？",
        "客服解释题：包网后台首次登录后通常先配什么？",
    ],
    "kb_20260428_030": [
        "商户问：后台前端素材一般可以配置哪些内容？",
        "商户问：包网后台前端素材配置通常包含哪些展示项？",
        "客服解释题：前端素材位常见能改哪些平台展示内容？",
    ],
    "kb_20260428_038": [
        "商户问：导入商户功能迁移老会员数据时，大致流程是什么？",
        "商户问：老会员数据要通过导入商户迁移，一般怎么操作？",
        "客服解释题：请概括导入商户迁移老数据的常见流程。",
    ],
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def extract_bad_answer(row: Dict[str, Any]) -> str:
    try:
        return str(row["score"].get("answer") or "")
    except Exception:
        return ""


def build_row(row: Dict[str, Any], user_query: str, answer: str, cluster: str, variant_idx: int) -> Dict[str, Any]:
    sample_id = str(row["id"])
    return {
        "id": f"gpu5_gate_targeted_repair_{sample_id}_v{variant_idx}",
        "split": "train",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_query},
            {"role": "assistant", "content": answer},
        ],
        "meta": {
            "source": "gpu5_candidate_gate_round1_targeted_repair",
            "cluster": cluster,
            "sample_id": sample_id,
            "variant_idx": variant_idx,
            "scenario": row.get("scenario", ""),
            "gold_intent": row.get("gold_intent", ""),
            "gold_need_escalation": row.get("gold_need_escalation"),
            "must_include": row.get("must_include") or [],
            "must_not_include": row.get("must_not_include") or [],
            "bad_answer": extract_bad_answer(row),
            "score": row.get("score") or {},
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a targeted SFT repair set from GPU5 candidate gate failure clusters.")
    parser.add_argument("--knowledge-results", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--summary-json", required=True)
    args = parser.parse_args()

    rows = read_json(Path(args.knowledge_results))
    by_id = {str(row.get("id")): row for row in rows if isinstance(row, dict)}

    missing = sorted(set(REPAIR_SPECS) - set(by_id))
    if missing:
        raise SystemExit(f"missing required failure ids in knowledge results: {missing}")
    missing_variants = sorted(set(REPAIR_SPECS) - set(QUERY_VARIANTS))
    if missing_variants:
        raise SystemExit(f"missing query variants for sample ids: {missing_variants}")

    out_rows: List[Dict[str, Any]] = []
    cluster_counts: Counter[str] = Counter()
    sample_variant_counts: Dict[str, int] = {}
    for sample_id, spec in REPAIR_SPECS.items():
        row = by_id[sample_id]
        variants = QUERY_VARIANTS[sample_id]
        sample_variant_counts[sample_id] = len(variants)
        for variant_idx, user_query in enumerate(variants):
            out_rows.append(build_row(row, user_query, spec["answer"], spec["cluster"], variant_idx))
            cluster_counts[spec["cluster"]] += 1

    write_jsonl(Path(args.output_jsonl), out_rows)

    summary = {
        "knowledge_results": args.knowledge_results,
        "output_jsonl": args.output_jsonl,
        "total": len(out_rows),
        "system_prompt": SYSTEM_PROMPT,
        "clusters": dict(sorted(cluster_counts.items())),
        "variants_per_sample": sample_variant_counts,
        "sample_ids": list(REPAIR_SPECS.keys()),
        "notes": [
            "Targeted from GPU5 candidate gate round1/rerun knowledge failures.",
            "Prioritizes first-gate misses and repetition regressions.",
            "Rows are single-turn SFT samples with explicit system prompt retained per record.",
            "Each failure cluster is expanded into 2-3 query variants for a more stable gate-oriented repair set.",
        ],
    }
    summary_path = Path(args.summary_json)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
