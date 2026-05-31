#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List


PROJECT_ROOT = Path("/home/ubuntu/qwen35a3b_finetune")
DEFAULT_INPUT = PROJECT_ROOT / "datasets" / "tydata_domain_multimodal_pack" / "multimodal_train.swift_vl.jsonl"
DEFAULT_OUTPUT = PROJECT_ROOT / "datasets" / "relation_patch_pack" / "vl_definition_misconception_patch_v1.swift_vl.jsonl"
DEFAULT_SUMMARY = PROJECT_ROOT / "datasets" / "relation_patch_pack" / "vl_definition_misconception_patch_v1.summary.json"

TOPIC_FALLBACK = {
    "资金审核": "页面能看到补单、订单、支付分组、充值审核这类字段。",
    "代理管理": "页面能看到会员ID、上级代理、VIP、报表或盈亏相关字段。",
    "独赢和1X2": "页面能看到盘口、赔率、独赢或 1X2 玩法结构。",
    "串关规则": "页面能看到综合过关、多个玩法或组合投注说明。",
    "危险球": "页面能看到危险球、待确认、取消或点球角球说明。",
}


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_text(text: str) -> str:
    return " ".join(str(text or "").replace("\r", " ").replace("\n", " ").split())


def clean_text(text: str) -> str:
    value = normalize_text(text)
    while "。。" in value:
        value = value.replace("。。", "。")
    return value.strip(" 。")


def compact(text: str, limit: int) -> str:
    text = clean_text(text).strip("，,；;")
    if not text:
        return ""
    if len(text) <= limit:
        return text
    return text[:limit].rstrip("，,；;。") + "…"


def low_signal_visible(text: str) -> bool:
    value = clean_text(text)
    if not value:
        return True
    cjk = len(re.findall(r"[\u4e00-\u9fff]", value))
    latin = len(re.findall(r"[A-Za-z]", value))
    if cjk < 6 and latin > cjk * 2:
        return True
    if "watio" in value.lower():
        return True
    return False


def extract_visible_info(response: str, topic: str) -> str:
    text = str(response or "")
    for pattern in [
        r"例如：(.+?)。图中没有直接写出",
        r"例如：(.+?)。这类",
        r"例如：(.+?)。图里",
        r"关键点：(.+?)。如落到具体",
        r"关键点：(.+?)。$",
    ]:
        match = re.search(pattern, text)
        if match:
            candidate = compact(match.group(1), 90)
            if not low_signal_visible(candidate):
                return candidate
    return TOPIC_FALLBACK.get(topic, "页面主要体现后台字段和业务场景。")


def infer_key(row: Dict[str, Any]) -> str | None:
    meta = row.get("meta") or {}
    term = str(meta.get("domain_term", "")).strip()
    topic = str(meta.get("topic", "")).strip()
    query = str(row.get("query", ""))
    blob = f"{term} {topic} {query}"
    if term == "三方":
        return "third_party"
    if term == "总代":
        return "master_agent"
    if term == "负盈利":
        return "negative_profit"
    if term == "体育包网":
        if "串关" in blob or "综合过关" in blob:
            return "parlay"
        if "危险球" in blob or "待确认" in blob or "取消" in blob:
            return "danger_ball"
        return "sports_package"
    if "串关" in blob or "综合过关" in blob:
        return "parlay"
    if "危险球" in blob or "待确认" in blob or "取消" in blob:
        return "danger_ball"
    return None


def should_use(row: Dict[str, Any], key: str) -> bool:
    meta = row.get("meta") or {}
    qtype = meta.get("domain_qtype")
    if key in {"third_party", "master_agent", "negative_profit"}:
        return qtype in {"definition", "misconception", "comparison"}
    if key == "sports_package":
        return qtype in {"definition", "misconception", "comparison"} and str(meta.get("topic", "")) == "独赢和1X2"
    if key in {"parlay", "danger_ball"}:
        topic = str(meta.get("topic", "")).strip()
        return topic in {"串关规则", "危险球"}
    return False


def dedupe_key(row: Dict[str, Any], key: str) -> str:
    meta = row.get("meta") or {}
    return "|".join(
        [
            key,
            str(meta.get("topic", "")),
            str(meta.get("domain_term", "")),
            "|".join(row.get("images") or []),
        ]
    )


def select_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    selected: List[Dict[str, Any]] = []
    seen = set()
    for row in rows:
        key = infer_key(row)
        if not key or not should_use(row, key):
            continue
        token = dedupe_key(row, key)
        if token in seen:
            continue
        seen.add(token)
        selected.append(row)
    return selected


def def_queries(key: str, topic: str, idx: int) -> List[str]:
    bank = {
        "third_party": [
            "三方是什么意思？这张补单页和三方是什么关系？先说页面可见信息，再解释业务含义。",
            "这页里的“三方名称”更接近第三方支付、第三方通道，还是平台自研模块？请分开说页面证据和业务解释。",
        ],
        "master_agent": [
            "总代在这类代理后台里通常指什么？请先说页面可见信息，再解释总代角色。",
            "这张代理页里，总代这个词应该怎么给用户解释才不会和平台混淆？",
        ],
        "negative_profit": [
            "负盈利在这类代理后台里通常指什么？先说页面可见信息，再解释业务口径。",
            "这张代理页里，负盈利这个词该怎么解释才不会和平台亏损混淆？",
        ],
        "sports_package": [
            "这张图能体现体育包网是什么吗？请先说页面可见信息，再解释体育包网能力。",
            "结合这张体育玩法图，体育包网通常指什么能力集合？不要只讲一个玩法。",
        ],
        "parlay": [
            "串关是什么意思？这张图里的综合过关和串关是什么关系？先说页面可见信息，再解释。",
            "这页规则图里，综合过关为什么可以按串关来理解？请分开说页面证据和业务解释。",
        ],
        "danger_ball": [
            "危险球是什么意思？为什么会待确认或被取消？先说页面可见信息，再解释。",
            "这页规则图里，危险球为什么会影响确认和取消？请分开说页面证据和业务解释。",
        ],
    }
    values = bank[key]
    return [values[idx % len(values)]]


def mis_queries(key: str, idx: int) -> List[str]:
    bank = {
        "third_party": [
            "从这类补单页面看，三方是不是平台自研系统？先说页面可见信息，再纠正误解。",
            "这页出现“三方名称”时，能不能直接理解成平台自己的风控模块？请先点页面证据，再纠正。",
        ],
        "master_agent": [
            "总代是不是平台？结合这张代理相关页面说明，先说页面可见信息，再纠正误解。",
            "有人把总代直接说成平台方，这个判断对吗？请先点页面证据，再纠正。",
        ],
        "negative_profit": [
            "负盈利是玩家亏损还是平台亏损？结合这张代理后台图说明，先说页面可见信息，再纠正误解。",
            "有人把负盈利理解成平台自己亏钱，这个说法对吗？请先点页面证据，再纠正。",
        ],
        "sports_package": [
            "从这张体育玩法图看，体育包网是不是只有足球玩法？先说页面可见信息，再纠正误解。",
            "有人把体育包网理解成只给一页赔率说明，这个判断对吗？请先点页面证据，再纠正。",
        ],
        "parlay": [
            "综合过关是不是和串关完全不同的另一个玩法？请先说页面可见信息，再纠正误解。",
            "有人把综合过关当成单场玩法，这个说法对吗？请先点页面证据，再纠正。",
        ],
        "danger_ball": [
            "危险球是不是一种单独玩法？为什么会待确认或被取消？请先说页面可见信息，再纠正误解。",
            "有人把危险球理解成注单一定无效，这个说法对吗？请先点页面证据，再纠正。",
        ],
    }
    values = bank[key]
    return [values[idx % len(values)]]


def biz_explanation(key: str, topic: str, idx: int) -> str:
    topic_hint = compact(topic, 12)
    bank = {
        "third_party": [
            "这类补单、充值审核和通道字段更接近外部支付或游戏服务接入场景，三方通常指第三方支付/通道或第三方内容提供方，不是平台自研模块。",
            "页面如果围绕订单、补单、支付分组展开，三方应优先按外部服务方理解，而不是按平台内部功能理解。",
        ],
        "master_agent": [
            "总代属于代理层级里的上级角色，可以管理下级代理并对接平台，但本身不等于平台系统提供方。",
            "这类代理后台更能说明层级关系和渠道角色，总代讲的是代理身份，不是平台技术主体。",
        ],
        "negative_profit": [
            "负盈利通常是代理分润和结算口径，常见含义是会员亏损、平台盈利后，代理按这部分亏损参与分润，不是说平台自己亏钱。",
            "这类代理报表页里，负盈利更接近会员输掉金额对应的代理分润口径，不是通用财务亏损概念。",
        ],
        "sports_package": [
            "体育包网不是单个玩法页，而是一整套体育博彩能力，包括赛事数据、赔率盘口、滚球、串关、结算引擎和后台管理等。",
            "这类玩法图只能说明体育能力的一部分，但能作为锚点去解释：体育包网交付的是完整体育系统能力，不是一张页面。",
        ],
        "parlay": [
            "综合过关通常就是串关在页面里的展示名称，本质是把多场或多个选项组合成一张注单，通常需要全赢才算命中，赔率会连乘。",
            "这类规则页强调的是组合投注逻辑，综合过关不是另一套独立体系，而是串关玩法的页面表达。",
        ],
        "danger_ball": [
            "危险球不是独立玩法，而是滚球结算里的高风险事件状态，常见于点球、角球、自由球或临门进攻，因此注单可能先待确认再决定是否取消。",
            "这类规则页说明的是赛事出现敏感进攻事件时的确认逻辑，待确认或取消是为避免错误结算，不等于所有注单必定作废。",
        ],
    }
    prefix = f"结合这页的“{topic_hint}”场景看，" if topic_hint else ""
    values = bank[key]
    return prefix + values[idx % len(values)]


def conclusion(key: str, idx: int) -> str:
    bank = {
        "third_party": [
            "结论：三方更偏外部通道或外部服务方，不是平台自研系统。",
            "结论：看到“三方名称”时，应先按外部服务接入理解。",
        ],
        "master_agent": [
            "结论：总代是代理层级角色，不是平台本身。",
            "结论：总代可以对接平台，但不能直接等同平台方。",
        ],
        "negative_profit": [
            "结论：这里的负盈利更接近玩家亏损、平台盈利后的代理分润口径。",
            "结论：负盈利不是平台自己亏钱的同义词。",
        ],
        "sports_package": [
            "结论：体育包网交付的是整套体育业务能力，不是一张玩法页。",
            "结论：盘口、赔率、滚球、串关、结算这些能力合起来，才更接近体育包网。",
        ],
        "parlay": [
            "结论：综合过关可以按串关理解，不是另一套完全独立玩法。",
            "结论：串关的核心是组合投注和赔率连乘。",
        ],
        "danger_ball": [
            "结论：危险球是影响确认和结算的事件状态，不是单独玩法。",
            "结论：待确认或取消是结算保护动作，不代表注单天然无效。",
        ],
    }
    values = bank[key]
    return values[idx % len(values)]


def build_response(row: Dict[str, Any], key: str, idx: int) -> str:
    meta = row.get("meta") or {}
    topic = str(meta.get("topic", "")).strip()
    visible = extract_visible_info(str(row.get("response", "")), topic)
    style = idx % 3
    if style == 0:
        parts = [
            f"页面可见信息：{visible}。",
            f"业务解释：{biz_explanation(key, topic, idx)}。",
            conclusion(key, idx),
        ]
    elif style == 1:
        parts = [
            f"图上可见：{visible}。",
            f"结合业务场景看：{biz_explanation(key, topic, idx)}。",
            conclusion(key, idx),
        ]
    else:
        parts = [
            f"页面证据：{visible}。",
            f"不要把图里没直接写出的概念当成页面字段，业务上应理解为：{biz_explanation(key, topic, idx)}。",
            conclusion(key, idx),
        ]
    return clean_text(" ".join(parts))


def build_targeted_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    targeted: List[Dict[str, Any]] = []
    for row in rows:
        meta = row.get("meta") or {}
        key = infer_key(row)
        if not key:
            continue
        common_meta = {
            **meta,
            "source_vl_id": row.get("id"),
            "patch_family": "vl_definition_misconception_patch_v1",
            "patch_focus": key,
        }
        for query in def_queries(key, str(meta.get("topic", "")).strip(), len(targeted)):
            targeted.append(
                {
                    **row,
                    "id": f"{row.get('id')}_defmis_def_v1_{len(targeted):04d}",
                    "query": query,
                    "response": build_response(row, key, len(targeted)),
                    "meta": {
                        **common_meta,
                        "domain_qtype": "definition",
                        "patch_type": "vl_definition_targeted",
                    },
                }
            )
        for query in mis_queries(key, len(targeted)):
            targeted.append(
                {
                    **row,
                    "id": f"{row.get('id')}_defmis_mis_v1_{len(targeted):04d}",
                    "query": query,
                    "response": build_response(row, key, len(targeted)),
                    "meta": {
                        **common_meta,
                        "domain_qtype": "misconception",
                        "patch_type": "vl_misconception_targeted",
                    },
                }
            )
    return targeted


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a targeted VL patch pack for definition + misconception recovery.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--summary", default=str(DEFAULT_SUMMARY))
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()
    summary_path = Path(args.summary).resolve()

    base_rows = read_jsonl(input_path)
    selected_rows = select_rows(base_rows)
    targeted_rows = build_targeted_rows(selected_rows)

    write_jsonl(output_path, targeted_rows)
    focus_counter = Counter((row.get("meta") or {}).get("patch_focus", "unknown") for row in targeted_rows)
    type_counter = Counter((row.get("meta") or {}).get("patch_type", "unknown") for row in targeted_rows)
    qtype_counter = Counter((row.get("meta") or {}).get("domain_qtype", "unknown") for row in targeted_rows)
    summary = {
        "input": str(input_path),
        "output": str(output_path),
        "selected_source_rows": len(selected_rows),
        "targeted_rows": len(targeted_rows),
        "patch_focus_counts": dict(focus_counter),
        "patch_type_counts": dict(type_counter),
        "domain_qtype_counts": dict(qtype_counter),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
