#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


PROJECT_ROOT = Path("/home/ubuntu/qwen35a3b_finetune")
DEFAULT_INPUT = PROJECT_ROOT / "datasets" / "relation_patch_pack" / "vl_relation_patch_v2.swift_vl.jsonl"
DEFAULT_OUTPUT = PROJECT_ROOT / "datasets" / "relation_patch_pack" / "vl_relation_narrow_patch_v1.swift_vl.jsonl"
DEFAULT_SUMMARY = PROJECT_ROOT / "datasets" / "relation_patch_pack" / "vl_relation_narrow_patch_v1.summary.json"

PAIR_ORDER = ["限红_vs_风控", "RTP_vs_PNL", "总代_vs_平台"]
PATCH_TYPE_ORDER = [
    "relation_vl",
    "relation_vl_compact",
    "relation_vl_targeted_misconception",
    "relation_vl_targeted_binding",
]
PAIR_PATCH_CAPS = {
    "限红_vs_风控": {
        "relation_vl": 24,
        "relation_vl_compact": 24,
        "relation_vl_targeted_misconception": 36,
        "relation_vl_targeted_binding": 54,
        "relation_vl_targeted_strict_binding": 36,
    },
    "RTP_vs_PNL": {
        "relation_vl": 12,
        "relation_vl_compact": 12,
        "relation_vl_targeted_misconception": 24,
        "relation_vl_targeted_binding": 30,
        "relation_vl_targeted_strict_binding": 18,
    },
    "总代_vs_平台": {
        "relation_vl": 8,
        "relation_vl_compact": 8,
        "relation_vl_targeted_misconception": 18,
        "relation_vl_targeted_binding": 24,
        "relation_vl_targeted_strict_binding": 8,
    },
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
    return value.strip()


def compact(text: str, limit: int) -> str:
    value = clean_text(text).strip("，,；;。 ")
    if not value:
        return ""
    if len(value) <= limit:
        return value
    return value[:limit].rstrip("，,；;。") + "…"


def sanitize_evidence_candidate(text: str) -> str:
    value = clean_text(text)
    for marker in [
        "关系结论：",
        "业务解释：",
        "结论：",
        "判断：",
        "原因：",
        "业务对应：",
        "风控管",
        "这类页面更像",
        "包网平台对",
    ]:
        if marker in value:
            value = value.split(marker, 1)[0].strip()
    return clean_text(value)


def low_signal_text(text: str) -> bool:
    value = clean_text(text)
    if not value:
        return True
    cjk_count = sum("\u4e00" <= ch <= "\u9fff" for ch in value)
    if cjk_count < 6:
        return True
    bad_markers = ["结论：", "业务解释：", "关系结论：", "http", "……"]
    return any(marker in value for marker in bad_markers)


def stable_sort_key(row: Dict[str, Any]) -> str:
    meta = row.get("meta") or {}
    payload = "|".join(
        [
            str(row.get("id", "")),
            str(row.get("query", "")),
            str(meta.get("patch_type", "")),
            str(meta.get("relation_pair", "")),
            str(meta.get("source_vl_id", "")),
        ]
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def infer_relation_pair(row: Dict[str, Any]) -> str:
    meta = row.get("meta") or {}
    relation_pair = str(meta.get("relation_pair", "")).strip()
    if relation_pair:
        return relation_pair
    query = str(row.get("query", ""))
    if "限红" in query and "风控" in query:
        return "限红_vs_风控"
    if "RTP" in query and "PNL" in query:
        return "RTP_vs_PNL"
    if "总代" in query and "平台" in query:
        return "总代_vs_平台"
    return ""


def group_existing_rows(rows: List[Dict[str, Any]]) -> Dict[Tuple[str, str], List[Dict[str, Any]]]:
    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        pair = infer_relation_pair(row)
        patch_type = str((row.get("meta") or {}).get("patch_type", "")).strip()
        if pair not in PAIR_PATCH_CAPS or patch_type not in PATCH_TYPE_ORDER:
            continue
        grouped[(pair, patch_type)].append(row)
    return grouped


def extract_section(answer: str, label: str, next_labels: List[str]) -> str:
    anchor = f"{label}："
    text = str(answer or "")
    start = text.find(anchor)
    if start == -1:
        return ""
    section = text[start + len(anchor):]
    end = len(section)
    for next_label in next_labels:
        idx = section.find(f"{next_label}：")
        if idx != -1:
            end = min(end, idx)
    return clean_text(section[:end].strip(" 。"))


def response_visible_text(row: Dict[str, Any]) -> str:
    answer = str(row.get("response", ""))
    visible = extract_section(answer, "页面可见信息", ["业务解释", "关系结论"])
    visible = sanitize_evidence_candidate(visible)
    if "。" in visible:
        visible = visible.split("。", 1)[0].strip()
    return clean_text(visible)


def page_evidence_text(row: Dict[str, Any]) -> str:
    candidates = [
        str(row.get("topic_hint", "")),
        str(row.get("ocr_context", "")),
        response_visible_text(row),
        str((row.get("meta") or {}).get("topic", "")),
    ]
    for candidate in candidates:
        value = compact(sanitize_evidence_candidate(candidate), 72)
        if not low_signal_text(value):
            return value
    return compact(str((row.get("meta") or {}).get("topic", "")).strip(), 24)


def visible_hint(row: Dict[str, Any]) -> str:
    return compact(page_evidence_text(row), 42)


def quality_priority(row: Dict[str, Any], pair: str) -> Tuple[int, int, int, str]:
    meta = row.get("meta") or {}
    family = str(meta.get("family", "")).strip()
    topic = str(meta.get("topic", "")).strip()
    evidence = page_evidence_text(row)
    admin_first = 0 if family == "admin" else 1
    if pair == "限红_vs_风控":
        keyword_first = 0 if any(token in evidence for token in ["额度", "审核", "补单", "限额", "风控"]) else 1
        topic_first = 0 if topic in {"资金审核", "代理管理"} else 1
    elif pair == "RTP_vs_PNL":
        keyword_first = 0 if any(token in evidence for token in ["仪表盘", "盈利", "数据", "赔率", "盘口", "返还"]) else 1
        topic_first = 0 if topic in {"仪表盘", "体育规则", "滚球规则"} else 1
    else:
        keyword_first = 0 if any(token in evidence for token in ["代理", "上级代理", "VIP", "域名", "推广"]) else 1
        topic_first = 0 if topic in {"代理管理"} else 1
    return (admin_first, keyword_first, topic_first, stable_sort_key(row))


def pair_business_explanation(pair: str, topic: str, idx: int) -> str:
    topic_hint = compact(topic, 18)
    if pair == "限红_vs_风控":
        variants = [
            "这类额度限制、审核、补单相关字段更像风控执行位，限红是把风险控制落到具体额度或下注限制上的动作。",
            "如果页面重点是额度限制和审核节点，通常应先按风控场景理解，再把限红解释成其中一类执行方式。",
            "风控负责识别和处理风险，限红负责把风险控制落实到额度、下注或账户限制上，两者不是同义词。",
        ]
    elif pair == "RTP_vs_PNL":
        variants = [
            "这类看板、排行、汇总页更接近 PNL/盈亏统计口径；RTP 是玩法或系统的长期返还率，不是这类实际统计页的主口径。",
            "如果页面在讲数据看板、运营统计或盈亏汇总，应优先按 PNL 场景理解，而不是说成 RTP 配置页。",
            "RTP 讲长期返还模型，PNL 讲一段时间内的实际盈亏结果，两者维度不同，不能混成一个概念。",
        ]
    else:
        variants = [
            "这类代理层级、上级代理、推广链路页面更能说明总代属于代理体系角色，不是平台系统主体本身。",
            "总代可以对接平台、管理下级代理，但平台负责底层系统、规则、风控和结算，二者不能直接等同。",
            "如果页面围绕代理层级和推广关系展开，应优先按总代/代理角色理解，不要直接拔高成平台身份。",
        ]
    prefix = f"结合这页的“{topic_hint}”场景看，" if topic_hint else ""
    return prefix + variants[idx % len(variants)]


def pair_conclusion(pair: str, idx: int) -> str:
    bank = {
        "限红_vs_风控": [
            "结论：风控是上层机制，限红是其中一种落地动作。",
            "结论：不能把限红和风控当成同义词，应先说风控，再说限红。",
            "结论：这页更支持“风控包含限红”，不支持把两者说成两套独立体系。",
        ],
        "RTP_vs_PNL": [
            "结论：RTP 是长期返还率口径，PNL 是实际盈亏口径。",
            "结论：统计/仪表盘页优先落到 PNL，不应误判成 RTP 配置页。",
            "结论：RTP 不是实际盈亏统计，PNL 也不是返奖率设置。",
        ],
        "总代_vs_平台": [
            "结论：总代属于代理层级角色，不等于平台方。",
            "结论：总代可以对接平台，但总代本身不是平台系统主体。",
            "结论：这页更支持“总代属于代理体系”，不支持直接等同平台身份。",
        ],
    }
    return bank[pair][idx % len(bank[pair])]


def strict_query(row: Dict[str, Any], pair: str, idx: int) -> str:
    hint = visible_hint(row)
    if pair == "限红_vs_风控":
        bank = [
            f"图里像“{hint}”这类内容，请严格分成“页面证据”和“业务解释”回答限红和风控的关系；不要编造“风控标签页”“账户状态”“账号设定”这类页面字段。",
            f"如果图里没有直接写“限红”或“风控”，该怎么解释两者关系？只许引用像“{hint}”这类可见内容，不要补出图里没有的标签页或状态字段。",
            f"结合像“{hint}”这类页面信息回答：先写页面证据，再写你基于额度限制/审核场景做的判断；不要把没出现的页面词硬说成证据。",
        ]
    elif pair == "RTP_vs_PNL":
        bank = [
            f"图里像“{hint}”这类内容，请严格分成“页面证据”和“业务解释”回答 RTP 和 PNL 的区别；不要编造“RTP配置”“返奖率设置”“赔率调整”这类页面字段。",
            f"如果图里没有直接写“RTP”或“PNL”，该怎么解释两者区别？只许引用像“{hint}”这类可见内容，不要补出图里没有的配置项。",
            f"结合像“{hint}”这类页面信息回答：先写页面证据，再写你基于统计/盈亏场景做的判断；不要把没出现的配置词硬说成证据。",
        ]
    else:
        bank = [
            f"图里像“{hint}”这类内容，请严格分成“页面证据”和“业务解释”回答总代和平台的关系；不要编造图里没有的“平台方”“总代字样”等页面字段。",
            f"如果图里没有直接写“总代”或“平台方”，该怎么解释两者关系？只许引用像“{hint}”这类可见内容，不要补出图里没有的身份字段。",
            f"结合像“{hint}”这类页面信息回答：先写页面证据，再写你基于代理层级场景做的判断；不要把没出现的身份词硬说成证据。",
        ]
    return bank[idx % len(bank)]


def strict_response(row: Dict[str, Any], pair: str, idx: int) -> str:
    visible = page_evidence_text(row)
    topic = str((row.get("meta") or {}).get("topic", "")).strip()
    cautions = {
        "限红_vs_风控": "图里未必直接出现“限红”或“风控”字样，这部分要明确说是基于额度限制、审核或补单场景做的业务判断，不要编造标签页或状态字段。",
        "RTP_vs_PNL": "图里未必直接出现“RTP”或“PNL”字样，这部分要明确说是基于统计看板、运营数据或盈亏场景做的业务判断，不要编造配置项。",
        "总代_vs_平台": "图里未必直接出现“总代”或“平台方”字样，这部分要明确说是基于上级代理、代理层级或推广关系做的业务判断，不要编造身份字段。",
    }
    parts = [
        f"页面证据：{visible}。",
        f"业务解释：{pair_business_explanation(pair, topic, idx)}",
        f"约束：{cautions[pair]}",
        pair_conclusion(pair, idx),
    ]
    return clean_text(" ".join(parts))


def sample_existing_rows(rows: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, List[Dict[str, Any]]]]:
    grouped = group_existing_rows(rows)
    sampled: List[Dict[str, Any]] = []
    strict_sources: Dict[str, List[Dict[str, Any]]] = {}
    for pair in PAIR_ORDER:
        strict_sources[pair] = []
        for patch_type in PATCH_TYPE_ORDER:
            candidates = sorted(grouped.get((pair, patch_type), []), key=lambda row: quality_priority(row, pair))
            cap = PAIR_PATCH_CAPS[pair][patch_type]
            picked = candidates[:cap]
            sampled.extend(picked)
            if patch_type == "relation_vl":
                if pair in {"限红_vs_风控", "总代_vs_平台"}:
                    strict_sources[pair].extend([row for row in picked if str((row.get("meta") or {}).get("family", "")).strip() == "admin"])
                else:
                    strict_sources[pair].extend(picked)
    return sampled, strict_sources


def build_strict_rows(strict_sources: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for pair in PAIR_ORDER:
        candidates = strict_sources[pair]
        cap = PAIR_PATCH_CAPS[pair]["relation_vl_targeted_strict_binding"]
        for idx, source_row in enumerate(candidates[:cap]):
            meta = source_row.get("meta") or {}
            source_id = str(meta.get("source_vl_id") or source_row.get("id"))
            rows.append(
                {
                    **source_row,
                    "id": f"{source_id}_strict_binding_narrow_v1",
                    "query": strict_query(source_row, pair, idx),
                    "response": strict_response(source_row, pair, idx),
                    "meta": {
                        **meta,
                        "source_vl_id": source_id,
                        "relation_pair": pair,
                        "patch_type": "relation_vl_targeted_strict_binding",
                        "targeted_round": "vl_relation_narrow_v1",
                    },
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a narrow relation-only VL patch pack with stronger anti-hallucination binding samples.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--summary", default=str(DEFAULT_SUMMARY))
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()
    summary_path = Path(args.summary).resolve()

    source_rows = read_jsonl(input_path)
    sampled_existing, strict_sources = sample_existing_rows(source_rows)
    strict_rows = build_strict_rows(strict_sources)
    all_rows = sorted([*sampled_existing, *strict_rows], key=stable_sort_key)

    pair_counts = Counter()
    patch_counts = Counter()
    for row in all_rows:
        pair_counts[infer_relation_pair(row) or "unknown"] += 1
        patch_counts[str((row.get("meta") or {}).get("patch_type", "unknown"))] += 1

    write_jsonl(output_path, all_rows)
    summary = {
        "input": str(input_path),
        "output": str(output_path),
        "total_rows": len(all_rows),
        "pair_counts": dict(pair_counts),
        "patch_type_counts": dict(patch_counts),
        "pair_patch_caps": PAIR_PATCH_CAPS,
        "strict_rows_added": len(strict_rows),
        "existing_rows_sampled": len(sampled_existing),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
