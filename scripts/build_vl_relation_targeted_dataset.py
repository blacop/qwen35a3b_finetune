#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List


PROJECT_ROOT = Path("/home/ubuntu/qwen35a3b_finetune")
DEFAULT_INPUT = PROJECT_ROOT / "datasets" / "relation_patch_pack" / "vl_relation_patch.swift_vl.jsonl"
DEFAULT_OUTPUT = PROJECT_ROOT / "datasets" / "relation_patch_pack" / "vl_relation_patch_v2.swift_vl.jsonl"
DEFAULT_SUMMARY = PROJECT_ROOT / "datasets" / "relation_patch_pack" / "vl_relation_patch_v2.summary.json"


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
    return normalize_text(section[:end].strip(" 。"))


def compact(text: str, max_chars: int) -> str:
    value = normalize_text(text).strip("。；;，, ")
    if not value:
        return ""
    if len(value) <= max_chars:
        return value
    return value[:max_chars].rstrip("，,；;。") + "…"


def clean_text(text: str) -> str:
    value = normalize_text(text)
    while "。。" in value:
        value = value.replace("。。", "。")
    value = value.replace(" .", ".").replace(" ,", ",")
    return value.strip()


def pair_key(row: Dict[str, Any]) -> str:
    query = str(row.get("query", ""))
    if "限红" in query and "风控" in query:
        return "limit_risk"
    if "RTP" in query and "PNL" in query:
        return "rtp_pnl"
    if "总代" in query and "平台" in query:
        return "master_platform"
    return "other"


def pair_name(pair: str) -> str:
    names = {
        "limit_risk": "限红_vs_风控",
        "rtp_pnl": "RTP_vs_PNL",
        "master_platform": "总代_vs_平台",
        "other": "other",
    }
    return names[pair]


def context_hint(row: Dict[str, Any]) -> str:
    answer = str(row.get("response", ""))
    visible = compact(extract_section(answer, "页面可见信息", ["业务解释", "关系结论"]), 32)
    topic = compact(str((row.get("meta") or {}).get("topic", "")).strip(), 12)
    if visible:
        return f"图里像“{visible}”这类内容"
    if topic:
        return f"这页偏“{topic}”场景"
    return "只看这张图"


def misconception_queries(row: Dict[str, Any], pair: str, idx: int) -> str:
    hint = context_hint(row)
    bank = {
        "limit_risk": [
            f"有人把这页说成“只是限红，不属于风控”，这个判断对吗？{hint}，先说页面证据，再纠正。",
            f"{hint}，这页看到的是单纯限红页，还是风控执行位？不要只下结论，要结合图里的字段。",
            f"{hint}，能不能把限红和风控当成同义词？请先点页面证据，再解释。",
        ],
        "rtp_pnl": [
            f"有人把这页直接当成 RTP 配置页，这个说法对吗？{hint}，先说页面证据，再纠正。",
            f"{hint}，这张图更像 RTP 页面还是 PNL/盈亏统计场景？别泛讲定义，要结合图里可见内容。",
            f"{hint}，能不能把 RTP 和 PNL 当成一回事？请先点页面证据，再解释。",
        ],
        "master_platform": [
            f"有人看到这页就说“总代就是平台方”，这个判断对吗？{hint}，先说页面证据，再纠正。",
            f"{hint}，这页更能说明总代身份，还是平台系统身份？不要空讲概念，要结合可见字段。",
            f"{hint}，能不能把总代和平台当成同一个角色？请先点页面证据，再解释。",
        ],
    }
    return bank[pair][idx % len(bank[pair])]


def binding_queries(row: Dict[str, Any], pair: str, idx: int) -> str:
    hint = context_hint(row)
    bank = {
        "limit_risk": [
            f"{hint}，把图里直接看得到的字段，和你对“限红/风控关系”的业务判断分开说，不要混写。",
            f"{hint}，先列页面证据，再列业务解释：哪些是图上直接写出来的，哪些只是你根据场景做的判断？",
            f"{hint}，请严格拆成“页面证据”和“业务解释”两部分回答，别把图里没写的内容当成页面字段。",
        ],
        "rtp_pnl": [
            f"{hint}，把图里直接看得到的内容，和你对“RTP/PNL区别”的业务判断分开说，不要混写。",
            f"{hint}，先列页面证据，再列业务解释：图里直接写了什么，哪些是你基于运营统计场景做的判断？",
            f"{hint}，请严格拆成“页面证据”和“业务解释”两部分回答，别把图里没写的 RTP/PNL 字样硬说成页面字段。",
        ],
        "master_platform": [
            f"{hint}，把图里直接看得到的代理/域名字段，和你对“总代/平台区别”的业务判断分开说，不要混写。",
            f"{hint}，先列页面证据，再列业务解释：页面直接写了什么，哪些是你基于代理层级做的判断？",
            f"{hint}，请严格拆成“页面证据”和“业务解释”两部分回答，别把图里没写的“平台方”当成页面字段。",
        ],
    }
    return bank[pair][idx % len(bank[pair])]


def pair_business_explanation(pair: str, topic: str, idx: int) -> str:
    topic_hint = compact(topic, 18)
    if pair == "limit_risk":
        tails = [
            "这类额度、审核、补单或资金处理字段通常更像风控执行位，限红只是其中一种落地动作。",
            "图里更像风险控制的执行入口，限红负责收紧额度，风控负责识别和处理风险。",
            "如果页面围绕额度限制、审核节点或补单处理展开，通常应先按风控场景理解，再落到限红动作。",
        ]
    elif pair == "rtp_pnl":
        tails = [
            "这类看板、排行、注单或资金统计更接近 PNL 场景；RTP 是玩法长期返还率，不是这类实际统计口径。",
            "图里如果主要是运营统计、充值提现、盈亏汇总，就应先按 PNL 理解，而不是说成 RTP 配置。",
            "RTP 讲长期返还模型，PNL 讲一段时间内的实际盈亏，统计页和配置页不能混成一个概念。",
        ]
    else:
        tails = [
            "图里出现代理层级、上级下级、推广域名这类信息时，应优先按代理角色理解，不要直接上升成平台身份。",
            "这类页面更像渠道或代理管理场景，总代可以对接平台，但本身不等于平台系统提供方。",
            "如果页面重点是代理关系和推广配置，通常说明的是总代/代理角色，而不是平台技术方身份。",
        ]
    if topic_hint:
        return f"结合这页的“{topic_hint}”场景看，{tails[idx % len(tails)]}"
    return tails[idx % len(tails)]


def pair_conclusion(pair: str, idx: int) -> str:
    bank = {
        "limit_risk": [
            "不能把限红和风控当成同义词，限红更像风控落地动作。",
            "结论上应先说风控是上层机制，再说限红是具体执行方式。",
            "这页更支持“风控包含限红”，不支持把两者并列成两套独立体系。",
        ],
        "rtp_pnl": [
            "结论上应区分长期返还率和实际盈亏，不能把 RTP 和 PNL 混说。",
            "这页如果是统计或排行场景，应优先落到 PNL，不要误判成 RTP 配置。",
            "RTP 不是实际盈亏统计，PNL 也不是返奖率配置，两者维度不同。",
        ],
        "master_platform": [
            "结论上应说总代是代理层级角色，不等于平台方。",
            "可以说总代可能对接平台，但总代本身不是平台系统提供方。",
            "这页更支持“总代属于代理体系”，不支持直接等同平台身份。",
        ],
    }
    return bank[pair][idx % len(bank[pair])]


def binding_caution(pair: str, visible: str) -> str:
    if pair == "limit_risk" and "风控" not in visible and "限红" not in visible:
        return "图里未必直接写出“风控”或“限红”字样，这部分要明确说是基于额度/审核字段做的业务判断。"
    if pair == "rtp_pnl" and "RTP" not in visible and "PNL" not in visible:
        return "图里未必直接写出“RTP”或“PNL”，这部分要明确说是基于统计/排行/盈亏场景做的业务判断。"
    if pair == "master_platform" and "平台" not in visible and "总代" not in visible:
        return "图里未必直接出现“平台方”字样，这部分要明确说是基于代理层级和域名字段做的业务判断。"
    return "页面字段和业务判断要分开说，不能把图里没有直接写出的词当成页面证据。"


def misconception_response(row: Dict[str, Any], pair: str, idx: int) -> str:
    answer = str(row.get("response", ""))
    topic = str((row.get("meta") or {}).get("topic", "")).strip()
    visible = compact(extract_section(answer, "页面可见信息", ["业务解释", "关系结论"]), 110)
    evidence_labels = ["页面证据", "图上可见", "页面可见信息"]
    correction_labels = ["纠正", "业务解释", "判断依据"]
    conclusion_labels = ["结论", "关系结论", "最终结论"]
    evidence = f"{evidence_labels[idx % len(evidence_labels)]}：{visible}。"
    correction = f"{correction_labels[idx % len(correction_labels)]}：{pair_business_explanation(pair, topic, idx)}。"
    conclusion = f"{conclusion_labels[idx % len(conclusion_labels)]}：{pair_conclusion(pair, idx)}。"
    return clean_text(" ".join([evidence, correction, conclusion]))


def binding_response(row: Dict[str, Any], pair: str, idx: int) -> str:
    answer = str(row.get("response", ""))
    visible = compact(extract_section(answer, "页面可见信息", ["业务解释", "关系结论"]), 110)
    topic = str((row.get("meta") or {}).get("topic", "")).strip()
    biz = pair_business_explanation(pair, topic, idx)
    caution = binding_caution(pair, visible)
    conclusion = pair_conclusion(pair, idx)
    styles = [
        (
            f"页面证据：{visible}。",
            f"业务解释：{biz}",
            f"结论：{conclusion}",
            f"注意：{caution}",
        ),
        (
            f"图上直接能看到：{visible}。",
            f"基于场景的业务判断：{biz}",
            f"不要混写：{caution}",
            f"最终结论：{conclusion}",
        ),
        (
            f"页面可见信息：{visible}。",
            f"页面没有直接写出的部分，要按业务场景解释：{biz}",
            f"说明：{caution}",
            f"关系结论：{conclusion}",
        ),
    ]
    return clean_text(" ".join(styles[idx % len(styles)]).strip())


def build_targeted_rows(base_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    targeted: List[Dict[str, Any]] = []
    for row in base_rows:
        meta = row.get("meta") or {}
        if meta.get("patch_type") != "relation_vl":
            continue
        pair = pair_key(row)
        if pair == "other":
            continue
        base_id = str(row["id"])
        common_meta = {
            **meta,
            "source_vl_id": base_id,
            "relation_pair": pair_name(pair),
            "targeted_round": "vl_relation_v2",
        }
        targeted.append(
            {
                **row,
                "id": f"{base_id}_misunderstand_v2",
                "query": misconception_queries(row, pair, len(targeted)),
                "response": misconception_response(row, pair, len(targeted)),
                "meta": {
                    **common_meta,
                    "patch_type": "relation_vl_targeted_misconception",
                },
            }
        )
        targeted.append(
            {
                **row,
                "id": f"{base_id}_binding_v2",
                "query": binding_queries(row, pair, len(targeted)),
                "response": binding_response(row, pair, len(targeted)),
                "meta": {
                    **common_meta,
                    "patch_type": "relation_vl_targeted_binding",
                },
            }
        )
    return targeted


def main() -> None:
    parser = argparse.ArgumentParser(description="Append targeted VL relation samples on top of the existing relation patch set.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--summary", default=str(DEFAULT_SUMMARY))
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()
    summary_path = Path(args.summary).resolve()

    base_rows = read_jsonl(input_path)
    targeted_rows = build_targeted_rows(base_rows)
    all_rows = [*base_rows, *targeted_rows]

    patch_counter = Counter((row.get("meta") or {}).get("patch_type", "?") for row in all_rows)
    pair_counter = Counter((row.get("meta") or {}).get("relation_pair", "base") for row in targeted_rows)

    write_jsonl(output_path, all_rows)
    summary = {
        "input": str(input_path),
        "output": str(output_path),
        "base_rows": len(base_rows),
        "targeted_rows": len(targeted_rows),
        "total_rows": len(all_rows),
        "patch_type_counts": dict(patch_counter),
        "targeted_pair_counts": dict(pair_counter),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
