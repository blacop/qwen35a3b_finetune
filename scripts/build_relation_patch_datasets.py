#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List


PROJECT_ROOT = Path("/home/ubuntu/qwen35a3b_finetune")
DEFAULT_MM_INPUT = PROJECT_ROOT / "datasets" / "tydata_domain_multimodal_pack" / "multimodal_train.swift_vl.jsonl"
DEFAULT_OUT_DIR = PROJECT_ROOT / "datasets" / "relation_patch_pack"
TEXT_SYSTEM_PROMPT = (
    "你是体育包网客服助手。请根据给定页面上下文，准确说明页面可见信息与业务解释的对应关系，"
    "不要编造图中没有的字段、按钮、流程或结论。不要输出思维链，不要重复堆砌同一字段、术语或句子；"
    "如果图里没有直接证据，要明确说是基于页面场景判断。"
)

RELATION_OFFTOPIC_MARKERS = [
    "短信提供商",
    "新版发布",
    "导航页配置",
    "充值任务",
    "充值分组",
    "域名解析",
    "添加CDN",
    "添加缓存",
    "站点管理",
    "域名盾",
    "证书",
    "跑量域名",
    "ITDOG",
    "上传APK",
    "安装完成",
]


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


def extract_visible_info(answer: str) -> str:
    answer = str(answer or "")
    match = re.search(r"页面可见信息：(.+?)业务解释：", answer, flags=re.S)
    if match:
        return normalize_text(match.group(1).strip(" 。"))
    return ""


def build_text_prompt(query: str, answer: str, topic: str) -> str:
    visible = extract_visible_info(answer)
    parts = []
    if visible:
        parts.append(f"【页面上下文】\n{visible}")
    if topic:
        parts.append(f"【页面主题】\n{topic}")
    parts.append(f"【问题】\n{query}")
    return "\n\n".join(parts)


def extract_section(answer: str, label: str, next_labels: Sequence[str]) -> str:
    answer = str(answer or "")
    pattern = re.escape(label) + r"：(.+)"
    match = re.search(pattern, answer, flags=re.S)
    if not match:
        return ""
    section = match.group(1)
    end = len(section)
    for next_label in next_labels:
        idx = section.find(f"{next_label}：")
        if idx != -1:
            end = min(end, idx)
    return normalize_text(section[:end].strip(" 。"))


def compact_sentence(text: str, max_chars: int) -> str:
    value = normalize_text(text).strip("。；;，, ")
    if not value:
        return ""
    if len(value) <= max_chars:
        return value
    return value[:max_chars].rstrip("，,；;。") + "…"


def repeated_span_score(text: str, span_len: int = 8) -> int:
    text = normalize_text(text)
    if len(text) < span_len * 2:
        return 1
    counts: Counter[str] = Counter(text[i : i + span_len] for i in range(len(text) - span_len + 1))
    return max(counts.values()) if counts else 1


def relation_row_allowed(row: Dict[str, Any]) -> bool:
    meta = row.get("meta") or {}
    topic = str(meta.get("topic", "")).strip()
    family = str(meta.get("family", "")).strip()
    response = str(row.get("response", ""))
    visible = extract_section(response, "页面可见信息", ["业务解释", "关系结论"])
    blob = "\n".join([topic, str(row.get("query", "")), response])
    signal_blob = "\n".join([topic, visible])
    if any(marker in blob for marker in RELATION_OFFTOPIC_MARKERS):
        return False

    term = str(meta.get("domain_term", "")).strip()
    signal_keywords = {
        "限红": ["额度", "审核", "补单", "冻结", "盘口", "限额", "风险"],
        "RTP": ["赔率", "盘口", "返奖", "返还", "盈亏", "统计", "盈利", "充值", "提现", "结算", "赛果", "仪表盘", "排行"],
        "总代": ["代理", "总代", "上级", "下级", "推广", "域名", "分润", "直属", "平台"],
    }
    keywords = signal_keywords.get(term, [])
    if keywords and not any(keyword in signal_blob for keyword in keywords):
        return False
    if term == "RTP":
        if any(marker in blob for marker in ["支付", "通道", "支付宝", "JPAI", "编辑", "禁止充值用户ID"]):
            return False
        if family == "admin" and topic not in {"仪表盘", "资金审核", "赛事结算", "用户充值排名描述"}:
            return False
        if topic not in {"仪表盘", "资金审核", "赛事结算", "用户充值排名描述", "滚球规则", "赔率盘型", "独赢和1X2", "串关规则"} and not any(
            keyword in signal_blob for keyword in ["统计", "排行", "盈亏", "赛果", "已结算注单", "未结算注单", "赔率", "盘口", "返奖", "仪表盘"]
        ):
            return False
    if repeated_span_score(response, span_len=8) >= 6:
        return False
    return True


def compact_relation_response(answer: str) -> str:
    visible = compact_sentence(extract_section(answer, "页面可见信息", ["业务解释", "关系结论"]), 72)
    biz = compact_sentence(extract_section(answer, "业务解释", ["关系结论"]), 120)
    conclusion = compact_sentence(extract_section(answer, "关系结论", []), 120)
    parts: List[str] = []
    if visible:
        parts.append(f"页面可见信息：{visible}。")
    if biz:
        parts.append(f"业务解释：{biz}。")
    if conclusion:
        parts.append(f"关系结论：{conclusion}。")
    return " ".join(parts).strip()


def build_compact_query(row: Dict[str, Any]) -> str:
    term = str((row.get("meta") or {}).get("domain_term", "")).strip()
    base = str(row.get("query", "")).strip()
    if term == "限红":
        return "请用不超过3句话回答：先说页面可见信息，再说风控和限红怎么绑定；不要重复同一个字段名。"
    if term == "RTP":
        return "请用不超过3句话回答：先说页面可见信息，再说 RTP 和 PNL 怎么区分；不要重复同一个字段名。"
    if term == "总代":
        return "请用不超过3句话回答：先说页面可见信息，再说总代和平台怎么区分；不要重复同一个字段名。"
    return f"请用不超过3句话回答，并先说页面可见信息再给业务解释：{base}"


def build_datasets(mm_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    relation_rows = [
        row
        for row in mm_rows
        if (row.get("meta") or {}).get("domain_qtype") == "relation" and relation_row_allowed(row)
    ]

    text_rows: List[Dict[str, Any]] = []
    vl_rows: List[Dict[str, Any]] = []
    q_counter: Counter[str] = Counter()
    term_counter: Counter[str] = Counter()
    compact_added = 0

    for row in relation_rows:
        meta = row.get("meta") or {}
        topic = str(meta.get("topic", "")).strip()
        term = str(meta.get("domain_term", "")).strip()
        q_counter[str(row.get("query", ""))] += 1
        if term:
            term_counter[term] += 1

        text_rows.append(
            {
                "id": f"text_{row['id']}",
                "split": row.get("split", "train"),
                "messages": [
                    {"role": "user", "content": build_text_prompt(str(row.get("query", "")), str(row.get("response", "")), topic)},
                    {"role": "assistant", "content": str(row.get("response", ""))},
                ],
                "meta": {
                    **meta,
                    "patch_type": "relation_text",
                    "source_mm_id": row.get("id"),
                },
            }
        )
        vl_rows.append(
            {
                **row,
                "meta": {
                    **meta,
                    "patch_type": "relation_vl",
                },
            }
        )

        compact_response = compact_relation_response(str(row.get("response", "")))
        if compact_response:
            compact_added += 1
            compact_query = build_compact_query(row)
            q_counter[compact_query] += 1
            if term:
                term_counter[term] += 1
            text_rows.append(
                {
                    "id": f"text_compact_{row['id']}",
                    "split": row.get("split", "train"),
                    "messages": [
                        {"role": "user", "content": build_text_prompt(compact_query, compact_response, topic)},
                        {"role": "assistant", "content": compact_response},
                    ],
                    "meta": {
                        **meta,
                        "patch_type": "relation_text_compact",
                        "source_mm_id": row.get("id"),
                    },
                }
            )
            vl_rows.append(
                {
                    **row,
                    "id": f"{row['id']}_compact",
                    "query": compact_query,
                    "response": compact_response,
                    "system": TEXT_SYSTEM_PROMPT,
                    "meta": {
                        **meta,
                        "patch_type": "relation_vl_compact",
                    },
                }
            )

    return {
        "text_rows": text_rows,
        "vl_rows": vl_rows,
        "query_counts": dict(q_counter),
        "term_counts": dict(term_counter),
        "relation_rows_kept": len(relation_rows),
        "compact_added": compact_added,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build relation-only text/VL patch datasets from the multimodal domain pack.")
    parser.add_argument("--mm-input", default=str(DEFAULT_MM_INPUT))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    args = parser.parse_args()

    mm_input = Path(args.mm_input).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    mm_rows = read_jsonl(mm_input)
    built = build_datasets(mm_rows)

    text_out = out_dir / "text_relation_patch.jsonl"
    vl_out = out_dir / "vl_relation_patch.swift_vl.jsonl"
    summary_out = out_dir / "summary.json"

    write_jsonl(text_out, built["text_rows"])
    write_jsonl(vl_out, built["vl_rows"])

    summary = {
        "mm_input": str(mm_input),
        "output_dir": str(out_dir),
        "text_output": str(text_out),
        "vl_output": str(vl_out),
        "relation_rows_kept": built["relation_rows_kept"],
        "compact_added": built["compact_added"],
        "text_rows": len(built["text_rows"]),
        "vl_rows": len(built["vl_rows"]),
        "text_train": sum(1 for row in built["text_rows"] if row["split"] == "train"),
        "text_val": sum(1 for row in built["text_rows"] if row["split"] == "val"),
        "vl_train": sum(1 for row in built["vl_rows"] if row["split"] == "train"),
        "vl_val": sum(1 for row in built["vl_rows"] if row["split"] == "val"),
        "query_counts": built["query_counts"],
        "term_counts": built["term_counts"],
        "text_system_prompt": TEXT_SYSTEM_PROMPT,
    }
    summary_out.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
