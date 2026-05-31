#!/usr/bin/env python3
"""Audit customer-service SFT data for POP / marketing / repetition pollution."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def now_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


KEYWORD_GROUPS = {
    "pop_redirect": [
        "POP",
        "下载POP",
        "下载APP",
        "下载app",
        "添加客服",
        "聊天软件",
        "联系客服",
    ],
    "marketing_bonus": [
        "彩金",
        "礼金",
        "优惠",
        "福利",
        "礼包",
        "签到",
        "大转盘",
        "首存",
        "次存",
        "返利",
    ],
    "promotion_push": [
        "参加活动",
        "参与活动",
        "上分一笔领取",
        "今日上分",
        "碰碰手气",
        "好运连连",
        "财源广进",
        "包赔100%",
        "极速到账",
        "游戏愉快",
    ],
    "off_policy_redirection": [
        "下载POP聊天软件",
        "添加客服查收",
        "请添加客服",
        "添加客服处理",
    ],
}

GROUP_PATTERNS = {
    group: re.compile("|".join(re.escape(term) for term in terms), re.I)
    for group, terms in KEYWORD_GROUPS.items()
}

ANALYSIS_PREFIX_PATTERNS = [
    re.compile(r"用户的问题是"),
    re.compile(r"回答需"),
    re.compile(r"根据中国法律法规"),
    re.compile(r"需要明确"),
    re.compile(r"需说明"),
]


@dataclass
class SampleAudit:
    sample_id: str
    assistant_text: str
    groups: list[str]
    keyword_hits: list[str]
    repeated_lines: list[str]
    analysis_prefix_hit: bool
    raw: dict[str, Any]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at line {line_no}: {exc}") from exc
            if not isinstance(obj, dict):
                raise ValueError(f"Line {line_no} is not a JSON object.")
            rows.append(obj)
    return rows


def extract_assistant_text(sample: dict[str, Any]) -> str:
    messages = sample.get("messages")
    if not isinstance(messages, list):
        return ""
    parts: list[str] = []
    for msg in messages:
        if isinstance(msg, dict) and msg.get("role") == "assistant":
            content = msg.get("content")
            if isinstance(content, str):
                parts.append(content)
    return "\n".join(parts).strip()


def normalize_line(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\s+", "", text)
    text = re.sub(r"[~!！。,.，?？]+$", "", text)
    return text


def find_repeated_lines(text: str) -> list[str]:
    counter: Counter[str] = Counter()
    original: dict[str, str] = {}
    for raw_line in text.splitlines():
        line = normalize_line(raw_line)
        if len(line) < 4:
            continue
        counter[line] += 1
        original.setdefault(line, raw_line.strip())
    return [original[line] for line, cnt in counter.items() if cnt >= 2]


def find_keyword_hits(text: str) -> tuple[list[str], list[str]]:
    groups: list[str] = []
    hits: list[str] = []
    for group, pattern in GROUP_PATTERNS.items():
        matched_terms: set[str] = set()
        for match in pattern.finditer(text):
            matched_terms.add(match.group(0))
        if matched_terms:
            groups.append(group)
            hits.extend(sorted(matched_terms))
    return groups, sorted(set(hits))


def analysis_prefix_hit(text: str) -> bool:
    prefix = "\n".join(text.splitlines()[:3])
    return any(p.search(prefix) for p in ANALYSIS_PREFIX_PATTERNS)


def build_report(
    input_path: Path,
    total_samples: int,
    contaminated: list[SampleAudit],
    group_counts: Counter[str],
    keyword_counts: Counter[str],
    repeated_line_counts: Counter[str],
    output_dir: Path,
) -> str:
    lines: list[str] = []
    contaminated_count = len(contaminated)
    lines.append(f"# GPU5 营销污染清洗报告：{input_path.name}")
    lines.append("")
    lines.append(f"- 输入文件：`{input_path}`")
    lines.append(f"- 总样本数：**{total_samples}**")
    lines.append(f"- 命中污染样本：**{contaminated_count}** ({contaminated_count / max(total_samples, 1):.2%})")
    lines.append(f"- 生成时间：`{now_ts()}`")
    lines.append("")
    lines.append("## 一、污染类型统计")
    lines.append("")
    lines.append("| 类型 | 命中数 | 占总样本比 |")
    lines.append("|---|---:|---:|")
    for group, count in sorted(group_counts.items(), key=lambda x: (-x[1], x[0])):
        lines.append(f"| {group} | {count} | {count / max(total_samples, 1):.2%} |")
    analysis_count = sum(1 for item in contaminated if item.analysis_prefix_hit)
    repeat_count = sum(1 for item in contaminated if item.repeated_lines)
    lines.append(f"| analysis_prefix | {analysis_count} | {analysis_count / max(total_samples, 1):.2%} |")
    lines.append(f"| repeated_lines | {repeat_count} | {repeat_count / max(total_samples, 1):.2%} |")
    lines.append("")
    lines.append("## 二、高频污染关键词 Top 20")
    lines.append("")
    lines.append("| 关键词 | 命中次数 |")
    lines.append("|---|---:|")
    for keyword, count in keyword_counts.most_common(20):
        lines.append(f"| {keyword} | {count} |")
    lines.append("")
    lines.append("## 三、高频重复短句 Top 20")
    lines.append("")
    lines.append("| 重复短句 | 命中样本数 |")
    lines.append("|---|---:|")
    for phrase, count in repeated_line_counts.most_common(20):
        safe_phrase = phrase.replace("\n", " ").replace("|", "\\|")
        lines.append(f"| {safe_phrase} | {count} |")
    lines.append("")
    lines.append("## 四、污染样本示例")
    lines.append("")
    grouped_examples: dict[str, list[SampleAudit]] = defaultdict(list)
    for item in contaminated:
        for group in item.groups:
            if len(grouped_examples[group]) < 3:
                grouped_examples[group].append(item)
    for group in sorted(grouped_examples):
        lines.append(f"### {group}")
        for item in grouped_examples[group]:
            text = item.assistant_text.replace("\n", " ")
            text = text[:280] + ("..." if len(text) > 280 else "")
            lines.append(f"- `{item.sample_id}`")
            lines.append(f"  - 命中词：`{', '.join(item.keyword_hits)}`")
            if item.repeated_lines:
                lines.append(f"  - 重复句：`{'; '.join(item.repeated_lines[:3])}`")
            if item.analysis_prefix_hit:
                lines.append("  - 命中 `analysis_prefix`")
            lines.append(f"  - 摘要：{text}")
        lines.append("")
    lines.append("## 五、建议")
    lines.append("")
    lines.append("- 先从训练集里删除或降权 `POP/下载/添加客服/签到福利/送彩金/活动参与` 相关 assistant 样本。")
    lines.append("- 对保留的活动类业务样本，统一改写为“查看官方规则/核实资格/不做外部引流”的客服口径。")
    lines.append("- 为 `analysis_prefix` 和 `重复短句` 单独构造 reject->chosen 修复集，后续做一轮 targeted SFT 或 DPO。")
    lines.append(f"- 详细污染样本清单见：`{output_dir / 'contaminated_samples.jsonl'}`")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", required=True, help="Training JSONL to audit.")
    parser.add_argument(
        "--output-dir",
        default="",
        help="Directory for summary/report outputs. Defaults to reports/marketing_audit_<ts>.",
    )
    args = parser.parse_args()

    input_path = Path(args.input_jsonl).expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else Path("/home/ubuntu/qwen35a3b_finetune/reports") / f"marketing_audit_{now_ts()}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = read_jsonl(input_path)
    total_samples = len(rows)
    contaminated: list[SampleAudit] = []
    group_counts: Counter[str] = Counter()
    keyword_counts: Counter[str] = Counter()
    repeated_line_counts: Counter[str] = Counter()

    for idx, sample in enumerate(rows, 1):
        assistant_text = extract_assistant_text(sample)
        if not assistant_text:
            continue
        groups, keyword_hits = find_keyword_hits(assistant_text)
        repeated_lines = find_repeated_lines(assistant_text)
        analysis_hit = analysis_prefix_hit(assistant_text)
        if not groups and not repeated_lines and not analysis_hit:
            continue

        sample_id = str(sample.get("id") or f"line_{idx}")
        item = SampleAudit(
            sample_id=sample_id,
            assistant_text=assistant_text,
            groups=groups,
            keyword_hits=keyword_hits,
            repeated_lines=repeated_lines,
            analysis_prefix_hit=analysis_hit,
            raw=sample,
        )
        contaminated.append(item)
        group_counts.update(groups)
        keyword_counts.update(keyword_hits)
        repeated_line_counts.update(repeated_lines)

    summary = {
        "input_jsonl": str(input_path),
        "total_samples": total_samples,
        "contaminated_samples": len(contaminated),
        "contaminated_ratio": round(len(contaminated) / max(total_samples, 1), 6),
        "group_counts": dict(group_counts),
        "keyword_top_50": keyword_counts.most_common(50),
        "repeated_line_top_50": repeated_line_counts.most_common(50),
        "analysis_prefix_samples": sum(1 for item in contaminated if item.analysis_prefix_hit),
        "repeated_line_samples": sum(1 for item in contaminated if item.repeated_lines),
    }

    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    with (output_dir / "contaminated_samples.jsonl").open("w", encoding="utf-8") as f:
        for item in contaminated:
            row = {
                "id": item.sample_id,
                "groups": item.groups,
                "keyword_hits": item.keyword_hits,
                "repeated_lines": item.repeated_lines,
                "analysis_prefix_hit": item.analysis_prefix_hit,
                "assistant_text": item.assistant_text,
                "raw": item.raw,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    report = build_report(
        input_path=input_path,
        total_samples=total_samples,
        contaminated=contaminated,
        group_counts=group_counts,
        keyword_counts=keyword_counts,
        repeated_line_counts=repeated_line_counts,
        output_dir=output_dir,
    )
    (output_dir / "REPORT.md").write_text(report, encoding="utf-8")

    print(f"[summary] {output_dir / 'summary.json'}")
    print(f"[report] {output_dir / 'REPORT.md'}")
    print(f"[samples] {output_dir / 'contaminated_samples.jsonl'}")


if __name__ == "__main__":
    main()
