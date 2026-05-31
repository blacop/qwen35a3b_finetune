#!/usr/bin/env python3
"""Clean POP / marketing / repetition pollution from customer-service SFT data."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def now_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


SEVERE_SAMPLE_PATTERNS = [
    re.compile(r"下载\s*POP", re.I),
    re.compile(r"POP聊天软件", re.I),
    re.compile(r"下载\s*APP", re.I),
    re.compile(r"添加客服"),
    re.compile(r"APP下载链接"),
    re.compile(r"Mee Yoo下载链接", re.I),
    re.compile(r"TT2\.COM", re.I),
    re.compile(r"包赔100%"),
    re.compile(r"红包雨"),
]

DROP_LINE_PATTERNS = [
    re.compile(r"签到福利"),
    re.compile(r"首存.*次存"),
    re.compile(r"大转盘"),
    re.compile(r"好运连连"),
    re.compile(r"财源广进"),
    re.compile(r"游戏愉快"),
    re.compile(r"参与.*活动"),
    re.compile(r"参加.*活动"),
    re.compile(r"今日上分"),
    re.compile(r"上分一笔领取"),
    re.compile(r"红包雨"),
    re.compile(r"开户链接"),
    re.compile(r"彩金已(为您)?派发"),
    re.compile(r"彩金已添加"),
    re.compile(r"祝您好运"),
    re.compile(r"极速到账"),
]

MILD_NOISE_PATTERNS = [
    re.compile(r"稍等(一下|片刻|一会|哦|呐|哟|哈|噢).*", re.I),
    re.compile(r"好的[♥~\-！!]*.*查询.*", re.I),
]


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


def normalize_line(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\s+", "", text)
    text = re.sub(r"[~!！。,.，?？]+$", "", text)
    return text


def assistant_messages(sample: dict[str, Any]) -> list[dict[str, Any]]:
    msgs = sample.get("messages")
    if not isinstance(msgs, list):
        return []
    return [msg for msg in msgs if isinstance(msg, dict) and msg.get("role") == "assistant"]


def user_messages(sample: dict[str, Any]) -> list[dict[str, Any]]:
    msgs = sample.get("messages")
    if not isinstance(msgs, list):
        return []
    return [msg for msg in msgs if isinstance(msg, dict) and msg.get("role") == "user"]


@dataclass
class CleanResult:
    sample_id: str
    action: str
    reasons: list[str]
    cleaned_sample: dict[str, Any] | None
    original_sample: dict[str, Any]


def sample_has_severe_pollution(sample: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    text = "\n".join(
        str(msg.get("content", "")) for msg in assistant_messages(sample) if isinstance(msg.get("content"), str)
    )
    for pattern in SEVERE_SAMPLE_PATTERNS:
        if pattern.search(text):
            reasons.append(f"severe:{pattern.pattern}")
    return reasons


def dedupe_lines(lines: list[str]) -> tuple[list[str], int]:
    seen: set[str] = set()
    out: list[str] = []
    removed = 0
    for line in lines:
        norm = normalize_line(line)
        if not norm:
            continue
        if norm in seen:
            removed += 1
            continue
        seen.add(norm)
        out.append(line.strip())
    return out, removed


def clean_assistant_text(text: str) -> tuple[str, list[str]]:
    reasons: list[str] = []
    raw_lines = [line.strip() for line in text.splitlines() if line.strip()]
    kept: list[str] = []
    for line in raw_lines:
        if any(pattern.search(line) for pattern in DROP_LINE_PATTERNS):
            reasons.append("drop_line:marketing")
            continue
        kept.append(line)

    deduped, removed_dup = dedupe_lines(kept)
    if removed_dup:
        reasons.append(f"dedupe_lines:{removed_dup}")

    # If only soft filler remains, keep only the first one.
    if len(deduped) >= 2:
        filler_flags = [any(p.match(line) for p in MILD_NOISE_PATTERNS) for line in deduped]
        if all(filler_flags):
            deduped = deduped[:1]
            reasons.append("collapse_filler_only")

    return "\n".join(deduped).strip(), reasons


def clean_sample(sample: dict[str, Any]) -> CleanResult:
    sample_id = str(sample.get("id") or "unknown")
    severe_reasons = sample_has_severe_pollution(sample)
    if severe_reasons:
        return CleanResult(
            sample_id=sample_id,
            action="drop",
            reasons=severe_reasons,
            cleaned_sample=None,
            original_sample=sample,
        )

    messages = sample.get("messages")
    if not isinstance(messages, list):
        return CleanResult(
            sample_id=sample_id,
            action="keep",
            reasons=[],
            cleaned_sample=sample,
            original_sample=sample,
        )

    new_messages: list[dict[str, Any]] = []
    reasons: list[str] = []
    assistant_kept_chars = 0
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        if msg.get("role") != "assistant" or not isinstance(msg.get("content"), str):
            new_messages.append(msg)
            continue

        cleaned_text, msg_reasons = clean_assistant_text(msg["content"])
        reasons.extend(msg_reasons)
        if cleaned_text:
            updated = dict(msg)
            updated["content"] = cleaned_text
            assistant_kept_chars += len(cleaned_text)
            new_messages.append(updated)

    if assistant_kept_chars == 0:
        return CleanResult(
            sample_id=sample_id,
            action="drop",
            reasons=reasons + ["empty_after_clean"],
            cleaned_sample=None,
            original_sample=sample,
        )

    # If the sample still ends up as pure filler without any substantive content, drop it.
    assistant_text = "\n".join(
        msg["content"] for msg in new_messages if isinstance(msg, dict) and msg.get("role") == "assistant"
    )
    if len(assistant_text) < 12 and user_messages(sample):
        return CleanResult(
            sample_id=sample_id,
            action="drop",
            reasons=reasons + ["too_short_after_clean"],
            cleaned_sample=None,
            original_sample=sample,
        )

    cleaned = dict(sample)
    cleaned["messages"] = new_messages
    action = "modify" if reasons else "keep"
    return CleanResult(
        sample_id=sample_id,
        action=action,
        reasons=reasons,
        cleaned_sample=cleaned,
        original_sample=sample,
    )


def write_report(
    input_path: Path,
    output_path: Path,
    summary: dict[str, Any],
    dropped_path: Path,
    modified_path: Path,
    report_path: Path,
) -> None:
    lines: list[str] = []
    lines.append("# GPU5 去污导出报告")
    lines.append("")
    lines.append(f"- 输入文件：`{input_path}`")
    lines.append(f"- 输出文件：`{output_path}`")
    lines.append(f"- 被删除样本：`{dropped_path}`")
    lines.append(f"- 被修改样本：`{modified_path}`")
    lines.append("")
    lines.append("## 一、总体统计")
    lines.append("")
    lines.append(f"- 原始样本数：**{summary['input_samples']}**")
    lines.append(f"- 导出样本数：**{summary['output_samples']}**")
    lines.append(f"- 删除样本数：**{summary['dropped_samples']}** ({summary['dropped_ratio']:.2%})")
    lines.append(f"- 修改样本数：**{summary['modified_samples']}** ({summary['modified_ratio']:.2%})")
    lines.append(f"- 保留不变样本数：**{summary['kept_samples']}**")
    lines.append("")
    lines.append("## 二、删除原因 Top 20")
    lines.append("")
    lines.append("| 原因 | 次数 |")
    lines.append("|---|---:|")
    for reason, count in summary["drop_reason_top_20"]:
        lines.append(f"| {reason} | {count} |")
    lines.append("")
    lines.append("## 三、修改原因 Top 20")
    lines.append("")
    lines.append("| 原因 | 次数 |")
    lines.append("|---|---:|")
    for reason, count in summary["modify_reason_top_20"]:
        lines.append(f"| {reason} | {count} |")
    lines.append("")
    lines.append("## 四、建议")
    lines.append("")
    lines.append("- 当前这版清洗以“删除强引流样本 + 删除强营销行 + 行级去重”为主，适合先做去毒基线。")
    lines.append("- 活动/彩金相关业务能力会同步下降，后续应用专项 repair 数据回灌合规版本。")
    lines.append("- 下一步建议直接用清洗后的数据再叠加 `repair_pop_marketing_repetition_sft_template.jsonl` 做 targeted SFT。")
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument(
        "--output-jsonl",
        default="",
        help="Cleaned dataset output. Defaults to sibling file with .decontam.jsonl suffix.",
    )
    parser.add_argument(
        "--report-dir",
        default="",
        help="Directory for summary/report/removed samples. Defaults to reports/marketing_clean_<ts>.",
    )
    args = parser.parse_args()

    input_path = Path(args.input_jsonl).expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    output_path = (
        Path(args.output_jsonl).expanduser().resolve()
        if args.output_jsonl
        else input_path.with_name(input_path.stem + ".decontam.jsonl")
    )
    report_dir = (
        Path(args.report_dir).expanduser().resolve()
        if args.report_dir
        else Path("/home/ubuntu/qwen35a3b_finetune/reports") / f"marketing_clean_{now_ts()}"
    )
    report_dir.mkdir(parents=True, exist_ok=True)

    rows = read_jsonl(input_path)
    cleaned_rows: list[dict[str, Any]] = []
    dropped_rows: list[dict[str, Any]] = []
    modified_rows: list[dict[str, Any]] = []

    drop_reason_counter: Counter[str] = Counter()
    modify_reason_counter: Counter[str] = Counter()

    for row in rows:
        result = clean_sample(row)
        if result.action == "drop":
            dropped_rows.append(
                {
                    "id": result.sample_id,
                    "reasons": result.reasons,
                    "raw": result.original_sample,
                }
            )
            drop_reason_counter.update(result.reasons)
            continue

        assert result.cleaned_sample is not None
        cleaned_rows.append(result.cleaned_sample)
        if result.action == "modify":
            modified_rows.append(
                {
                    "id": result.sample_id,
                    "reasons": result.reasons,
                    "raw": result.original_sample,
                    "cleaned": result.cleaned_sample,
                }
            )
            modify_reason_counter.update(result.reasons)

    with output_path.open("w", encoding="utf-8") as f:
        for row in cleaned_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    dropped_path = report_dir / "dropped_samples.jsonl"
    with dropped_path.open("w", encoding="utf-8") as f:
        for row in dropped_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    modified_path = report_dir / "modified_samples.jsonl"
    with modified_path.open("w", encoding="utf-8") as f:
        for row in modified_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "input_jsonl": str(input_path),
        "output_jsonl": str(output_path),
        "input_samples": len(rows),
        "output_samples": len(cleaned_rows),
        "dropped_samples": len(dropped_rows),
        "dropped_ratio": len(dropped_rows) / max(len(rows), 1),
        "modified_samples": len(modified_rows),
        "modified_ratio": len(modified_rows) / max(len(rows), 1),
        "kept_samples": len(cleaned_rows) - len(modified_rows),
        "drop_reason_top_20": drop_reason_counter.most_common(20),
        "modify_reason_top_20": modify_reason_counter.most_common(20),
    }
    summary_path = report_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    report_path = report_dir / "REPORT.md"
    write_report(
        input_path=input_path,
        output_path=output_path,
        summary=summary,
        dropped_path=dropped_path,
        modified_path=modified_path,
        report_path=report_path,
    )

    print(f"[cleaned] {output_path}")
    print(f"[summary] {summary_path}")
    print(f"[report] {report_path}")
    print(f"[dropped] {dropped_path}")
    print(f"[modified] {modified_path}")


if __name__ == "__main__":
    main()
