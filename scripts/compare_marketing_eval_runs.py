#!/usr/bin/env python3
"""Compare two marketing eval runs and emit REPORT.md + diff_cases.csv."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def fmt_float(value: Any, ndigits: int = 4) -> str:
    try:
        return f"{float(value):.{ndigits}f}"
    except Exception:
        return "-"


def fmt_delta(a_value: Any, b_value: Any, ndigits: int = 4) -> str:
    try:
        delta = float(a_value) - float(b_value)
    except Exception:
        return "-"
    sign = "+" if delta > 0 else ""
    arrow = " ↑" if delta > 1e-9 else (" ↓" if delta < -1e-9 else " ·")
    return f"{sign}{delta:.{ndigits}f}{arrow}"


def case_score(pred: dict[str, Any]) -> float:
    request_ok = float(pred.get("request_ok", 0))
    must_include = float(pred.get("must_include_score", 0.0))
    forbidden = float(pred.get("forbidden_hit", 0))
    repetition = float(pred.get("repetition_hit", 0))
    analysis_prefix = float(pred.get("analysis_prefix_hit", 0))
    think_leak = float(pred.get("think_leak_hit", 0))
    return request_ok * 2.0 + must_include - forbidden - repetition - analysis_prefix - think_leak


def case_pass(pred: dict[str, Any]) -> bool:
    return (
        int(pred.get("request_ok", 0)) == 1
        and int(pred.get("forbidden_hit", 0)) == 0
        and int(pred.get("repetition_hit", 0)) == 0
        and int(pred.get("analysis_prefix_hit", 0)) == 0
        and int(pred.get("think_leak_hit", 0)) == 0
    )


def render_case(
    sample_id: str,
    winner: dict[str, Any],
    loser: dict[str, Any],
    winner_name: str,
    loser_name: str,
) -> str:
    lines = [
        f"### {sample_id} — {winner.get('scenario', '')}",
        f"- query: {winner.get('user_query', '')}",
        (
            f"- {winner_name}: score={case_score(winner):.4f} "
            f"must_include={float(winner.get('must_include_score', 0.0)):.4f} "
            f"forbidden={winner.get('forbidden_hit', 0)} "
            f"repetition={winner.get('repetition_hit', 0)} "
            f"analysis_prefix={winner.get('analysis_prefix_hit', 0)} "
            f"think_leak={winner.get('think_leak_hit', 0)}"
        ),
        f"- {winner_name} answer: {(winner.get('answer') or '').strip()}",
        (
            f"- {loser_name}: score={case_score(loser):.4f} "
            f"must_include={float(loser.get('must_include_score', 0.0)):.4f} "
            f"forbidden={loser.get('forbidden_hit', 0)} "
            f"repetition={loser.get('repetition_hit', 0)} "
            f"analysis_prefix={loser.get('analysis_prefix_hit', 0)} "
            f"think_leak={loser.get('think_leak_hit', 0)}"
        ),
        f"- {loser_name} answer: {(loser.get('answer') or '').strip()}",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--a-dir", required=True)
    parser.add_argument("--a-name", default="A")
    parser.add_argument("--b-dir", required=True)
    parser.add_argument("--b-name", default="B")
    parser.add_argument("--output", required=True)
    parser.add_argument("--bad-case-limit", type=int, default=15)
    args = parser.parse_args()

    a_dir = Path(args.a_dir).expanduser().resolve()
    b_dir = Path(args.b_dir).expanduser().resolve()

    a_summary = json.loads((a_dir / "summary.json").read_text(encoding="utf-8"))
    b_summary = json.loads((b_dir / "summary.json").read_text(encoding="utf-8"))
    a_predictions = {row["id"]: row for row in load_jsonl(a_dir / "predictions.jsonl")}
    b_predictions = {row["id"]: row for row in load_jsonl(b_dir / "predictions.jsonl")}

    common_ids = sorted(set(a_predictions) & set(b_predictions))
    a_better: list[str] = []
    b_better: list[str] = []
    ties: list[str] = []
    both_pass: list[str] = []
    both_fail: list[str] = []

    for sample_id in common_ids:
        a_pred = a_predictions[sample_id]
        b_pred = b_predictions[sample_id]
        a_score = case_score(a_pred)
        b_score = case_score(b_pred)
        if abs(a_score - b_score) <= 1e-9:
            ties.append(sample_id)
        elif a_score > b_score:
            a_better.append(sample_id)
        else:
            b_better.append(sample_id)

        a_pass = case_pass(a_pred)
        b_pass = case_pass(b_pred)
        if a_pass and b_pass:
            both_pass.append(sample_id)
        elif (not a_pass) and (not b_pass):
            both_fail.append(sample_id)

    metric_rows = [
        ("request_ok_rate", "请求成功率", "↑"),
        ("forbidden_hit_rate", "禁词泄漏率", "↓"),
        ("repetition_hit_rate", "复读命中率", "↓"),
        ("analysis_prefix_hit_rate", "分析前缀命中率", "↓"),
        ("think_leak_hit_rate", "思维外溢命中率", "↓"),
        ("must_include_avg", "must_include 平均命中率", "↑"),
        ("failure_count", "失败样本数", "↓"),
        ("avg_completion_chars", "平均回复长度", "≈"),
    ]

    a_by_category = a_summary.get("by_category", {})
    b_by_category = b_summary.get("by_category", {})
    categories = sorted(set(a_by_category) | set(b_by_category))

    lines: list[str] = []
    lines.append(f"# POP / 营销 / 复读专项对比报告：{args.a_name} vs {args.b_name}")
    lines.append("")
    lines.append(f"- A = `{args.a_name}` (`{a_dir}`)")
    lines.append(f"- B = `{args.b_name}` (`{b_dir}`)")
    lines.append(f"- 对齐样本数：**{len(common_ids)}**")
    lines.append("")
    lines.append("## 一、总体指标")
    lines.append("")
    lines.append(f"| 指标 | 方向 | {args.a_name} | {args.b_name} | Δ (A - B) |")
    lines.append("|---|:---:|---:|---:|---:|")
    for key, label, direction in metric_rows:
        lines.append(
            f"| {label} | {direction} | {fmt_float(a_summary.get(key))} | "
            f"{fmt_float(b_summary.get(key))} | {fmt_delta(a_summary.get(key), b_summary.get(key))} |"
        )

    lines.append("")
    lines.append("## 二、逐条胜负")
    lines.append("")
    total = max(len(common_ids), 1)
    lines.append(f"- `{args.a_name}` 独胜：{len(a_better)} 条 ({len(a_better) / total:.1%})")
    lines.append(f"- `{args.b_name}` 独胜：{len(b_better)} 条 ({len(b_better) / total:.1%})")
    lines.append(f"- 平手：{len(ties)} 条 ({len(ties) / total:.1%})")
    lines.append(f"- 双方都 pass：{len(both_pass)} 条 ({len(both_pass) / total:.1%})")
    lines.append(f"- 双方都 fail：{len(both_fail)} 条 ({len(both_fail) / total:.1%})")
    if a_better or b_better:
        decisive = len(a_better) + len(b_better)
        lines.append(
            f"- 去平手后胜率：`{args.a_name}` = {len(a_better) / decisive:.1%}，"
            f"`{args.b_name}` = {len(b_better) / decisive:.1%}"
        )

    lines.append("")
    lines.append("## 三、分类指标")
    lines.append("")
    lines.append(
        f"| 分类 | count | {args.a_name} forbidden | {args.b_name} forbidden | Δ | "
        f"{args.a_name} repetition | {args.b_name} repetition | Δ | "
        f"{args.a_name} think | {args.b_name} think | Δ | "
        f"{args.a_name} must_include | {args.b_name} must_include | Δ |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for category in categories:
        a_row = a_by_category.get(category, {})
        b_row = b_by_category.get(category, {})
        count = a_row.get("count", b_row.get("count", "-"))
        lines.append(
            f"| {category} | {count} | "
            f"{fmt_float(a_row.get('forbidden_hit_rate'))} | {fmt_float(b_row.get('forbidden_hit_rate'))} | {fmt_delta(a_row.get('forbidden_hit_rate'), b_row.get('forbidden_hit_rate'))} | "
            f"{fmt_float(a_row.get('repetition_hit_rate'))} | {fmt_float(b_row.get('repetition_hit_rate'))} | {fmt_delta(a_row.get('repetition_hit_rate'), b_row.get('repetition_hit_rate'))} | "
            f"{fmt_float(a_row.get('think_leak_hit_rate'))} | {fmt_float(b_row.get('think_leak_hit_rate'))} | {fmt_delta(a_row.get('think_leak_hit_rate'), b_row.get('think_leak_hit_rate'))} | "
            f"{fmt_float(a_row.get('must_include_avg'))} | {fmt_float(b_row.get('must_include_avg'))} | {fmt_delta(a_row.get('must_include_avg'), b_row.get('must_include_avg'))} |"
        )

    limit = args.bad_case_limit
    lines.append("")
    lines.append(f"## 四、{args.a_name} 独胜 Top {limit}")
    lines.append("")
    for sample_id in a_better[:limit]:
        lines.append(render_case(sample_id, a_predictions[sample_id], b_predictions[sample_id], args.a_name, args.b_name))

    lines.append("")
    lines.append(f"## 五、{args.b_name} 独胜 Top {limit}")
    lines.append("")
    for sample_id in b_better[:limit]:
        lines.append(render_case(sample_id, b_predictions[sample_id], a_predictions[sample_id], args.b_name, args.a_name))

    lines.append("")
    lines.append(f"## 六、双方都 fail Top {limit}")
    lines.append("")
    for sample_id in both_fail[:limit]:
        a_pred = a_predictions[sample_id]
        b_pred = b_predictions[sample_id]
        lines.append(render_case(sample_id, a_pred, b_pred, args.a_name, args.b_name))

    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    csv_path = output_path.parent / "diff_cases.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "id",
                "bucket",
                "category",
                "scenario",
                f"{args.a_name}_score",
                f"{args.a_name}_must_include",
                f"{args.a_name}_forbidden",
                f"{args.a_name}_repetition",
                f"{args.a_name}_analysis_prefix",
                f"{args.a_name}_think_leak",
                f"{args.b_name}_score",
                f"{args.b_name}_must_include",
                f"{args.b_name}_forbidden",
                f"{args.b_name}_repetition",
                f"{args.b_name}_analysis_prefix",
                f"{args.b_name}_think_leak",
                "user_query",
            ]
        )

        def write_bucket(bucket: str, ids: list[str]) -> None:
            for sample_id in ids:
                a_pred = a_predictions[sample_id]
                b_pred = b_predictions[sample_id]
                writer.writerow(
                    [
                        sample_id,
                        bucket,
                        a_pred.get("category", ""),
                        a_pred.get("scenario", ""),
                        f"{case_score(a_pred):.4f}",
                        a_pred.get("must_include_score", 0.0),
                        a_pred.get("forbidden_hit", 0),
                        a_pred.get("repetition_hit", 0),
                        a_pred.get("analysis_prefix_hit", 0),
                        a_pred.get("think_leak_hit", 0),
                        f"{case_score(b_pred):.4f}",
                        b_pred.get("must_include_score", 0.0),
                        b_pred.get("forbidden_hit", 0),
                        b_pred.get("repetition_hit", 0),
                        b_pred.get("analysis_prefix_hit", 0),
                        b_pred.get("think_leak_hit", 0),
                        a_pred.get("user_query", ""),
                    ]
                )

        write_bucket("a_better", a_better)
        write_bucket("b_better", b_better)
        write_bucket("both_fail", both_fail)

    print(f"[report] {output_path}")
    print(f"[diff csv] {csv_path}")


if __name__ == "__main__":
    main()
