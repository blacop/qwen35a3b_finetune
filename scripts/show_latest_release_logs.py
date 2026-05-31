#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


DEFAULT_INDEX = Path("/home/ubuntu/qwen35a3b_finetune/tracking/logs/release_log_index.jsonl")


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def filter_rows(rows: List[Dict[str, Any]], pipeline: str) -> List[Dict[str, Any]]:
    if not pipeline:
        return rows
    return [row for row in rows if str(row.get("pipeline", "")) == pipeline]


def sort_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(rows, key=lambda row: str(row.get("ts", "")), reverse=True)


def filter_by_time(rows: List[Dict[str, Any]], since: str, until: str) -> List[Dict[str, Any]]:
    filtered: List[Dict[str, Any]] = []
    for row in rows:
        ts = str(row.get("ts", ""))
        if since and ts < since:
            continue
        if until and ts > until:
            continue
        filtered.append(row)
    return filtered


def format_row(row: Dict[str, Any]) -> str:
    lines = [
        f"ts: {row.get('ts', '')}",
        f"pipeline: {row.get('pipeline', '')}",
    ]
    for key in [
        "model_dir",
        "merged_model_dir",
        "served_model_name",
        "ab_summary_path",
        "release_gate_summary_path",
        "knowledge_gate_out_dir",
        "knowledge_gate_gpu5_summary",
        "knowledge_gate_gpu7_summary",
        "knowledge_gate_kb40_summary",
    ]:
        value = row.get(key)
        if value:
            lines.append(f"{key}: {value}")
    extras = {
        k: v
        for k, v in row.items()
        if k
        not in {
            "ts",
            "pipeline",
            "model_dir",
            "merged_model_dir",
            "served_model_name",
            "ab_summary_path",
            "release_gate_summary_path",
            "knowledge_gate_out_dir",
            "knowledge_gate_gpu5_summary",
            "knowledge_gate_gpu7_summary",
            "knowledge_gate_kb40_summary",
        }
    }
    if extras:
        lines.append(f"extra: {json.dumps(extras, ensure_ascii=False, sort_keys=True)}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Show latest release log index entries with summary paths.")
    parser.add_argument("--index-file", default=str(DEFAULT_INDEX))
    parser.add_argument("--pipeline", default="")
    parser.add_argument("--since", default="", help="Only include rows with ts >= this value, e.g. 20260508T000000Z")
    parser.add_argument("--until", default="", help="Only include rows with ts <= this value, e.g. 20260508T235959Z")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--json", action="store_true", help="Print matched rows as JSON instead of plain text.")
    args = parser.parse_args()

    index_path = Path(args.index_file).expanduser().resolve()
    rows = read_jsonl(index_path)
    rows = filter_rows(rows, args.pipeline)
    rows = filter_by_time(rows, args.since, args.until)
    rows = sort_rows(rows)
    rows = rows[: max(args.limit, 0)]

    if args.json:
        print(json.dumps(rows, ensure_ascii=False, indent=2))
        return

    print(f"index_file: {index_path}")
    print(f"matched: {len(rows)}")
    if not rows:
        print("no release records found")
        return
    for idx, row in enumerate(rows, 1):
        print(f"\n[{idx}]")
        print(format_row(row))


if __name__ == "__main__":
    main()
