#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List


DEFAULT_INPUT = "/home/ubuntu/qwen35a3b_finetune/datasets/ops_policy_platform_abc_regression_20260520.jsonl"


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            raw = line.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid jsonl at line {line_no}: {exc}") from exc
            if not isinstance(obj, dict):
                raise ValueError(f"jsonl row at line {line_no} must be an object")
            rows.append(obj)
    return rows


def dump_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def text_from_messages(messages: Any, role: str) -> str:
    if not isinstance(messages, list):
        return ""
    for msg in messages:
        if isinstance(msg, dict) and str(msg.get("role", "")).strip() == role:
            return str(msg.get("content", "") or "")
    return ""


def to_json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def row_to_csv(row: Dict[str, Any]) -> Dict[str, str]:
    messages = row.get("messages") or []
    meta = row.get("meta") or {}
    platform_pattern = str(row.get("platform_pattern") or "")
    system_prompt = text_from_messages(messages, "system")
    user_query = str(row.get("user_query") or text_from_messages(messages, "user"))
    gold_answer = str(row.get("gold_answer") or text_from_messages(messages, "assistant"))
    must_include = row.get("must_include") or []
    must_not_include = row.get("must_not_include") or []
    tags = row.get("tags") or []

    return {
        "id": str(row.get("id") or ""),
        "split": str(row.get("split") or ""),
        "scenario": str(row.get("scenario") or ""),
        "platform_pattern": platform_pattern,
        "proxy_mode": str(meta.get("proxy_mode") or ""),
        "settlement_mode": str(meta.get("settlement_mode") or ""),
        "needs_backend": "true" if bool(meta.get("needs_backend")) else "false",
        "priority": str(row.get("priority") or ""),
        "query": user_query,
        "answer": gold_answer,
        "user_query": user_query,
        "gold_answer": gold_answer,
        "system_prompt": system_prompt,
        "must_include_json": to_json_text(must_include),
        "must_not_include_json": to_json_text(must_not_include),
        "tags_json": to_json_text(tags),
        "messages_json": to_json_text(messages),
        "meta_json": to_json_text(meta),
    }


def write_csv(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    rows = list(rows)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    csv_rows = [row_to_csv(row) for row in rows]
    fieldnames = list(csv_rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Split ops_policy platform ABC regression set into A/B/C JSONL files and a flat CSV export.",
    )
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Combined JSONL input file")
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory. Defaults to the input file directory.",
    )
    parser.add_argument(
        "--csv-name",
        default="ops_policy_platform_abc_regression_20260520.csv",
        help="CSV output filename",
    )
    args = parser.parse_args()

    input_path = Path(args.input).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else input_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_jsonl(input_path)
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("platform_pattern") or "").strip()].append(row)

    outputs = {}
    for platform in ("A", "B", "C"):
        platform_rows = grouped.get(platform, [])
        jsonl_path = out_dir / f"{input_path.stem}_{platform}.jsonl"
        dump_jsonl(jsonl_path, platform_rows)
        outputs[platform] = {"jsonl": str(jsonl_path), "count": len(platform_rows)}

    csv_path = out_dir / args.csv_name
    write_csv(csv_path, rows)

    summary = {
        "input": str(input_path),
        "out_dir": str(out_dir),
        "total": len(rows),
        "outputs": outputs,
        "csv": str(csv_path),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
