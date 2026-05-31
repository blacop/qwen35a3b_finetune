#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path("/home/ubuntu/qwen35a3b_finetune")
DEFAULT_BASE = (
    ROOT
    / "datasets"
    / "weak_intent_targeted_12k_20260416T055426Z"
    / "sft_openai_messages.cleaned.v3_merged.single75_exact.repair_v3.tydata_pdfcurated.nometa.weak12k_merged.decontam.jsonl"
)
DEFAULT_REPAIR = ROOT / "templates" / "repair_pop_marketing_repetition_sft_template.jsonl"


def now_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at line {line_no} in {path}: {exc}") from exc
            if not isinstance(obj, dict):
                raise ValueError(f"Line {line_no} in {path} is not a JSON object.")
            rows.append(obj)
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def merge_by_id(base_rows: list[dict[str, Any]], repair_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int, int]:
    merged = list(base_rows)
    seen_ids = {row.get("id") for row in base_rows if row.get("id")}
    added = 0
    skipped_duplicates = 0

    for row in repair_rows:
        row_id = row.get("id")
        if row_id and row_id in seen_ids:
            skipped_duplicates += 1
            continue
        merged.append(row)
        if row_id:
            seen_ids.add(row_id)
        added += 1

    return merged, added, skipped_duplicates


def build_summary(
    base_path: Path,
    repair_path: Path,
    output_path: Path,
    summary_path: Path,
    base_rows: list[dict[str, Any]],
    repair_rows: list[dict[str, Any]],
    merged_rows: list[dict[str, Any]],
    added: int,
    skipped_duplicates: int,
) -> dict[str, Any]:
    return {
        "generated_at": now_ts(),
        "base_dataset": str(base_path),
        "repair_template": str(repair_path),
        "output_dataset": str(output_path),
        "summary_path": str(summary_path),
        "counts": {
            "base": len(base_rows),
            "repair_template": len(repair_rows),
            "added": added,
            "skipped_duplicates": skipped_duplicates,
            "output": len(merged_rows),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge decontaminated GPU5 SFT data with POP/marketing/repetition repair samples.")
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE, help="Base cleaned SFT JSONL.")
    parser.add_argument("--repair", type=Path, default=DEFAULT_REPAIR, help="Repair template JSONL.")
    parser.add_argument("--output", type=Path, help="Output merged JSONL path.")
    parser.add_argument("--summary", type=Path, help="Output summary JSON path.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="If set, auto-generate output/summary filenames under this directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    base_path: Path = args.base
    repair_path: Path = args.repair

    if not base_path.exists():
        raise FileNotFoundError(f"Missing base dataset: {base_path}")
    if not repair_path.exists():
        raise FileNotFoundError(f"Missing repair template: {repair_path}")

    if args.output_dir:
        output_dir = args.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{base_path.stem}.marketing_repair_merged.jsonl"
        summary_path = output_dir / "marketing_repair_merge_summary.json"
    else:
        output_path = args.output or base_path.with_name(f"{base_path.stem}.marketing_repair_merged.jsonl")
        summary_path = args.summary or output_path.with_suffix(".summary.json")

    base_rows = read_jsonl(base_path)
    repair_rows = read_jsonl(repair_path)
    merged_rows, added, skipped_duplicates = merge_by_id(base_rows, repair_rows)

    write_jsonl(output_path, merged_rows)

    summary = build_summary(
        base_path=base_path,
        repair_path=repair_path,
        output_path=output_path,
        summary_path=summary_path,
        base_rows=base_rows,
        repair_rows=repair_rows,
        merged_rows=merged_rows,
        added=added,
        skipped_duplicates=skipped_duplicates,
    )
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
