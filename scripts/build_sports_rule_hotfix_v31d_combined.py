#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path("/home/ubuntu/qwen35a3b_finetune")
STORAGE_ROOT = Path("/video-storage/ai-customer/qwen35a3b_finetune")

DEFAULT_DATASETS = [
    # v3 base JSON-aligned dataset
    str(STORAGE_ROOT / "datasets" / "sports_rule_hotfix_v3_json_20260429T071807Z" / "sft_sports_rule_hotfix_v3_json.jsonl"),
    # v31c patch dataset
    str(STORAGE_ROOT / "datasets" / "sports_rule_hotfix_v31c_json_mixed_patch_20260429T100809Z" / "sft_sports_rule_hotfix_v31c_mixed.jsonl"),
]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc
    return rows


def ensure_no_thinking_in_assistant(row: dict[str, Any]) -> dict[str, Any]:
    """Ensure assistant message contains only JSON, no thinking process"""
    messages = row.get("messages", [])
    for msg in messages:
        if msg.get("role") == "assistant":
            content = str(msg.get("content", "")).strip()
            # Remove any thinking prefix if present
            if content.lower().startswith("think") or "<think>" in content or "thinking" in content.lower():
                # Try to extract JSON from the response
                import re
                json_match = re.search(r"\{.*\}", content, re.DOTALL)
                if json_match:
                    msg["content"] = json_match.group(0)
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Build v31d combined SFT dataset (v3 base + v31c patches)")
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--output-jsonl", default=str(STORAGE_ROOT / "datasets" / "sports_rule_hotfix_v31d_combined_json_20260430" / "sft_sports_rule_hotfix_v31d_combined.jsonl"))
    args = parser.parse_args()

    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    for dataset_path in args.datasets:
        path = Path(dataset_path)
        if not path.exists():
            print(f"[WARN] Skipping missing dataset: {path}")
            continue

        rows = read_jsonl(path)
        print(f"[INFO] Loaded {len(rows)} rows from {path.name}")

        for row in rows:
            row_id = str(row.get("id", ""))
            if row_id and row_id in seen_ids:
                continue
            seen_ids.add(row_id)
            row = ensure_no_thinking_in_assistant(row)
            all_rows.append(row)

    with output_path.open("w", encoding="utf-8") as f:
        for row in all_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "output_jsonl": str(output_path),
        "total_rows": len(all_rows),
        "datasets": args.datasets,
    }

    summary_path = output_path.parent / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
