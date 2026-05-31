#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict


def maybe_set(row: Dict[str, Any], key: str, value: str) -> None:
    if value:
        row[key] = value


def main() -> None:
    parser = argparse.ArgumentParser(description="Append one release record to tracking/logs/release_log_index.jsonl")
    parser.add_argument("--index-file", required=True)
    parser.add_argument("--ts", required=True)
    parser.add_argument("--pipeline", required=True)
    parser.add_argument("--model-dir", default="")
    parser.add_argument("--merged-model-dir", default="")
    parser.add_argument("--served-model-name", default="")
    parser.add_argument("--ab-summary-path", default="")
    parser.add_argument("--release-gate-summary-path", default="")
    parser.add_argument("--knowledge-gate-out-dir", default="")
    parser.add_argument("--knowledge-gate-gpu5-summary", default="")
    parser.add_argument("--knowledge-gate-gpu7-summary", default="")
    parser.add_argument("--knowledge-gate-kb40-summary", default="")
    parser.add_argument("--extra-json", default="")
    args = parser.parse_args()

    row: Dict[str, Any] = {
        "ts": args.ts,
        "pipeline": args.pipeline,
    }
    maybe_set(row, "model_dir", args.model_dir)
    maybe_set(row, "merged_model_dir", args.merged_model_dir)
    maybe_set(row, "served_model_name", args.served_model_name)
    maybe_set(row, "ab_summary_path", args.ab_summary_path)
    maybe_set(row, "release_gate_summary_path", args.release_gate_summary_path)
    maybe_set(row, "knowledge_gate_out_dir", args.knowledge_gate_out_dir)
    maybe_set(row, "knowledge_gate_gpu5_summary", args.knowledge_gate_gpu5_summary)
    maybe_set(row, "knowledge_gate_gpu7_summary", args.knowledge_gate_gpu7_summary)
    maybe_set(row, "knowledge_gate_kb40_summary", args.knowledge_gate_kb40_summary)

    if args.extra_json:
        extra = json.loads(args.extra_json)
        if not isinstance(extra, dict):
            raise SystemExit("--extra-json must decode to a JSON object")
        row.update(extra)

    path = Path(args.index_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
