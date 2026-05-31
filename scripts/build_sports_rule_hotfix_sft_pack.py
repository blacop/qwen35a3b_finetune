#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path("/home/ubuntu/qwen35a3b_finetune")
DEFAULT_BASE = (
    PROJECT_ROOT
    / "datasets"
    / "nohallucination_hotfix_pack_20260425"
    / "sft_openai_messages.cleaned.v3_merged.single75_exact.repair_v3.tydata_pdfcurated.nometa.marketingfix.opsclean.20260425.nohallucination_hotfix_merged.jsonl"
)
DEFAULT_RULE_DISTILL = (
    PROJECT_ROOT
    / "datasets"
    / "sports_rule_distill_gpu7_to_gpu5_20260428"
    / "sft_openai_messages.jsonl"
)
DEFAULT_OUT_DIR = PROJECT_ROOT / "datasets" / "sports_rule_hotfix_pack_20260428"


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
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
            if not isinstance(obj, dict):
                raise ValueError(f"Expected object at {path}:{line_no}")
            rows.append(obj)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_messages(row: dict[str, Any]) -> list[dict[str, str]]:
    messages = row.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError(f"missing messages: {row.get('id')}")
    out: list[dict[str, str]] = []
    for msg in messages:
        if not isinstance(msg, dict):
            raise ValueError(f"invalid message object: {row.get('id')}")
        role = str(msg.get("role", "")).strip()
        content = str(msg.get("content", "")).strip()
        if role not in {"system", "user", "assistant"}:
            raise ValueError(f"invalid role {role!r}: {row.get('id')}")
        if not content:
            raise ValueError(f"empty content: {row.get('id')}")
        out.append({"role": role, "content": content})
    return out


def signature(messages: list[dict[str, str]]) -> str:
    payload = json.dumps(messages, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def normalized_row(row: dict[str, Any], sample_id: str) -> dict[str, Any]:
    return {
        "id": str(row.get("id") or sample_id),
        "split": str(row.get("split") or "train"),
        "messages": normalize_messages(row),
        "meta": row.get("meta", {}),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build GPU5 sports-rule hotfix SFT pack.")
    parser.add_argument("--base-jsonl", default=str(DEFAULT_BASE))
    parser.add_argument("--rule-distill-jsonl", default=str(DEFAULT_RULE_DISTILL))
    parser.add_argument("--output-jsonl", default=str(DEFAULT_OUT_DIR / "sft_sports_rule_hotfix_merged.jsonl"))
    parser.add_argument("--summary-json", default=str(DEFAULT_OUT_DIR / "summary.json"))
    parser.add_argument(
        "--rule-repeat",
        type=int,
        default=40,
        help="Repeat corrected rule samples to give the small patch enough training weight.",
    )
    args = parser.parse_args()

    base_path = Path(args.base_jsonl).expanduser().resolve()
    rule_path = Path(args.rule_distill_jsonl).expanduser().resolve()
    output_path = Path(args.output_jsonl).expanduser().resolve()
    summary_path = Path(args.summary_json).expanduser().resolve()

    base_rows_raw = read_jsonl(base_path)
    rule_rows_raw = read_jsonl(rule_path)

    merged: list[dict[str, Any]] = []
    seen: set[str] = set()
    base_added = 0
    base_skipped = 0
    for idx, row in enumerate(base_rows_raw, 1):
        norm = normalized_row(row, f"base_{idx:06d}")
        sig = signature(norm["messages"])
        if sig in seen:
            base_skipped += 1
            continue
        seen.add(sig)
        merged.append(norm)
        base_added += 1

    rule_added = 0
    rule_skipped = 0
    repeat = max(1, args.rule_repeat)
    for rep in range(repeat):
        for idx, row in enumerate(rule_rows_raw, 1):
            norm = normalized_row(row, f"rule_{idx:04d}")
            norm["id"] = f"{norm['id']}_repeat{rep + 1:02d}"
            meta = dict(norm.get("meta") or {})
            meta["sports_rule_hotfix_repeat"] = rep + 1
            meta["sports_rule_hotfix_source_id"] = row.get("id")
            norm["meta"] = meta
            # Keep repeats intentionally; only skip exact id/signature duplicates
            # inside the same repeat would indicate malformed input.
            sig = signature(norm["messages"] + [{"role": "system", "content": f"repeat={rep + 1}"}])
            if sig in seen:
                rule_skipped += 1
                continue
            seen.add(sig)
            merged.append(norm)
            rule_added += 1

    write_jsonl(output_path, merged)
    summary = {
        "base_jsonl": str(base_path),
        "rule_distill_jsonl": str(rule_path),
        "output_jsonl": str(output_path),
        "base_rows_raw": len(base_rows_raw),
        "base_rows_added": base_added,
        "base_rows_skipped": base_skipped,
        "rule_rows_raw": len(rule_rows_raw),
        "rule_repeat": repeat,
        "rule_rows_added": rule_added,
        "rule_rows_skipped": rule_skipped,
        "total_rows": len(merged),
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
