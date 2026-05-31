#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


PROJECT_ROOT = Path("/home/ubuntu/qwen35a3b_finetune")
RELATION_INPUT = PROJECT_ROOT / "datasets" / "relation_patch_pack" / "vl_relation_patch_v2.swift_vl.jsonl"
DEFMIS_INPUT = PROJECT_ROOT / "datasets" / "relation_patch_pack" / "vl_definition_misconception_patch_v1.swift_vl.jsonl"
DEFAULT_OUTPUT = PROJECT_ROOT / "datasets" / "relation_patch_pack" / "vl_balanced_patch_v1.swift_vl.jsonl"
DEFAULT_SUMMARY = PROJECT_ROOT / "datasets" / "relation_patch_pack" / "vl_balanced_patch_v1.summary.json"

RELATION_CAPS = {
    "relation_vl": 18,
    "relation_vl_compact": 18,
    "relation_vl_targeted_misconception": 30,
    "relation_vl_targeted_binding": 30,
}


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


def stable_sort_key(row: Dict[str, Any]) -> str:
    payload = "|".join(
        [
            str(row.get("id", "")),
            str(row.get("query", "")),
            str((row.get("meta") or {}).get("patch_type", "")),
            str((row.get("meta") or {}).get("relation_pair", "")),
            str((row.get("meta") or {}).get("patch_focus", "")),
        ]
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def infer_relation_pair(row: Dict[str, Any]) -> str:
    meta = row.get("meta") or {}
    relation_pair = str(meta.get("relation_pair", "")).strip()
    if relation_pair:
        return relation_pair
    query = str(row.get("query", ""))
    if "限红" in query and "风控" in query:
        return "限红_vs_风控"
    if "RTP" in query and "PNL" in query:
        return "RTP_vs_PNL"
    if "总代" in query and "平台" in query:
        return "总代_vs_平台"
    return ""


def group_relation_rows(rows: List[Dict[str, Any]]) -> Dict[Tuple[str, str], List[Dict[str, Any]]]:
    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        meta = row.get("meta") or {}
        patch_type = str(meta.get("patch_type", ""))
        relation_pair = infer_relation_pair(row)
        if patch_type not in RELATION_CAPS or not relation_pair:
            continue
        grouped[(relation_pair, patch_type)].append(row)
    return grouped


def sample_relation_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped = group_relation_rows(rows)
    sampled: List[Dict[str, Any]] = []
    for key in sorted(grouped):
        relation_pair, patch_type = key
        candidates = sorted(grouped[key], key=stable_sort_key)
        cap = RELATION_CAPS[patch_type]
        sampled.extend(candidates[:cap])
    return sampled


def build_balanced_rows(relation_rows: List[Dict[str, Any]], defmis_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    sampled_relation = sample_relation_rows(relation_rows)
    all_rows = [*sampled_relation, *defmis_rows]
    return sorted(all_rows, key=stable_sort_key)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a balanced VL patch pack from relation_v2 and def+mis_v1.")
    parser.add_argument("--relation-input", default=str(RELATION_INPUT))
    parser.add_argument("--defmis-input", default=str(DEFMIS_INPUT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--summary", default=str(DEFAULT_SUMMARY))
    args = parser.parse_args()

    relation_input = Path(args.relation_input).resolve()
    defmis_input = Path(args.defmis_input).resolve()
    output_path = Path(args.output).resolve()
    summary_path = Path(args.summary).resolve()

    relation_rows = read_jsonl(relation_input)
    defmis_rows = read_jsonl(defmis_input)
    balanced_rows = build_balanced_rows(relation_rows, defmis_rows)

    relation_pair_counts = Counter()
    relation_type_counts = Counter()
    defmis_focus_counts = Counter()
    domain_qtype_counts = Counter()
    source_family_counts = Counter()

    for row in balanced_rows:
        meta = row.get("meta") or {}
        patch_type = meta.get("patch_type", "unknown")
        domain_qtype_counts[str(meta.get("domain_qtype", "unknown"))] += 1
        source_family_counts["relation_v2" if str(patch_type).startswith("relation_vl") else "defmis_v1"] += 1
        if str(patch_type).startswith("relation_vl"):
            relation_pair_counts[infer_relation_pair(row) or "unknown"] += 1
            relation_type_counts[str(patch_type)] += 1
        else:
            defmis_focus_counts[str(meta.get("patch_focus", "unknown"))] += 1

    write_jsonl(output_path, balanced_rows)
    summary = {
        "relation_input": str(relation_input),
        "defmis_input": str(defmis_input),
        "output": str(output_path),
        "total_rows": len(balanced_rows),
        "source_family_counts": dict(source_family_counts),
        "domain_qtype_counts": dict(domain_qtype_counts),
        "relation_patch_type_counts": dict(relation_type_counts),
        "relation_pair_counts": dict(relation_pair_counts),
        "defmis_focus_counts": dict(defmis_focus_counts),
        "relation_caps_per_pair": RELATION_CAPS,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
