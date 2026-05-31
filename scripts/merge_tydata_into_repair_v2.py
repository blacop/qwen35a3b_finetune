#!/usr/bin/env python3
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


ROOT = Path("/home/ubuntu/qwen35a3b_finetune")
DATASETS = ROOT / "datasets"

BASE_SFT = DATASETS / "sft_openai_messages.cleaned.v3_merged.single75_exact.repair_v1.jsonl"
BASE_DPO = DATASETS / "dpo_pairs.repair_v1.jsonl"

SOCCER_SFT = DATASETS / "tydata_soccer_pack_v3" / "soccer_sft_candidates.jsonl"
SOCCER_DPO = DATASETS / "tydata_soccer_pack_v3" / "soccer_dpo_candidates.jsonl"
GAME_SFT = DATASETS / "tydata_gameplay_pack_v3" / "soccer_sft_candidates.jsonl"
GAME_DPO = DATASETS / "tydata_gameplay_pack_v3" / "soccer_dpo_candidates.jsonl"

GAME_FILTERED_DIR = DATASETS / "tydata_gameplay_pack_v3_filtered"
GAME_FILTERED_SFT = GAME_FILTERED_DIR / "soccer_sft_candidates.filtered.jsonl"
GAME_FILTERED_DPO = GAME_FILTERED_DIR / "soccer_dpo_candidates.filtered.jsonl"
GAME_FILTER_SUMMARY = GAME_FILTERED_DIR / "filter_summary.json"

OUT_SFT = DATASETS / "sft_openai_messages.cleaned.v3_merged.single75_exact.repair_v2.tydata.jsonl"
OUT_DPO = DATASETS / "dpo_pairs.repair_v2.tydata.jsonl"
OUT_SUMMARY = DATASETS / "tydata_merge_into_repair_v2_summary.json"

# Fixed 3 rows requested for gameplay pack filtering.
DROP_SFT_IDS = {
    "tygameplay_slide_01",
    "tygameplay_slide_02",
    "tygameplay_slide_07",
}
DROP_DPO_IDS = {f"dpo_{x}" for x in DROP_SFT_IDS}


def read_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def filter_gameplay_pack() -> Dict:
    sft_rows = read_jsonl(GAME_SFT)
    dpo_rows = read_jsonl(GAME_DPO)

    sft_keep = [r for r in sft_rows if r.get("id") not in DROP_SFT_IDS]
    dpo_keep = [r for r in dpo_rows if r.get("id") not in DROP_DPO_IDS]

    write_jsonl(GAME_FILTERED_SFT, sft_keep)
    write_jsonl(GAME_FILTERED_DPO, dpo_keep)

    summary = {
        "source_sft": str(GAME_SFT),
        "source_dpo": str(GAME_DPO),
        "drop_sft_ids": sorted(DROP_SFT_IDS),
        "drop_dpo_ids": sorted(DROP_DPO_IDS),
        "sft_before": len(sft_rows),
        "sft_after": len(sft_keep),
        "dpo_before": len(dpo_rows),
        "dpo_after": len(dpo_keep),
        "output_sft": str(GAME_FILTERED_SFT),
        "output_dpo": str(GAME_FILTERED_DPO),
    }
    GAME_FILTER_SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def merge_by_id(base_rows: List[Dict], extra_rows: List[Dict]) -> Tuple[List[Dict], int]:
    seen = {r.get("id") for r in base_rows if r.get("id")}
    merged = list(base_rows)
    added = 0
    for row in extra_rows:
        rid = row.get("id")
        if rid and rid in seen:
            continue
        merged.append(row)
        if rid:
            seen.add(rid)
        added += 1
    return merged, added


def main() -> None:
    for p in [BASE_SFT, BASE_DPO, SOCCER_SFT, SOCCER_DPO, GAME_SFT, GAME_DPO]:
        if not p.exists():
            raise FileNotFoundError(f"missing required input: {p}")

    filter_summary = filter_gameplay_pack()

    base_sft = read_jsonl(BASE_SFT)
    base_dpo = read_jsonl(BASE_DPO)
    add_soccer_sft = read_jsonl(SOCCER_SFT)
    add_soccer_dpo = read_jsonl(SOCCER_DPO)
    add_game_sft = read_jsonl(GAME_FILTERED_SFT)
    add_game_dpo = read_jsonl(GAME_FILTERED_DPO)

    merged_sft, add_sft_1 = merge_by_id(base_sft, add_soccer_sft)
    merged_sft, add_sft_2 = merge_by_id(merged_sft, add_game_sft)
    merged_dpo, add_dpo_1 = merge_by_id(base_dpo, add_soccer_dpo)
    merged_dpo, add_dpo_2 = merge_by_id(merged_dpo, add_game_dpo)

    write_jsonl(OUT_SFT, merged_sft)
    write_jsonl(OUT_DPO, merged_dpo)

    summary = {
        "base_sft": str(BASE_SFT),
        "base_dpo": str(BASE_DPO),
        "soccer_pack_sft": str(SOCCER_SFT),
        "soccer_pack_dpo": str(SOCCER_DPO),
        "gameplay_pack_sft_filtered": str(GAME_FILTERED_SFT),
        "gameplay_pack_dpo_filtered": str(GAME_FILTERED_DPO),
        "base_counts": {"sft": len(base_sft), "dpo": len(base_dpo)},
        "added_counts": {
            "sft_soccer": add_sft_1,
            "sft_gameplay_filtered": add_sft_2,
            "dpo_soccer": add_dpo_1,
            "dpo_gameplay_filtered": add_dpo_2,
        },
        "output_counts": {"sft": len(merged_sft), "dpo": len(merged_dpo)},
        "output_sft": str(OUT_SFT),
        "output_dpo": str(OUT_DPO),
        "filter_summary": filter_summary,
    }
    OUT_SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
