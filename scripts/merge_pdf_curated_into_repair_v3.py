#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


ROOT = Path("/home/ubuntu/qwen35a3b_finetune")
DATASETS = ROOT / "datasets"

BASE_SFT = DATASETS / "sft_openai_messages.cleaned.v3_merged.single75_exact.repair_v2.tydata.jsonl"
BASE_DPO = DATASETS / "dpo_pairs.repair_v2.tydata.jsonl"

PDF_SFT = DATASETS / "tydata_pdf_ocr_pack_v2_curated" / "pdf_ocr_curated_sft_candidates.jsonl"
PDF_DPO = DATASETS / "tydata_pdf_ocr_pack_v2_curated" / "pdf_ocr_curated_dpo_candidates.jsonl"

OUT_SFT = DATASETS / "sft_openai_messages.cleaned.v3_merged.single75_exact.repair_v3.tydata_pdfcurated.jsonl"
OUT_DPO = DATASETS / "dpo_pairs.repair_v3.tydata_pdfcurated.jsonl"
OUT_SUMMARY = DATASETS / "pdf_curated_merge_into_repair_v3_summary.json"


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


def merge_by_id(base_rows: List[Dict], extra_rows: List[Dict]) -> Tuple[List[Dict], int, int]:
    seen = {r.get("id") for r in base_rows if r.get("id")}
    merged = list(base_rows)
    added = 0
    skipped_dup = 0
    for row in extra_rows:
        rid = row.get("id")
        if rid and rid in seen:
            skipped_dup += 1
            continue
        merged.append(row)
        if rid:
            seen.add(rid)
        added += 1
    return merged, added, skipped_dup


def main() -> None:
    for p in [BASE_SFT, BASE_DPO, PDF_SFT, PDF_DPO]:
        if not p.exists():
            raise FileNotFoundError(f"missing required input: {p}")

    base_sft = read_jsonl(BASE_SFT)
    base_dpo = read_jsonl(BASE_DPO)
    pdf_sft = read_jsonl(PDF_SFT)
    pdf_dpo = read_jsonl(PDF_DPO)

    merged_sft, added_sft, dup_sft = merge_by_id(base_sft, pdf_sft)
    merged_dpo, added_dpo, dup_dpo = merge_by_id(base_dpo, pdf_dpo)

    write_jsonl(OUT_SFT, merged_sft)
    write_jsonl(OUT_DPO, merged_dpo)

    summary = {
        "base_sft": str(BASE_SFT),
        "base_dpo": str(BASE_DPO),
        "pdf_curated_sft": str(PDF_SFT),
        "pdf_curated_dpo": str(PDF_DPO),
        "base_counts": {"sft": len(base_sft), "dpo": len(base_dpo)},
        "incoming_counts": {"sft": len(pdf_sft), "dpo": len(pdf_dpo)},
        "added_counts": {"sft": added_sft, "dpo": added_dpo},
        "duplicate_skipped": {"sft": dup_sft, "dpo": dup_dpo},
        "output_counts": {"sft": len(merged_sft), "dpo": len(merged_dpo)},
        "output_sft": str(OUT_SFT),
        "output_dpo": str(OUT_DPO),
    }
    OUT_SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
