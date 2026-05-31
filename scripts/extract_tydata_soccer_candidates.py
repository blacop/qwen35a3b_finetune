#!/usr/bin/env python3
"""
Extract training candidates from a soccer-rule PPTX file.

Outputs:
  - soccer_sft_candidates.jsonl
  - soccer_dpo_candidates.jsonl
  - soccer_extracted_slides.jsonl
  - soccer_extract_summary.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import zipfile
import xml.etree.ElementTree as ET
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


PPT_NS = {"a": "http://schemas.openxmlformats.org/drawingml/2006/main"}

NOISE_TOKENS = {
    "|",
    "/",
    "……",
    "………………",
    "Contents",
    "01",
    "02",
    "03",
    "04",
    "05",
}

BANNED_PATTERNS = [
    r"彩金",
    r"包赢|稳赚|必赢|稳赢",
    r"代理|返佣|拉新",
    r"洗钱|跑分|代收|代付",
    r"假球|官方纯赚",
]
BANNED_RE = re.compile("|".join(BANNED_PATTERNS))


def now_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _is_noise_token(tok: str) -> bool:
    if not tok:
        return True
    t = tok.strip()
    if not t:
        return True
    if t in NOISE_TOKENS:
        return True
    if re.fullmatch(r"[0-9]+", t):
        return True
    if re.fullmatch(r"[A-Z]{1,2}", t):
        return True
    if re.fullmatch(r"[^\w\u4e00-\u9fff]+", t):
        return True
    return False


def _clean_tokens(tokens: List[str]) -> List[str]:
    cleaned: List[str] = []
    for tok in tokens:
        t = tok.strip()
        if _is_noise_token(t):
            continue
        if BANNED_RE.search(t):
            continue
        # normalize whitespace and punctuation tails
        t = re.sub(r"\s+", " ", t)
        t = t.strip(" ，,;；")
        if not t:
            continue
        cleaned.append(t)

    # de-dup while keeping order
    seen = set()
    unique: List[str] = []
    for t in cleaned:
        if t in seen:
            continue
        seen.add(t)
        unique.append(t)
    return unique


def extract_slides(pptx_path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with zipfile.ZipFile(pptx_path) as zf:
        slides = [n for n in zf.namelist() if n.startswith("ppt/slides/slide") and n.endswith(".xml")]
        slides = sorted(slides, key=lambda x: int(re.search(r"slide(\d+)\.xml", x).group(1)))
        for s in slides:
            idx = int(re.search(r"slide(\d+)\.xml", s).group(1))
            root = ET.fromstring(zf.read(s))
            raw_tokens = [t.text.strip() for t in root.findall(".//a:t", PPT_NS) if t.text and t.text.strip()]
            clean_tokens = _clean_tokens(raw_tokens)
            rows.append(
                {
                    "slide": idx,
                    "raw_token_count": len(raw_tokens),
                    "clean_token_count": len(clean_tokens),
                    "raw_tokens": raw_tokens,
                    "clean_tokens": clean_tokens,
                }
            )
    return rows


def _topic_from_tokens(tokens: List[str]) -> str:
    text = " ".join(tokens)
    rules = [
        ("危险球确认", r"危险球"),
        ("滚球与今日赛事", r"滚球|今日赛事"),
        ("让球盘结算", r"让球盘|让球"),
        ("大小盘结算", r"大\s*/\s*小|大小盘"),
        ("单双与波胆", r"单双|波胆"),
        ("独赢盘1X2", r"1X2|独赢"),
        ("半全场与净胜球", r"半.?全场|净胜球"),
        ("复式串关", r"复式串关|串关"),
        ("赔率盘型", r"欧洲盘|香港盘|马来盘|印尼盘|赔率"),
        ("足球比赛规则", r"比赛时间|伤停补时|加时赛|点球大战|赛事中断"),
    ]
    for topic, pat in rules:
        if re.search(pat, text):
            return topic
    return tokens[0] if tokens else "体育规则"


def _build_answer(tokens: List[str]) -> str:
    # Keep concise, factual, and service-friendly.
    content = "；".join(tokens[:6])
    content = re.sub(r"[；]{2,}", "；", content).strip("；")
    answer = f"{content}。如需核实具体注单结果，请提供账号与注单号，我们会按规则协助查询。"
    return answer


def _source_tag(source_file: str) -> str:
    lower = source_file.lower()
    if "足球" in source_file:
        return "tysoccer"
    if "玩法" in source_file:
        return "tygameplay"
    ascii_stem = re.sub(r"[^a-z0-9]+", "_", lower).strip("_")
    if ascii_stem:
        return f"ty_{ascii_stem}"
    digest = hashlib.md5(source_file.encode("utf-8")).hexdigest()[:8]
    return f"ty_{digest}"


def build_candidates(
    slides: List[Dict[str, Any]],
    source_file: str,
    min_tokens: int = 4,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    extracted_rows: List[Dict[str, Any]] = []
    sft_rows: List[Dict[str, Any]] = []
    dpo_rows: List[Dict[str, Any]] = []
    excluded: List[Dict[str, Any]] = []
    tag = _source_tag(source_file)

    for row in slides:
        slide = row["slide"]
        tokens = row["clean_tokens"]
        if len(tokens) < min_tokens:
            excluded.append({"slide": slide, "reason": "low_information", "clean_tokens": tokens})
            continue

        topic = _topic_from_tokens(tokens)
        query = f"{topic}怎么判定和结算？请按规则说明。"
        chosen = _build_answer(tokens)
        rejected = "这个要看运气，先随便下注即可，具体规则不用管。"

        sid = f"{tag}_slide_{slide:02d}"
        extracted_rows.append(
            {
                "id": sid,
                "slide": slide,
                "topic": topic,
                "source": source_file,
                "clean_tokens": tokens,
            }
        )
        sft_rows.append(
            {
                "id": sid,
                "split": "train",
                "messages": [
                    {"role": "user", "content": query},
                    {"role": "assistant", "content": chosen},
                ],
                "meta": {
                    "source_file": source_file,
                    "slide": slide,
                    "topic": topic,
                    "tydata_candidate": True,
                },
            }
        )
        dpo_rows.append(
            {
                "id": f"dpo_{sid}",
                "split": "train",
                "prompt": query,
                "chosen": chosen,
                "rejected": rejected,
                "meta": {
                    "source_file": source_file,
                    "slide": slide,
                    "topic": topic,
                    "tydata_candidate": True,
                },
            }
        )

    topic_counter = Counter(r["topic"] for r in extracted_rows)
    summary = {
        "source_file": source_file,
        "total_slides": len(slides),
        "accepted_samples": len(extracted_rows),
        "excluded_slides": len(excluded),
        "topics": dict(topic_counter),
        "excluded": excluded,
    }
    return extracted_rows, sft_rows, dpo_rows, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract training candidates from tydata soccer pptx.")
    parser.add_argument("--input-pptx", default="/tmp/tydata/体育足球.pptx")
    parser.add_argument("--output-dir", default=f"/home/ubuntu/qwen35a3b_finetune/datasets/tydata_soccer_pack_{now_ts()}")
    parser.add_argument("--min-tokens", type=int, default=4)
    args = parser.parse_args()

    in_path = Path(args.input_pptx).resolve()
    if not in_path.exists():
        raise FileNotFoundError(f"input not found: {in_path}")

    slides = extract_slides(in_path)
    extracted_rows, sft_rows, dpo_rows, summary = build_candidates(
        slides,
        source_file=in_path.name,
        min_tokens=args.min_tokens,
    )

    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    extracted_path = out_dir / "soccer_extracted_slides.jsonl"
    sft_path = out_dir / "soccer_sft_candidates.jsonl"
    dpo_path = out_dir / "soccer_dpo_candidates.jsonl"
    summary_path = out_dir / "soccer_extract_summary.json"

    write_jsonl(extracted_path, extracted_rows)
    write_jsonl(sft_path, sft_rows)
    write_jsonl(dpo_path, dpo_rows)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(json.dumps({"output_dir": str(out_dir), **summary}, ensure_ascii=False, indent=2))
    print(f"[INFO] extracted: {extracted_path}")
    print(f"[INFO] sft: {sft_path}")
    print(f"[INFO] dpo: {dpo_path}")
    print(f"[INFO] summary: {summary_path}")


if __name__ == "__main__":
    main()
