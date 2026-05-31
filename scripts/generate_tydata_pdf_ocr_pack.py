#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


NOISE_PATTERNS = [
    r"PLEASE ENTER YOUR TEXT HERE",
    r"PLEASE ENTER YOUR TEX",
    r"CONTENTS?",
    r"^\W+$",
]
NOISE_RE = re.compile("|".join(NOISE_PATTERNS), re.IGNORECASE)

BANNED_PATTERNS = [
    r"假球",
    r"官方纯赚",
    r"包赢|稳赚|必赢|稳赢",
    r"代理|返佣|拉新",
    r"洗钱|跑分|代收|代付",
]
BANNED_RE = re.compile("|".join(BANNED_PATTERNS))


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def run_cmd(cmd: List[str]) -> str:
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
    return p.stdout


def has_cjk(s: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", s))


def clean_ocr_text(text: str) -> List[str]:
    lines: List[str] = []
    for raw in text.splitlines():
        t = re.sub(r"\s+", " ", raw).strip(" \t\r\n")
        if not t:
            continue
        if NOISE_RE.search(t):
            continue
        if len(t) < 7:
            continue
        if not has_cjk(t):
            continue
        cjk_count = len(re.findall(r"[\u4e00-\u9fff]", t))
        compact = re.sub(r"\s+", "", t)
        if cjk_count < 6:
            continue
        if compact and (cjk_count / max(len(compact), 1)) < 0.35:
            continue
        if re.search(r"[=|<>~@#$%^&*_]{2,}", t):
            continue
        if BANNED_RE.search(t):
            continue
        lines.append(t)
    dedup: List[str] = []
    seen = set()
    for t in lines:
        if t in seen:
            continue
        seen.add(t)
        dedup.append(t)
    return dedup


def infer_topic(text: str) -> str:
    rules = [
        ("赛事变更", r"中断|改期|延期|取消|停赛|退赛|弃赛"),
        ("赔率异常", r"赔率|盘口|水位|异常|跳动|变化"),
        ("限红风控", r"限额|限红|风控|拒绝下注|拒单|购期货"),
        ("滚球延迟", r"滚球|即时|危险球|待确认|延迟"),
        ("串关结算", r"串关|过关|综合过关|一串|三串"),
        ("赛事规则", r"联赛|冠军赛|罚牌|球场|比分|波胆|总入球"),
    ]
    for topic, pat in rules:
        if re.search(pat, text):
            return topic
    return "赛事规则"


def prompt_by_topic(topic: str) -> str:
    mapping = {
        "赛事变更": "比赛改期/中断后，注单如何判定与结算？",
        "赔率异常": "下注时赔率突然变化，注单按哪个赔率结算？",
        "限红风控": "为什么我被限红或拒单？该怎么处理？",
        "滚球延迟": "滚球注单一直待确认，通常多久结算？",
        "串关结算": "串关里有一场取消或走盘，整单怎么计算？",
        "赛事规则": "这个玩法规则和结算口径怎么判定？",
    }
    return mapping.get(topic, "该玩法规则和结算口径怎么判定？")


def answer_from_lines(topic: str, lines: List[str]) -> str:
    body = "；".join(lines[:6]).strip("；")
    if topic in {"赛事变更", "赔率异常", "限红风控", "滚球延迟"}:
        tail = "若涉及注单异常或争议，我先为您升级专员核查，请提供账号与注单号。"
    else:
        tail = "如需核实具体注单结果，请提供账号与注单号，我们会按规则协助查询。"
    return f"{body}。{tail}"


def rejected_by_topic(topic: str) -> str:
    if topic in {"赛事变更", "赔率异常", "限红风控", "滚球延迟"}:
        return "这是正常情况，不需要升级处理，您自行等待即可。"
    return "这个我不清楚，建议您自行处理。"


def ocr_pdf(pdf: Path, image_dir: Path) -> List[Tuple[int, str, List[str]]]:
    image_dir.mkdir(parents=True, exist_ok=True)
    prefix = image_dir / "page"
    run_cmd(["pdftoppm", "-r", "220", "-png", str(pdf), str(prefix)])
    pages: List[Tuple[int, str, List[str]]] = []
    for img in sorted(image_dir.glob("page-*.png")):
        m = re.search(r"page-(\d+)\.png$", img.name)
        if not m:
            continue
        page = int(m.group(1))
        raw = run_cmd(["tesseract", str(img), "stdout", "-l", "chi_sim+eng", "--psm", "6"])
        clean = clean_ocr_text(raw)
        pages.append((page, raw, clean))
    return pages


def page_quality_ok(clean: List[str]) -> bool:
    if len(clean) < 2:
        return False
    cjk_total = sum(len(re.findall(r"[\u4e00-\u9fff]", s)) for s in clean)
    long_lines = sum(1 for s in clean if len(s) >= 12)
    if cjk_total < 40:
        return False
    if long_lines < 2:
        return False
    return True


def build_pack(input_dir: Path, output_dir: Path, work_dir: Path) -> Dict[str, Any]:
    pdfs = sorted(input_dir.glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(f"no pdf files under: {input_dir}")

    pages_rows: List[Dict[str, Any]] = []
    sft_rows: List[Dict[str, Any]] = []
    dpo_rows: List[Dict[str, Any]] = []
    excluded: List[Dict[str, Any]] = []
    topic_counter: Counter[str] = Counter()
    seen_text = set()

    for pdf in pdfs:
        stem_tag = f"pdf_{abs(hash(pdf.name)) % 100000:05d}"
        pages = ocr_pdf(pdf, work_dir / pdf.stem)
        for page, raw, clean in pages:
            page_id = f"{stem_tag}_p{page:03d}"
            pages_rows.append(
                {
                    "id": page_id,
                    "source_file": pdf.name,
                    "page": page,
                    "raw_preview": raw[:1500],
                    "clean_lines": clean,
                }
            )
            if len(clean) < 3:
                excluded.append({"id": page_id, "source_file": pdf.name, "page": page, "reason": "low_information"})
                continue
            if not page_quality_ok(clean):
                excluded.append({"id": page_id, "source_file": pdf.name, "page": page, "reason": "low_quality_ocr"})
                continue

            text_all = " ".join(clean)
            if BANNED_RE.search(text_all):
                excluded.append({"id": page_id, "source_file": pdf.name, "page": page, "reason": "banned_content"})
                continue

            sig = " ".join(clean[:4])
            if sig in seen_text:
                excluded.append({"id": page_id, "source_file": pdf.name, "page": page, "reason": "duplicate"})
                continue
            seen_text.add(sig)

            topic = infer_topic(text_all)
            topic_counter[topic] += 1
            prompt = prompt_by_topic(topic)
            chosen = answer_from_lines(topic, clean)
            rejected = rejected_by_topic(topic)

            sid = f"typdf_{page_id}"
            sft_rows.append(
                {
                    "id": sid,
                    "split": "train",
                    "messages": [
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": chosen},
                    ],
                    "meta": {
                        "source_file": pdf.name,
                        "page": page,
                        "topic": topic,
                        "ocr_generated": True,
                    },
                }
            )
            dpo_rows.append(
                {
                    "id": f"dpo_{sid}",
                    "split": "train",
                    "prompt": prompt,
                    "chosen": chosen,
                    "rejected": rejected,
                    "meta": {
                        "source_file": pdf.name,
                        "page": page,
                        "topic": topic,
                        "ocr_generated": True,
                    },
                }
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "pdf_ocr_pages.jsonl", pages_rows)
    write_jsonl(output_dir / "pdf_ocr_sft_candidates.jsonl", sft_rows)
    write_jsonl(output_dir / "pdf_ocr_dpo_candidates.jsonl", dpo_rows)

    summary = {
        "input_dir": str(input_dir),
        "pdf_files": [p.name for p in pdfs],
        "total_pages_ocr": len(pages_rows),
        "accepted_samples": len(sft_rows),
        "excluded_pages": len(excluded),
        "topics": dict(topic_counter),
        "excluded_preview": excluded[:120],
    }
    (output_dir / "pdf_ocr_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="OCR tydata PDFs and generate SFT/DPO augmentation pack.")
    parser.add_argument("--input-dir", default="/tmp/tydata")
    parser.add_argument(
        "--output-dir",
        default="/home/ubuntu/qwen35a3b_finetune/datasets/tydata_pdf_ocr_pack_v1",
    )
    parser.add_argument("--work-dir", default="/tmp/tydata_pdf_ocr_work")
    args = parser.parse_args()

    build_pack(Path(args.input_dir), Path(args.output_dir), Path(args.work_dir))


if __name__ == "__main__":
    main()
