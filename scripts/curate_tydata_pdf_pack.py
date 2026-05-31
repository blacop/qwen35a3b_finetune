#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List


KW_RE = re.compile(r"注单|结算|取消|危险球|串关|过关|赔率|盘口|滚球|改期|中断|异常|限额|限红|风控|走盘|退回|拒绝")
BANNED_RE = re.compile(r"假球|官方纯赚|包赢|稳赚|必赢|稳赢|代理|返佣|拉新|洗钱|跑分")


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def infer_topic(text: str) -> str:
    rules = [
        ("赛事变更", r"中断|改期|延期|取消|停赛|退赛|弃赛"),
        ("赔率异常", r"赔率|盘口|水位|异常|跳动|变化"),
        ("限红风控", r"限额|限红|风控|拒绝下注|拒单"),
        ("滚球延迟", r"滚球|危险球|待确认|延迟|退回"),
        ("串关结算", r"串关|过关|综合过关|一串|三串|走盘"),
    ]
    for topic, pat in rules:
        if re.search(pat, text):
            return topic
    return "赛事规则"


def prompt_by_topic(topic: str) -> str:
    mapping = {
        "赛事变更": "比赛中断或取消后，注单如何判定和结算？",
        "赔率异常": "下注时盘口/赔率变化，结算按哪个口径？",
        "限红风控": "为什么出现限红或拒单，后续怎么处理？",
        "滚球延迟": "滚球注单显示待确认或延迟，通常多久结算？",
        "串关结算": "串关里有取消/走盘时，整单派彩怎么计算？",
        "赛事规则": "该玩法的判定与结算规则是什么？",
    }
    return mapping.get(topic, "该玩法的判定与结算规则是什么？")


def chosen_by_topic(topic: str, lines: List[str]) -> str:
    core = "；".join(lines[:6]).strip("；")
    if topic in {"赛事变更", "赔率异常", "限红风控", "滚球延迟"}:
        tail = "若涉及注单争议或异常，我先为您升级专员核查，请提供账号与注单号。"
    else:
        tail = "如需核实具体注单结果，请提供账号与注单号，我们会按规则协助查询。"
    return f"{core}。{tail}"


def rejected_by_topic(topic: str) -> str:
    if topic in {"赛事变更", "赔率异常", "限红风控", "滚球延迟"}:
        return "这是正常波动，不需要升级处理，您再等等。"
    return "这个不清楚，您自己看规则吧。"


def line_quality(line: str) -> bool:
    t = re.sub(r"\s+", "", line)
    if len(t) < 8:
        return False
    cjk = len(re.findall(r"[\u4e00-\u9fff]", t))
    if cjk < 6:
        return False
    if cjk / max(len(t), 1) < 0.45:
        return False
    if BANNED_RE.search(t):
        return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Curate OCR pages into high-relevance tydata PDF pack.")
    parser.add_argument(
        "--pages-jsonl",
        default="/home/ubuntu/qwen35a3b_finetune/datasets/tydata_pdf_ocr_pack_v2/pdf_ocr_pages.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        default="/home/ubuntu/qwen35a3b_finetune/datasets/tydata_pdf_ocr_pack_v2_curated",
    )
    parser.add_argument("--source-file", default="体育规则串关玩法介绍.pdf")
    parser.add_argument("--min-kw-hits", type=int, default=2)
    args = parser.parse_args()

    rows = read_jsonl(Path(args.pages_jsonl))
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sft_rows: List[Dict[str, Any]] = []
    dpo_rows: List[Dict[str, Any]] = []
    excluded: List[Dict[str, Any]] = []
    topics = Counter()

    for r in rows:
        if r.get("source_file") != args.source_file:
            continue
        page = int(r["page"])
        clean_lines = [x for x in r.get("clean_lines", []) if line_quality(x)]
        if not clean_lines:
            excluded.append({"page": page, "reason": "no_quality_lines"})
            continue

        scored = sorted(clean_lines, key=lambda x: (len(KW_RE.findall(x)), len(x)), reverse=True)
        kw_hits = len(KW_RE.findall(" ".join(scored)))
        if kw_hits < args.min_kw_hits:
            excluded.append({"page": page, "reason": "low_keyword_hits", "kw_hits": kw_hits})
            continue

        selected = scored[:8]
        text_all = " ".join(selected)
        if BANNED_RE.search(text_all):
            excluded.append({"page": page, "reason": "banned_content"})
            continue

        topic = infer_topic(text_all)
        topics[topic] += 1
        sid = f"typdfcur_p{page:03d}"
        prompt = prompt_by_topic(topic)
        chosen = chosen_by_topic(topic, selected)
        rejected = rejected_by_topic(topic)

        sft_rows.append(
            {
                "id": sid,
                "split": "train",
                "messages": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": chosen},
                ],
                "meta": {
                    "source_file": args.source_file,
                    "page": page,
                    "topic": topic,
                    "ocr_curated": True,
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
                    "source_file": args.source_file,
                    "page": page,
                    "topic": topic,
                    "ocr_curated": True,
                },
            }
        )

    write_jsonl(out_dir / "pdf_ocr_curated_sft_candidates.jsonl", sft_rows)
    write_jsonl(out_dir / "pdf_ocr_curated_dpo_candidates.jsonl", dpo_rows)
    summary = {
        "source_file": args.source_file,
        "input_pages_file": args.pages_jsonl,
        "accepted_samples": len(sft_rows),
        "excluded_pages": len(excluded),
        "topics": dict(topics),
        "excluded_preview": excluded[:80],
    }
    (out_dir / "pdf_ocr_curated_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
