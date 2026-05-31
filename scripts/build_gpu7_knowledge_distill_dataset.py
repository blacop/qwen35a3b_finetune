#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


SYSTEM_PROMPT = "你是体育包网智能客服。回答要简洁、合规、不要输出思考过程。"


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


def assistant_content(row: Dict[str, Any]) -> str:
    try:
        content = row["response"]["choices"][0]["message"].get("content") or ""
    except Exception:
        content = ""
    return str(content).strip()


def score_overall(row: Dict[str, Any]) -> float:
    score = row.get("score") or {}
    try:
        return float(score.get("overall", 0.0))
    except Exception:
        return 0.0


def make_chat_row(row_id: str, user_query: str, answer: str, meta: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": row_id,
        "split": "train",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_query},
            {"role": "assistant", "content": answer},
        ],
        "meta": meta,
    }


def query_variants(query: str) -> List[str]:
    clean = query.strip()
    return [
        clean,
        f"请直接按体育包网客服口吻说明：{clean}",
        f"会员咨询这个规则，麻烦用简洁客服话术解释清楚：{clean}",
    ]


def build_knowledge_rows(
    gpu7_rows: List[Dict[str, Any]],
    gpu5_raw_rows: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    gpu5_by_id = {str(row.get("id")): row for row in gpu5_raw_rows}
    out: List[Dict[str, Any]] = []
    for row in gpu7_rows:
        sid = str(row.get("id") or f"kb_{len(out):04d}")
        answer = assistant_content(row)
        if not answer:
            continue
        gpu5_score = score_overall(gpu5_by_id.get(sid, {}))
        gpu7_score = score_overall(row)
        # Heavier weight when GPU5 lagged GPU7 or missed core terms.
        repeat = 4 if gpu5_score < 0.75 or gpu7_score - gpu5_score >= 0.20 else 2
        variants = query_variants(str(row.get("user_query", "")))
        for rep in range(repeat):
            for vidx, user_query in enumerate(variants):
                if not user_query.strip():
                    continue
                if vidx > 0 and rep >= max(1, repeat // 2):
                    continue
                out.append(
                    make_chat_row(
                        f"gpu7_kb_distill_{sid}_{rep}_{vidx}",
                        user_query,
                        answer,
                        {
                            "source": "gpu7_teacher",
                            "teacher_model": "qwen35a3b-sft-v5-combined",
                            "sample_id": sid,
                            "scenario": row.get("scenario", ""),
                            "gpu5_raw_overall": gpu5_score,
                            "gpu7_overall": gpu7_score,
                            "must_include": row.get("must_include") or [],
                        },
                    )
                )
    return out


def build_reference_rows(rule_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in rule_rows:
        sid = str(row.get("id") or f"rule_{len(out):04d}")
        answer = str(row.get("reference_answer") or "").strip()
        query = str(row.get("user_query") or "").strip()
        if not answer or not query:
            continue
        variants = query_variants(query)
        for rep in range(4):
            for vidx, user_query in enumerate(variants):
                if vidx > 0 and rep >= 2:
                    continue
                out.append(
                    make_chat_row(
                        f"reference_rule_distill_{sid}_{rep}_{vidx}",
                        user_query,
                        answer,
                        {
                            "source": "sports_rule_reference",
                            "sample_id": sid,
                            "scenario": row.get("scenario", ""),
                            "must_include": row.get("must_include") or [],
                        },
                    )
                )
    return out


def take_tool_guard_rows(path: Path, limit: int) -> List[Dict[str, Any]]:
    if limit <= 0 or not path.exists():
        return []
    rows = read_jsonl(path)
    selected: List[Dict[str, Any]] = []
    for row in rows:
        meta = row.get("meta") or {}
        if str(meta.get("source", "")).startswith("tool_dialogue") or "tool" in str(meta.get("kind", "")):
            selected.append(row)
        if len(selected) >= limit:
            break
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description="Build GPU7-to-GPU5 raw sports knowledge distillation SFT dataset.")
    parser.add_argument("--gpu7-knowledge-results", required=True)
    parser.add_argument("--gpu5-raw-results", required=True)
    parser.add_argument("--sports-rule-jsonl", required=True)
    parser.add_argument("--tool-guard-jsonl", required=True)
    parser.add_argument("--tool-guard-limit", type=int, default=160)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--summary-json", required=True)
    args = parser.parse_args()

    gpu7_rows = json.loads(Path(args.gpu7_knowledge_results).read_text(encoding="utf-8"))
    gpu5_raw_rows = json.loads(Path(args.gpu5_raw_results).read_text(encoding="utf-8"))
    rule_rows = read_jsonl(Path(args.sports_rule_jsonl))

    knowledge_rows = build_knowledge_rows(gpu7_rows, gpu5_raw_rows)
    reference_rows = build_reference_rows(rule_rows)
    tool_rows = take_tool_guard_rows(Path(args.tool_guard_jsonl), args.tool_guard_limit)

    all_rows = knowledge_rows + reference_rows + tool_rows
    write_jsonl(Path(args.output_jsonl), all_rows)

    summary = {
        "output_jsonl": args.output_jsonl,
        "total": len(all_rows),
        "gpu7_teacher_knowledge": len(knowledge_rows),
        "sports_rule_reference": len(reference_rows),
        "tool_guard": len(tool_rows),
        "system_prompt": SYSTEM_PROMPT,
        "notes": [
            "Dataset keeps per-sample system messages.",
            "No global --system override should be used during training.",
            "Teacher rows are weighted more when GPU5 raw score lagged GPU7.",
        ],
    }
    Path(args.summary_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary_json).write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
