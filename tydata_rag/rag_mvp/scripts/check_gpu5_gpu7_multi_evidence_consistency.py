#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def now_utc_compact() -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


def normalize_text(text: str) -> str:
    text = str(text or "").strip().lower()
    text = re.sub(r"\s+", "", text)
    return text


def evidence_matches(hit: Dict[str, Any], evidence: Dict[str, Any]) -> bool:
    source_file = str(evidence.get("source_file") or "").strip()
    if source_file and str(hit.get("source_file") or "").strip() != source_file:
        return False
    contains = normalize_text(str(evidence.get("contains") or ""))
    if not contains:
        return True
    return contains in normalize_text(str(hit.get("text") or ""))


def coverage_metrics(hits: Sequence[Dict[str, Any]], evidences: Sequence[Dict[str, Any]]) -> Tuple[int, int, float, float]:
    matched = 0
    matched_sources = set()
    for evidence in evidences:
        if any(evidence_matches(hit, evidence) for hit in hits):
            matched += 1
            matched_sources.add(str(evidence.get("source_file") or ""))
    evidence_total = len(evidences)
    source_total = len({str(item.get("source_file") or "") for item in evidences if str(item.get("source_file") or "")})
    evidence_rate = float(matched) / float(evidence_total) if evidence_total else 0.0
    source_rate = float(len(matched_sources)) / float(source_total) if source_total else 0.0
    return matched, len(matched_sources), evidence_rate, source_rate


def classify_answer(text: str) -> str:
    norm = normalize_text(text)
    if not norm:
        return "empty"

    has_fact = any(
        token in norm
        for token in (
            "8副牌",
            "按1点",
            "按0点",
            "算作1点",
            "计为0点",
            "退回",
            "5%",
            "黑杰克>五小龙",
            "黑杰克排在五小龙之前",
        )
    )
    has_unknown = any(
        token in norm
        for token in (
            "无法确定",
            "不能确定",
            "未明确",
            "未说明",
            "没有说明",
            "资料不足",
            "无法判断",
            "不确定",
        )
    )
    has_negative = any(
        token in norm
        for token in (
            "不都",
            "并非都",
            "不是都",
            "并不都",
            "不一致",
            "不完全一致",
            "不相同",
            "不同",
        )
    )
    has_positive = any(
        token in norm
        for token in (
            "是一致",
            "一致",
            "都使用",
            "都写明",
            "都明确",
            "都显示",
            "都为",
            "都是",
            "都按",
            "都属于",
        )
    ) or norm.startswith("是") or norm.startswith("按现有资料，")
    has_partial = any(
        token in norm
        for token in (
            "其中",
            "其余",
            "但",
            "不过",
            "只有",
            "部分",
            "分别",
        )
    )

    if has_unknown and (has_positive or has_negative or has_partial or has_fact):
        return "partial"
    if has_negative and has_positive:
        return "partial"
    if has_unknown:
        return "unknown"
    if has_negative:
        return "no"
    if has_partial:
        return "partial"
    if has_positive:
        return "yes"
    return "other"


def judge_answer(
    *,
    gold_stance: str,
    pred_stance: str,
    evidence_rate: float,
    source_rate: float,
) -> str:
    full_support = evidence_rate >= 0.999 and source_rate >= 0.999
    strong_support = evidence_rate >= 0.8 and source_rate >= 0.8
    source_ready = source_rate >= 0.8
    evidence_ready = evidence_rate >= 0.8
    source_partial = source_rate >= 0.5
    evidence_partial = evidence_rate >= 0.5

    def low_support_suffix() -> str:
        if source_ready and not evidence_ready:
            return "generation_underexplained"
        if source_partial and not evidence_partial:
            return "retrieval_missing_key_evidence"
        if not source_partial:
            return "retrieval_missing_sources"
        return "mixed_low_support"

    if pred_stance == gold_stance:
        if full_support:
            return "correct_supported"
        if strong_support:
            return "correct_partial_support"
        return f"correct_low_support__{low_support_suffix()}"

    if pred_stance in {"unknown", "partial"}:
        if full_support or strong_support:
            return "conservative_under_answer"
        return f"conservative_low_support__{low_support_suffix()}"

    if pred_stance == "other":
        if full_support or strong_support:
            return "ambiguous_with_support"
        return f"ambiguous_low_support__{low_support_suffix()}"

    if gold_stance in {"yes", "no"} and pred_stance in {"yes", "no"} and pred_stance != gold_stance:
        if full_support or strong_support:
            return "wrong_conflict_with_evidence"
        return "wrong_conflict_low_support"

    if pred_stance == "yes" and gold_stance in {"unknown", "partial"}:
        return "wrong_overclaim"
    if pred_stance == "no" and gold_stance in {"unknown", "partial"}:
        return "wrong_overreject"

    if not source_partial or not evidence_partial:
        return "retrieval_insufficient_other"
    return "judge_uncertain"


def source_list(resp: Dict[str, Any]) -> List[str]:
    contexts = resp.get("rag_contexts") or []
    out: List[str] = []
    for item in contexts:
        if not isinstance(item, dict):
            continue
        source = str(item.get("source_file") or "").strip()
        if source:
            out.append(source)
    return out


def parse_answer(resp: Dict[str, Any]) -> str:
    choices = resp.get("choices") or []
    if not choices:
        return ""
    message = (choices[0] or {}).get("message") or {}
    return str(message.get("content") or "").strip()


def context_hits(resp: Dict[str, Any]) -> List[Dict[str, Any]]:
    contexts = resp.get("rag_contexts") or []
    if not isinstance(contexts, list):
        return []
    return [item for item in contexts if isinstance(item, dict)]


def call_chat(
    *,
    gateway_base_url: str,
    api_key: str,
    model: str,
    question: str,
    temperature: float,
    max_tokens: int,
    timeout: float,
) -> Dict[str, Any]:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": question}],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "include_context": True,
    }
    resp = requests.post(
        gateway_base_url.rstrip("/") + "/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=timeout,
    )
    resp.raise_for_status()
    data = resp.json()
    if not isinstance(data, dict):
        raise ValueError(f"Unexpected response type for model={model}: {type(data)}")
    return data


def overlap_ratio(a: Sequence[str], b: Sequence[str]) -> float:
    a_set = {x for x in a if x}
    b_set = {x for x in b if x}
    if not a_set and not b_set:
        return 1.0
    union = a_set | b_set
    return float(len(a_set & b_set)) / float(len(union)) if union else 0.0


def main() -> int:
    parser = argparse.ArgumentParser(description="Check GPU5/GPU7 aggregate-answer consistency through gateway")
    parser.add_argument("--dataset", default=str(PROJECT_ROOT / "data" / "eval" / "eval_questions_multi_evidence.jsonl"))
    parser.add_argument("--gateway-base-url", default="http://127.0.0.1:8025")
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--gpu5-model", default="baowang-gpu5-rag")
    parser.add_argument("--gpu7-model", default="baowang-gpu7-rag")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "outputs" / "gpu5_gpu7_consistency"))
    args = parser.parse_args()

    rows = load_jsonl(Path(args.dataset))
    run_dir = Path(args.output_dir) / now_utc_compact()
    ensure_dir(run_dir)

    case_rows: List[Dict[str, Any]] = []
    exact_match_count = 0
    same_stance_count = 0
    both_gold_match_count = 0
    source_overlap_sum = 0.0
    gpu5_supported_correct_count = 0
    gpu7_supported_correct_count = 0
    gpu5_conservative_count = 0
    gpu7_conservative_count = 0
    gpu5_wrong_count = 0
    gpu7_wrong_count = 0

    for row in rows:
        question = str(row.get("question") or "").strip()
        gold_answer = str(row.get("gold_answer") or "").strip()
        gold_evidence = row.get("gold_evidence", [])
        gpu5_resp = call_chat(
            gateway_base_url=args.gateway_base_url,
            api_key=args.api_key,
            model=args.gpu5_model,
            question=question,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            timeout=args.timeout,
        )
        gpu7_resp = call_chat(
            gateway_base_url=args.gateway_base_url,
            api_key=args.api_key,
            model=args.gpu7_model,
            question=question,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            timeout=args.timeout,
        )

        gpu5_answer = parse_answer(gpu5_resp)
        gpu7_answer = parse_answer(gpu7_resp)
        gold_stance = classify_answer(gold_answer)
        gpu5_stance = classify_answer(gpu5_answer)
        gpu7_stance = classify_answer(gpu7_answer)
        exact_match = normalize_text(gpu5_answer) == normalize_text(gpu7_answer)
        same_stance = gpu5_stance == gpu7_stance
        both_gold_match = gpu5_stance == gold_stance and gpu7_stance == gold_stance
        gpu5_sources = source_list(gpu5_resp)
        gpu7_sources = source_list(gpu7_resp)
        gpu5_hits = context_hits(gpu5_resp)
        gpu7_hits = context_hits(gpu7_resp)
        gpu5_matched_evidence, gpu5_matched_sources, gpu5_evidence_rate, gpu5_source_rate = coverage_metrics(gpu5_hits, gold_evidence)
        gpu7_matched_evidence, gpu7_matched_sources, gpu7_evidence_rate, gpu7_source_rate = coverage_metrics(gpu7_hits, gold_evidence)
        gpu5_judge = judge_answer(
            gold_stance=gold_stance,
            pred_stance=gpu5_stance,
            evidence_rate=gpu5_evidence_rate,
            source_rate=gpu5_source_rate,
        )
        gpu7_judge = judge_answer(
            gold_stance=gold_stance,
            pred_stance=gpu7_stance,
            evidence_rate=gpu7_evidence_rate,
            source_rate=gpu7_source_rate,
        )
        source_overlap = overlap_ratio(gpu5_sources, gpu7_sources)

        exact_match_count += int(exact_match)
        same_stance_count += int(same_stance)
        both_gold_match_count += int(both_gold_match)
        source_overlap_sum += source_overlap
        gpu5_supported_correct_count += int(gpu5_judge == "correct_supported")
        gpu7_supported_correct_count += int(gpu7_judge == "correct_supported")
        gpu5_conservative_count += int(gpu5_judge.startswith("conservative_"))
        gpu7_conservative_count += int(gpu7_judge.startswith("conservative_"))
        gpu5_wrong_count += int(gpu5_judge.startswith("wrong_"))
        gpu7_wrong_count += int(gpu7_judge.startswith("wrong_"))

        case_rows.append(
            {
                "qid": row.get("id", ""),
                "question": question,
                "gold_answer": gold_answer,
                "gold_stance": gold_stance,
                "gpu5_stance": gpu5_stance,
                "gpu7_stance": gpu7_stance,
                "gpu5_judge": gpu5_judge,
                "gpu7_judge": gpu7_judge,
                "same_stance": int(same_stance),
                "exact_match": int(exact_match),
                "both_match_gold_stance": int(both_gold_match),
                "gold_evidence_total": len(gold_evidence),
                "gpu5_matched_evidence": gpu5_matched_evidence,
                "gpu5_matched_sources": gpu5_matched_sources,
                "gpu5_evidence_coverage_rate": round(gpu5_evidence_rate, 6),
                "gpu5_source_coverage_rate": round(gpu5_source_rate, 6),
                "gpu7_matched_evidence": gpu7_matched_evidence,
                "gpu7_matched_sources": gpu7_matched_sources,
                "gpu7_evidence_coverage_rate": round(gpu7_evidence_rate, 6),
                "gpu7_source_coverage_rate": round(gpu7_source_rate, 6),
                "gpu5_total_hits": int(((gpu5_resp.get("rag_retrieval") or {}).get("total_hits") or 0)),
                "gpu7_total_hits": int(((gpu7_resp.get("rag_retrieval") or {}).get("total_hits") or 0)),
                "source_overlap_ratio": round(source_overlap, 6),
                "gpu5_sources_top": " | ".join(gpu5_sources[:8]),
                "gpu7_sources_top": " | ".join(gpu7_sources[:8]),
                "gpu5_answer": gpu5_answer,
                "gpu7_answer": gpu7_answer,
            }
        )

    total = len(case_rows)
    summary = {
        "dataset": str(Path(args.dataset)),
        "gateway_base_url": args.gateway_base_url,
        "gpu5_model": args.gpu5_model,
        "gpu7_model": args.gpu7_model,
        "total": total,
        "same_stance_rate": float(same_stance_count) / float(total) if total else 0.0,
        "exact_match_rate": float(exact_match_count) / float(total) if total else 0.0,
        "both_match_gold_stance_rate": float(both_gold_match_count) / float(total) if total else 0.0,
        "avg_source_overlap_ratio": source_overlap_sum / float(total) if total else 0.0,
        "gpu5_correct_supported_rate": float(gpu5_supported_correct_count) / float(total) if total else 0.0,
        "gpu7_correct_supported_rate": float(gpu7_supported_correct_count) / float(total) if total else 0.0,
        "gpu5_conservative_rate": float(gpu5_conservative_count) / float(total) if total else 0.0,
        "gpu7_conservative_rate": float(gpu7_conservative_count) / float(total) if total else 0.0,
        "gpu5_wrong_rate": float(gpu5_wrong_count) / float(total) if total else 0.0,
        "gpu7_wrong_rate": float(gpu7_wrong_count) / float(total) if total else 0.0,
        "mismatch_qids": [row["qid"] for row in case_rows if not row["same_stance"]],
        "gpu5_judge_breakdown": {label: sum(1 for row in case_rows if row["gpu5_judge"] == label) for label in sorted({row["gpu5_judge"] for row in case_rows})},
        "gpu7_judge_breakdown": {label: sum(1 for row in case_rows if row["gpu7_judge"] == label) for label in sorted({row["gpu7_judge"] for row in case_rows})},
    }

    csv_path = run_dir / "cases.csv"
    summary_path = run_dir / "summary.json"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(case_rows[0].keys()) if case_rows else [])
        if case_rows:
            writer.writeheader()
            writer.writerows(case_rows)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(str(csv_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
