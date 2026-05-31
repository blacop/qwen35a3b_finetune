#!/usr/bin/python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import requests
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
DEFAULT_TYPLAY_INDEX_DIR = "/video-storage/ai-customer/qwen35a3b_finetune/rag/typlay/rag_index_typlay_default_560_112"


def now_utc_compact() -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


def normalize_text(text: str) -> str:
    return "".join(str(text or "").split())


def percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    rank = max(0.0, min(100.0, p)) / 100.0 * (len(ordered) - 1)
    lo = int(math.floor(rank))
    hi = int(math.ceil(rank))
    if lo == hi:
        return float(ordered[lo])
    frac = rank - lo
    return float(ordered[lo] * (1.0 - frac) + ordered[hi] * frac)


def load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, dict) else {}


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def evidence_matches_hit(hit: Dict[str, Any], evidence: Dict[str, Any], require_contains: bool = True) -> bool:
    source_file = str(evidence.get("source_file") or evidence.get("file") or "").strip()
    if source_file:
        hit_source = str(hit.get("source_file", "")).strip()
        if hit_source != source_file:
            hit_stem = Path(hit_source).stem
            evidence_stem = Path(source_file).stem
            if not hit_stem or hit_stem != evidence_stem:
                return False

    chunk_id = evidence.get("chunk_id")
    if chunk_id is not None:
        try:
            if int(hit.get("chunk_id")) != int(chunk_id):
                return False
        except Exception:
            return False

    contains = str(evidence.get("contains") or evidence.get("must_include") or "").strip()
    if not contains:
        return True
    if not require_contains:
        return True

    hay = normalize_text(str(hit.get("text", "")))
    needle = normalize_text(contains)
    return bool(needle) and needle in hay


def first_match_rank(
    hits: Sequence[Dict[str, Any]],
    evidences: Sequence[Dict[str, Any]],
    require_contains: bool = True,
) -> Tuple[Optional[int], Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    for rank, hit in enumerate(hits, start=1):
        for evidence in evidences:
            if evidence_matches_hit(hit, evidence, require_contains=require_contains):
                return rank, hit, evidence
    return None, None, None


def answer_supported(
    hits: Sequence[Dict[str, Any]],
    gold_answer: str,
    evidences: Sequence[Dict[str, Any]],
    require_contains: bool = True,
) -> bool:
    gold = normalize_text(gold_answer)
    if not gold:
        gold = ""
    for hit in hits:
        hit_text = normalize_text(str(hit.get("text", "")))
        if gold and gold in hit_text:
            return True
        for evidence in evidences:
            if evidence_matches_hit(hit, evidence, require_contains=require_contains):
                return True
    return False


def load_local_retriever(index_dir: Path) -> Any:
    from retrieval_hybrid import HybridRetriever  # type: ignore

    return HybridRetriever(index_dir)


def retrieve_local(
    retriever: Any,
    query: str,
    search_cfg: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], float]:
    started = time.perf_counter()
    hits = retriever.search(query=query, **search_cfg)
    latency_ms = (time.perf_counter() - started) * 1000.0
    return hits, latency_ms


def retrieve_api(
    base_url: str,
    api_key: str,
    query: str,
    search_cfg: Dict[str, Any],
    timeout_s: float,
) -> Tuple[List[Dict[str, Any]], float]:
    url = base_url.rstrip("/") + "/retrieve"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    payload = {"query": query, **search_cfg}
    started = time.perf_counter()
    resp = requests.post(url, headers=headers, json=payload, timeout=timeout_s)
    resp.raise_for_status()
    data = resp.json()
    latency_ms = (time.perf_counter() - started) * 1000.0
    hits = data.get("hits") if isinstance(data, dict) else []
    return hits if isinstance(hits, list) else [], latency_ms


def build_search_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    retrieval = cfg.get("retrieval", {}) if isinstance(cfg.get("retrieval", {}), dict) else {}
    return {
        "top_k": int(retrieval.get("top_k", 5)),
        "min_score": float(retrieval.get("min_score", 0.01)),
        "min_score_vec": float(retrieval.get("min_score_vec", 0.0)),
        "mode": str(retrieval.get("mode", "hybrid")),
        "fusion": str(retrieval.get("fusion", "rrf")),
        "candidate_k": int(retrieval.get("candidate_k", 40)),
        "rrf_k": int(retrieval.get("rrf_k", 60)),
        "weight_lex": float(retrieval.get("weight_lex", 0.55)),
        "weight_vec": float(retrieval.get("weight_vec", 0.45)),
        "enable_rerank": bool(retrieval.get("enable_rerank", True)),
        "enable_diversity": bool(retrieval.get("enable_diversity", False)),
        "max_per_source": int(retrieval.get("max_per_source", 1)),
        "max_per_family": int(retrieval.get("max_per_family", 3)),
        "diversity_lambda": float(retrieval.get("diversity_lambda", 0.7)),
        "source_match_weight": float(retrieval.get("source_match_weight", 0.0)),
    }


@dataclass
class EvalRow:
    qid: str
    question: str
    qtype: str
    tags: List[str]
    gold_answer: str
    hit_rank: Optional[int]
    hit_at_1: bool
    hit_at_3: bool
    hit_at_5: bool
    answer_supported_at_5: bool
    top1_source_file: str
    top1_score: float
    matched_source_file: str
    matched_contains: str
    latency_ms: float


def build_group_summary(
    rows: Sequence[EvalRow],
    recall_k: Sequence[int],
) -> Dict[str, Any]:
    total = len(rows)
    denom = max(1, total)
    latencies = [float(r.latency_ms) for r in rows]
    out: Dict[str, Any] = {
        "total": total,
        "mrr": sum((1.0 / r.hit_rank) if r.hit_rank else 0.0 for r in rows) / denom,
        "answer_supported_rate": sum(1 for r in rows if r.answer_supported_at_5) / denom,
        "latency_ms_avg": sum(latencies) / denom if latencies else 0.0,
        "latency_ms_p95": percentile(latencies, 95) if latencies else 0.0,
    }
    for k in recall_k:
        out[f"recall@{k}"] = sum(1 for r in rows if r.hit_rank is not None and r.hit_rank <= k) / denom
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Minimal RAG eval runner")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "default.yaml"))
    parser.add_argument("--dataset", default=None, help="Override dataset path")
    parser.add_argument("--backend", choices=["local", "api"], default=None)
    parser.add_argument("--index-dir", default=None)
    parser.add_argument("--api-base", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    cfg = load_yaml(Path(args.config))
    paths = cfg.get("paths", {}) if isinstance(cfg.get("paths", {}), dict) else {}
    retrieval = cfg.get("retrieval", {}) if isinstance(cfg.get("retrieval", {}), dict) else {}
    api_cfg = cfg.get("api", {}) if isinstance(cfg.get("api", {}), dict) else {}
    eval_cfg = cfg.get("evaluation", {}) if isinstance(cfg.get("evaluation", {}), dict) else {}

    dataset_path = Path(args.dataset or paths.get("eval_dataset") or PROJECT_ROOT / "data" / "eval" / "eval_questions.jsonl")
    output_dir = Path(args.output_dir or paths.get("output_dir") or PROJECT_ROOT / "outputs")
    ensure_dir(output_dir)

    backend = (args.backend or retrieval.get("backend") or "local").strip().lower()
    index_dir = Path(args.index_dir or paths.get("index_dir") or DEFAULT_TYPLAY_INDEX_DIR)
    api_base = str(args.api_base or api_cfg.get("base_url") or "http://127.0.0.1:18080")
    api_key = str(args.api_key or api_cfg.get("api_key") or "")
    timeout_s = float(api_cfg.get("timeout_s", 30))

    recall_k = eval_cfg.get("recall_k", [1, 3, 5])
    if not isinstance(recall_k, list) or not recall_k:
        recall_k = [1, 3, 5]
    recall_k = sorted({max(1, int(k)) for k in recall_k})
    top_k = max(recall_k)
    support_top_k = max(top_k, int(eval_cfg.get("support_top_k", top_k) or top_k))
    require_contains = bool(eval_cfg.get("require_evidence_contains", True))
    max_rows = int(eval_cfg.get("max_rows", 0) or 0)

    search_cfg = build_search_cfg(cfg)
    search_cfg["top_k"] = support_top_k

    if backend == "local":
        if not index_dir.exists():
            raise SystemExit(f"Index directory not found: {index_dir}")
        try:
            retriever = load_local_retriever(index_dir)
        except Exception as exc:
            raise SystemExit(f"Failed to load local retriever from {index_dir}: {exc}") from exc
    else:
        retriever = None

    rows = load_jsonl(dataset_path)
    if max_rows > 0:
        rows = rows[:max_rows]
    if not rows:
        raise SystemExit(f"No rows found in dataset: {dataset_path}")

    results: List[EvalRow] = []
    latencies: List[float] = []
    hit_counts = {k: 0 for k in recall_k}
    answer_supported_count = 0

    for row in rows:
        qid = str(row.get("id") or row.get("qid") or "")
        question = str(row.get("question") or "").strip()
        gold_answer = str(row.get("gold_answer") or "").strip()
        qtype = str(row.get("type") or "").strip()
        tags = [str(x).strip() for x in (row.get("tags") or []) if str(x).strip()]
        evidences = row.get("gold_evidence") or []
        if not isinstance(evidences, list):
            evidences = []

        if backend == "local":
            hits, latency_ms = retrieve_local(retriever, question, search_cfg)  # type: ignore[arg-type]
        else:
            hits, latency_ms = retrieve_api(api_base, api_key, question, search_cfg, timeout_s)

        latencies.append(latency_ms)
        hit_rank, matched_hit, matched_evidence = first_match_rank(hits, evidences, require_contains=require_contains)

        top1_source = str(hits[0].get("source_file", "")) if hits else ""
        top1_score = float(hits[0].get("score", 0.0) or 0.0) if hits else 0.0
        matched_source_file = ""
        matched_contains = ""
        if matched_evidence:
            matched_source_file = str(matched_evidence.get("source_file") or matched_evidence.get("file") or "")
            matched_contains = str(matched_evidence.get("contains") or matched_evidence.get("must_include") or "")

        hit_at = {k: bool(hit_rank is not None and hit_rank <= k) for k in recall_k}
        for k, ok in hit_at.items():
            if ok:
                hit_counts[k] += 1

        supported = answer_supported(hits[:support_top_k], gold_answer, evidences, require_contains=require_contains)
        if supported:
            answer_supported_count += 1

        results.append(
            EvalRow(
                qid=qid,
                question=question,
                qtype=qtype,
                tags=tags,
                gold_answer=gold_answer,
                hit_rank=hit_rank,
                hit_at_1=hit_at.get(1, False),
                hit_at_3=hit_at.get(3, False),
                hit_at_5=hit_at.get(5, False),
                answer_supported_at_5=supported,
                top1_source_file=top1_source,
                top1_score=top1_score,
                matched_source_file=matched_source_file,
                matched_contains=matched_contains,
                latency_ms=latency_ms,
            )
        )

    total = max(1, len(results))
    summary: Dict[str, Any] = {
        "dataset": str(dataset_path),
        "backend": backend,
        "index_dir": str(index_dir) if backend == "local" else None,
        "api_base": api_base if backend == "api" else None,
        "total": len(results),
        "recall_k": recall_k,
        "latency_ms_avg": sum(latencies) / total,
        "latency_ms_p50": percentile(latencies, 50),
        "latency_ms_p95": percentile(latencies, 95),
        "mrr": sum((1.0 / r.hit_rank) if r.hit_rank else 0.0 for r in results) / total,
        "answer_supported_rate": answer_supported_count / total,
    }
    for k in recall_k:
        summary[f"recall@{k}"] = hit_counts[k] / total

    by_type: Dict[str, List[EvalRow]] = defaultdict(list)
    by_tag: Dict[str, List[EvalRow]] = defaultdict(list)
    for row in results:
        if row.qtype:
            by_type[row.qtype].append(row)
        for tag in row.tags:
            by_tag[tag].append(row)

    summary["by_type"] = {
        key: build_group_summary(group_rows, recall_k)
        for key, group_rows in sorted(by_type.items())
    }
    summary["by_tag"] = {
        key: build_group_summary(group_rows, recall_k)
        for key, group_rows in sorted(by_tag.items())
    }
    summary["failures"] = {
        "recall_at_1": [
            {
                "qid": r.qid,
                "question": r.question,
                "qtype": r.qtype,
                "tags": r.tags,
                "top1_source_file": r.top1_source_file,
                "matched_source_file": r.matched_source_file,
                "hit_rank": r.hit_rank,
            }
            for r in results
            if not r.hit_at_1
        ],
        "answer_supported": [
            {
                "qid": r.qid,
                "question": r.question,
                "qtype": r.qtype,
                "tags": r.tags,
                "top1_source_file": r.top1_source_file,
                "matched_source_file": r.matched_source_file,
                "hit_rank": r.hit_rank,
            }
            for r in results
            if not r.answer_supported_at_5
        ],
    }

    stamp = now_utc_compact()
    summary_path = output_dir / f"summary_{stamp}.json"
    detail_path = output_dir / f"results_{stamp}.jsonl"
    csv_path = output_dir / f"results_{stamp}.csv"

    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    with detail_path.open("w", encoding="utf-8") as f:
        for row in results:
            f.write(json.dumps(row.__dict__, ensure_ascii=False) + "\n")

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].__dict__.keys()))
        writer.writeheader()
        for row in results:
            writer.writerow(row.__dict__)

    if not args.quiet:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        print(f"summary: {summary_path}")
        print(f"details: {detail_path}")
        print(f"csv: {csv_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
