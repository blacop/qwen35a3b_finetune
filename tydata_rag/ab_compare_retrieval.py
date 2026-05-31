#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests

from retrieval_hybrid import term_coverage_score


def now_utc_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    v = sorted(values)
    if len(v) == 1:
        return float(v[0])
    p = max(0.0, min(100.0, p))
    rank = (p / 100.0) * (len(v) - 1)
    lo = int(math.floor(rank))
    hi = int(math.ceil(rank))
    if lo == hi:
        return float(v[lo])
    frac = rank - lo
    return float(v[lo] * (1.0 - frac) + v[hi] * frac)


def latest_user_query(messages: Any) -> str:
    if not isinstance(messages, list):
        return ""
    for m in reversed(messages):
        if not isinstance(m, dict):
            continue
        if str(m.get("role", "")).lower() != "user":
            continue
        content = m.get("content")
        if isinstance(content, str) and content.strip():
            return content.strip()
        if isinstance(content, list):
            parts: List[str] = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    txt = item.get("text")
                    if isinstance(txt, str) and txt.strip():
                        parts.append(txt.strip())
            if parts:
                return "\n".join(parts)
    return ""


def load_queries(dataset_path: Path, query_field: str, id_field: str, limit: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with dataset_path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            if limit > 0 and len(rows) >= limit:
                break
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            query = str(obj.get(query_field, "")).strip()
            if not query:
                query = latest_user_query(obj.get("messages"))
            if not query:
                continue
            row_id = str(obj.get(id_field) or f"row_{i:06d}")
            rows.append(
                {
                    "id": row_id,
                    "query": query,
                    "scenario": str(obj.get("scenario", "")),
                    "gold_intent": str(obj.get("gold_intent", "")),
                }
            )
    return rows


@dataclass
class ModeMetrics:
    mode: str
    total: int = 0
    request_ok: int = 0
    non_empty: int = 0
    total_hits_sum: int = 0
    top1_score_sum: float = 0.0
    top1_cov_sum: float = 0.0
    latencies_ms: List[float] = None

    def __post_init__(self) -> None:
        if self.latencies_ms is None:
            self.latencies_ms = []

    def update(self, ok: bool, latency_ms: float, total_hits: int, top1_score: float, top1_cov: float) -> None:
        self.total += 1
        self.latencies_ms.append(float(latency_ms))
        if ok:
            self.request_ok += 1
        if total_hits > 0:
            self.non_empty += 1
        self.total_hits_sum += int(total_hits)
        self.top1_score_sum += float(top1_score)
        self.top1_cov_sum += float(top1_cov)

    def to_summary(self) -> Dict[str, Any]:
        total = max(1, self.total)
        return {
            "mode": self.mode,
            "total": self.total,
            "request_ok_rate": self.request_ok / total,
            "non_empty_rate": self.non_empty / total,
            "avg_total_hits": self.total_hits_sum / total,
            "avg_top1_score": self.top1_score_sum / total,
            "avg_top1_term_coverage": self.top1_cov_sum / total,
            "latency_ms_p50": percentile(self.latencies_ms, 50),
            "latency_ms_p95": percentile(self.latencies_ms, 95),
            "latency_ms_mean": (sum(self.latencies_ms) / len(self.latencies_ms)) if self.latencies_ms else 0.0,
        }


def run_mode(
    api_base: str,
    api_key: str,
    rows: List[Dict[str, Any]],
    mode: str,
    top_k: int,
    min_score: float,
    min_score_vec: float,
    fusion: str,
    candidate_k: int,
    rrf_k: int,
    weight_lex: float,
    weight_vec: float,
    enable_rerank: bool,
    timeout_s: float,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    url = api_base.rstrip("/") + "/retrieve"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    metrics = ModeMetrics(mode=mode)
    out_rows: List[Dict[str, Any]] = []

    for row in rows:
        payload = {
            "query": row["query"],
            "top_k": top_k,
            "min_score": min_score,
            "min_score_vec": min_score_vec,
            "retrieval_mode": mode,
            "fusion": fusion,
            "candidate_k": candidate_k,
            "rrf_k": rrf_k,
            "weight_lex": weight_lex,
            "weight_vec": weight_vec,
            "enable_rerank": enable_rerank,
        }

        started = time.perf_counter()
        ok = False
        status_code = 0
        err = ""
        data: Dict[str, Any] = {}
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=timeout_s)
            status_code = int(resp.status_code)
            if resp.status_code == 200:
                data = resp.json()
                ok = True
            else:
                err = resp.text[:500]
        except Exception as exc:
            err = str(exc)
        latency_ms = (time.perf_counter() - started) * 1000.0

        hits = data.get("hits") if isinstance(data, dict) else None
        if not isinstance(hits, list):
            hits = []
        total_hits = int(data.get("total_hits", len(hits))) if isinstance(data, dict) else 0

        top1_score = 0.0
        top1_cov = 0.0
        top1_source = ""
        effective_mode = ""
        if hits:
            top1 = hits[0]
            top1_score = float(top1.get("score", 0.0) or 0.0)
            top1_cov = term_coverage_score(row["query"], str(top1.get("text", "")))
            top1_source = str(top1.get("source_file", ""))
            effective_mode = str(top1.get("retrieval_mode", ""))
        elif isinstance(data, dict):
            effective_mode = str(data.get("effective_retrieval_mode", ""))

        metrics.update(
            ok=ok,
            latency_ms=latency_ms,
            total_hits=total_hits,
            top1_score=top1_score,
            top1_cov=top1_cov,
        )

        out_rows.append(
            {
                "id": row["id"],
                "scenario": row["scenario"],
                "gold_intent": row["gold_intent"],
                "query": row["query"],
                "mode": mode,
                "ok": ok,
                "status_code": status_code,
                "latency_ms": round(latency_ms, 3),
                "total_hits": total_hits,
                "top1_score": round(top1_score, 6),
                "top1_term_coverage": round(top1_cov, 6),
                "top1_source_file": top1_source,
                "effective_mode": effective_mode,
                "error": err,
            }
        )

    return out_rows, metrics.to_summary()


def build_pair_rows(long_rows: List[Dict[str, Any]], mode_a: str, mode_b: str) -> List[Dict[str, Any]]:
    by_id_mode: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for r in long_rows:
        by_id_mode[(str(r["id"]), str(r["mode"]))] = r

    ids = sorted({str(r["id"]) for r in long_rows})
    out: List[Dict[str, Any]] = []
    for qid in ids:
        ra = by_id_mode.get((qid, mode_a), {})
        rb = by_id_mode.get((qid, mode_b), {})
        if not ra and not rb:
            continue
        row = {
            "id": qid,
            "query": str(ra.get("query") or rb.get("query") or ""),
            f"{mode_a}_ok": bool(ra.get("ok", False)),
            f"{mode_b}_ok": bool(rb.get("ok", False)),
            f"{mode_a}_latency_ms": float(ra.get("latency_ms", 0.0) or 0.0),
            f"{mode_b}_latency_ms": float(rb.get("latency_ms", 0.0) or 0.0),
            f"{mode_a}_total_hits": int(ra.get("total_hits", 0) or 0),
            f"{mode_b}_total_hits": int(rb.get("total_hits", 0) or 0),
            f"{mode_a}_top1_term_coverage": float(ra.get("top1_term_coverage", 0.0) or 0.0),
            f"{mode_b}_top1_term_coverage": float(rb.get("top1_term_coverage", 0.0) or 0.0),
        }
        row["delta_latency_ms"] = row[f"{mode_b}_latency_ms"] - row[f"{mode_a}_latency_ms"]
        row["delta_total_hits"] = row[f"{mode_b}_total_hits"] - row[f"{mode_a}_total_hits"]
        row["delta_top1_term_coverage"] = (
            row[f"{mode_b}_top1_term_coverage"] - row[f"{mode_a}_top1_term_coverage"]
        )
        out.append(row)
    return out


def write_csv(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    rows = list(rows)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="A/B compare retrieval modes via RAG /retrieve API")
    parser.add_argument("--api-base", default="http://127.0.0.1:18080", help="RAG API base URL")
    parser.add_argument("--api-key", default=os.getenv("RAG_API_KEY", ""), help="RAG API key (Bearer)")
    parser.add_argument(
        "--dataset",
        default="/home/ubuntu/qwen35a3b_finetune/datasets/eval_sports_customer_prod_500.jsonl",
        help="JSONL dataset path",
    )
    parser.add_argument("--query-field", default="user_query", help="Query field name in JSONL")
    parser.add_argument("--id-field", default="id", help="ID field name in JSONL")
    parser.add_argument("--limit", type=int, default=0, help="Limit rows (0 means all)")
    parser.add_argument("--modes", default="tfidf,hybrid", help="Comma-separated retrieval modes to compare")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--min-score", type=float, default=0.01)
    parser.add_argument("--min-score-vec", type=float, default=0.0)
    parser.add_argument("--fusion", default="rrf", choices=["rrf", "weighted"])
    parser.add_argument("--candidate-k", type=int, default=40)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--weight-lex", type=float, default=0.55)
    parser.add_argument("--weight-vec", type=float, default=0.45)
    parser.add_argument("--disable-rerank", action="store_true")
    parser.add_argument("--timeout-s", type=float, default=20.0)
    parser.add_argument("--output-dir", default="", help="Output directory")
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    rows = load_queries(
        dataset_path=dataset_path,
        query_field=args.query_field,
        id_field=args.id_field,
        limit=max(0, int(args.limit)),
    )
    if not rows:
        raise RuntimeError("No valid query rows found in dataset")

    modes = [m.strip().lower() for m in str(args.modes).split(",") if m.strip()]
    if not modes:
        raise RuntimeError("No retrieval mode provided")

    run_tag = now_utc_compact()
    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        out_dir = Path("/home/ubuntu/generate/tydata_rag/ab_outputs") / f"retrieval_ab_{run_tag}"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_long_rows: List[Dict[str, Any]] = []
    summaries: List[Dict[str, Any]] = []
    for mode in modes:
        mode_rows, mode_summary = run_mode(
            api_base=args.api_base,
            api_key=args.api_key,
            rows=rows,
            mode=mode,
            top_k=args.top_k,
            min_score=args.min_score,
            min_score_vec=args.min_score_vec,
            fusion=args.fusion,
            candidate_k=args.candidate_k,
            rrf_k=args.rrf_k,
            weight_lex=args.weight_lex,
            weight_vec=args.weight_vec,
            enable_rerank=not args.disable_rerank,
            timeout_s=args.timeout_s,
        )
        all_long_rows.extend(mode_rows)
        summaries.append(mode_summary)

    long_csv = out_dir / "ab_results_long.csv"
    write_csv(long_csv, all_long_rows)

    pair_csv: Optional[Path] = None
    pair_rows: List[Dict[str, Any]] = []
    if len(modes) >= 2:
        pair_rows = build_pair_rows(all_long_rows, modes[0], modes[1])
        pair_csv = out_dir / "ab_results_pair.csv"
        write_csv(pair_csv, pair_rows)

    summary: Dict[str, Any] = {
        "run_tag": run_tag,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "api_base": args.api_base,
        "dataset": str(dataset_path),
        "num_queries": len(rows),
        "modes": modes,
        "retrieval_params": {
            "top_k": args.top_k,
            "min_score": args.min_score,
            "min_score_vec": args.min_score_vec,
            "fusion": args.fusion,
            "candidate_k": args.candidate_k,
            "rrf_k": args.rrf_k,
            "weight_lex": args.weight_lex,
            "weight_vec": args.weight_vec,
            "enable_rerank": not args.disable_rerank,
            "timeout_s": args.timeout_s,
        },
        "per_mode": summaries,
        "outputs": {
            "ab_results_long_csv": str(long_csv),
            "ab_results_pair_csv": str(pair_csv) if pair_csv else "",
        },
    }

    if len(summaries) >= 2:
        first = summaries[0]
        second = summaries[1]
        summary["delta_1_vs_0"] = {
            "mode_0": first["mode"],
            "mode_1": second["mode"],
            "request_ok_rate": float(second["request_ok_rate"]) - float(first["request_ok_rate"]),
            "non_empty_rate": float(second["non_empty_rate"]) - float(first["non_empty_rate"]),
            "avg_total_hits": float(second["avg_total_hits"]) - float(first["avg_total_hits"]),
            "avg_top1_term_coverage": float(second["avg_top1_term_coverage"])
            - float(first["avg_top1_term_coverage"]),
            "latency_ms_p95": float(second["latency_ms_p95"]) - float(first["latency_ms_p95"]),
        }
        if pair_rows:
            better_cov = sum(1 for r in pair_rows if float(r["delta_top1_term_coverage"]) > 1e-9)
            better_hits = sum(1 for r in pair_rows if float(r["delta_total_hits"]) > 0)
            summary["pairwise_counts"] = {
                "num_pairs": len(pair_rows),
                "mode_1_better_top1_coverage": better_cov,
                "mode_1_better_total_hits": better_hits,
            }

    summary_json = out_dir / "summary.json"
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
