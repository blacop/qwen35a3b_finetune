#!/usr/bin/python3
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


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


def normalize_text(text: str) -> str:
    return "".join(str(text or "").split())


def load_retriever(index_dir: Path) -> Any:
    from retrieval_hybrid import HybridRetriever  # type: ignore

    return HybridRetriever(index_dir)


def evidence_matches(hit: Dict[str, Any], evidence: Dict[str, Any]) -> bool:
    source_file = str(evidence.get("source_file") or "").strip()
    if source_file and str(hit.get("source_file") or "").strip() != source_file:
        return False
    contains = normalize_text(str(evidence.get("contains") or ""))
    if not contains:
        return True
    return contains in normalize_text(str(hit.get("text") or ""))


def coverage_metrics(hits: Sequence[Dict[str, Any]], evidences: Sequence[Dict[str, Any]]) -> Tuple[int, int, float]:
    matched = 0
    matched_sources = set()
    for evidence in evidences:
        if any(evidence_matches(hit, evidence) for hit in hits):
            matched += 1
            matched_sources.add(str(evidence.get("source_file") or ""))
    total = len(evidences)
    rate = float(matched) / float(total) if total > 0 else 0.0
    return matched, len(matched_sources), rate


def build_search_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    retrieval = cfg.get("retrieval", {}) if isinstance(cfg.get("retrieval", {}), dict) else {}
    return {
        "top_k": int(retrieval.get("top_k", 5)),
        "min_score": float(retrieval.get("min_score", 0.0)),
        "min_score_vec": float(retrieval.get("min_score_vec", 0.0)),
        "mode": str(retrieval.get("mode", "hybrid")),
        "fusion": str(retrieval.get("fusion", "rrf")),
        "candidate_k": int(retrieval.get("candidate_k", 60)),
        "rrf_k": int(retrieval.get("rrf_k", 80)),
        "weight_lex": float(retrieval.get("weight_lex", 0.7)),
        "weight_vec": float(retrieval.get("weight_vec", 0.3)),
        "enable_rerank": bool(retrieval.get("enable_rerank", True)),
        "enable_diversity": bool(retrieval.get("enable_diversity", True)),
        "max_per_source": int(retrieval.get("max_per_source", 1)),
        "max_per_family": int(retrieval.get("max_per_family", 3)),
        "diversity_lambda": float(retrieval.get("diversity_lambda", 0.7)),
        "source_match_weight": float(retrieval.get("source_match_weight", 0.0)),
    }


def now_utc_compact() -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


def main() -> int:
    p = argparse.ArgumentParser(description="Evaluate multi-evidence retrieval coverage")
    p.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "default.yaml"))
    p.add_argument("--dataset", default=str(PROJECT_ROOT / "data" / "eval" / "eval_questions_multi_evidence.jsonl"))
    p.add_argument("--index-dir", required=True)
    p.add_argument("--output-dir", required=True)
    args = p.parse_args()

    cfg = load_yaml(Path(args.config))
    rows = load_jsonl(Path(args.dataset))
    ensure_dir(Path(args.output_dir))
    retriever = load_retriever(Path(args.index_dir))
    search_cfg = build_search_cfg(cfg)
    search_cfg["top_k"] = max(int(search_cfg["top_k"]), 8)

    out_rows: List[Dict[str, Any]] = []
    total = len(rows)
    full_cover = 0
    evidence_rate_sum = 0.0
    source_rate_sum = 0.0

    for row in rows:
        hits = retriever.search(query=row["question"], **search_cfg)
        evidences = row.get("gold_evidence", [])
        matched_evidence, matched_sources, evidence_rate = coverage_metrics(hits, evidences)
        source_target = len({str(x.get("source_file") or "") for x in evidences})
        source_rate = float(matched_sources) / float(source_target) if source_target else 0.0
        if matched_evidence >= len(evidences):
            full_cover += 1
        evidence_rate_sum += evidence_rate
        source_rate_sum += source_rate
        top_sources = [str(h.get("source_file") or "") for h in hits[:5]]
        out_rows.append(
            {
                "qid": row["id"],
                "question": row["question"],
                "evidence_total": len(evidences),
                "matched_evidence": matched_evidence,
                "matched_sources": matched_sources,
                "evidence_coverage_rate": round(evidence_rate, 6),
                "source_coverage_rate": round(source_rate, 6),
                "top_sources": " | ".join(top_sources),
            }
        )

    run_tag = now_utc_compact()
    csv_path = Path(args.output_dir) / f"multi_evidence_{run_tag}.csv"
    summary_path = Path(args.output_dir) / f"multi_evidence_summary_{run_tag}.json"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        writer.writeheader()
        writer.writerows(out_rows)

    summary = {
        "dataset": str(Path(args.dataset)),
        "index_dir": str(Path(args.index_dir)),
        "total": total,
        "full_evidence_cover_rate": float(full_cover) / float(total) if total else 0.0,
        "avg_evidence_coverage_rate": evidence_rate_sum / float(total) if total else 0.0,
        "avg_source_coverage_rate": source_rate_sum / float(total) if total else 0.0,
        "csv": str(csv_path),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
