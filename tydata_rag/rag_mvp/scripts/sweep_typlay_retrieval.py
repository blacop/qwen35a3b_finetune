#!/usr/bin/python3
from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = Path(__file__).resolve().parents[1]
BUILD_SCRIPT = REPO_ROOT / "build_rag_index.py"
EVAL_SCRIPT = PROJECT_ROOT / "scripts" / "eval_runner.py"
MULTI_EVAL_SCRIPT = PROJECT_ROOT / "scripts" / "eval_multi_evidence.py"
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "default.yaml"
DEFAULT_DATASET = PROJECT_ROOT / "data" / "eval" / "eval_questions.jsonl"
DEFAULT_MULTI_DATASET = PROJECT_ROOT / "data" / "eval" / "eval_questions_multi_evidence.jsonl"
DEFAULT_PYTHON = os.environ.get("RAG_SWEEP_PYTHON", "/usr/bin/python3")
DEFAULT_TYPLAY_TEXT_DIR = "/video-storage/ai-customer/qwen35a3b_finetune/rag/typlay/text_all"


def load_yaml(path: Path) -> Dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def run_cmd(cmd: List[str]) -> None:
    print("+ " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def now_utc_compact() -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


def parse_csv_ints(raw: str) -> List[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def parse_csv_floats(raw: str) -> List[float]:
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def parse_chunk_pairs(raw: str) -> List[tuple[int, int]]:
    out: List[tuple[int, int]] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        left, _, right = item.partition(":")
        if not left or not right:
            raise ValueError(f"Invalid chunk pair: {item}")
        out.append((int(left), int(right)))
    return out


def latest_file(directory: Path, pattern: str) -> Path:
    files = sorted(directory.glob(pattern), key=lambda p: p.stat().st_mtime)
    if not files:
        raise FileNotFoundError(f"No file matched {pattern} in {directory}")
    return files[-1]


def write_yaml(path: Path, data: Dict[str, Any]) -> None:
    path.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")


def build_variant_cfg(
    base_cfg: Dict[str, Any],
    *,
    index_dir: Path,
    output_dir: Path,
    dataset: Path,
    chunk_size: int,
    overlap: int,
    min_score: float,
    min_score_vec: float,
    candidate_k: int,
    rrf_k: int,
    weight_lex: float,
    weight_vec: float,
) -> Dict[str, Any]:
    cfg = json.loads(json.dumps(base_cfg, ensure_ascii=False))
    cfg.setdefault("chunk", {})
    cfg.setdefault("index", {})
    cfg.setdefault("paths", {})
    cfg.setdefault("retrieval", {})
    cfg.setdefault("evaluation", {})

    cfg["chunk"]["chunk_size"] = int(chunk_size)
    cfg["chunk"]["overlap"] = int(overlap)
    cfg["index"]["output_dir"] = str(index_dir)
    cfg["paths"]["index_dir"] = str(index_dir)
    cfg["paths"]["output_dir"] = str(output_dir)
    cfg["paths"]["eval_dataset"] = str(dataset)

    retrieval = cfg["retrieval"]
    retrieval["min_score"] = float(min_score)
    retrieval["min_score_vec"] = float(min_score_vec)
    retrieval["candidate_k"] = int(candidate_k)
    retrieval["rrf_k"] = int(rrf_k)
    retrieval["weight_lex"] = float(weight_lex)
    retrieval["weight_vec"] = float(weight_vec)
    return cfg


def summarize_group(group: Dict[str, Any], key: str) -> Dict[str, Any]:
    row = group.get(key, {}) if isinstance(group, dict) else {}
    return {
        "total": int(row.get("total", 0) or 0),
        "recall@1": float(row.get("recall@1", 0.0) or 0.0),
        "answer_supported_rate": float(row.get("answer_supported_rate", 0.0) or 0.0),
    }


def main() -> int:
    p = argparse.ArgumentParser(description="Sweep typlay chunk/retrieval params and compare main vs multi-evidence metrics")
    p.add_argument("--config", default=str(DEFAULT_CONFIG))
    p.add_argument("--dataset", default=str(DEFAULT_DATASET))
    p.add_argument("--multi-dataset", default=str(DEFAULT_MULTI_DATASET))
    p.add_argument("--work-dir", default=str(PROJECT_ROOT / "outputs" / "sweeps"))
    p.add_argument("--chunk-pairs", default="")
    p.add_argument("--chunk-sizes", default="480,560")
    p.add_argument("--overlaps", default="96,112")
    p.add_argument("--min-scores", default="0.0,0.01")
    p.add_argument("--min-score-vecs", default="0.0")
    p.add_argument("--candidate-ks", default="60,80")
    p.add_argument("--rrf-ks", default="80")
    p.add_argument("--weight-lexes", default="0.7")
    p.add_argument("--weight-vecs", default="0.3")
    p.add_argument("--keep-indexes", action="store_true")
    args = p.parse_args()

    base_cfg = load_yaml(Path(args.config))
    work_dir = Path(args.work_dir)
    run_dir = work_dir / f"sweep_{now_utc_compact()}"
    ensure_dir(run_dir)

    chunk_sizes = parse_csv_ints(args.chunk_sizes)
    overlaps = parse_csv_ints(args.overlaps)
    min_scores = parse_csv_floats(args.min_scores)
    min_score_vecs = parse_csv_floats(args.min_score_vecs)
    candidate_ks = parse_csv_ints(args.candidate_ks)
    rrf_ks = parse_csv_ints(args.rrf_ks)
    weight_lexes = parse_csv_floats(args.weight_lexes)
    weight_vecs = parse_csv_floats(args.weight_vecs)

    rows: List[Dict[str, Any]] = []
    if args.chunk_pairs.strip():
        chunk_pairs = parse_chunk_pairs(args.chunk_pairs)
    else:
        chunk_pairs = list(itertools.product(chunk_sizes, overlaps))

    combos: Iterable[Any] = itertools.product(
        chunk_pairs,
        min_scores,
        min_score_vecs,
        candidate_ks,
        rrf_ks,
        weight_lexes,
        weight_vecs,
    )

    for idx, combo in enumerate(combos, start=1):
        (chunk_size, overlap), min_score, min_score_vec, candidate_k, rrf_k, weight_lex, weight_vec = combo
        if weight_lex + weight_vec <= 0:
            continue
        tag = f"c{chunk_size}_o{overlap}_ms{min_score}_mv{min_score_vec}_ck{candidate_k}_rrf{rrf_k}_wl{weight_lex}_wv{weight_vec}"
        variant_dir = run_dir / tag
        index_dir = variant_dir / "index"
        eval_out_dir = variant_dir / "main_eval"
        multi_out_dir = variant_dir / "multi_eval"
        ensure_dir(variant_dir)
        ensure_dir(eval_out_dir)
        ensure_dir(multi_out_dir)

        cfg = build_variant_cfg(
            base_cfg,
            index_dir=index_dir,
            output_dir=eval_out_dir,
            dataset=Path(args.dataset),
            chunk_size=chunk_size,
            overlap=overlap,
            min_score=min_score,
            min_score_vec=min_score_vec,
            candidate_k=candidate_k,
            rrf_k=rrf_k,
            weight_lex=weight_lex,
            weight_vec=weight_vec,
        )
        cfg_path = variant_dir / "config.yaml"
        write_yaml(cfg_path, cfg)

        build_cmd = [
            DEFAULT_PYTHON,
            str(BUILD_SCRIPT),
            "--input-dir",
            str(base_cfg.get("ingest", {}).get("source_dir", DEFAULT_TYPLAY_TEXT_DIR)),
            "--output-dir",
            str(index_dir),
            "--chunk-size",
            str(chunk_size),
            "--overlap",
            str(overlap),
            "--ngram-min",
            str(base_cfg.get("index", {}).get("ngram_min", 1)),
            "--ngram-max",
            str(base_cfg.get("index", {}).get("ngram_max", 3)),
            "--dense-backend",
            str(base_cfg.get("index", {}).get("dense_backend", "svd")),
            "--dense-dim",
            str(base_cfg.get("index", {}).get("dense_dim", 256)),
        ]
        run_cmd(build_cmd)

        eval_cmd = [
            DEFAULT_PYTHON,
            str(EVAL_SCRIPT),
            "--config",
            str(cfg_path),
            "--dataset",
            str(args.dataset),
            "--index-dir",
            str(index_dir),
            "--output-dir",
            str(eval_out_dir),
            "--quiet",
        ]
        run_cmd(eval_cmd)
        summary_path = latest_file(eval_out_dir, "summary_*.json")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))

        multi_cfg = json.loads(json.dumps(cfg, ensure_ascii=False))
        multi_cfg["paths"]["eval_dataset"] = str(args.multi_dataset)
        multi_cfg["retrieval"]["top_k"] = max(int(multi_cfg["retrieval"].get("top_k", 5)), 8)
        multi_cfg["retrieval"]["enable_diversity"] = True
        multi_cfg["retrieval"]["max_per_source"] = 1
        multi_cfg["retrieval"]["max_per_family"] = 4
        multi_cfg["retrieval"]["diversity_lambda"] = 0.5
        multi_cfg["retrieval"]["source_match_weight"] = 0.12
        multi_cfg["retrieval"]["candidate_k"] = max(int(candidate_k), 80)
        multi_cfg_path = variant_dir / "config_multi.yaml"
        write_yaml(multi_cfg_path, multi_cfg)

        multi_cmd = [
            DEFAULT_PYTHON,
            str(MULTI_EVAL_SCRIPT),
            "--config",
            str(multi_cfg_path),
            "--dataset",
            str(args.multi_dataset),
            "--index-dir",
            str(index_dir),
            "--output-dir",
            str(multi_out_dir),
        ]
        run_cmd(multi_cmd)
        multi_summary_path = latest_file(multi_out_dir, "multi_evidence_summary_*.json")
        multi_summary = json.loads(multi_summary_path.read_text(encoding="utf-8"))

        row = {
            "variant": tag,
            "chunk_size": chunk_size,
            "overlap": overlap,
            "min_score": min_score,
            "min_score_vec": min_score_vec,
            "candidate_k": candidate_k,
            "rrf_k": rrf_k,
            "weight_lex": weight_lex,
            "weight_vec": weight_vec,
            "main_recall@1": summary.get("recall@1", 0.0),
            "main_answer_supported_rate": summary.get("answer_supported_rate", 0.0),
            "main_mrr": summary.get("mrr", 0.0),
            "main_latency_ms_p95": summary.get("latency_ms_p95", 0.0),
            "main_cross_doc_recall@1": summarize_group(summary.get("by_tag", {}), "跨文档对比").get("recall@1", 0.0),
            "main_cross_rule_recall@1": summarize_group(summary.get("by_tag", {}), "规则差异").get("recall@1", 0.0),
            "multi_full_evidence_cover_rate": multi_summary.get("full_evidence_cover_rate", 0.0),
            "multi_avg_evidence_coverage_rate": multi_summary.get("avg_evidence_coverage_rate", 0.0),
            "multi_avg_source_coverage_rate": multi_summary.get("avg_source_coverage_rate", 0.0),
            "main_summary": str(summary_path),
            "multi_summary": str(multi_summary_path),
        }
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False, indent=2))

        if not args.keep_indexes:
            shutil.rmtree(index_dir, ignore_errors=True)

    rows.sort(
        key=lambda x: (
            -float(x["main_recall@1"]),
            -float(x["main_answer_supported_rate"]),
            -float(x["multi_full_evidence_cover_rate"]),
            -float(x["multi_avg_evidence_coverage_rate"]),
            float(x["main_latency_ms_p95"]),
        )
    )

    csv_path = run_dir / "sweep_results.csv"
    summary_path = run_dir / "best_variants.json"
    if rows:
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    summary = {
        "run_dir": str(run_dir),
        "variants": len(rows),
        "best_overall": rows[:5],
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
