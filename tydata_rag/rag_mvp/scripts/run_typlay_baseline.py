#!/usr/bin/python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = Path(__file__).resolve().parents[1]
BUILD_SCRIPT = REPO_ROOT / "build_rag_index.py"
EVAL_SCRIPT = PROJECT_ROOT / "scripts" / "eval_runner.py"
DEFAULT_TYPLAY_TEXT_DIR = "/video-storage/ai-customer/qwen35a3b_finetune/rag/typlay/text_all"
DEFAULT_TYPLAY_INDEX_DIR = "/video-storage/ai-customer/qwen35a3b_finetune/rag/typlay/rag_index_typlay_baseline_480_96"


def load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"Config not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def as_path(value: Any, fallback: str) -> Path:
    raw = str(value or fallback).strip()
    return Path(raw).expanduser()


def run_cmd(cmd: list[str]) -> None:
    print("+ " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def latest_file(directory: Path, pattern: str) -> Path | None:
    files = sorted(directory.glob(pattern), key=lambda p: p.stat().st_mtime)
    return files[-1] if files else None


def main() -> int:
    parser = argparse.ArgumentParser(description="Build and evaluate the typlay RAG baseline")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "typlay_baseline.yaml"))
    parser.add_argument("--rebuild", action="store_true", help="Remove the target index dir before building")
    parser.add_argument("--limit", type=int, default=0, help="Limit eval rows")
    args = parser.parse_args()

    cfg = load_yaml(Path(args.config))
    ingest = cfg.get("ingest", {}) if isinstance(cfg.get("ingest", {}), dict) else {}
    chunk = cfg.get("chunk", {}) if isinstance(cfg.get("chunk", {}), dict) else {}
    index = cfg.get("index", {}) if isinstance(cfg.get("index", {}), dict) else {}
    paths = cfg.get("paths", {}) if isinstance(cfg.get("paths", {}), dict) else {}
    eval_cfg = cfg.get("evaluation", {}) if isinstance(cfg.get("evaluation", {}), dict) else {}

    input_dir = as_path(ingest.get("source_dir") or paths.get("corpus_dir"), DEFAULT_TYPLAY_TEXT_DIR)
    output_dir = as_path(index.get("output_dir") or paths.get("index_dir"), DEFAULT_TYPLAY_INDEX_DIR)
    eval_out_dir = as_path(paths.get("output_dir"), str(PROJECT_ROOT / "outputs" / "typlay_baseline_480_96"))
    dataset = as_path(paths.get("eval_dataset"), str(PROJECT_ROOT / "data" / "eval" / "eval_questions.jsonl"))

    chunk_size = int(chunk.get("chunk_size", 480))
    overlap = int(chunk.get("overlap", 96))
    ngram_min = int(index.get("ngram_min", 1))
    ngram_max = int(index.get("ngram_max", 3))
    dense_backend = str(index.get("dense_backend", "svd"))
    dense_dim = int(index.get("dense_dim", 256))

    if args.rebuild and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    eval_out_dir.mkdir(parents=True, exist_ok=True)

    build_cmd = [
        "/usr/bin/python3",
        str(BUILD_SCRIPT),
        "--input-dir",
        str(input_dir),
        "--output-dir",
        str(output_dir),
        "--chunk-size",
        str(chunk_size),
        "--overlap",
        str(overlap),
        "--ngram-min",
        str(ngram_min),
        "--ngram-max",
        str(ngram_max),
        "--dense-backend",
        dense_backend,
        "--dense-dim",
        str(dense_dim),
    ]
    run_cmd(build_cmd)

    eval_cmd = [
        "/usr/bin/python3",
        str(EVAL_SCRIPT),
        "--config",
        str(args.config),
        "--dataset",
        str(dataset),
        "--index-dir",
        str(output_dir),
        "--output-dir",
        str(eval_out_dir),
    ]
    if args.limit > 0:
        eval_cmd.extend(["--limit", str(args.limit)])
    run_cmd(eval_cmd)

    summary = latest_file(eval_out_dir, "summary_*.json")
    if not summary:
        raise SystemExit(f"No summary file produced in {eval_out_dir}")

    data = json.loads(summary.read_text(encoding="utf-8"))
    print(json.dumps(data, ensure_ascii=False, indent=2))
    print(f"summary_file: {summary}")
    print(f"index_dir: {output_dir}")
    print(f"eval_output_dir: {eval_out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
