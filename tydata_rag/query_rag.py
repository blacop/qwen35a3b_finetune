#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from retrieval_hybrid import HybridRetriever


def format_prompt(query: str, hits: list[dict], max_chars: int) -> str:
    blocks = []
    for i, h in enumerate(hits, start=1):
        text = str(h.get("text", ""))[:max_chars]
        ref = (
            f"{h.get('source_file')} {h.get('unit_type')}#{h.get('unit_id')} "
            f"chunk#{h.get('chunk_index')} score={float(h.get('score', 0.0)):.4f}"
        )
        blocks.append(f"[Context {i}] {ref}\n{text}")

    joined = "\n\n".join(blocks)
    return (
        "You are given retrieved context from a local knowledge base.\n"
        "Use the context first. If context is insufficient, state uncertainty explicitly.\n\n"
        f"Question:\n{query}\n\n"
        f"Retrieved Context:\n{joined}\n\n"
        "Answer in Chinese with concise reasoning."
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Query local hybrid RAG index")
    parser.add_argument("--index-dir", default="/video-storage/ai-customer/qwen35a3b_finetune/rag/typlay/rag_index_typlay_default_560_112", help="Index directory")
    parser.add_argument("--query", required=True, help="User query")
    parser.add_argument("--top-k", type=int, default=5, help="Top-k chunks")
    parser.add_argument("--min-score", type=float, default=0.01, help="Min score for lexical search")
    parser.add_argument("--min-score-vec", type=float, default=0.0, help="Min score for vector search")
    parser.add_argument(
        "--mode",
        choices=["tfidf", "vector", "hybrid"],
        default="hybrid",
        help="Retrieval mode",
    )
    parser.add_argument(
        "--fusion",
        choices=["rrf", "weighted"],
        default="rrf",
        help="Fusion strategy for hybrid mode",
    )
    parser.add_argument("--candidate-k", type=int, default=40, help="Candidate size before rerank")
    parser.add_argument("--rrf-k", type=int, default=60, help="RRF K")
    parser.add_argument("--weight-lex", type=float, default=0.55, help="Weighted fusion lexical weight")
    parser.add_argument("--weight-vec", type=float, default=0.45, help="Weighted fusion vector weight")
    parser.add_argument("--disable-rerank", action="store_true", help="Disable lightweight rerank")
    parser.add_argument(
        "--output",
        choices=["text", "json", "prompt"],
        default="text",
        help="Output format",
    )
    parser.add_argument("--max-context-chars", type=int, default=600, help="Chars per context when output=prompt")
    args = parser.parse_args()

    retriever = HybridRetriever(Path(args.index_dir))
    hits = retriever.search(
        query=args.query,
        top_k=args.top_k,
        min_score=args.min_score,
        min_score_vec=args.min_score_vec,
        mode=args.mode,
        fusion=args.fusion,
        candidate_k=args.candidate_k,
        rrf_k=args.rrf_k,
        weight_lex=args.weight_lex,
        weight_vec=args.weight_vec,
        enable_rerank=not args.disable_rerank,
    )

    if not hits:
        print("No hit above thresholds.")
        return 0

    if args.output == "json":
        print(json.dumps(hits, ensure_ascii=False, indent=2))
        return 0

    if args.output == "prompt":
        print(format_prompt(args.query, hits, args.max_context_chars))
        return 0

    for i, h in enumerate(hits, start=1):
        print(
            f"[{i}] mode={h.get('retrieval_mode')} score={float(h.get('score', 0.0)):.4f} "
            f"lex={float(h.get('score_lex', 0.0)):.4f} vec={float(h.get('score_vec', 0.0)):.4f} "
            f"file={h.get('source_file')} {h.get('unit_type')}#{h.get('unit_id')} chunk#{h.get('chunk_index')}"
        )
        print(h.get("text", ""))
        print("-" * 80)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
