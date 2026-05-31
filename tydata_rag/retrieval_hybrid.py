#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import pickle
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import sparse


def normalize_int(value: Any, default: int, min_value: int, max_value: int) -> int:
    try:
        v = int(value)
    except Exception:
        v = default
    if v < min_value:
        return min_value
    if v > max_value:
        return max_value
    return v


def normalize_float(value: Any, default: float, min_value: float, max_value: float) -> float:
    try:
        v = float(value)
    except Exception:
        v = default
    if v < min_value:
        return min_value
    if v > max_value:
        return max_value
    return v


def l2_normalize_rows(x: np.ndarray) -> np.ndarray:
    eps = 1e-12
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    return x / norms


def load_chunks(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_context_block(hits: List[Dict[str, Any]], max_context_chars: int) -> str:
    blocks = []
    for i, h in enumerate(hits, start=1):
        ref = (
            f"[{i}] {h['source_file']} {h['unit_type']}#{h['unit_id']} "
            f"chunk#{h['chunk_index']} score={float(h.get('score', 0.0)):.4f}"
        )
        text = str(h.get("text", ""))[:max_context_chars]
        blocks.append(f"{ref}\n{text}")
    return "\n\n".join(blocks)


_ALNUM_RE = re.compile(r"[A-Za-z0-9_]{2,}")
_QUERY_ALNUM_RE = re.compile(r"[A-Za-z0-9_]{1,}")
_CJK_CHAR_RE = re.compile(r"[\u4e00-\u9fff]")
_SOURCE_TOKEN_RE = re.compile(r"[A-Za-z]{2,}|\d+|[\u4e00-\u9fff]{2,}")
_GENERIC_SOURCE_TERM_RE = re.compile(
    r"(视讯|彩票|棋牌|游戏公平机制|游戏规则|游戏|百家乐|规则|玩法|介绍|澳门|经典|传统|2\.0)",
    re.IGNORECASE,
)
_QUERY_STOP_TERMS = {
    "规则",
    "玩法",
    "游戏",
    "视讯",
    "棋牌",
    "彩票",
    "百家乐",
    "这四",
    "四份",
    "这三",
    "三份",
    "这几",
    "几份",
    "两份",
    "是否",
    "一致",
    "明确",
    "采用",
}


def extract_query_terms(query: str) -> List[str]:
    q = (query or "").strip()
    if not q:
        return []
    terms: List[str] = []
    terms.extend(t.lower() for t in _QUERY_ALNUM_RE.findall(q))
    cjk_seq = "".join(_CJK_CHAR_RE.findall(q))
    if len(cjk_seq) >= 2:
        terms.extend(
            bg
            for bg in (cjk_seq[i : i + 2] for i in range(len(cjk_seq) - 1))
            if bg not in _QUERY_STOP_TERMS
        )
    # keep order while de-duplicating
    seen = set()
    ordered: List[str] = []
    for t in terms:
        if t in _QUERY_STOP_TERMS:
            continue
        if t not in seen:
            seen.add(t)
            ordered.append(t)
    return ordered


def term_coverage_score(query: str, text: str) -> float:
    terms = extract_query_terms(query)
    if not terms:
        return 0.0
    hay = (text or "").lower()
    hit = sum(1 for t in terms if t and t in hay)
    return float(hit) / float(len(terms))


def _source_tokens(source_file: str) -> List[str]:
    raw = Path(source_file or "").stem
    toks = [t.lower() for t in _SOURCE_TOKEN_RE.findall(raw)]
    seen = set()
    out: List[str] = []
    for t in toks:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def source_aliases(source_file: str) -> List[str]:
    raw = Path(source_file or "").stem.strip()
    if not raw:
        return []
    simplified = _GENERIC_SOURCE_TERM_RE.sub(" ", raw)
    simplified = re.sub(r"[\s_\-./()（）【】\[\]<>]+", " ", simplified).strip()
    aliases: List[str] = []
    aliases.extend(t.lower() for t in _ALNUM_RE.findall(simplified))
    aliases.extend(t for t in re.findall(r"[\u4e00-\u9fff]{2,4}", simplified) if t.strip())
    seen = set()
    out: List[str] = []
    for alias in aliases:
        alias = alias.strip()
        if not alias or alias in seen:
            continue
        seen.add(alias)
        out.append(alias)
    return out


def explicit_source_match_score(query: str, source_file: str) -> float:
    aliases = source_aliases(source_file)
    if not aliases:
        return 0.0
    q = (query or "").lower()
    if not q:
        return 0.0
    matched = sum(1 for alias in aliases if alias and alias.lower() in q)
    return float(matched) / float(max(1, len(aliases)))


def query_mentioned_source_aliases(query: str, source_files: List[str]) -> List[str]:
    q = (query or "").lower()
    if not q:
        return []
    seen = set()
    out: List[str] = []
    for source_file in source_files:
        for alias in source_aliases(source_file):
            lowered = alias.lower()
            if lowered in q and lowered not in seen:
                seen.add(lowered)
                out.append(lowered)
    return out


def source_name_match_score(query: str, source_file: str) -> float:
    q_terms = extract_query_terms(query)
    if not q_terms:
        return 0.0
    file_terms = _source_tokens(source_file)
    if not file_terms:
        return 0.0
    score = 0.0
    for ft in file_terms:
        if len(ft) < 2:
            continue
        if any(ft in qt or qt in ft for qt in q_terms):
            score += 1.0
    return score / float(max(1, len(file_terms)))


def normalize_source_family(source_file: str) -> str:
    stem = Path(source_file or "").stem
    stem = re.sub(r"^[A-Za-z]+", "", stem)
    stem = re.sub(r"^[\u4e00-\u9fff]{2,4}", "", stem)
    stem = stem.strip(" _-")
    if stem:
        return stem.lower()
    return Path(source_file or "").stem.lower()


def char_bigram_jaccard(a: str, b: str) -> float:
    sa = (a or "").strip()
    sb = (b or "").strip()
    if len(sa) < 2 or len(sb) < 2:
        return 0.0
    ga = {sa[i : i + 2] for i in range(len(sa) - 1)}
    gb = {sb[i : i + 2] for i in range(len(sb) - 1)}
    if not ga or not gb:
        return 0.0
    inter = len(ga & gb)
    union = len(ga | gb)
    if union <= 0:
        return 0.0
    return float(inter) / float(union)


class HybridRetriever:
    def __init__(self, index_dir: str | Path) -> None:
        self.index_dir = Path(index_dir)
        self.chunks = load_chunks(self.index_dir / "chunks.jsonl")
        self.matrix = sparse.load_npz(self.index_dir / "tfidf_matrix.npz")
        with (self.index_dir / "vectorizer.pkl").open("rb") as f:
            self.vectorizer = pickle.load(f)

        self.dense_available = False
        self.dense_backend = "none"
        self.dense_matrix: Optional[np.ndarray] = None
        self._dense_meta: Dict[str, Any] = {}
        self._dense_encoder: Any = None
        self._dense_word_vectorizer: Any = None
        self._dense_svd: Any = None
        self._load_dense_index()

    def _load_dense_index(self) -> None:
        dense_vec_path = self.index_dir / "dense_vectors.npy"
        dense_meta_path = self.index_dir / "dense_meta.json"
        if not dense_vec_path.exists() or not dense_meta_path.exists():
            return

        self.dense_matrix = np.asarray(np.load(dense_vec_path), dtype=np.float32)
        self._dense_meta = json.loads(dense_meta_path.read_text(encoding="utf-8"))
        self.dense_backend = str(self._dense_meta.get("backend", "none"))

        if self.dense_backend == "svd":
            vec_path = self.index_dir / "dense_word_vectorizer.pkl"
            svd_path = self.index_dir / "dense_svd.pkl"
            if not vec_path.exists() or not svd_path.exists():
                return
            with vec_path.open("rb") as f:
                self._dense_word_vectorizer = pickle.load(f)
            with svd_path.open("rb") as f:
                self._dense_svd = pickle.load(f)
            self.dense_available = True
            return

        if self.dense_backend == "sbert":
            model_name = str(self._dense_meta.get("model_name", "")).strip()
            if not model_name:
                return
            try:
                from sentence_transformers import SentenceTransformer  # type: ignore

                self._dense_encoder = SentenceTransformer(model_name, device="cpu")
                self.dense_available = True
            except Exception:
                self.dense_available = False
            return

        # Unknown backend
        self.dense_available = False

    def _encode_dense_query(self, query: str) -> Optional[np.ndarray]:
        q = (query or "").strip()
        if not q or not self.dense_available:
            return None

        if self.dense_backend == "svd":
            if self._dense_word_vectorizer is None or self._dense_svd is None:
                return None
            q_sparse = self._dense_word_vectorizer.transform([q])
            q_vec = self._dense_svd.transform(q_sparse)
            q_vec = np.asarray(q_vec, dtype=np.float32)
            q_vec = l2_normalize_rows(q_vec)
            return q_vec[0]

        if self.dense_backend == "sbert":
            if self._dense_encoder is None:
                return None
            q_vec = self._dense_encoder.encode(
                [q],
                normalize_embeddings=True,
                convert_to_numpy=True,
            )
            q_vec = np.asarray(q_vec, dtype=np.float32)
            return q_vec[0]
        return None

    def lexical_search(
        self,
        query: str,
        top_k: int,
        min_score: float,
    ) -> List[Dict[str, Any]]:
        q = (query or "").strip()
        if not q:
            return []
        qvec = self.vectorizer.transform([q])
        scores = (self.matrix @ qvec.T).toarray().ravel()
        order = scores.argsort()[::-1]

        hits: List[Dict[str, Any]] = []
        for idx in order:
            score = float(scores[int(idx)])
            if score < min_score:
                continue
            row = dict(self.chunks[int(idx)])
            row["score_lex"] = score
            hits.append(row)
            if len(hits) >= top_k:
                break
        return hits

    def vector_search(
        self,
        query: str,
        top_k: int,
        min_score: float,
    ) -> List[Dict[str, Any]]:
        if not self.dense_available or self.dense_matrix is None:
            return []
        q_vec = self._encode_dense_query(query)
        if q_vec is None:
            return []

        scores = self.dense_matrix @ q_vec
        order = np.argsort(scores)[::-1]
        hits: List[Dict[str, Any]] = []
        for idx in order:
            score = float(scores[int(idx)])
            if score < min_score:
                continue
            row = dict(self.chunks[int(idx)])
            row["score_vec"] = score
            hits.append(row)
            if len(hits) >= top_k:
                break
        return hits

    @staticmethod
    def _normalize_scores(score_map: Dict[int, float]) -> Dict[int, float]:
        if not score_map:
            return {}
        vals = list(score_map.values())
        hi = max(vals)
        lo = min(vals)
        if hi - lo < 1e-12:
            return {k: 1.0 for k in score_map.keys()}
        return {k: (v - lo) / (hi - lo) for k, v in score_map.items()}

    def _fuse_rrf(
        self,
        lex_hits: List[Dict[str, Any]],
        vec_hits: List[Dict[str, Any]],
        rrf_k: int,
    ) -> Dict[int, Dict[str, float]]:
        fused: Dict[int, Dict[str, float]] = {}

        for rank, hit in enumerate(lex_hits, start=1):
            idx = int(hit["chunk_id"]) - 1
            slot = fused.setdefault(idx, {"score_lex": 0.0, "score_vec": 0.0, "score_fusion": 0.0})
            slot["score_lex"] = float(hit.get("score_lex", 0.0))
            slot["score_fusion"] += 1.0 / float(rrf_k + rank)

        for rank, hit in enumerate(vec_hits, start=1):
            idx = int(hit["chunk_id"]) - 1
            slot = fused.setdefault(idx, {"score_lex": 0.0, "score_vec": 0.0, "score_fusion": 0.0})
            slot["score_vec"] = float(hit.get("score_vec", 0.0))
            slot["score_fusion"] += 1.0 / float(rrf_k + rank)
        return fused

    def _fuse_weighted(
        self,
        lex_hits: List[Dict[str, Any]],
        vec_hits: List[Dict[str, Any]],
        weight_lex: float,
        weight_vec: float,
    ) -> Dict[int, Dict[str, float]]:
        lex_map = {int(h["chunk_id"]) - 1: float(h.get("score_lex", 0.0)) for h in lex_hits}
        vec_map = {int(h["chunk_id"]) - 1: float(h.get("score_vec", 0.0)) for h in vec_hits}
        norm_lex = self._normalize_scores(lex_map)
        norm_vec = self._normalize_scores(vec_map)
        all_ids = set(norm_lex.keys()) | set(norm_vec.keys())

        fused: Dict[int, Dict[str, float]] = {}
        for idx in all_ids:
            s_lex = float(lex_map.get(idx, 0.0))
            s_vec = float(vec_map.get(idx, 0.0))
            f = float(weight_lex) * float(norm_lex.get(idx, 0.0)) + float(weight_vec) * float(norm_vec.get(idx, 0.0))
            fused[idx] = {"score_lex": s_lex, "score_vec": s_vec, "score_fusion": f}
        return fused

    def _rerank(
        self,
        query: str,
        fused_rows: List[Tuple[int, Dict[str, float]]],
        source_match_weight: float,
    ) -> List[Tuple[int, Dict[str, float]]]:
        out: List[Tuple[int, Dict[str, float]]] = []
        source_match_weight = min(1.0, max(0.0, float(source_match_weight)))
        candidate_source_files = [
            str(self.chunks[int(idx)].get("source_file", ""))
            for idx, _ in fused_rows
        ]
        mentioned_aliases = query_mentioned_source_aliases(query, candidate_source_files)
        explicit_focus = len(mentioned_aliases)
        for idx, score_obj in fused_rows:
            chunk = self.chunks[int(idx)]
            text = str(chunk.get("text", ""))
            source_file = str(chunk.get("source_file", ""))
            fusion_score = float(score_obj.get("score_fusion", 0.0))
            cov = term_coverage_score(query, text)
            jac = char_bigram_jaccard(query, text)
            source_match = source_name_match_score(query, source_file)
            explicit_source_match = explicit_source_match_score(query, source_file)
            if explicit_focus > 0:
                explicit_bonus = 0.18 * explicit_source_match
                explicit_penalty = 0.06 if explicit_source_match <= 0.0 else 0.0
            else:
                explicit_bonus = 0.0
                explicit_penalty = 0.0
            base_fusion_weight = max(0.0, 0.70 - source_match_weight)
            rerank_score = (
                base_fusion_weight * fusion_score
                + 0.20 * cov
                + 0.10 * jac
                + source_match_weight * source_match
                + explicit_bonus
                - explicit_penalty
            )

            merged = dict(score_obj)
            merged["score_rerank"] = float(rerank_score)
            merged["score_source_match"] = float(source_match)
            merged["score_explicit_source_match"] = float(explicit_source_match)
            out.append((idx, merged))
        out.sort(key=lambda x: x[1].get("score_rerank", 0.0), reverse=True)
        return out

    def _apply_diversity(
        self,
        fused_rows: List[Tuple[int, Dict[str, float]]],
        top_k: int,
        max_per_source: int,
        max_per_family: int,
        diversity_lambda: float,
    ) -> List[Tuple[int, Dict[str, float]]]:
        max_per_source = max(1, int(max_per_source))
        max_per_family = max(1, int(max_per_family))
        diversity_lambda = min(1.0, max(0.0, float(diversity_lambda)))

        source_counts: Dict[str, int] = {}
        family_counts: Dict[str, int] = {}
        selected: List[Tuple[int, Dict[str, float]]] = []
        deferred: List[Tuple[float, int, Dict[str, float]]] = []

        for idx, scores in fused_rows:
            chunk = self.chunks[int(idx)]
            source_file = str(chunk.get("source_file", ""))
            family = normalize_source_family(source_file)
            text = str(chunk.get("text", ""))
            s_count = source_counts.get(source_file, 0)
            f_count = family_counts.get(family, 0)
            if s_count >= max_per_source or f_count >= max_per_family:
                penalty = math.exp(-(s_count + f_count + 1) * max(0.05, diversity_lambda))
                deferred.append((penalty, idx, scores))
                continue
            if s_count > 0:
                same_source_texts = [
                    str(self.chunks[int(sel_idx)].get("text", ""))
                    for sel_idx, _ in selected
                    if str(self.chunks[int(sel_idx)].get("source_file", "")) == source_file
                ]
                if same_source_texts:
                    max_sim = max(char_bigram_jaccard(text, prior) for prior in same_source_texts)
                    if max_sim >= 0.35:
                        penalty = max(0.05, 1.0 - diversity_lambda * max_sim)
                        deferred.append((penalty, idx, scores))
                        continue
            source_counts[source_file] = s_count + 1
            family_counts[family] = f_count + 1
            selected.append((idx, scores))
            if len(selected) >= top_k:
                return selected

        if len(selected) >= top_k:
            return selected[:top_k]

        deferred.sort(
            key=lambda item: item[2].get("score_rerank", item[2].get("score_fusion", 0.0)) * item[0],
            reverse=True,
        )
        for _, idx, scores in deferred:
            selected.append((idx, scores))
            if len(selected) >= top_k:
                break
        return selected[:top_k]

    def search(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.01,
        mode: str = "hybrid",
        min_score_vec: Optional[float] = None,
        candidate_k: Optional[int] = None,
        fusion: str = "rrf",
        rrf_k: int = 60,
        weight_lex: float = 0.55,
        weight_vec: float = 0.45,
        enable_rerank: bool = True,
        enable_diversity: bool = True,
        max_per_source: int = 1,
        max_per_family: int = 3,
        diversity_lambda: float = 0.7,
        source_match_weight: float = 0.0,
    ) -> List[Dict[str, Any]]:
        q = (query or "").strip()
        if not q:
            return []

        top_k = normalize_int(top_k, default=5, min_value=1, max_value=100)
        min_score = normalize_float(min_score, default=0.01, min_value=0.0, max_value=1.0)
        if min_score_vec is None:
            min_score_vec = max(0.0, min_score * 0.2)
        min_score_vec = normalize_float(min_score_vec, default=0.0, min_value=-1.0, max_value=1.0)
        candidate_k = normalize_int(candidate_k, default=max(top_k * 4, 40), min_value=top_k, max_value=300)
        rrf_k = normalize_int(rrf_k, default=60, min_value=1, max_value=2000)
        weight_lex = normalize_float(weight_lex, default=0.55, min_value=0.0, max_value=1.0)
        weight_vec = normalize_float(weight_vec, default=0.45, min_value=0.0, max_value=1.0)
        w_sum = weight_lex + weight_vec
        if w_sum <= 1e-12:
            weight_lex, weight_vec = 0.55, 0.45
            w_sum = 1.0
        weight_lex /= w_sum
        weight_vec /= w_sum

        mode = (mode or "hybrid").strip().lower()
        fusion = (fusion or "rrf").strip().lower()

        if mode == "tfidf":
            lex_hits = self.lexical_search(q, top_k=top_k, min_score=min_score)
            out: List[Dict[str, Any]] = []
            for h in lex_hits:
                row = dict(h)
                row["score"] = float(row.get("score_lex", 0.0))
                row["retrieval_mode"] = "tfidf"
                out.append(row)
            return out

        if mode == "vector":
            vec_hits = self.vector_search(q, top_k=top_k, min_score=min_score_vec)
            out = []
            for h in vec_hits:
                row = dict(h)
                row["score"] = float(row.get("score_vec", 0.0))
                row["retrieval_mode"] = "vector"
                out.append(row)
            return out

        # hybrid
        lex_hits = self.lexical_search(q, top_k=candidate_k, min_score=min_score)
        vec_hits = self.vector_search(q, top_k=candidate_k, min_score=min_score_vec)
        if not vec_hits:
            # fallback
            out = []
            for h in lex_hits[:top_k]:
                row = dict(h)
                row["score"] = float(row.get("score_lex", 0.0))
                row["retrieval_mode"] = "tfidf_fallback"
                out.append(row)
            return out

        if fusion == "weighted":
            fused = self._fuse_weighted(
                lex_hits=lex_hits,
                vec_hits=vec_hits,
                weight_lex=weight_lex,
                weight_vec=weight_vec,
            )
        else:
            fused = self._fuse_rrf(
                lex_hits=lex_hits,
                vec_hits=vec_hits,
                rrf_k=rrf_k,
            )

        fused_rows = list(fused.items())
        fused_rows.sort(key=lambda x: x[1].get("score_fusion", 0.0), reverse=True)

        if enable_rerank:
            fused_rows = self._rerank(
                query=q,
                fused_rows=fused_rows,
                source_match_weight=source_match_weight,
            )

        if enable_diversity:
            candidate_source_files = [
                str(self.chunks[int(idx)].get("source_file", ""))
                for idx, _ in fused_rows
            ]
            mentioned_aliases = query_mentioned_source_aliases(q, candidate_source_files)
            if len(mentioned_aliases) == 1:
                max_per_source = max(int(max_per_source), min(top_k, 4))
                max_per_family = max(int(max_per_family), min(top_k, 4))
            elif len(mentioned_aliases) >= 2:
                max_per_family = max(int(max_per_family), len(mentioned_aliases))
            fused_rows = self._apply_diversity(
                fused_rows=fused_rows,
                top_k=top_k,
                max_per_source=max_per_source,
                max_per_family=max_per_family,
                diversity_lambda=diversity_lambda,
            )

        out: List[Dict[str, Any]] = []
        for idx, scores in fused_rows[:top_k]:
            row = dict(self.chunks[int(idx)])
            row.update(scores)
            row["score"] = float(scores.get("score_rerank", scores.get("score_fusion", 0.0)))
            row["retrieval_mode"] = "hybrid"
            row["source_family"] = normalize_source_family(str(row.get("source_file", "")))
            out.append(row)
        return out
