#!/usr/bin/env python3
from __future__ import annotations

import json
import pickle
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import numpy as np
from PIL import Image
from scipy import sparse


def clean_text(text: str) -> str:
    return " ".join((text or "").replace("\r", " ").replace("\n", " ").split()).strip()


def l2_normalize(x: np.ndarray) -> np.ndarray:
    eps = 1e-12
    norm = np.linalg.norm(x)
    if norm < eps:
        return x
    return x / norm


def normalize_map(score_map: Dict[int, float]) -> Dict[int, float]:
    if not score_map:
        return {}
    values = list(score_map.values())
    lo = min(values)
    hi = max(values)
    if abs(hi - lo) < 1e-12:
        return {k: 1.0 for k in score_map}
    return {k: (v - lo) / (hi - lo) for k, v in score_map.items()}


ALNUM_RE = re.compile(r"[A-Za-z0-9_]{2,}")
CJK_RE = re.compile(r"[\u4e00-\u9fff]")


def extract_query_terms(query: str) -> List[str]:
    q = clean_text(query)
    if not q:
        return []
    terms: List[str] = [t.lower() for t in ALNUM_RE.findall(q)]
    cjk = "".join(CJK_RE.findall(q))
    if len(cjk) >= 2:
        terms.extend(cjk[i : i + 2] for i in range(len(cjk) - 1))
    seen: Set[str] = set()
    out: List[str] = []
    for t in terms:
        if t not in seen:
            out.append(t)
            seen.add(t)
    return out


def term_coverage_score(query: str, text: str) -> float:
    terms = extract_query_terms(query)
    if not terms:
        return 0.0
    hay = clean_text(text).lower()
    hit = sum(1 for t in terms if t in hay)
    return float(hit) / float(len(terms))


def text_signature(text: str, sig_len: int = 100) -> str:
    return clean_text(text).lower()[:sig_len]


def image_hist_feature(image_path: str, bins: int = 32, size: int = 224) -> np.ndarray:
    with Image.open(image_path) as img:
        img = img.convert("RGB").resize((size, size))
        arr = np.asarray(img, dtype=np.float32)
    feats: List[np.ndarray] = []
    for c in range(3):
        hist, _ = np.histogram(arr[:, :, c], bins=bins, range=(0, 255), density=True)
        feats.append(hist.astype(np.float32))
    gray = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
    hgray, _ = np.histogram(gray, bins=bins, range=(0, 255), density=True)
    feats.append(hgray.astype(np.float32))
    feat = np.concatenate(feats).astype(np.float32)
    return l2_normalize(feat)


def build_context_block(hits: List[Dict[str, Any]], max_chars: int = 600) -> str:
    lines: List[str] = []
    for i, h in enumerate(hits, start=1):
        unit = h.get("unit", {})
        title = clean_text(str(unit.get("title", "") or unit.get("source_file", "")))
        text = clean_text(str(unit.get("text", "")))
        preview = text[:max_chars]
        image_path = str(unit.get("image_path", ""))
        meta = f"[{i}] score={float(h.get('score', 0.0)):.4f} modality={unit.get('modality')}"
        if title:
            meta += f" title={title}"
        if image_path:
            meta += f" image={image_path}"
        lines.append(meta)
        if preview:
            lines.append(preview)
    return "\n\n".join(lines)


class MultiModalRetriever:
    def __init__(self, index_dir: str | Path) -> None:
        self.index_dir = Path(index_dir)
        self.meta = json.loads((self.index_dir / "meta.json").read_text(encoding="utf-8"))
        self.units = self._load_jsonl(self.index_dir / "units.jsonl")

        with (self.index_dir / "text_vectorizer.pkl").open("rb") as f:
            self.text_vectorizer = pickle.load(f)
        self.text_matrix = sparse.load_npz(self.index_dir / "text_matrix.npz")

        self.image_features = np.asarray(np.load(self.index_dir / "image_features.npy"), dtype=np.float32)
        self.image_unit_indices = np.asarray(np.load(self.index_dir / "image_unit_indices.npy"), dtype=np.int32)
        self.image_feature_backend = str(self.meta.get("image_feature_backend", "hist"))
        self.unit_modalities = [str(u.get("modality", "text")) for u in self.units]
        self.unit_to_image_feat_idx: Dict[int, int] = {
            int(unit_idx): int(i) for i, unit_idx in enumerate(self.image_unit_indices.tolist())
        }

    @staticmethod
    def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        if not path.exists():
            return rows
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if s:
                    rows.append(json.loads(s))
        return rows

    def search_text(
        self,
        query: str,
        top_k: int = 12,
        min_score: float = 0.01,
        allowed_modalities: Optional[Set[str]] = None,
    ) -> List[Dict[str, Any]]:
        q = clean_text(query)
        if not q:
            return []
        qvec = self.text_vectorizer.transform([q])
        scores = (self.text_matrix @ qvec.T).toarray().ravel()
        order = scores.argsort()[::-1]

        hits: List[Dict[str, Any]] = []
        for idx in order:
            score = float(scores[int(idx)])
            if score < min_score:
                continue
            unit = self.units[int(idx)]
            if allowed_modalities is not None:
                if str(unit.get("modality", "text")) not in allowed_modalities:
                    continue
            hits.append({"unit_idx": int(idx), "unit": unit, "score": score, "score_text": score, "score_image": 0.0})
            if len(hits) >= top_k:
                break
        return hits

    def search_image(
        self,
        query_image_path: str,
        top_k: int = 12,
        min_score: float = 0.05,
    ) -> List[Dict[str, Any]]:
        if self.image_features.size == 0:
            return []
        qf = image_hist_feature(query_image_path)
        scores = self.image_features @ qf
        order = np.argsort(scores)[::-1]
        hits: List[Dict[str, Any]] = []
        for i in order:
            score = float(scores[int(i)])
            if score < min_score:
                continue
            unit_idx = int(self.image_unit_indices[int(i)])
            unit = self.units[unit_idx]
            hits.append({"unit_idx": unit_idx, "unit": unit, "score": score, "score_text": 0.0, "score_image": score})
            if len(hits) >= top_k:
                break
        return hits

    def search(
        self,
        query: str = "",
        query_image_path: str = "",
        top_k: int = 8,
        top_k_text: int = 20,
        top_k_image: int = 20,
        text_weight: float = 0.65,
        image_weight: float = 0.35,
        min_score_text: float = 0.01,
        min_score_image: float = 0.05,
        prefer_text_for_text_query: bool = True,
        enable_rerank: bool = True,
        enable_dedup: bool = True,
        dedup_text_signature_len: int = 100,
        image_dedup_sim_threshold: float = 0.985,
    ) -> Dict[str, Any]:
        q = clean_text(query)
        qimg = query_image_path.strip()
        if not q and not qimg:
            return {"query": q, "query_image_path": qimg, "text_hits": [], "image_hits": [], "fused_hits": []}

        allowed_modalities = None
        if q and (not qimg) and prefer_text_for_text_query:
            allowed_modalities = {"text"}

        text_hits = (
            self.search_text(query=q, top_k=top_k_text, min_score=min_score_text, allowed_modalities=allowed_modalities)
            if q
            else []
        )
        image_hits = self.search_image(query_image_path=qimg, top_k=top_k_image, min_score=min_score_image) if qimg else []

        text_map = {int(h["unit_idx"]): float(h["score"]) for h in text_hits}
        image_map = {int(h["unit_idx"]): float(h["score"]) for h in image_hits}
        ntext = normalize_map(text_map)
        nimage = normalize_map(image_map)

        fused: Dict[int, Dict[str, float]] = {}
        all_ids = set(ntext.keys()) | set(nimage.keys())
        if not qimg:
            image_weight = 0.0
            text_weight = 1.0

        sum_w = max(1e-12, float(text_weight) + float(image_weight))
        text_weight = float(text_weight) / sum_w
        image_weight = float(image_weight) / sum_w

        for uid in all_ids:
            st = ntext.get(uid, 0.0)
            si = nimage.get(uid, 0.0)
            fused_score = text_weight * st + image_weight * si
            fused[uid] = {
                "score": float(fused_score),
                "score_text": float(text_map.get(uid, 0.0)),
                "score_image": float(image_map.get(uid, 0.0)),
            }

        fused_hits: List[Dict[str, Any]] = []
        for uid, scores in sorted(fused.items(), key=lambda x: x[1]["score"], reverse=True)[: max(top_k * 4, top_k)]:
            row = {
                "unit_idx": uid,
                "unit": self.units[uid],
                "score": float(scores["score"]),
                "score_text": float(scores["score_text"]),
                "score_image": float(scores["score_image"]),
            }
            fused_hits.append(row)

        if enable_rerank and q:
            reranked: List[Dict[str, Any]] = []
            for h in fused_hits:
                unit = h.get("unit", {})
                base = float(h.get("score", 0.0))
                text_body = clean_text(str(unit.get("text", "")))
                title = clean_text(str(unit.get("title", "")))
                cov_text = term_coverage_score(q, text_body)
                cov_title = term_coverage_score(q, title)
                modality = str(unit.get("modality", "text"))

                # textual query should prioritize text units to reduce image noise
                modality_boost = 1.0
                if (not qimg) and prefer_text_for_text_query and modality == "text":
                    modality_boost = 1.08
                if (not qimg) and prefer_text_for_text_query and modality == "image":
                    modality_boost = 0.92

                rerank_score = (0.55 * base + 0.35 * cov_text + 0.10 * cov_title) * modality_boost
                row = dict(h)
                row["score_rerank"] = float(rerank_score)
                reranked.append(row)
            reranked.sort(key=lambda x: float(x.get("score_rerank", 0.0)), reverse=True)
            for h in reranked:
                h["score"] = float(h.get("score_rerank", h.get("score", 0.0)))
            fused_hits = reranked

        if enable_dedup:
            deduped: List[Dict[str, Any]] = []
            seen_text_sigs: Set[str] = set()
            kept_image_feat_rows: List[np.ndarray] = []

            for h in fused_hits:
                unit = h.get("unit", {})
                uid = int(h.get("unit_idx", -1))
                modality = str(unit.get("modality", "text"))
                sig = text_signature(
                    " ".join(
                        [
                            str(unit.get("title", "")),
                            str(unit.get("text", "")),
                        ]
                    ),
                    sig_len=max(20, int(dedup_text_signature_len)),
                )
                if sig and sig in seen_text_sigs:
                    continue

                if modality == "image" and uid in self.unit_to_image_feat_idx:
                    feat_idx = self.unit_to_image_feat_idx[uid]
                    cand = self.image_features[int(feat_idx)]
                    is_dup_img = False
                    for kept in kept_image_feat_rows:
                        sim = float(np.dot(cand, kept))
                        if sim >= float(image_dedup_sim_threshold):
                            is_dup_img = True
                            break
                    if is_dup_img:
                        continue
                    kept_image_feat_rows.append(cand)

                if sig:
                    seen_text_sigs.add(sig)
                deduped.append(h)
                if len(deduped) >= top_k:
                    break
            fused_hits = deduped
        else:
            fused_hits = fused_hits[:top_k]

        return {
            "query": q,
            "query_image_path": qimg,
            "text_hits": text_hits[:top_k],
            "image_hits": image_hits[:top_k],
            "fused_hits": fused_hits,
            "weights": {"text_weight": text_weight, "image_weight": image_weight},
            "settings": {
                "prefer_text_for_text_query": prefer_text_for_text_query,
                "enable_rerank": enable_rerank,
                "enable_dedup": enable_dedup,
                "dedup_text_signature_len": dedup_text_signature_len,
                "image_dedup_sim_threshold": image_dedup_sim_threshold,
            },
        }
