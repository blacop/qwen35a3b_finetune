#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

from mm_retrieval import clean_text, image_hist_feature


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                rows.append(json.loads(s))
    return rows


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    text = clean_text(text)
    if not text:
        return []
    if chunk_size <= overlap:
        raise ValueError("chunk_size must be greater than overlap")
    out: List[str] = []
    step = chunk_size - overlap
    for i in range(0, len(text), step):
        c = clean_text(text[i : i + chunk_size])
        if len(c) >= 20:
            out.append(c)
        if i + chunk_size >= len(text):
            break
    return out


def build_units(docx_dir: Path, web_dir: Path, chunk_size: int, overlap: int) -> List[Dict[str, Any]]:
    units: List[Dict[str, Any]] = []

    # docx text
    for row in load_jsonl(docx_dir / "text_chunks.jsonl"):
        text = clean_text(str(row.get("text", "")))
        if not text:
            continue
        units.append(
            {
                "unit_id": len(units) + 1,
                "modality": "text",
                "source_type": "docx",
                "source_file": str(row.get("source_file", "")),
                "source_url": "",
                "title": "docx_text_chunk",
                "text": text,
                "image_path": "",
            }
        )

    # docx images
    for row in load_jsonl(docx_dir / "image_records.jsonl"):
        image_path = str(row.get("image_path", "")).strip()
        if not image_path or not Path(image_path).exists():
            continue
        caption = clean_text(str(row.get("caption", "")))
        context_text = clean_text(str(row.get("context_text", "")))
        text = clean_text(" ".join(x for x in [caption, context_text] if x))
        if not text:
            text = f"docx image {Path(image_path).name}"
        units.append(
            {
                "unit_id": len(units) + 1,
                "modality": "image",
                "source_type": "docx",
                "source_file": str(row.get("source_file", "")),
                "source_url": "",
                "title": caption or Path(image_path).name,
                "text": text,
                "image_path": image_path,
            }
        )

    # web pages + screenshots/crops
    page_rows = load_jsonl(web_dir / "page_records.jsonl")
    for pr in page_rows:
        url = str(pr.get("url", ""))
        title = clean_text(str(pr.get("title", "")))
        body = clean_text(str(pr.get("text", "")))

        for i, c in enumerate(chunk_text(body, chunk_size=chunk_size, overlap=overlap), start=1):
            units.append(
                {
                    "unit_id": len(units) + 1,
                    "modality": "text",
                    "source_type": "web",
                    "source_file": "",
                    "source_url": url,
                    "title": title or f"web_page_{i}",
                    "text": c,
                    "image_path": "",
                }
            )

        screenshot_path = str(pr.get("screenshot_path", "")).strip()
        candidate_images = []
        if screenshot_path:
            candidate_images.append(screenshot_path)
        for cp in pr.get("crop_paths", []):
            p = str(cp).strip()
            if p:
                candidate_images.append(p)

        image_caption = clean_text(" ".join(x for x in [title, url, body[:600]] if x))
        for p in candidate_images:
            if not Path(p).exists():
                continue
            units.append(
                {
                    "unit_id": len(units) + 1,
                    "modality": "image",
                    "source_type": "web",
                    "source_file": "",
                    "source_url": url,
                    "title": title or Path(p).name,
                    "text": image_caption or Path(p).name,
                    "image_path": p,
                }
            )
    return units


def main() -> int:
    parser = argparse.ArgumentParser(description="Build multimodal index from extracted docx + web assets")
    parser.add_argument("--docx-dir", default="/home/ubuntu/generate/tydata_rag/mm_rag/work/docx_assets")
    parser.add_argument("--web-dir", default="/home/ubuntu/generate/tydata_rag/mm_rag/work/web_assets")
    parser.add_argument("--output-dir", default="/home/ubuntu/generate/tydata_rag/mm_rag/index")
    parser.add_argument("--chunk-size", type=int, default=360)
    parser.add_argument("--overlap", type=int, default=80)
    parser.add_argument("--ngram-min", type=int, default=1)
    parser.add_argument("--ngram-max", type=int, default=3)
    args = parser.parse_args()

    docx_dir = Path(args.docx_dir)
    web_dir = Path(args.web_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    units = build_units(docx_dir=docx_dir, web_dir=web_dir, chunk_size=args.chunk_size, overlap=args.overlap)
    if not units:
        raise RuntimeError("No units found. Run extract_docx_assets.py and/or capture_site_assets.py first.")

    corpus = [clean_text(str(u.get("text", ""))) or "empty" for u in units]
    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(args.ngram_min, args.ngram_max),
        lowercase=False,
        norm="l2",
    )
    text_matrix = vectorizer.fit_transform(corpus)

    image_unit_indices: List[int] = []
    image_features: List[np.ndarray] = []
    for i, u in enumerate(units):
        if str(u.get("modality", "")) != "image":
            continue
        image_path = str(u.get("image_path", "")).strip()
        if not image_path:
            continue
        p = Path(image_path)
        if not p.exists():
            continue
        try:
            feat = image_hist_feature(str(p))
            image_unit_indices.append(i)
            image_features.append(feat)
        except Exception:
            continue

    image_feature_matrix = np.vstack(image_features).astype(np.float32) if image_features else np.zeros((0, 128), dtype=np.float32)
    image_idx_arr = np.asarray(image_unit_indices, dtype=np.int32)

    write_jsonl(output_dir / "units.jsonl", units)
    sparse.save_npz(output_dir / "text_matrix.npz", text_matrix)
    with (output_dir / "text_vectorizer.pkl").open("wb") as f:
        pickle.dump(vectorizer, f)
    np.save(output_dir / "image_features.npy", image_feature_matrix)
    np.save(output_dir / "image_unit_indices.npy", image_idx_arr)

    meta = {
        "num_units": len(units),
        "num_text_units": int(sum(1 for u in units if u.get("modality") == "text")),
        "num_image_units": int(sum(1 for u in units if u.get("modality") == "image")),
        "num_image_features": int(image_feature_matrix.shape[0]),
        "text_vectorizer": "tfidf_char_ngram",
        "image_feature_backend": "hist_rgb_gray",
        "chunk_size": int(args.chunk_size),
        "overlap": int(args.overlap),
        "ngram_range": [int(args.ngram_min), int(args.ngram_max)],
        "docx_dir": str(docx_dir),
        "web_dir": str(web_dir),
        "output_dir": str(output_dir),
    }
    (output_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
