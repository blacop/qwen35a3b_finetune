#!/usr/bin/env python3
import argparse
import json
import pickle
import re
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from lxml import etree
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer


def clean_text(text: str) -> str:
    text = text.replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def extract_pdf_units(pdf_path: Path) -> List[Dict]:
    try:
        proc = subprocess.run(
            ["pdftotext", "-enc", "UTF-8", str(pdf_path), "-"],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"pdftotext failed for {pdf_path}: {exc}") from exc

    raw_text = proc.stdout
    pages = raw_text.split("\f")
    units = []
    for i, page in enumerate(pages, start=1):
        page_text = clean_text(page)
        if page_text:
            units.append({"unit_type": "page", "unit_id": i, "text": page_text})

    # If text extraction is nearly empty, the PDF is likely scan-based; fallback to OCR.
    direct_chars = sum(len(u["text"]) for u in units)
    if direct_chars >= 120:
        return units

    ocr_units = extract_pdf_units_by_ocr(pdf_path)
    ocr_chars = sum(len(u["text"]) for u in ocr_units)
    if ocr_chars > direct_chars:
        print(
            f"[INFO] OCR used for {pdf_path.name}: direct_chars={direct_chars}, ocr_chars={ocr_chars}",
            file=sys.stderr,
        )
        return ocr_units
    return units


def get_pdf_page_count(pdf_path: Path) -> int:
    proc = subprocess.run(
        ["pdfinfo", str(pdf_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    for line in proc.stdout.splitlines():
        if line.startswith("Pages:"):
            return int(line.split(":", 1)[1].strip())
    raise RuntimeError(f"Unable to parse page count from pdfinfo for {pdf_path}")


def extract_pdf_units_by_ocr(pdf_path: Path) -> List[Dict]:
    pages = get_pdf_page_count(pdf_path)
    units: List[Dict] = []

    with tempfile.TemporaryDirectory(prefix="tydata_ocr_") as tmp_dir:
        tmp = Path(tmp_dir)
        for page in range(1, pages + 1):
            prefix = tmp / f"page_{page}"
            subprocess.run(
                [
                    "pdftoppm",
                    "-f",
                    str(page),
                    "-l",
                    str(page),
                    "-r",
                    "220",
                    "-png",
                    str(pdf_path),
                    str(prefix),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            images = sorted(tmp.glob(f"page_{page}-*.png"))
            if not images:
                continue
            image = images[0]
            ocr = subprocess.run(
                [
                    "tesseract",
                    str(image),
                    "stdout",
                    "-l",
                    "chi_sim+eng",
                    "--psm",
                    "6",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            page_text = clean_text(ocr.stdout)
            if page_text:
                units.append({"unit_type": "page", "unit_id": page, "text": page_text})
    return units


def _slide_num(slide_name: str) -> int:
    m = re.search(r"slide(\d+)\.xml$", slide_name)
    return int(m.group(1)) if m else 0


def extract_pptx_units(pptx_path: Path) -> List[Dict]:
    units: List[Dict] = []
    with zipfile.ZipFile(pptx_path, "r") as zf:
        slide_files = [
            n
            for n in zf.namelist()
            if n.startswith("ppt/slides/slide") and n.endswith(".xml")
        ]
        slide_files.sort(key=_slide_num)

        ns = {"a": "http://schemas.openxmlformats.org/drawingml/2006/main"}
        for slide_file in slide_files:
            slide_id = _slide_num(slide_file)
            xml_bytes = zf.read(slide_file)
            root = etree.fromstring(xml_bytes)
            texts = root.xpath(".//a:t/text()", namespaces=ns)
            joined = clean_text("\n".join(texts))
            if joined:
                units.append({"unit_type": "slide", "unit_id": slide_id, "text": joined})
    return units


def extract_text_units(file_path: Path) -> List[Dict]:
    suffix = file_path.suffix.lower()
    if suffix == ".pdf":
        return extract_pdf_units(file_path)
    if suffix == ".pptx":
        return extract_pptx_units(file_path)
    if suffix in {".txt", ".md"}:
        text = clean_text(file_path.read_text(encoding="utf-8", errors="ignore"))
        if text:
            return [{"unit_type": "doc", "unit_id": 1, "text": text}]
        return []
    return []


def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    if chunk_size <= overlap:
        raise ValueError("chunk_size must be greater than overlap")

    chunks: List[str] = []
    step = chunk_size - overlap
    for start in range(0, len(text), step):
        chunk = text[start : start + chunk_size]
        chunk = clean_text(chunk)
        if len(chunk) >= 20:
            chunks.append(chunk)
        if start + chunk_size >= len(text):
            break
    return chunks


def build_chunk_records(input_dir: Path, chunk_size: int, overlap: int) -> List[Dict]:
    supported = {".pdf", ".pptx", ".txt", ".md"}
    files = sorted([p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in supported])

    records: List[Dict] = []
    chunk_id = 1

    for fp in files:
        try:
            units = extract_text_units(fp)
        except Exception as exc:
            print(f"[WARN] skip {fp}: {exc}", file=sys.stderr)
            continue

        for unit in units:
            chunks = chunk_text(unit["text"], chunk_size=chunk_size, overlap=overlap)
            for cidx, chunk in enumerate(chunks, start=1):
                records.append(
                    {
                        "chunk_id": chunk_id,
                        "source_file": fp.name,
                        "source_path": str(fp),
                        "source_type": fp.suffix.lower().lstrip("."),
                        "unit_type": unit["unit_type"],
                        "unit_id": unit["unit_id"],
                        "chunk_index": cidx,
                        "text": chunk,
                    }
                )
                chunk_id += 1

    return records


def l2_normalize_rows(x: np.ndarray) -> np.ndarray:
    eps = 1e-12
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    return x / norms


def _build_dense_with_sbert(corpus: List[str], model_name: str) -> Tuple[np.ndarray, Dict]:
    from sentence_transformers import SentenceTransformer  # type: ignore

    model = SentenceTransformer(model_name, device="cpu")
    embs = model.encode(
        corpus,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
        batch_size=32,
    )
    arr = np.asarray(embs, dtype=np.float32)
    arr = l2_normalize_rows(arr)
    meta = {
        "backend": "sbert",
        "model_name": model_name,
        "dim": int(arr.shape[1]),
    }
    return arr, meta


def _build_dense_with_svd(corpus: List[str], dense_dim: int) -> Tuple[np.ndarray, Dict, Dict[str, Any]]:
    word_vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        lowercase=True,
        norm="l2",
        max_features=200000,
    )
    word_matrix = word_vectorizer.fit_transform(corpus)

    max_comp = min(word_matrix.shape[0] - 1, word_matrix.shape[1] - 1)
    n_components = min(dense_dim, max_comp)
    if n_components < 2:
        raise RuntimeError(f"Dense SVD cannot be built: max_comp={max_comp}")

    svd = TruncatedSVD(n_components=n_components, random_state=42)
    dense = svd.fit_transform(word_matrix)
    dense = np.asarray(dense, dtype=np.float32)
    dense = l2_normalize_rows(dense)

    meta = {
        "backend": "svd",
        "model_name": "local_word_tfidf_svd",
        "dim": int(dense.shape[1]),
        "explained_variance_ratio_sum": float(np.sum(svd.explained_variance_ratio_)),
    }
    artifacts = {
        "dense_word_vectorizer": word_vectorizer,
        "dense_svd": svd,
    }
    return dense, meta, artifacts


def build_dense_embeddings(
    corpus: List[str],
    output_dir: Path,
    dense_backend: str,
    sbert_model: str,
    dense_dim: int,
) -> Dict:
    dense_backend = (dense_backend or "auto").strip().lower()
    dense_dim = max(8, int(dense_dim))
    dense_meta: Dict = {"backend": "none", "available": False}

    dense_vectors: Optional[np.ndarray] = None
    extra_artifacts: Dict[str, object] = {}

    if dense_backend in {"auto", "sbert"}:
        try:
            dense_vectors, m = _build_dense_with_sbert(corpus=corpus, model_name=sbert_model)
            dense_meta.update(m)
            dense_meta["available"] = True
        except Exception as exc:
            if dense_backend == "sbert":
                raise
            print(f"[WARN] sentence-transformers unavailable, fallback to SVD dense embeddings: {exc}", file=sys.stderr)

    if dense_vectors is None and dense_backend in {"auto", "svd"}:
        dense_vectors, m, extra_artifacts = _build_dense_with_svd(corpus=corpus, dense_dim=dense_dim)
        dense_meta.update(m)
        dense_meta["available"] = True

    if dense_vectors is None:
        dense_meta["available"] = False
        (output_dir / "dense_meta.json").write_text(
            json.dumps(dense_meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return dense_meta

    np.save(output_dir / "dense_vectors.npy", dense_vectors.astype(np.float32))
    if "dense_word_vectorizer" in extra_artifacts:
        with (output_dir / "dense_word_vectorizer.pkl").open("wb") as f:
            pickle.dump(extra_artifacts["dense_word_vectorizer"], f)
    if "dense_svd" in extra_artifacts:
        with (output_dir / "dense_svd.pkl").open("wb") as f:
            pickle.dump(extra_artifacts["dense_svd"], f)

    (output_dir / "dense_meta.json").write_text(
        json.dumps(dense_meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return dense_meta


def save_index(
    records: List[Dict],
    output_dir: Path,
    ngram_min: int,
    ngram_max: int,
    dense_backend: str,
    sbert_model: str,
    dense_dim: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    corpus = [r["text"] for r in records]
    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(ngram_min, ngram_max),
        lowercase=False,
        norm="l2",
    )
    matrix = vectorizer.fit_transform(corpus)

    chunks_path = output_dir / "chunks.jsonl"
    with chunks_path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    sparse.save_npz(output_dir / "tfidf_matrix.npz", matrix)
    with (output_dir / "vectorizer.pkl").open("wb") as f:
        pickle.dump(vectorizer, f)

    dense_meta = build_dense_embeddings(
        corpus=corpus,
        output_dir=output_dir,
        dense_backend=dense_backend,
        sbert_model=sbert_model,
        dense_dim=dense_dim,
    )

    stats = {
        "num_chunks": len(records),
        "num_docs": len({r["source_file"] for r in records}),
        "files": sorted(list({r["source_file"] for r in records})),
        "dense_backend": dense_meta.get("backend", "none"),
        "dense_available": bool(dense_meta.get("available", False)),
        "dense_dim": int(dense_meta.get("dim", 0) or 0),
    }
    (output_dir / "meta.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a local TF-IDF RAG index from typlay persistent files")
    parser.add_argument("--input-dir", default="/video-storage/ai-customer/qwen35a3b_finetune/rag/typlay/raw", help="Input directory with source files")
    parser.add_argument("--output-dir", default="/home/ubuntu/generate/tydata_rag/index", help="Output index directory")
    parser.add_argument("--chunk-size", type=int, default=600, help="Chunk size (characters)")
    parser.add_argument("--overlap", type=int, default=120, help="Chunk overlap (characters)")
    parser.add_argument("--ngram-min", type=int, default=1, help="Min char ngram")
    parser.add_argument("--ngram-max", type=int, default=3, help="Max char ngram")
    parser.add_argument(
        "--dense-backend",
        default="auto",
        choices=["auto", "sbert", "svd", "none"],
        help="Dense embedding backend for vector retrieval",
    )
    parser.add_argument(
        "--sbert-model",
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        help="sentence-transformers model id when dense-backend=sbert/auto",
    )
    parser.add_argument("--dense-dim", type=int, default=256, help="Dense embedding dim for SVD backend")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists() or not input_dir.is_dir():
        print(f"Input directory not found: {input_dir}", file=sys.stderr)
        return 2

    records = build_chunk_records(
        input_dir=input_dir,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
    )

    if not records:
        print("No extractable text found. Nothing indexed.", file=sys.stderr)
        return 3

    save_index(
        records=records,
        output_dir=Path(args.output_dir),
        ngram_min=args.ngram_min,
        ngram_max=args.ngram_max,
        dense_backend=args.dense_backend,
        sbert_model=args.sbert_model,
        dense_dim=args.dense_dim,
    )

    print(f"Indexed {len(records)} chunks into {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
