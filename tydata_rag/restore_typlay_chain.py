#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
from pathlib import Path, PurePosixPath
from typing import Iterator, List, Sequence
from zipfile import ZipFile

import pdfplumber
from docx import Document
from docx.document import Document as DocxDocument
from docx.oxml.table import CT_Tbl
from docx.oxml.text.paragraph import CT_P
from docx.table import Table, _Cell
from docx.text.paragraph import Paragraph


DEFAULT_ZIP_PATH = Path("/tmp/ragdata/各类型玩法.zip")
DEFAULT_ROOT = Path("/video-storage/ai-customer/qwen35a3b_finetune/rag/typlay")
DEFAULT_INDEX_NAME = "rag_index_typlay_default_560_112"
PROJECT_ROOT = Path("/home/ubuntu/generate/tydata_rag")
BUILD_SCRIPT = PROJECT_ROOT / "build_rag_index.py"


def clean_text(text: str) -> str:
    text = text.replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def decode_zip_name(name: str) -> str:
    try:
        return name.encode("cp437").decode("utf-8")
    except UnicodeError:
        return name


def safe_relative_name(name: str) -> Path:
    parts = [part for part in PurePosixPath(name).parts if part not in {"", ".", ".."}]
    if parts and parts[0] == "各类型玩法":
        parts = parts[1:]
    if not parts:
        raise ValueError(f"invalid archive member name: {name!r}")
    return Path(*parts)


def iter_block_items(parent: DocxDocument | _Cell) -> Iterator[Paragraph | Table]:
    container = parent.element.body if isinstance(parent, DocxDocument) else parent._tc
    for child in container.iterchildren():
        if isinstance(child, CT_P):
            yield Paragraph(child, parent)
        elif isinstance(child, CT_Tbl):
            yield Table(child, parent)


def table_to_text(table: Table) -> str:
    rows: List[str] = []
    for row in table.rows:
        cells = [clean_text(cell.text) for cell in row.cells]
        cells = [cell for cell in cells if cell]
        if cells:
            rows.append("\t".join(cells))
    return "\n".join(rows)


def extract_docx_text(path: Path) -> str:
    doc = Document(path)
    blocks: List[str] = []
    for block in iter_block_items(doc):
        text = clean_text(block.text if isinstance(block, Paragraph) else table_to_text(block))
        if text:
            blocks.append(text)
    return clean_text("\n\n".join(blocks))


def extract_pdf_text(path: Path) -> str:
    try:
        proc = subprocess.run(
            ["pdftotext", "-enc", "UTF-8", str(path), "-"],
            check=True,
            capture_output=True,
            text=True,
        )
        text = clean_text(proc.stdout)
        if text:
            return text
    except Exception:
        pass

    pages: List[str] = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            text = clean_text(page.extract_text() or "")
            if text:
                pages.append(text)
    return clean_text("\n\n".join(pages))


def extract_text(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".docx":
        return extract_docx_text(path)
    if suffix == ".pdf":
        return extract_pdf_text(path)
    raise ValueError(f"unsupported source type: {path}")


def ensure_clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def extract_archive(zip_path: Path, archive_path: Path, raw_dir: Path) -> List[Path]:
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(zip_path, archive_path)

    ensure_clean_dir(raw_dir)
    restored: List[Path] = []
    with ZipFile(zip_path) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            decoded = decode_zip_name(info.filename)
            rel = safe_relative_name(decoded)
            dest = raw_dir / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(zf.read(info))
            restored.append(dest)
    return restored


def build_text_corpus(raw_files: Sequence[Path], text_dir: Path) -> List[Path]:
    ensure_clean_dir(text_dir)
    outputs: List[Path] = []
    for src in sorted(raw_files):
        text = extract_text(src)
        if not text:
            continue
        dest = text_dir / f"{src.stem}.txt"
        dest.write_text(text + "\n", encoding="utf-8")
        outputs.append(dest)
    return outputs


def build_default_index(text_dir: Path, index_dir: Path) -> None:
    if index_dir.exists():
        shutil.rmtree(index_dir)
    index_dir.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "/usr/bin/python3",
        str(BUILD_SCRIPT),
        "--input-dir",
        str(text_dir),
        "--output-dir",
        str(index_dir),
        "--chunk-size",
        "560",
        "--overlap",
        "112",
        "--ngram-min",
        "1",
        "--ngram-max",
        "3",
        "--dense-backend",
        "svd",
        "--dense-dim",
        "256",
    ]
    subprocess.run(cmd, check=True)


def write_manifest(
    manifest_path: Path,
    zip_path: Path,
    archive_path: Path,
    raw_files: Sequence[Path],
    text_files: Sequence[Path],
    index_dir: Path,
) -> None:
    manifest = {
        "source_zip": str(zip_path),
        "archive_copy": str(archive_path),
        "raw_dir": str(raw_files[0].parent if raw_files else ""),
        "text_dir": str(text_files[0].parent if text_files else ""),
        "default_index_dir": str(index_dir),
        "raw_files": [str(p) for p in raw_files],
        "text_files": [str(p) for p in text_files],
        "counts": {
            "raw_files": len(raw_files),
            "text_files": len(text_files),
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Restore the typlay RAG corpus to persistent storage.")
    parser.add_argument("--zip-path", default=str(DEFAULT_ZIP_PATH))
    parser.add_argument("--root", default=str(DEFAULT_ROOT))
    parser.add_argument("--skip-index", action="store_true")
    args = parser.parse_args()

    zip_path = Path(args.zip_path)
    root = Path(args.root)
    archive_path = root / "archives" / zip_path.name
    raw_dir = root / "raw"
    text_dir = root / "text_all"
    index_dir = root / DEFAULT_INDEX_NAME
    manifest_path = root / "restore_manifest.json"

    if not zip_path.exists():
        raise SystemExit(f"zip file not found: {zip_path}")

    root.mkdir(parents=True, exist_ok=True)
    raw_files = extract_archive(zip_path, archive_path, raw_dir)
    text_files = build_text_corpus(raw_files, text_dir)
    if not args.skip_index:
        build_default_index(text_dir, index_dir)
    write_manifest(manifest_path, zip_path, archive_path, raw_files, text_files, index_dir)

    summary = {
        "root": str(root),
        "archive_copy": str(archive_path),
        "raw_dir": str(raw_dir),
        "text_dir": str(text_dir),
        "default_index_dir": str(index_dir),
        "raw_files": len(raw_files),
        "text_files": len(text_files),
        "manifest": str(manifest_path),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
