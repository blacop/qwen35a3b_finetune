#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import shutil
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Tuple
from xml.etree import ElementTree as ET


NS = {
    "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    "pr": "http://schemas.openxmlformats.org/package/2006/relationships",
}
RID_KEY = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}embed"
TEXT_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    text = text.replace("\r", "\n")
    text = TEXT_RE.sub(" ", text)
    return text.strip()


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_relationships(zf: zipfile.ZipFile) -> Tuple[Dict[str, str], List[str]]:
    rid_to_target: Dict[str, str] = {}
    links: List[str] = []
    rels_path = "word/_rels/document.xml.rels"
    if rels_path not in zf.namelist():
        return rid_to_target, links

    root = ET.fromstring(zf.read(rels_path))
    for rel in root.findall("pr:Relationship", NS):
        rid = rel.attrib.get("Id", "").strip()
        target = rel.attrib.get("Target", "").strip()
        rel_type = rel.attrib.get("Type", "").strip()
        if not rid:
            continue
        if target.startswith("http://") or target.startswith("https://"):
            links.append(target)
        if rel_type.endswith("/image") and target:
            rid_to_target[rid] = target
    return rid_to_target, sorted(list(set(links)))


def extract_media(zf: zipfile.ZipFile, output_dir: Path) -> List[Path]:
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    copied: List[Path] = []
    for name in zf.namelist():
        if not name.startswith("word/media/"):
            continue
        base = Path(name).name
        target_path = images_dir / base
        with zf.open(name, "r") as src, target_path.open("wb") as dst:
            shutil.copyfileobj(src, dst)
        copied.append(target_path)
    copied.sort()
    return copied


def parse_paragraphs(zf: zipfile.ZipFile, rid_to_target: Dict[str, str]) -> List[Dict[str, Any]]:
    doc_path = "word/document.xml"
    if doc_path not in zf.namelist():
        return []
    root = ET.fromstring(zf.read(doc_path))

    paragraphs: List[Dict[str, Any]] = []
    for idx, p in enumerate(root.findall(".//w:p", NS)):
        texts = [(t.text or "") for t in p.findall(".//w:t", NS)]
        joined = clean_text("".join(texts))
        rids: List[str] = []
        for blip in p.findall(".//a:blip", NS):
            rid = blip.attrib.get(RID_KEY, "").strip()
            if rid:
                rids.append(rid)
        image_targets = []
        for rid in rids:
            target = rid_to_target.get(rid, "")
            if not target:
                continue
            image_targets.append(Path(target).name)
        paragraphs.append(
            {
                "paragraph_id": idx + 1,
                "text": joined,
                "image_names": image_targets,
            }
        )
    return paragraphs


def nearest_non_empty_text(paragraphs: List[Dict[str, Any]], start: int, step: int) -> str:
    i = start + step
    while 0 <= i < len(paragraphs):
        txt = str(paragraphs[i].get("text", "")).strip()
        if txt:
            return txt
        i += step
    return ""


def build_image_records(paragraphs: List[Dict[str, Any]], output_dir: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for i, p in enumerate(paragraphs):
        image_names = p.get("image_names") or []
        if not image_names:
            continue
        current_text = str(p.get("text", "")).strip()
        prev_text = nearest_non_empty_text(paragraphs, i, -1)
        next_text = nearest_non_empty_text(paragraphs, i, +1)
        for image_name in image_names:
            image_path = output_dir / "images" / image_name
            caption = current_text if current_text else f"image {image_name}"
            merged_context = clean_text(" ".join(x for x in [prev_text, current_text, next_text] if x))
            rows.append(
                {
                    "record_id": f"docx_img_{len(rows) + 1:05d}",
                    "source_type": "docx",
                    "source_file": "",
                    "paragraph_id": int(p["paragraph_id"]),
                    "image_name": image_name,
                    "image_path": str(image_path),
                    "caption": caption,
                    "context_text": merged_context,
                }
            )
    return rows


def chunk_text_rows(paragraphs: List[Dict[str, Any]], chunk_size: int, overlap: int) -> List[Dict[str, Any]]:
    if chunk_size <= overlap:
        raise ValueError("chunk_size must be greater than overlap")
    merged = []
    for p in paragraphs:
        text = str(p.get("text", "")).strip()
        if text:
            merged.append(f"[P{int(p['paragraph_id'])}] {text}")
    text_all = "\n".join(merged).strip()
    if not text_all:
        return []

    rows: List[Dict[str, Any]] = []
    step = chunk_size - overlap
    for start in range(0, len(text_all), step):
        chunk = clean_text(text_all[start : start + chunk_size])
        if len(chunk) < 20:
            continue
        rows.append(
            {
                "chunk_id": f"docx_txt_{len(rows) + 1:05d}",
                "source_type": "docx",
                "text": chunk,
                "char_start": start,
                "char_end": min(len(text_all), start + chunk_size),
            }
        )
        if start + chunk_size >= len(text_all):
            break
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract text + images + contexts from a .docx file")
    parser.add_argument("--docx", required=True, help="Input docx path")
    parser.add_argument("--output-dir", default="/home/ubuntu/generate/tydata_rag/mm_rag/work/docx_assets", help="Output dir")
    parser.add_argument("--chunk-size", type=int, default=360, help="Text chunk size")
    parser.add_argument("--overlap", type=int, default=80, help="Chunk overlap")
    args = parser.parse_args()

    docx_path = Path(args.docx)
    if not docx_path.exists():
        raise FileNotFoundError(f"docx not found: {docx_path}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(docx_path, "r") as zf:
        rid_to_target, external_links = parse_relationships(zf)
        media_paths = extract_media(zf, output_dir)
        paragraphs = parse_paragraphs(zf, rid_to_target)

    image_rows = build_image_records(paragraphs, output_dir)
    text_rows = chunk_text_rows(paragraphs, chunk_size=args.chunk_size, overlap=args.overlap)

    for row in image_rows:
        row["source_file"] = docx_path.name
    for row in text_rows:
        row["source_file"] = docx_path.name

    write_jsonl(output_dir / "paragraphs.jsonl", paragraphs)
    write_jsonl(output_dir / "text_chunks.jsonl", text_rows)
    write_jsonl(output_dir / "image_records.jsonl", image_rows)
    (output_dir / "external_links.json").write_text(
        json.dumps({"links": external_links}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    summary = {
        "docx": str(docx_path),
        "num_paragraphs": len(paragraphs),
        "num_text_chunks": len(text_rows),
        "num_images": len(media_paths),
        "num_image_records": len(image_rows),
        "num_external_links": len(external_links),
        "output_dir": str(output_dir),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
