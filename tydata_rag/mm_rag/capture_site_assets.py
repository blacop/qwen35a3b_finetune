#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import re
import time
from html import unescape
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple
from urllib.parse import urljoin, urlparse

import requests
from PIL import Image


HREF_RE = re.compile(r"""href=["']([^"'#]+)["']""", re.IGNORECASE)
SRC_RE = re.compile(r"""<img[^>]+src=["']([^"']+)["']""", re.IGNORECASE)
TAG_RE = re.compile(r"<[^>]+>")
SPACE_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    text = unescape(text)
    text = TAG_RE.sub(" ", text)
    text = SPACE_RE.sub(" ", text)
    return text.strip()


def safe_name(url: str) -> str:
    parsed = urlparse(url)
    path = parsed.path.strip("/") or "root"
    raw = f"{parsed.netloc}_{path}"
    raw = raw.replace("/", "_")
    raw = re.sub(r"[^A-Za-z0-9._-]+", "_", raw)
    if len(raw) > 150:
        raw = raw[:150]
    return raw


def same_host(base_url: str, target_url: str) -> bool:
    b = urlparse(base_url).netloc.lower()
    t = urlparse(target_url).netloc.lower()
    return bool(t) and b == t


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def generate_grid_crops_from_image(
    screenshot_path: Path,
    output_dir: Path,
    slug: str,
    max_crops: int = 6,
) -> List[str]:
    if max_crops <= 0 or not screenshot_path.exists():
        return []
    out: List[str] = []
    try:
        with Image.open(screenshot_path) as img:
            w, h = img.size
            if w < 200 or h < 200:
                return out
            # Split vertically into several large strips to preserve readability.
            strips = min(max_crops, max(2, h // 700))
            strip_h = max(240, h // strips)
            for i in range(strips):
                top = i * strip_h
                if top >= h:
                    break
                bottom = min(h, top + strip_h)
                crop = img.crop((0, top, w, bottom))
                p = output_dir / f"{slug}_grid_{i + 1:02d}.png"
                crop.save(p)
                out.append(str(p))
                if len(out) >= max_crops:
                    break
    except Exception:
        return []
    return out


async def capture_with_playwright(args: argparse.Namespace, output_dir: Path) -> Dict[str, Any]:
    from playwright.async_api import async_playwright

    pages_dir = output_dir / "pages"
    shots_dir = output_dir / "screenshots"
    crops_dir = output_dir / "crops"
    pages_dir.mkdir(parents=True, exist_ok=True)
    shots_dir.mkdir(parents=True, exist_ok=True)
    crops_dir.mkdir(parents=True, exist_ok=True)

    queue: List[Tuple[str, int]] = [(args.base_url, 0)]
    visited: Set[str] = set()
    rows: List[Dict[str, Any]] = []

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=not args.no_headless)
        context = await browser.new_context(ignore_https_errors=True)
        page = await context.new_page()

        if args.login_url:
            await page.goto(args.login_url, wait_until="domcontentloaded", timeout=args.timeout_ms)
            await page.wait_for_timeout(args.settle_ms)
            if args.username and args.password and args.user_selector and args.pass_selector and args.submit_selector:
                await page.fill(args.user_selector, args.username)
                await page.fill(args.pass_selector, args.password)
                await page.click(args.submit_selector)
                await page.wait_for_timeout(2000)

        while queue and len(rows) < args.max_pages:
            url, depth = queue.pop(0)
            if url in visited:
                continue
            visited.add(url)
            try:
                # Some sites (for example Google Docs) keep background requests alive
                # and never satisfy "networkidle", so use domcontentloaded + settle wait.
                await page.goto(url, wait_until="domcontentloaded", timeout=args.timeout_ms)
                await page.wait_for_timeout(args.settle_ms)
            except Exception:
                continue

            idx = len(rows) + 1
            slug = f"{idx:04d}_{safe_name(url)}"
            html = await page.content()
            text = await page.evaluate("() => document.body ? document.body.innerText : ''")
            title = await page.title()

            html_path = pages_dir / f"{slug}.html"
            shot_path = shots_dir / f"{slug}.png"
            html_path.write_text(html, encoding="utf-8", errors="ignore")
            await page.screenshot(path=str(shot_path), full_page=True)

            crop_paths: List[str] = []
            selectors = ["main", "section", "table", ".card", ".panel", ".modal"]
            crop_id = 0
            for sel in selectors:
                elements = await page.query_selector_all(sel)
                for e in elements[: args.max_crops_per_selector]:
                    if crop_id >= args.max_crops_total:
                        break
                    try:
                        box = await e.bounding_box()
                        if not box:
                            continue
                        if box.get("width", 0) < 120 or box.get("height", 0) < 80:
                            continue
                        crop_id += 1
                        p = crops_dir / f"{slug}_crop_{crop_id:02d}.png"
                        await e.screenshot(path=str(p))
                        crop_paths.append(str(p))
                    except Exception:
                        continue
            if not crop_paths:
                crop_paths.extend(
                    generate_grid_crops_from_image(
                        screenshot_path=shot_path,
                        output_dir=crops_dir,
                        slug=slug,
                        max_crops=args.fallback_grid_crops,
                    )
                )

            rows.append(
                {
                    "record_id": f"web_page_{idx:05d}",
                    "source_type": "web",
                    "url": url,
                    "title": clean_text(title),
                    "text": clean_text(text),
                    "html_path": str(html_path),
                    "screenshot_path": str(shot_path),
                    "crop_paths": crop_paths,
                    "depth": depth,
                    "captured_at": int(time.time()),
                }
            )

            if depth >= args.max_depth:
                continue
            hrefs = await page.eval_on_selector_all(
                "a[href]",
                "els => els.map(e => e.href).filter(Boolean)",
            )
            for href in hrefs:
                if not isinstance(href, str):
                    continue
                if not href.startswith("http"):
                    continue
                if not same_host(args.base_url, href):
                    continue
                if href in visited:
                    continue
                queue.append((href, depth + 1))

        await browser.close()

    write_jsonl(output_dir / "page_records.jsonl", rows)
    summary = {
        "mode": "playwright",
        "num_pages": len(rows),
        "output_dir": str(output_dir),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def capture_with_requests(args: argparse.Namespace, output_dir: Path) -> Dict[str, Any]:
    pages_dir = output_dir / "pages"
    images_dir = output_dir / "downloaded_images"
    pages_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    queue: List[Tuple[str, int]] = [(args.base_url, 0)]
    visited: Set[str] = set()
    rows: List[Dict[str, Any]] = []

    while queue and len(rows) < args.max_pages:
        url, depth = queue.pop(0)
        if url in visited:
            continue
        visited.add(url)
        try:
            resp = session.get(url, timeout=max(5, args.timeout_ms // 1000))
        except Exception:
            continue
        if resp.status_code >= 400:
            continue

        html = resp.text
        idx = len(rows) + 1
        slug = f"{idx:04d}_{safe_name(url)}"
        html_path = pages_dir / f"{slug}.html"
        html_path.write_text(html, encoding="utf-8", errors="ignore")

        downloaded_images: List[str] = []
        for src in SRC_RE.findall(html):
            img_url = src if src.startswith("http") else urljoin(url, src)
            if not img_url.startswith("http"):
                continue
            ext = Path(urlparse(img_url).path).suffix.lower()
            if ext not in {".png", ".jpg", ".jpeg", ".webp", ".gif"}:
                ext = ".bin"
            img_name = f"{slug}_img_{len(downloaded_images) + 1:02d}{ext}"
            img_path = images_dir / img_name
            try:
                r = session.get(img_url, timeout=max(5, args.timeout_ms // 1000))
                if r.status_code < 400 and r.content:
                    img_path.write_bytes(r.content)
                    downloaded_images.append(str(img_path))
            except Exception:
                continue

        rows.append(
            {
                "record_id": f"web_page_{idx:05d}",
                "source_type": "web",
                "url": url,
                "title": "",
                "text": clean_text(html),
                "html_path": str(html_path),
                "screenshot_path": "",
                "crop_paths": downloaded_images,
                "depth": depth,
                "captured_at": int(time.time()),
            }
        )

        if depth >= args.max_depth:
            continue
        for href in HREF_RE.findall(html):
            next_url = href if href.startswith("http") else urljoin(url, href)
            if not next_url.startswith("http"):
                continue
            if not same_host(args.base_url, next_url):
                continue
            if next_url in visited:
                continue
            queue.append((next_url, depth + 1))

    write_jsonl(output_dir / "page_records.jsonl", rows)
    summary = {
        "mode": "requests_fallback",
        "num_pages": len(rows),
        "output_dir": str(output_dir),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Capture backend site assets (screenshots + page texts)")
    parser.add_argument("--base-url", required=True, help="Site base URL")
    parser.add_argument("--output-dir", default="/home/ubuntu/generate/tydata_rag/mm_rag/work/web_assets", help="Output dir")
    parser.add_argument("--max-pages", type=int, default=40)
    parser.add_argument("--max-depth", type=int, default=1)
    parser.add_argument("--timeout-ms", type=int, default=30000)
    parser.add_argument("--settle-ms", type=int, default=5000)
    parser.add_argument("--no-headless", action="store_true")

    parser.add_argument("--login-url", default="")
    parser.add_argument("--username", default="")
    parser.add_argument("--password", default="")
    parser.add_argument("--user-selector", default="")
    parser.add_argument("--pass-selector", default="")
    parser.add_argument("--submit-selector", default="")

    parser.add_argument("--max-crops-total", type=int, default=12)
    parser.add_argument("--max-crops-per-selector", type=int, default=3)
    parser.add_argument("--fallback-grid-crops", type=int, default=6)

    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        import playwright  # type: ignore # noqa: F401

        summary = asyncio.run(capture_with_playwright(args, output_dir))
    except Exception:
        summary = capture_with_requests(args, output_dir)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
