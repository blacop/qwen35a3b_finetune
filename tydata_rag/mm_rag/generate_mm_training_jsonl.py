#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            rows.append(json.loads(s))
    return rows


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


SPACE_RE = re.compile(r"\s+")
URL_RE = re.compile(r"https?://\S+")
ACCOUNT_RE = re.compile(r"((?:后台|前台)?账号)\s*[：:]\s*[^\s，,;；]+")
PASS_RE = re.compile(r"((?:后台|前台)?密码)\s*[：:]\s*[^\s，,;；]+")
HANDLE_RE = re.compile(r"(?<![\w@])@[A-Za-z0-9_]{2,}")
TOKEN_RE = re.compile(r"\b[A-Za-z0-9]{24,}\b")
IMG_TOKEN_RE = re.compile(r"\bimage\s*image?\d+\.(?:png|jpg|jpeg|webp)\b", re.IGNORECASE)
IMG_FILE_RE = re.compile(r"\bimage\d+\.(?:png|jpg|jpeg|webp)\b", re.IGNORECASE)


def clean_text(text: str) -> str:
    return SPACE_RE.sub(" ", (text or "").replace("\r", " ").replace("\n", " ")).strip()


def strip_image_tokens(text: str) -> str:
    s = text or ""
    s = IMG_TOKEN_RE.sub(" ", s)
    s = IMG_FILE_RE.sub(" ", s)
    return clean_text(s)


def redact_text(text: str, enable: bool) -> str:
    s = clean_text(text)
    if not enable or not s:
        return s
    s = URL_RE.sub("[REDACTED_URL]", s)
    s = ACCOUNT_RE.sub(r"\1：[REDACTED_ACCOUNT]", s)
    s = PASS_RE.sub(r"\1：[REDACTED_PASSWORD]", s)
    s = HANDLE_RE.sub("[REDACTED_HANDLE]", s)
    s = TOKEN_RE.sub("[REDACTED_TOKEN]", s)
    return clean_text(s)


TOPIC_KEYWORDS: Sequence[tuple[str, Sequence[str]]] = [
    ("权限管理", ["权限", "角色", "管理员"]),
    ("系统配置", ["系统配置", "验证码", "重置", "冻结比例"]),
    ("域名与CDN", ["域名", "CNAME", "CDN", "证书", "解析"]),
    ("客服与通知", ["客服", "站内信", "通知", "短信"]),
    ("资金审核", ["资金", "审核", "提款", "充值", "补单"]),
    ("活动配置", ["活动", "弹窗", "轮播", "红包", "签到"]),
    ("游戏配置", ["游戏", "平台", "维护", "注单"]),
    ("统计报表", ["报表", "导出", "查询条件"]),
]


def infer_topic(text: str) -> str:
    for topic, kws in TOPIC_KEYWORDS:
        if any(k in text for k in kws):
            return topic
    return "后台操作"


def split_sentences(text: str) -> List[str]:
    parts = re.split(r"[。；;！!？?]|(?<=\])\s+", text)
    out: List[str] = []
    seen = set()
    for p in parts:
        s = clean_text(p)
        if len(s) < 6:
            continue
        if s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


def extract_steps(text: str, max_steps: int) -> List[str]:
    sents = split_sentences(text)
    scored: List[tuple[int, str]] = []
    for s in sents:
        score = 0
        if any(x in s for x in ["点击", "选择", "输入", "添加", "配置", "开启", "关闭", "编辑", "删除", "提交"]):
            score += 3
        if re.search(r"\b\d+[.、]\b", s) or re.search(r"第[一二三四五六七八九十\d]", s):
            score += 2
        if len(s) <= 40:
            score += 1
        scored.append((score, s))
    scored.sort(key=lambda x: (x[0], -len(x[1])), reverse=True)
    steps = [s for _, s in scored[:max_steps]]
    if not steps:
        steps = sents[:max_steps]
    return steps


PARA_RE = re.compile(r"\[P(\d+)\]")


def build_paragraph_chunk_index(text_chunk_rows: Sequence[Dict[str, Any]]) -> Dict[int, List[str]]:
    out: Dict[int, List[str]] = {}
    for row in text_chunk_rows:
        text = clean_text(str(row.get("text", "")))
        if not text:
            continue
        for p in PARA_RE.findall(text):
            pid = int(p)
            out.setdefault(pid, []).append(text)
    return out


QUESTION_TEMPLATES: Sequence[str] = [
    "请根据这张后台截图，说明这是哪个模块，以及应该如何操作。",
    "这张图对应后台什么功能？请给出简明操作步骤。",
    "用户发来这张后台页面截图时，客服应如何引导处理？",
]

WEB_QUESTION_TEMPLATES: Sequence[str] = [
    "根据该后台网页截图，说明页面用途并给出操作指引。",
    "请识别该网页页面的模块，并给出推荐的处理步骤。",
]

SYSTEM_PROMPT = (
    "你是包网后台智能客服助手。请基于截图和上下文给出准确、可执行、合规的中文指引，"
    "不要编造不存在的按钮或路径。"
)


@dataclass
class CanonicalSample:
    sample_id: str
    split: str
    source_type: str
    source_file: str
    source_url: str
    image_path: str
    question: str
    answer: str
    context: str
    topic: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.sample_id,
            "split": self.split,
            "source_type": self.source_type,
            "source_file": self.source_file,
            "source_url": self.source_url,
            "image_path": self.image_path,
            "topic": self.topic,
            "question": self.question,
            "answer": self.answer,
            "context": self.context,
            "system": SYSTEM_PROMPT,
        }


def pick_split(key: str, val_ratio: float) -> str:
    h = hashlib.md5(key.encode("utf-8")).hexdigest()
    v = int(h[:8], 16) / 0xFFFFFFFF
    return "val" if v < val_ratio else "train"


def make_answer(topic: str, context: str, max_steps: int, max_context_chars: int) -> str:
    ctx = clean_text(context)[:max_context_chars]
    steps = extract_steps(ctx, max_steps=max_steps)
    if not steps:
        return f"该截图主要涉及{topic}。请按后台当前版本核对对应功能入口并执行。"

    lines = [f"该截图对应「{topic}」相关操作。", "建议处理步骤："]
    for i, step in enumerate(steps, start=1):
        lines.append(f"{i}. {step}")
    lines.append("如页面字段与当前版本不一致，以实际后台为准并先做权限核验。")
    return "\n".join(lines)


def build_docx_samples(
    image_rows: Sequence[Dict[str, Any]],
    paragraph_chunk_index: Dict[int, List[str]],
    seed: int,
    redact: bool,
    variants_per_image: int,
    max_steps: int,
    max_context_chars: int,
    val_ratio: float,
    min_context_chars: int,
) -> List[CanonicalSample]:
    rng = random.Random(seed)
    out: List[CanonicalSample] = []

    for row in image_rows:
        image_path = str(row.get("image_path", "")).strip()
        if not image_path or not Path(image_path).exists():
            continue

        caption = strip_image_tokens(redact_text(str(row.get("caption", "")), redact))
        ctx = strip_image_tokens(redact_text(str(row.get("context_text", "")), redact))
        pid = int(row.get("paragraph_id", 0) or 0)
        enrich = ""
        if pid > 0:
            chunk_candidates = paragraph_chunk_index.get(pid, [])
            if chunk_candidates:
                enrich = strip_image_tokens(redact_text(chunk_candidates[0], redact))
        combined = clean_text(" ".join(x for x in [caption, ctx, enrich[:220]] if x))
        if len(combined) < min_context_chars:
            continue

        topic = infer_topic(combined)
        answer = make_answer(topic=topic, context=combined, max_steps=max_steps, max_context_chars=max_context_chars)
        templates = list(QUESTION_TEMPLATES)
        rng.shuffle(templates)

        count = max(1, variants_per_image)
        for i in range(count):
            q = templates[i % len(templates)]
            sid = f"mm_docx_{row.get('record_id', 'unknown')}_{i+1:02d}"
            split = pick_split(sid, val_ratio)
            out.append(
                CanonicalSample(
                    sample_id=sid,
                    split=split,
                    source_type="docx",
                    source_file=str(row.get("source_file", "")),
                    source_url="",
                    image_path=image_path,
                    question=q,
                    answer=answer,
                    context=combined,
                    topic=topic,
                )
            )
    return out


def iter_web_images(row: Dict[str, Any]) -> List[str]:
    paths: List[str] = []
    screenshot = str(row.get("screenshot_path", "")).strip()
    if screenshot:
        paths.append(screenshot)
    for p in row.get("crop_paths", []) or []:
        x = str(p).strip()
        if x:
            paths.append(x)
    uniq: List[str] = []
    seen = set()
    for p in paths:
        if p in seen:
            continue
        seen.add(p)
        uniq.append(p)
    return uniq


def build_web_samples(
    page_rows: Sequence[Dict[str, Any]],
    seed: int,
    redact: bool,
    max_web_images_per_page: int,
    max_steps: int,
    max_context_chars: int,
    val_ratio: float,
    min_context_chars: int,
) -> List[CanonicalSample]:
    rng = random.Random(seed + 17)
    out: List[CanonicalSample] = []

    for idx, row in enumerate(page_rows, start=1):
        title = redact_text(str(row.get("title", "")), redact)
        text = redact_text(str(row.get("text", "")), redact)
        url = redact_text(str(row.get("url", "")), redact)
        context = clean_text(" ".join(x for x in [title, text] if x))
        if len(context) < min_context_chars:
            continue

        topic = infer_topic(context)
        answer = make_answer(topic=topic, context=context, max_steps=max_steps, max_context_chars=max_context_chars)
        templates = list(WEB_QUESTION_TEMPLATES)
        rng.shuffle(templates)

        image_paths = [p for p in iter_web_images(row) if Path(p).exists()]
        if not image_paths:
            continue
        image_paths = image_paths[: max(1, max_web_images_per_page)]

        for j, image_path in enumerate(image_paths, start=1):
            sid = f"mm_web_{idx:04d}_{j:02d}"
            split = pick_split(sid, val_ratio)
            q = templates[(j - 1) % len(templates)]
            out.append(
                CanonicalSample(
                    sample_id=sid,
                    split=split,
                    source_type="web",
                    source_file="",
                    source_url=url,
                    image_path=image_path,
                    question=q,
                    answer=answer,
                    context=context[:max_context_chars],
                    topic=topic,
                )
            )
    return out


def to_swift_vl(sample: CanonicalSample) -> Dict[str, Any]:
    return {
        "id": sample.sample_id,
        "split": sample.split,
        "system": SYSTEM_PROMPT,
        "query": sample.question,
        "response": sample.answer,
        "images": [sample.image_path],
        "meta": {
            "source_type": sample.source_type,
            "source_file": sample.source_file,
            "source_url": sample.source_url,
            "topic": sample.topic,
        },
    }


def to_llava(sample: CanonicalSample) -> Dict[str, Any]:
    return {
        "id": sample.sample_id,
        "split": sample.split,
        "image": sample.image_path,
        "conversations": [
            {"from": "human", "value": f"<image>\n{sample.question}"},
            {"from": "gpt", "value": sample.answer},
        ],
        "meta": {
            "source_type": sample.source_type,
            "source_file": sample.source_file,
            "source_url": sample.source_url,
            "topic": sample.topic,
        },
    }


def to_openai_vision(sample: CanonicalSample) -> Dict[str, Any]:
    return {
        "id": sample.sample_id,
        "split": sample.split,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": sample.question},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"file://{sample.image_path}",
                        },
                    },
                ],
            },
            {"role": "assistant", "content": sample.answer},
        ],
        "meta": {
            "source_type": sample.source_type,
            "source_file": sample.source_file,
            "source_url": sample.source_url,
            "topic": sample.topic,
            "image_path": sample.image_path,
        },
    }


def main() -> int:
    p = argparse.ArgumentParser(description="Generate multimodal training JSONL from extracted docx/web assets")
    p.add_argument("--docx-image-records", default="/home/ubuntu/generate/tydata_rag/mm_rag/work/docx_assets/image_records.jsonl")
    p.add_argument("--docx-text-chunks", default="/home/ubuntu/generate/tydata_rag/mm_rag/work/docx_assets/text_chunks.jsonl")
    p.add_argument("--web-page-records", default="/home/ubuntu/generate/tydata_rag/mm_rag/work/web_assets/page_records.jsonl")
    p.add_argument("--output-dir", default="/home/ubuntu/generate/tydata_rag/mm_rag/train_datasets")
    p.add_argument("--dataset-name", default="jt_manual_mm_train")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--redact", dest="redact", action="store_true")
    p.add_argument("--no-redact", dest="redact", action="store_false")
    p.set_defaults(redact=True)
    p.add_argument("--variants-per-image", type=int, default=2)
    p.add_argument("--max-web-images-per-page", type=int, default=3)
    p.add_argument("--max-steps", type=int, default=4)
    p.add_argument("--max-context-chars", type=int, default=360)
    p.add_argument("--min-context-chars", type=int, default=12)
    p.add_argument("--val-ratio", type=float, default=0.02)
    args = p.parse_args()

    redact = bool(args.redact)
    docx_rows = load_jsonl(Path(args.docx_image_records))
    text_chunk_rows = load_jsonl(Path(args.docx_text_chunks))
    web_rows = load_jsonl(Path(args.web_page_records))
    paragraph_chunk_index = build_paragraph_chunk_index(text_chunk_rows)

    docx_samples = build_docx_samples(
        image_rows=docx_rows,
        paragraph_chunk_index=paragraph_chunk_index,
        seed=args.seed,
        redact=redact,
        variants_per_image=args.variants_per_image,
        max_steps=args.max_steps,
        max_context_chars=args.max_context_chars,
        val_ratio=args.val_ratio,
        min_context_chars=args.min_context_chars,
    )
    web_samples = build_web_samples(
        page_rows=web_rows,
        seed=args.seed,
        redact=redact,
        max_web_images_per_page=args.max_web_images_per_page,
        max_steps=args.max_steps,
        max_context_chars=args.max_context_chars,
        val_ratio=args.val_ratio,
        min_context_chars=args.min_context_chars,
    )

    samples = docx_samples + web_samples

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(args.output_dir) / f"{args.dataset_name}_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    canonical_path = out_dir / "multimodal_train.canonical.jsonl"
    swift_path = out_dir / "multimodal_train.swift_vl.jsonl"
    llava_path = out_dir / "multimodal_train.llava.jsonl"
    openai_path = out_dir / "multimodal_train.openai_vision.jsonl"

    n_canonical = write_jsonl(canonical_path, (s.to_dict() for s in samples))
    n_swift = write_jsonl(swift_path, (to_swift_vl(s) for s in samples))
    n_llava = write_jsonl(llava_path, (to_llava(s) for s in samples))
    n_openai = write_jsonl(openai_path, (to_openai_vision(s) for s in samples))

    by_split = {"train": 0, "val": 0}
    by_source = {"docx": 0, "web": 0}
    for s in samples:
        by_split[s.split] = by_split.get(s.split, 0) + 1
        by_source[s.source_type] = by_source.get(s.source_type, 0) + 1

    summary = {
        "generated_at_utc": ts,
        "output_dir": str(out_dir),
        "redact": redact,
        "counts": {
            "canonical": n_canonical,
            "swift_vl": n_swift,
            "llava": n_llava,
            "openai_vision": n_openai,
        },
        "by_split": by_split,
        "by_source": by_source,
        "inputs": {
            "docx_image_records": str(Path(args.docx_image_records)),
            "docx_text_chunks": str(Path(args.docx_text_chunks)),
            "web_page_records": str(Path(args.web_page_records)),
        },
        "params": {
            "seed": args.seed,
            "variants_per_image": args.variants_per_image,
            "max_web_images_per_page": args.max_web_images_per_page,
            "max_steps": args.max_steps,
            "max_context_chars": args.max_context_chars,
            "min_context_chars": args.min_context_chars,
            "val_ratio": args.val_ratio,
        },
        "files": {
            "canonical": str(canonical_path),
            "swift_vl": str(swift_path),
            "llava": str(llava_path),
            "openai_vision": str(openai_path),
        },
    }

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
