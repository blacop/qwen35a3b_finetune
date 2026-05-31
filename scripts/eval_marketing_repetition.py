#!/usr/bin/env python3
"""Targeted evaluator for POP / marketing leakage / repetition issues."""

from __future__ import annotations

import argparse
import json
import re
import statistics
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


DEFAULT_SYSTEM_PROMPT = (
    "你是体育包网客服。回答必须简洁、合规、中文自然。"
    "禁止输出营销引流、禁止推荐下载APP、禁止让用户添加外部联系方式、"
    "禁止空话复读、禁止先写分析再给答案。"
    "遇到充值/提款/注单异常等问题时，优先说明核实路径、需要的信息和处理时效。"
)


ANALYSIS_PREFIX_PATTERNS = [
    re.compile(r"^用户的问题是"),
    re.compile(r"^回答需"),
    re.compile(r"^根据中国法律法规"),
    re.compile(r"^用户提到"),
]

THINK_LEAK_PATTERNS = [
    re.compile(r"<\s*think\s*>", re.I),
    re.compile(r"<\s*/\s*think\s*>", re.I),
]


def now_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at line {line_no}: {exc}") from exc
    return rows


def to_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x) for x in value]
    return [str(value)]


def normalize_line(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\s+", "", text)
    text = re.sub(r"[~!！。,.，?？]+$", "", text)
    return text


def detect_repetition(text: str) -> list[str]:
    counter: Counter[str] = Counter()
    raw_map: dict[str, str] = {}
    for raw_line in text.splitlines():
        line = normalize_line(raw_line)
        if len(line) < 4:
            continue
        counter[line] += 1
        raw_map.setdefault(line, raw_line.strip())
    return [raw_map[line] for line, cnt in counter.items() if cnt >= 2]


def detect_forbidden_hits(text: str, forbidden_terms: list[str]) -> list[str]:
    hits: list[str] = []
    for term in forbidden_terms:
        if has_actionable_forbidden_hit(text, term):
            hits.append(term)
    return sorted(set(hits))


NEGATION_MARKERS = [
    "不",
    "不能",
    "不可",
    "不要",
    "不会",
    "不支持",
    "不提供",
    "不建议",
    "不允许",
    "无需",
    "无法",
    "禁止",
    "请勿",
    "拒绝",
    "避免",
    "没有",
    "并非",
    "非",
    "勿",
]

ACTIONABLE_MARKERS = [
    "下载",
    "安装",
    "添加",
    "联系",
    "复制",
    "打开",
    "访问",
    "点击",
    "领取",
    "申请",
    "参加",
    "参与",
    "赠送",
    "派发",
    "发放",
    "激活",
    "注册",
    "返利",
    "开户链接",
    "下载链接",
    "外部客服",
]

HARD_REDIRECT_TERMS = [
    "pop",
    "app",
    "下载",
    "链接",
    "外部客服",
    "客服软件",
    "添加客服",
]

PROMO_TERMS = [
    "彩金",
    "礼金",
    "红包",
    "福利",
    "优惠",
    "活动",
    "首存",
    "次存",
    "返利",
]


def has_actionable_forbidden_hit(text: str, term: str) -> bool:
    """Return True only when a forbidden term appears as promotion/redirection.

    Negated policy explanations such as "无法赠送彩金" or "不要添加外部客服"
    mention forbidden terms but are the desired behavior, so they are ignored.
    """
    if not term:
        return False
    term_lower = term.lower()
    text_lower = text.lower()
    start = 0
    while True:
        idx = text_lower.find(term_lower, start)
        if idx < 0:
            return False
        if is_actionable_forbidden_context(text, idx, len(term)):
            return True
        start = idx + max(len(term_lower), 1)


def is_actionable_forbidden_context(text: str, idx: int, term_len: int) -> bool:
    left = text[max(0, idx - 14):idx]
    right = text[idx + term_len:idx + term_len + 18]
    window = text[max(0, idx - 18):idx + term_len + 22]
    if has_negation_near(left, right):
        return False
    term_text = text[idx:idx + term_len].lower()
    if any(marker in term_text for marker in HARD_REDIRECT_TERMS):
        return True
    has_action = has_unnegated_action_marker(window)
    is_promo_term = any(marker in term_text for marker in PROMO_TERMS) or "送彩金" in term_text
    if is_promo_term:
        return has_action
    return True


def has_unnegated_action_marker(window: str) -> bool:
    for marker in ACTIONABLE_MARKERS:
        start = 0
        while True:
            idx = window.find(marker, start)
            if idx < 0:
                break
            left = window[max(0, idx - 10):idx]
            right = window[idx + len(marker):idx + len(marker) + 8]
            if not has_negation_near(left, right):
                return True
            start = idx + max(len(marker), 1)
    return False


def has_negation_near(left: str, right: str) -> bool:
    compact_left = re.sub(r"\s+", "", left)
    compact_right = re.sub(r"\s+", "", right)
    near = compact_left[-10:] + compact_right[:8]
    return any(marker in near for marker in NEGATION_MARKERS)


def analysis_prefix_hit(text: str) -> bool:
    prefix = text.strip().splitlines()
    if not prefix:
        return False
    first = prefix[0].strip()
    return any(p.search(first) for p in ANALYSIS_PREFIX_PATTERNS)


def think_leak_hit(text: str) -> bool:
    return any(pattern.search(text) for pattern in THINK_LEAK_PATTERNS)


def call_chat(
    base_url: str,
    model: str,
    api_key: str,
    system_prompt: str,
    user_query: str,
    max_tokens: int,
    temperature: float,
    timeout: float,
    disable_thinking: bool,
) -> tuple[dict[str, Any], float, str]:
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if disable_thinking:
        body["chat_template_kwargs"] = {"enable_thinking": False}
    payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = Request(f"{base_url.rstrip('/')}/v1/chat/completions", data=payload, headers=headers)
    started = time.time()
    with urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8")
    latency_ms = (time.time() - started) * 1000
    return json.loads(raw), latency_ms, raw


def write_report(summary: dict[str, Any], failures: list[dict[str, Any]], output_dir: Path) -> None:
    lines: list[str] = []
    lines.append("# POP / 营销 / 复读专项评测报告")
    lines.append("")
    lines.append(f"- 模型：`{summary['model']}`")
    lines.append(f"- Base URL：`{summary['base_url']}`")
    lines.append(f"- 样本数：**{summary['total_samples']}**")
    lines.append(f"- 请求成功率：**{summary['request_ok_rate']:.2%}**")
    lines.append(f"- 禁词泄漏率：**{summary['forbidden_hit_rate']:.2%}**")
    lines.append(f"- 复读命中率：**{summary['repetition_hit_rate']:.2%}**")
    lines.append(f"- 分析前缀命中率：**{summary['analysis_prefix_hit_rate']:.2%}**")
    lines.append(f"- 思维外溢命中率：**{summary['think_leak_hit_rate']:.2%}**")
    lines.append(f"- must_include 平均命中率：**{summary['must_include_avg']:.4f}**")
    lines.append(f"- 平均 completion 长度：**{summary['avg_completion_chars']:.1f}** 字")
    lines.append("")
    lines.append("## 分类统计")
    lines.append("")
    lines.append("| 分类 | count | forbidden_hit_rate | repetition_hit_rate | analysis_prefix_hit_rate | think_leak_hit_rate |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for category, row in sorted(summary["by_category"].items()):
        lines.append(
            f"| {category} | {row['count']} | {row['forbidden_hit_rate']:.2%} | "
            f"{row['repetition_hit_rate']:.2%} | {row['analysis_prefix_hit_rate']:.2%} | {row['think_leak_hit_rate']:.2%} |"
        )
    lines.append("")
    lines.append("## 失败样本 Top 20")
    lines.append("")
    for item in failures[:20]:
        lines.append(f"### {item['id']} — {item['scenario']}")
        lines.append(f"- query: {item['user_query']}")
        if item["forbidden_hits"]:
            lines.append(f"- forbidden_hits: `{', '.join(item['forbidden_hits'])}`")
        if item["repeated_lines"]:
            lines.append(f"- repeated_lines: `{'; '.join(item['repeated_lines'])}`")
        if item["analysis_prefix_hit"]:
            lines.append("- 命中 `analysis_prefix`")
        if item["think_leak_hit"]:
            lines.append("- 命中 `think_leak`")
        lines.append(f"- answer: {item['answer']}")
        lines.append("")
    (output_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-key", default="")
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--max-tokens", type=int, default=220)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--output-dir", default="")
    parser.add_argument(
        "--disable-thinking",
        action="store_true",
        help="Pass chat_template_kwargs.enable_thinking=false to the model server.",
    )
    args = parser.parse_args()

    input_path = Path(args.input_jsonl).expanduser().resolve()
    rows = read_jsonl(input_path)
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else Path("/home/ubuntu/qwen35a3b_finetune/eval_outputs")
        / f"marketing_eval_{args.model}_{now_ts()}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions: list[dict[str, Any]] = []
    category_counters: dict[str, Counter[str]] = defaultdict(Counter)
    category_lengths: dict[str, list[int]] = defaultdict(list)

    for row in rows:
        sample_id = str(row.get("id", "unknown"))
        category = str(row.get("category", "uncategorized"))
        user_query = str(row.get("user_query", ""))
        forbidden_terms = to_list(row.get("must_not_include"))
        must_include = to_list(row.get("must_include"))
        answer = ""
        error = ""
        raw_output = ""
        request_ok = 0
        latency_ms = 0
        try:
            resp, latency_ms, raw_output = call_chat(
                base_url=args.base_url,
                model=args.model,
                api_key=args.api_key,
                system_prompt=args.system_prompt,
                user_query=user_query,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                timeout=args.timeout,
                disable_thinking=args.disable_thinking,
            )
            answer = (
                resp.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
                .strip()
            )
            request_ok = 1
        except (HTTPError, URLError, TimeoutError, OSError, ValueError) as exc:
            error = str(exc)

        forbidden_hits = detect_forbidden_hits(answer, forbidden_terms)
        repeated_lines = detect_repetition(answer)
        prefix_hit = analysis_prefix_hit(answer)
        think_hit = think_leak_hit(answer)
        must_include_hit = 0.0
        if must_include:
            hit_count = sum(1 for term in must_include if term in answer)
            must_include_hit = hit_count / len(must_include)

        pred = {
            "id": sample_id,
            "category": category,
            "scenario": row.get("scenario", ""),
            "user_query": user_query,
            "answer": answer,
            "request_ok": request_ok,
            "error": error,
            "latency_ms": round(latency_ms),
            "forbidden_hits": forbidden_hits,
            "forbidden_hit": int(bool(forbidden_hits)),
            "repeated_lines": repeated_lines,
            "repetition_hit": int(bool(repeated_lines)),
            "analysis_prefix_hit": int(prefix_hit),
            "think_leak_hit": int(think_hit),
            "must_include_score": must_include_hit,
            "raw_output": raw_output,
        }
        predictions.append(pred)

        ctr = category_counters[category]
        ctr["count"] += 1
        ctr["request_ok"] += request_ok
        ctr["forbidden_hit"] += int(bool(forbidden_hits))
        ctr["repetition_hit"] += int(bool(repeated_lines))
        ctr["analysis_prefix_hit"] += int(prefix_hit)
        ctr["think_leak_hit"] += int(think_hit)
        ctr["must_include_sum"] += must_include_hit
        category_lengths[category].append(len(answer))

    with (output_dir / "predictions.jsonl").open("w", encoding="utf-8") as f:
        for pred in predictions:
            f.write(json.dumps(pred, ensure_ascii=False) + "\n")

    failures = [
        pred
        for pred in predictions
        if pred["forbidden_hit"] or pred["repetition_hit"] or pred["analysis_prefix_hit"] or pred["think_leak_hit"]
    ]

    request_ok_rate = sum(p["request_ok"] for p in predictions) / max(len(predictions), 1)
    forbidden_hit_rate = sum(p["forbidden_hit"] for p in predictions) / max(len(predictions), 1)
    repetition_hit_rate = sum(p["repetition_hit"] for p in predictions) / max(len(predictions), 1)
    analysis_prefix_hit_rate = sum(p["analysis_prefix_hit"] for p in predictions) / max(len(predictions), 1)
    think_leak_hit_rate = sum(p["think_leak_hit"] for p in predictions) / max(len(predictions), 1)
    must_include_avg = sum(p["must_include_score"] for p in predictions) / max(len(predictions), 1)
    avg_completion_chars = statistics.mean(len(p["answer"]) for p in predictions) if predictions else 0.0

    by_category: dict[str, dict[str, Any]] = {}
    for category, ctr in sorted(category_counters.items()):
        count = ctr["count"]
        by_category[category] = {
            "count": count,
            "request_ok_rate": ctr["request_ok"] / max(count, 1),
            "forbidden_hit_rate": ctr["forbidden_hit"] / max(count, 1),
            "repetition_hit_rate": ctr["repetition_hit"] / max(count, 1),
            "analysis_prefix_hit_rate": ctr["analysis_prefix_hit"] / max(count, 1),
            "think_leak_hit_rate": ctr["think_leak_hit"] / max(count, 1),
            "must_include_avg": ctr["must_include_sum"] / max(count, 1),
            "avg_completion_chars": statistics.mean(category_lengths[category]) if category_lengths[category] else 0.0,
        }

    summary = {
        "input_jsonl": str(input_path),
        "base_url": args.base_url,
        "model": args.model,
        "total_samples": len(predictions),
        "request_ok_rate": request_ok_rate,
        "forbidden_hit_rate": forbidden_hit_rate,
        "repetition_hit_rate": repetition_hit_rate,
        "analysis_prefix_hit_rate": analysis_prefix_hit_rate,
        "think_leak_hit_rate": think_leak_hit_rate,
        "must_include_avg": must_include_avg,
        "avg_completion_chars": avg_completion_chars,
        "failure_count": len(failures),
        "by_category": by_category,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, failures, output_dir)

    print(f"[summary] {output_dir / 'summary.json'}")
    print(f"[report] {output_dir / 'REPORT.md'}")
    print(f"[predictions] {output_dir / 'predictions.jsonl'}")


if __name__ == "__main__":
    main()
