#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import csv
import json
import statistics
import subprocess
import tempfile
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence


PROJECT_ROOT = Path("/home/ubuntu/qwen35a3b_finetune")
DEFAULT_EVAL_SET = PROJECT_ROOT / "rag" / "eval" / "domain_text_vl_ab" / "eval_set.jsonl"
DEFAULT_OUT_ROOT = PROJECT_ROOT / "rag" / "eval" / "domain_text_vl_ab"
DEFAULT_TEXT_SYSTEM = "你是体育包网客服助手。请只基于给定截图上下文回答，不要编造图中未给出的字段或按钮。"
DEFAULT_VL_SYSTEM = "你是体育包网客服助手。请基于截图直接回答，不要编造看不到的字段、按钮、流程或结论。"


def now_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_text(text: str) -> str:
    return " ".join(str(text or "").replace("\r", " ").replace("\n", " ").split()).lower()


def score_answer(answer: str, rubric_groups: Sequence[Sequence[str]]) -> Dict[str, Any]:
    norm = normalize_text(answer)
    hits = 0
    detail: List[Dict[str, Any]] = []
    for group in rubric_groups:
        group_hit = any(normalize_text(term) in norm for term in group)
        if group_hit:
            hits += 1
        detail.append({"group": list(group), "hit": group_hit})
    score = hits / len(rubric_groups) if rubric_groups else 0.0
    return {"group_hits": hits, "group_total": len(rubric_groups), "score": round(score, 4), "detail": detail}


def curl_chat(url: str, api_key: str, body: Dict[str, Any], timeout: int) -> Dict[str, Any]:
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".json", delete=False) as tmp:
        tmp.write(json.dumps(body, ensure_ascii=False))
        tmp_path = tmp.name

    safe_url = url.rstrip("/") + "/v1/chat/completions"
    auth_part = f"-H 'Authorization: Bearer {api_key}' " if api_key else ""
    cmd = (
        f"curl -sS -m {timeout} "
        f"-H 'Content-Type: application/json' "
        f"{auth_part}"
        f"'{safe_url}' --data-binary '@{tmp_path}'"
    )

    t0 = time.time()
    proc = subprocess.run(["bash", "-lc", cmd], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    Path(tmp_path).unlink(missing_ok=True)
    latency_s = round(time.time() - t0, 3)
    if proc.returncode != 0:
        return {"ok": False, "error": proc.stderr.strip() or f"curl_exit_{proc.returncode}", "latency_s": latency_s}
    try:
        data = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return {"ok": False, "error": "invalid_json_response", "raw": proc.stdout[:1000], "latency_s": latency_s}

    try:
        content = data["choices"][0]["message"]["content"]
    except Exception:
        return {"ok": False, "error": "missing_choices", "raw": data, "latency_s": latency_s}
    return {
        "ok": True,
        "content": content,
        "usage": data.get("usage", {}),
        "latency_s": latency_s,
    }


def call_text(sample: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    prompt = (
        f"【截图上下文】\n{sample['ocr_context']}\n\n"
        f"【补充提示】\n{sample['topic_hint']}\n\n"
        f"【问题】\n{sample['query']}"
    )
    body = {
        "model": args.text_model,
        "messages": [
            {"role": "system", "content": args.text_system},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
    }
    if args.disable_thinking:
        body["chat_template_kwargs"] = {"enable_thinking": False}
    return curl_chat(args.text_base_url, args.text_api_key, body, args.timeout)


def call_vl(sample: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    b64 = base64.b64encode(Path(sample["image_path"]).read_bytes()).decode()
    body = {
        "model": args.vl_model,
        "messages": [
            {
                "role": "system",
                "content": args.vl_system,
            },
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                    {"type": "text", "text": sample["query"]},
                ],
            },
        ],
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
    }
    if args.disable_thinking:
        body["chat_template_kwargs"] = {"enable_thinking": False}
    return curl_chat(args.vl_base_url, args.vl_api_key, body, args.timeout)


def build_markdown_report(path: Path, rows: Sequence[Dict[str, Any]], summary: Dict[str, Any]) -> None:
    lines: List[str] = []
    lines.append("# Domain Text vs VL AB")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- text_model: `{summary['config']['text_model']}` @ `{summary['config']['text_base_url']}`")
    lines.append(f"- vl_model: `{summary['config']['vl_model']}` @ `{summary['config']['vl_base_url']}`")
    lines.append(f"- total_samples: `{summary['total_samples']}`")
    lines.append(f"- text_avg_score: `{summary['text_avg_score']}`")
    lines.append(f"- vl_avg_score: `{summary['vl_avg_score']}`")
    lines.append(f"- text_avg_latency_s: `{summary['text_avg_latency_s']}`")
    lines.append(f"- vl_avg_latency_s: `{summary['vl_avg_latency_s']}`")
    lines.append("")
    lines.append("## Per Sample")
    lines.append("")

    for row in rows:
        lines.append(f"### {row['id']} {row['dimension']}")
        lines.append("")
        lines.append(f"- query: {row['query']}")
        lines.append(f"- image: `{row['image_path']}`")
        lines.append(f"- context: {row['topic_hint']}")
        lines.append(f"- text_score: `{row['text_eval']['score']}`")
        lines.append(f"- vl_score: `{row['vl_eval']['score']}`")
        lines.append("")
        lines.append("**Text**")
        lines.append("")
        lines.append(row["text_result"].get("content", ""))
        lines.append("")
        lines.append("**VL**")
        lines.append("")
        lines.append(row["vl_result"].get("content", ""))
        lines.append("")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a small AB comparison between text and VL domain QA.")
    parser.add_argument("--eval-set", default=str(DEFAULT_EVAL_SET))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_ROOT / f"run_{now_ts()}"))
    parser.add_argument("--text-base-url", default="http://127.0.0.1:8001")
    parser.add_argument("--text-model", default="qwen35a3b-domain-text-vl-gpu5")
    parser.add_argument("--text-api-key", default="sk-h2byzMZ66c53C8lZcz-9wcvN0Cwhny0AC7gf0qcOeksYCj-k")
    parser.add_argument("--vl-base-url", default="http://127.0.0.1:8001")
    parser.add_argument("--vl-model", default="qwen35a3b-domain-text-vl-gpu5")
    parser.add_argument("--vl-api-key", default="sk-h2byzMZ66c53C8lZcz-9wcvN0Cwhny0AC7gf0qcOeksYCj-k")
    parser.add_argument("--text-system", default=DEFAULT_TEXT_SYSTEM)
    parser.add_argument("--vl-system", default=DEFAULT_VL_SYSTEM)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--disable-thinking", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    eval_rows = read_jsonl(Path(args.eval_set))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results: List[Dict[str, Any]] = []
    for sample in eval_rows:
        text_result = call_text(sample, args)
        vl_result = call_vl(sample, args)
        text_eval = score_answer(text_result.get("content", ""), sample["rubric_groups"])
        vl_eval = score_answer(vl_result.get("content", ""), sample["rubric_groups"])
        results.append(
            {
                **sample,
                "text_result": text_result,
                "vl_result": vl_result,
                "text_eval": text_eval,
                "vl_eval": vl_eval,
            }
        )

    results_path = out_dir / "results.jsonl"
    write_jsonl(results_path, results)

    with (out_dir / "results.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "id",
                "dimension",
                "query",
                "text_score",
                "vl_score",
                "text_latency_s",
                "vl_latency_s",
            ]
        )
        for row in results:
            writer.writerow(
                [
                    row["id"],
                    row["dimension"],
                    row["query"],
                    row["text_eval"]["score"],
                    row["vl_eval"]["score"],
                    row["text_result"].get("latency_s", ""),
                    row["vl_result"].get("latency_s", ""),
                ]
            )

    text_scores = [row["text_eval"]["score"] for row in results]
    vl_scores = [row["vl_eval"]["score"] for row in results]
    text_latency = [row["text_result"].get("latency_s", 0) for row in results if row["text_result"].get("ok")]
    vl_latency = [row["vl_result"].get("latency_s", 0) for row in results if row["vl_result"].get("ok")]

    by_dimension: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: {"text": [], "vl": []})
    for row in results:
        by_dimension[row["dimension"]]["text"].append(row["text_eval"]["score"])
        by_dimension[row["dimension"]]["vl"].append(row["vl_eval"]["score"])

    summary = {
        "total_samples": len(results),
        "text_avg_score": round(statistics.mean(text_scores), 4) if text_scores else 0.0,
        "vl_avg_score": round(statistics.mean(vl_scores), 4) if vl_scores else 0.0,
        "text_avg_latency_s": round(statistics.mean(text_latency), 4) if text_latency else 0.0,
        "vl_avg_latency_s": round(statistics.mean(vl_latency), 4) if vl_latency else 0.0,
        "wins": dict(
            Counter(
                "text" if row["text_eval"]["score"] > row["vl_eval"]["score"]
                else "vl" if row["vl_eval"]["score"] > row["text_eval"]["score"]
                else "tie"
                for row in results
            )
        ),
        "by_dimension": {
            dim: {
                "text_avg_score": round(statistics.mean(scores["text"]), 4) if scores["text"] else 0.0,
                "vl_avg_score": round(statistics.mean(scores["vl"]), 4) if scores["vl"] else 0.0,
            }
            for dim, scores in by_dimension.items()
        },
        "config": {
            "eval_set": str(Path(args.eval_set)),
            "text_base_url": args.text_base_url,
            "text_model": args.text_model,
            "vl_base_url": args.vl_base_url,
            "vl_model": args.vl_model,
            "disable_thinking": args.disable_thinking,
        },
        "output_dir": str(out_dir),
    }

    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    build_markdown_report(out_dir / "REPORT.md", results, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
