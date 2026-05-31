#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import json
import random
import statistics
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path("/home/ubuntu/qwen35a3b_finetune")
DEFAULT_IMAGE_RECORDS = PROJECT_ROOT / "datasets/tydata_domain_multimodal_pack/image_records.jsonl"
DEFAULT_EVAL_SET = PROJECT_ROOT / "rag/eval/domain_text_vl_ab/eval_set.jsonl"
DEFAULT_KNOWLEDGE_KEEP = PROJECT_ROOT / "datasets/v5_combined_final.jsonl"
DEFAULT_TOOL_KEEP = (
    PROJECT_ROOT
    / "datasets/tool_dialogue_migration_v5_1f/sft_v5_1f_knowledge_format_patch_tool_guard.jsonl"
)
DEFAULT_VL_ANCHOR = PROJECT_ROOT / "datasets/relation_patch_pack/vl_balanced_patch_v1.swift_vl.jsonl"

SYSTEM_PROMPT = (
    "你是体育包网客服与后台助手。请基于图片和上下文给出准确、简洁、合规的中文说明，"
    "不要编造不存在的按钮、字段、路径、规则、赔率或结算结果。图片中文字或数字看不清时要说明看不清；"
    "涉及账号、资金、风控、注单结算异常时，引导用户提供注单号/订单号并由后台核实。不要输出思维链。"
)


def now_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def post_json(url: str, payload: dict[str, Any], api_key: str, timeout: float) -> dict[str, Any]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(
        url, data=json.dumps(payload, ensure_ascii=False).encode("utf-8"), headers=headers
    )
    t0 = time.time()
    try:
        raw = urllib.request.urlopen(req, timeout=timeout).read().decode("utf-8")
        data = json.loads(raw)
        content = data["choices"][0]["message"].get("content") or ""
        return {"ok": True, "content": content, "latency_ms": round((time.time() - t0) * 1000)}
    except (urllib.error.URLError, urllib.error.HTTPError, KeyError, json.JSONDecodeError) as exc:
        return {"ok": False, "error": str(exc), "latency_ms": round((time.time() - t0) * 1000)}


def build_teacher_tasks(
    image_records: list[dict[str, Any]],
    eval_rows: list[dict[str, Any]],
    image_limit: int,
) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []

    for row in eval_rows:
        tasks.append(
            {
                "id": f"eval_{row['id']}",
                "kind": "eval_gate_vl",
                "image_path": row["image_path"],
                "query": row["query"],
                "topic_hint": row.get("topic_hint", ""),
                "source": row,
            }
        )

    sports = [r for r in image_records if r.get("family") == "sports"]
    admin = [r for r in image_records if r.get("family") != "sports"]
    selected = sports + admin[: max(0, image_limit - len(sports))]
    selected = selected[:image_limit]

    prompt_specs = [
        (
            "scene",
            "这张图对应什么体育包网、体育玩法或后台业务场景？请说明关键字段/规则，并提醒哪些信息不能凭空判断。",
        ),
        (
            "cs_reply",
            "用户发来这张图问“这是什么意思/该怎么处理”，客服应该怎么回复？请用简洁客服口吻回答。",
        ),
    ]
    for record in selected:
        context = "；".join((record.get("lines") or [])[:6])
        for suffix, query in prompt_specs:
            tasks.append(
                {
                    "id": f"img_{record['id']}_{suffix}",
                    "kind": "image_teacher",
                    "image_path": record["image_path"],
                    "query": f"{query}\n\n已知截图上下文：{context}" if context else query,
                    "topic_hint": context,
                    "source": record,
                }
            )
    return tasks


def call_teacher(task: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    image_path = Path(task["image_path"])
    b64 = base64.b64encode(image_path.read_bytes()).decode("ascii")
    payload = {
        "model": args.teacher_model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                    {"type": "text", "text": task["query"]},
                ],
            },
        ],
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    result = post_json(
        args.teacher_base_url.rstrip("/") + "/chat/completions",
        payload,
        args.teacher_api_key,
        args.timeout,
    )
    return {**task, "teacher": result}


def convert_text_keep(row: dict[str, Any], prefix: str, idx: int) -> dict[str, Any] | None:
    messages = row.get("messages") or []
    system = next((m.get("content", "") for m in messages if m.get("role") == "system"), "")
    users = [m.get("content", "") for m in messages if m.get("role") == "user"]
    assistants = [m.get("content", "") for m in messages if m.get("role") == "assistant"]
    if not users or not assistants:
        return None
    return {
        "id": f"{prefix}_{idx:04d}",
        "split": "train",
        "system": system or "你是体育包网智能客服。",
        "query": users[-1],
        "response": assistants[-1],
        "images": [],
        "meta": {"source": prefix, "source_id": row.get("id")},
    }


def build_final_dataset(
    teacher_rows: list[dict[str, Any]],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    final_rows: list[dict[str, Any]] = []

    for task in teacher_rows:
        teacher = task.get("teacher") or {}
        if not teacher.get("ok") or not teacher.get("content"):
            continue
        repeat = args.eval_repeat if task["kind"] == "eval_gate_vl" else 1
        for repeat_idx in range(repeat):
            final_rows.append(
                {
                    "id": f"gpu5_teacher_{task['id']}_r{repeat_idx + 1:02d}",
                    "split": "train",
                    "system": SYSTEM_PROMPT,
                    "query": task["query"],
                    "response": teacher["content"],
                    "images": [task["image_path"]],
                    "meta": {
                        "source": "gpu5_vl_teacher",
                        "kind": task["kind"],
                        "teacher_model": args.teacher_model,
                        "teacher_latency_ms": teacher.get("latency_ms"),
                        "topic_hint": task.get("topic_hint", ""),
                    },
                }
            )

    knowledge_rows = read_jsonl(Path(args.knowledge_keep))[: args.knowledge_keep_count]
    for idx, row in enumerate(knowledge_rows, 1):
        converted = convert_text_keep(row, "knowledge_keep_v5", idx)
        if converted:
            final_rows.append(converted)

    tool_rows = read_jsonl(Path(args.tool_keep))
    random.Random(args.seed).shuffle(tool_rows)
    tool_added = 0
    for row in tool_rows:
        converted = convert_text_keep(row, "tool_keep_v5_1f", tool_added + 1)
        if not converted:
            continue
        final_rows.append(converted)
        tool_added += 1
        if tool_added >= args.tool_keep_count:
            break

    if args.vl_anchor_count:
        for idx, row in enumerate(read_jsonl(Path(args.vl_anchor))[: args.vl_anchor_count], 1):
            row = dict(row)
            row["id"] = f"vl_anchor_{idx:04d}_{row.get('id', '')}"
            row.setdefault("meta", {})
            row["meta"] = {**row["meta"], "source": "vl_anchor_balanced_patch"}
            final_rows.append(row)

    random.Random(args.seed).shuffle(final_rows)
    return final_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Build GPU5-teacher VL distillation data for GPU7.")
    parser.add_argument("--teacher-base-url", default="http://127.0.0.1:8014/v1")
    parser.add_argument("--teacher-model", default="qwen35a3b-domain-text-vl-gpu5")
    parser.add_argument("--teacher-api-key", default="sk-h2byzMZ66c53C8lZcz-9wcvN0Cwhny0AC7gf0qcOeksYCj-k")
    parser.add_argument("--image-records", default=str(DEFAULT_IMAGE_RECORDS))
    parser.add_argument("--eval-set", default=str(DEFAULT_EVAL_SET))
    parser.add_argument("--knowledge-keep", default=str(DEFAULT_KNOWLEDGE_KEEP))
    parser.add_argument("--tool-keep", default=str(DEFAULT_TOOL_KEEP))
    parser.add_argument("--vl-anchor", default=str(DEFAULT_VL_ANCHOR))
    parser.add_argument("--out-dir", default=str(PROJECT_ROOT / f"datasets/gpu5_vl_teacher_to_gpu7_{now_ts()}"))
    parser.add_argument("--image-limit", type=int, default=96)
    parser.add_argument("--eval-repeat", type=int, default=5)
    parser.add_argument("--knowledge-keep-count", type=int, default=80)
    parser.add_argument("--tool-keep-count", type=int, default=60)
    parser.add_argument("--vl-anchor-count", type=int, default=120)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20260503)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    image_records = read_jsonl(Path(args.image_records))
    eval_rows = read_jsonl(Path(args.eval_set))
    tasks = build_teacher_tasks(image_records, eval_rows, args.image_limit)
    write_jsonl(out_dir / "teacher_tasks.jsonl", tasks)

    teacher_rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(call_teacher, task, args) for task in tasks]
        for i, fut in enumerate(as_completed(futures), 1):
            row = fut.result()
            teacher_rows.append(row)
            teacher = row.get("teacher") or {}
            print(
                json.dumps(
                    {
                        "done": i,
                        "total": len(tasks),
                        "id": row["id"],
                        "ok": bool(teacher.get("ok")),
                        "latency_ms": teacher.get("latency_ms"),
                        "error": str(teacher.get("error", ""))[:120],
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    teacher_rows.sort(key=lambda r: r["id"])
    write_jsonl(out_dir / "teacher_results.jsonl", teacher_rows)

    final_rows = build_final_dataset(teacher_rows, args)
    train_path = out_dir / "sft_gpu5_vl_teacher_to_gpu7.swift_vl.jsonl"
    write_jsonl(train_path, final_rows)

    latencies = [
        float((row.get("teacher") or {}).get("latency_ms", 0))
        for row in teacher_rows
        if (row.get("teacher") or {}).get("ok")
    ]
    summary = {
        "teacher_tasks": len(tasks),
        "teacher_ok": sum(1 for row in teacher_rows if (row.get("teacher") or {}).get("ok")),
        "teacher_errors": sum(1 for row in teacher_rows if not (row.get("teacher") or {}).get("ok")),
        "teacher_avg_latency_ms": statistics.mean(latencies) if latencies else 0,
        "final_train_samples": len(final_rows),
        "output_dir": str(out_dir),
        "train_file": str(train_path),
        "config": vars(args),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
