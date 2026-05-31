#!/usr/bin/env python3
"""Check GPU7 VL consistency across direct vLLM, chat proxy, and agent proxy."""

from __future__ import annotations

import argparse
import base64
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict
from urllib.request import Request, urlopen


PROJECT_ROOT = Path("/home/ubuntu/qwen35a3b_finetune")
DEFAULT_IMAGE = (
    PROJECT_ROOT
    / "datasets/tydata_domain_multimodal_pack/assets/docx/JT包网后台操作手册/images/image235.png"
)
DEFAULT_OUT_DIR = PROJECT_ROOT / "reports"


def now_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def post_json(url: str, payload: Dict[str, Any], api_key: str, timeout: int) -> Dict[str, Any]:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = Request(url, data=data, headers=headers, method="POST")
    t0 = time.perf_counter()
    try:
        with urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
            status = resp.status
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}", "latency_ms": int((time.perf_counter() - t0) * 1000)}
    latency_ms = int((time.perf_counter() - t0) * 1000)
    try:
        body = json.loads(raw)
    except Exception:
        return {"ok": False, "status": status, "error": "invalid_json", "raw": raw[:500], "latency_ms": latency_ms}
    message = ((body.get("choices") or [{}])[0].get("message") or {}) if isinstance(body, dict) else {}
    content = message.get("content") or ""
    return {
        "ok": bool(status < 400 and body.get("choices")),
        "status": status,
        "error": body.get("error"),
        "model": body.get("model"),
        "content_preview": str(content)[:300],
        "latency_ms": latency_ms,
    }


def build_payload(model: str, image_path: Path, query: str, max_tokens: int) -> Dict[str, Any]:
    image_b64 = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": "你是体育包网客服助手。请基于截图直接回答。"},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
                    {"type": "text", "text": query},
                ],
            },
        ],
        "max_tokens": max_tokens,
        "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose GPU7 visual request path consistency.")
    parser.add_argument("--direct-base-url", default="http://127.0.0.1:8013/v1")
    parser.add_argument("--proxy-base-url", default="http://127.0.0.1:8010/v1")
    parser.add_argument("--direct-model", default="qwen35a3b-sft-v5-combined")
    parser.add_argument("--proxy-model", default="gpu7-v5-combined")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--image", default=str(DEFAULT_IMAGE))
    parser.add_argument("--query", default="这张图更接近 PNL 统计页还是 RTP 配置页？请说明理由。")
    parser.add_argument("--max-tokens", type=int, default=160)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    image_path = Path(args.image)
    direct_payload = build_payload(args.direct_model, image_path, args.query, args.max_tokens)
    proxy_payload = build_payload(args.proxy_model, image_path, args.query, args.max_tokens)

    report = {
        "ts": now_ts(),
        "image": str(image_path),
        "checks": {
            "direct_vllm": post_json(
                args.direct_base_url.rstrip("/") + "/chat/completions",
                direct_payload,
                args.api_key,
                args.timeout,
            ),
            "proxy_chat": post_json(
                args.proxy_base_url.rstrip("/") + "/chat/completions",
                proxy_payload,
                args.api_key,
                args.timeout,
            ),
            "proxy_agent": post_json(
                args.proxy_base_url.rstrip("/") + "/chat/completions/agent",
                proxy_payload,
                args.api_key,
                args.timeout,
            ),
        },
        "expected": {
            "direct_model": args.direct_model,
            "public_proxy_model": args.proxy_model,
            "proxy_should_route_to": args.direct_model,
        },
    }
    report["passed"] = all(item.get("ok") for item in report["checks"].values())

    out_path = Path(args.out) if args.out else DEFAULT_OUT_DIR / f"gpu7_vl_proxy_chain_{report['ts']}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"passed": report["passed"], "out": str(out_path), "checks": report["checks"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
