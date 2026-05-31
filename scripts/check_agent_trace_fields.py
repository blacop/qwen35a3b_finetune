#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List
from urllib.request import Request, urlopen


def post_json(url: str, payload: Dict[str, Any], api_key: str, timeout: float) -> Dict[str, Any]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = Request(url, data=json.dumps(payload, ensure_ascii=False).encode("utf-8"), headers=headers)
    t0 = time.time()
    try:
        raw = urlopen(req, timeout=timeout).read().decode("utf-8")
        out = json.loads(raw)
        out["_latency_ms"] = round((time.time() - t0) * 1000, 2)
        return out
    except Exception as e:
        return {"error": str(e), "_latency_ms": round((time.time() - t0) * 1000, 2)}


def bool_field(row: Dict[str, Any], key: str) -> int:
    return int(bool(row.get(key)))


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate _agent trace key fields are present.")
    parser.add_argument("--base-url", required=True, help="OpenAI-compatible base URL, e.g. http://127.0.0.1:8110/v1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-key", default="")
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--min-pass-rate", type=float, default=1.0)
    parser.add_argument("--output-json", required=True)
    parser.add_argument(
        "--queries-json",
        default="",
        help="Optional JSON array of user queries; defaults to built-in tool-routing prompts.",
    )
    args = parser.parse_args()

    if args.queries_json:
        raw_items = json.loads(args.queries_json)
        if not isinstance(raw_items, list):
            raise SystemExit("--queries-json must be a JSON array")
        checks_input = []
        for item in raw_items:
            if isinstance(item, str):
                checks_input.append({"query": item})
            elif isinstance(item, dict):
                checks_input.append(item)
            else:
                raise SystemExit("--queries-json items must be string or object")
    else:
        checks_input = [
            {
                "query": "请务必先调用 get_transfer_log_list，再回复三行：工具名、参数、结果摘要。查询最近24小时转账记录，页码1，每页10，状态-1，类型-1，开始时间1746403200000，结束时间1746489600000。",
                "tool_name": "get_transfer_log_list",
                "require_trace": True,
                "require_content": False,
                "require_trace_args_non_empty": True,
            },
            {
                "query": "请务必先调用 list_bet_orders，再回复三行：工具名、参数、结果摘要。查询最近体育投注订单，页码1，每页10，状态settled，开始时间1746403200000，结束时间1746489600000。",
                "tool_name": "list_bet_orders",
                "require_trace": True,
                "require_content": False,
                "require_trace_args_non_empty": True,
            },
            {
                "query": "请用一句话回复：充值未到账一般先核对什么信息？",
                "require_trace": False,
                "require_content": True,
            },
        ]

    url = args.base_url.rstrip("/") + "/chat/completions/agent"
    rows: List[Dict[str, Any]] = []

    for idx, item in enumerate(checks_input, 1):
        if isinstance(item, str):
            item = {"query": item}
        tool_name = str(item.get("tool_name") or "").strip()
        query = str(item.get("query") or "").strip()
        require_trace = bool(item.get("require_trace", True if tool_name else False))
        require_content = bool(item.get("require_content", False if tool_name else True))
        require_trace_args = bool(item.get("require_trace_args", True))
        require_trace_args_non_empty = bool(item.get("require_trace_args_non_empty", False))
        payload = {
            "model": args.model,
            "messages": [{"role": "user", "content": query}],
            "max_tokens": 180,
            "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        if tool_name:
            payload["tool_choice"] = {"type": "function", "function": {"name": tool_name}}
        resp = post_json(url, payload, args.api_key, args.timeout)

        choices = resp.get("choices") or []
        message = (choices[0].get("message") if choices and isinstance(choices[0], dict) else {}) or {}
        agent = resp.get("_agent") or {}
        trace = agent.get("trace") or []
        trace0 = trace[0] if trace and isinstance(trace[0], dict) else {}
        tool_routing = agent.get("tool_routing") or {}
        trace_args = trace0.get("args")
        args_dict = isinstance(trace_args, dict)
        args_non_empty = args_dict and len(trace_args) > 0
        args_field_ok = args_non_empty if require_trace_args_non_empty else args_dict

        checks = {
            "content": int(bool(message.get("content"))),
            "trace_non_empty": int(len(trace) > 0),
            "trace_tool": int(bool(trace0.get("tool"))),
            "trace_args": int(args_field_ok),
            "trace_result_preview": int(bool(trace0.get("result_preview"))),
            "selected_group": int(bool(tool_routing.get("selected_group"))),
        }
        required_checks: List[str] = []
        if require_content:
            required_checks.append("content")
        if require_trace:
            required_checks.extend(["trace_non_empty", "trace_tool", "trace_result_preview", "selected_group"])
            if require_trace_args:
                required_checks.append("trace_args")
        row = {
            "sample_id": idx,
            "query": query,
            "latency_ms": resp.get("_latency_ms", 0),
            "error": resp.get("error", ""),
            "require_trace": require_trace,
            "require_content": require_content,
            "require_trace_args": require_trace_args,
            "require_trace_args_non_empty": require_trace_args_non_empty,
            "required_checks": required_checks,
            "checks": checks,
            "passed": int(all(checks.get(k, 0) == 1 for k in required_checks)),
        }
        rows.append(row)

    n = len(rows)
    pass_rate = sum(r["passed"] for r in rows) / max(n, 1)
    coverage = {
        "content": sum(bool_field(r["checks"], "content") for r in rows) / max(n, 1),
        "trace_non_empty": sum(bool_field(r["checks"], "trace_non_empty") for r in rows) / max(n, 1),
        "trace_tool": sum(bool_field(r["checks"], "trace_tool") for r in rows) / max(n, 1),
        "trace_args": sum(bool_field(r["checks"], "trace_args") for r in rows) / max(n, 1),
        "trace_result_preview": sum(bool_field(r["checks"], "trace_result_preview") for r in rows) / max(n, 1),
        "selected_group": sum(bool_field(r["checks"], "selected_group") for r in rows) / max(n, 1),
    }

    summary = {
        "samples": n,
        "pass_count": int(sum(r["passed"] for r in rows)),
        "pass_rate": pass_rate,
        "min_pass_rate": args.min_pass_rate,
        "field_coverage": coverage,
        "passed": pass_rate >= args.min_pass_rate,
        "results": rows,
    }

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if not summary["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
