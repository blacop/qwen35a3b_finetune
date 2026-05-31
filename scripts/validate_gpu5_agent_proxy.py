#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from urllib import error, request


DEFAULT_BASE_URL = "http://127.0.0.1:8001"
DEFAULT_MODEL = "qwen35a3b-domain-text-vl-gpu5"
DEFAULT_TIMEOUT = 120.0


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str
    payload: dict[str, Any] | None = None


def http_json(
    url: str,
    *,
    method: str = "GET",
    payload: dict[str, Any] | None = None,
    timeout: float = DEFAULT_TIMEOUT,
    headers: dict[str, str] | None = None,
) -> dict[str, Any]:
    data = None
    req_headers = dict(headers or {})
    if payload is not None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req_headers.setdefault("Content-Type", "application/json")
    req = request.Request(url=url, data=data, headers=req_headers, method=method)
    with request.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode("utf-8")
    return json.loads(body)


def wait_for_health(base_url: str, timeout: float) -> dict[str, Any]:
    deadline = time.time() + timeout
    last_error = ""
    while time.time() < deadline:
        try:
            payload = http_json(f"{base_url}/health", timeout=5)
            if payload.get("ok") is True:
                return payload
            last_error = json.dumps(payload, ensure_ascii=False)
        except Exception as exc:  # noqa: BLE001
            last_error = repr(exc)
        time.sleep(2)
    raise RuntimeError(f"health check timeout: {last_error}")


def extract_text(response: dict[str, Any]) -> str:
    choices = response.get("choices") or []
    if not choices:
        return ""
    message = choices[0].get("message") or {}
    content = message.get("content")
    return content if isinstance(content, str) else ""


def run_glossary_check(base_url: str, model: str, timeout: float) -> CheckResult:
    resp = http_json(
        f"{base_url}/v1/chat/completions",
        method="POST",
        timeout=timeout,
        payload={
            "model": model,
            "messages": [{"role": "user", "content": "RTP是什么意思？请直接两句话回答。"}],
            "max_tokens": 128,
        },
    )
    text = extract_text(resp)
    rtp_meaning_ok = "返奖率" in text or "理论投注比例" in text or "理论返奖率" in text
    ok = rtp_meaning_ok
    detail = text.strip() or "empty response"
    return CheckResult(name="glossary_rtp", ok=ok, detail=detail, payload=resp)


def run_agent_check(
    *,
    name: str,
    base_url: str,
    model: str,
    user_query: str,
    expected_tool: str,
    timeout: float,
) -> CheckResult:
    resp = http_json(
        f"{base_url}/v1/chat/completions/agent",
        method="POST",
        timeout=timeout,
        payload={
            "model": model,
            "messages": [{"role": "user", "content": user_query}],
            "max_tokens": 256,
        },
    )
    trace = ((resp.get("_agent") or {}).get("trace")) or []
    trace_redacted = bool((resp.get("_agent") or {}).get("trace_redacted"))
    tool_names = [item.get("tool") for item in trace if isinstance(item, dict)]
    text = extract_text(resp)
    ok = bool(text.strip()) and (trace_redacted or expected_tool in tool_names)
    detail = f"tools={tool_names}; trace_redacted={trace_redacted}; answer={text.strip()}"
    return CheckResult(name=name, ok=ok, detail=detail, payload=resp)


def run_all(base_url: str, model: str, timeout: float) -> list[CheckResult]:
    health = wait_for_health(base_url, timeout)
    results: list[CheckResult] = [
        CheckResult(
            name="health",
            ok=health.get("ok") is True and (
                str(health.get("upstream_base_url", "")).endswith("/v1")
                or health.get("diagnostics_redacted") is True
            ),
            detail=json.dumps(health, ensure_ascii=False),
            payload=health,
        )
    ]

    glossary_stats = http_json(f"{base_url}/glossary_stats", timeout=timeout)
    sample_terms = glossary_stats.get("sample_terms") or []
    glossary_ok = glossary_stats.get("enabled") is True and any(term == "包网" for term in sample_terms)
    results.append(
        CheckResult(
            name="glossary_stats",
            ok=glossary_ok,
            detail=json.dumps(glossary_stats, ensure_ascii=False),
            payload=glossary_stats,
        )
    )

    agent_health = http_json(f"{base_url}/v1/agent/health", timeout=timeout)
    tools = [item.get("function", {}).get("name") for item in (agent_health.get("tools") or [])]
    required_tools = {
        "query_member",
        "query_order",
        "query_recharge",
        "get_game_type_list",
        "get_game_vendor_list",
        "search_game_vendor_list",
        "list_bet_orders",
        "get_bet_order_stats",
        "get_order_status_list",
        "get_activity_money_tasks",
        "get_deposit_task_list",
        "check_ip_blacklist",
        "get_user_account_list",
        "handover_to_human",
    }
    agent_ok = agent_health.get("skills_enabled") and agent_health.get("skills_registered") and (
        agent_health.get("diagnostics_redacted") is True
        or required_tools.issubset(set(tools))
    )
    results.append(
        CheckResult(
            name="agent_health",
            ok=bool(agent_ok),
            detail=json.dumps(agent_health, ensure_ascii=False),
            payload=agent_health,
        )
    )

    results.append(run_glossary_check(base_url, model, timeout))
    results.append(
        run_agent_check(
            name="agent_member",
            base_url=base_url,
            model=model,
            user_query="请使用工具查询会员 user_10086 的信息，并简要返回结果。",
            expected_tool="query_member",
            timeout=timeout,
        )
    )
    results.append(
        run_agent_check(
            name="agent_order",
            base_url=base_url,
            model=model,
            user_query="请调用工具查询订单 BET202604270001，并返回订单状态。",
            expected_tool="query_order",
            timeout=timeout,
        )
    )
    results.append(
        run_agent_check(
            name="agent_recharge",
            base_url=base_url,
            model=model,
            user_query="请使用工具查询充值订单 RC202604270001 的结果，并简要说明。",
            expected_tool="query_recharge",
            timeout=timeout,
        )
    )
    results.append(
        run_agent_check(
            name="agent_game_type_list",
            base_url=base_url,
            model=model,
            user_query="请调用工具查询游戏类型列表。",
            expected_tool="get_game_type_list",
            timeout=timeout,
        )
    )
    results.append(
        run_agent_check(
            name="agent_order_status_list",
            base_url=base_url,
            model=model,
            user_query="请调用工具查询订单状态列表。",
            expected_tool="get_order_status_list",
            timeout=timeout,
        )
    )
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate gpu5 proxy glossary and agent mock skills.")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT)
    parser.add_argument("--report-json", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        results = run_all(args.base_url.rstrip("/"), args.model, args.timeout)
    except Exception as exc:  # noqa: BLE001
        print(f"[FAIL] validation bootstrap error: {exc}", file=sys.stderr)
        return 1

    failed = [item for item in results if not item.ok]
    for item in results:
        prefix = "PASS" if item.ok else "FAIL"
        print(f"[{prefix}] {item.name}: {item.detail}")

    if args.report_json:
        report_path = Path(args.report_json)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report = {
            "base_url": args.base_url,
            "model": args.model,
            "all_passed": not failed,
            "results": [asdict(item) for item in results],
        }
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[INFO] report_json={report_path}")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
