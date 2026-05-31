#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
from urllib import error, request


UTC = timezone.utc
ROOT = Path(__file__).resolve().parent
REPORTS_DIR = ROOT / "reports"


@dataclass(frozen=True)
class ServiceDef:
    key: str
    label: str
    unit: str
    role: str
    health_url: Optional[str] = None


@dataclass(frozen=True)
class ProxyDef:
    key: str
    label: str
    env_path: Path
    unit: str
    health_url: str
    rag_stats_url: str
    vl_rag_stats_url: str
    audit_path: Path


@dataclass(frozen=True)
class RagDef:
    key: str
    label: str
    env_path: Path
    unit: str
    health_url: str


@dataclass(frozen=True)
class Finding:
    level: str
    text: str
    tag: str = ""


SERVICE_DEFS: List[ServiceDef] = [
    ServiceDef(
        key="gpu5_upstream",
        label="GPU5 vLLM 上游",
        unit="vllm-qwen35-gpu5-agent-upstream.service",
        role="模型上游 8014",
    ),
    ServiceDef(
        key="gpu5_proxy",
        label="GPU5 Guardrails Proxy",
        unit="dpo-guardrails-proxy-gpu5.service",
        role="代理层 8001",
        health_url="http://127.0.0.1:8001/health",
    ),
    ServiceDef(
        key="gpu5_rag",
        label="GPU5 RAG API",
        unit="tydata-rag-api.service",
        role="文本 RAG 18080",
        health_url="http://127.0.0.1:18080/health",
    ),
    ServiceDef(
        key="gpu7_upstream",
        label="GPU7 vLLM 上游",
        unit="vllm-qwen35-gpu7-baseline.service",
        role="模型上游 8013",
    ),
    ServiceDef(
        key="gpu7_proxy",
        label="GPU7 Guardrails Proxy",
        unit="dpo-guardrails-proxy.service",
        role="代理层 8010",
        health_url="http://127.0.0.1:8010/health",
    ),
    ServiceDef(
        key="gpu7_rag",
        label="GPU7 RAG API",
        unit="tydata-rag-api-gpu7.service",
        role="文本 RAG 18081",
        health_url="http://127.0.0.1:18081/health",
    ),
    ServiceDef(
        key="mm_rag",
        label="MM RAG API",
        unit="mm-rag-api.service",
        role="多模态 RAG 18100",
        health_url="http://127.0.0.1:18100/health",
    ),
    ServiceDef(
        key="gateway",
        label="External API Gateway",
        unit="external-api-gateway.service",
        role="对外网关 8025",
        health_url="http://127.0.0.1:8025/health",
    ),
]


PROXY_DEFS: List[ProxyDef] = [
    ProxyDef(
        key="gpu5",
        label="GPU5 Proxy",
        env_path=Path("/home/ubuntu/generate/qwen35a3b_finetune/services/dpo-guardrails-proxy-gpu5.env"),
        unit="dpo-guardrails-proxy-gpu5.service",
        health_url="http://127.0.0.1:8001/health",
        rag_stats_url="http://127.0.0.1:8001/rag_stats",
        vl_rag_stats_url="http://127.0.0.1:8001/vl_rag_stats",
        audit_path=Path("/home/ubuntu/generate/qwen35a3b_finetune/runtime/gpu5_dpo_guardrails_proxy/audit.jsonl"),
    ),
    ProxyDef(
        key="gpu7",
        label="GPU7 Proxy",
        env_path=Path("/home/ubuntu/generate/qwen35a3b_finetune/services/dpo-guardrails-proxy.env"),
        unit="dpo-guardrails-proxy.service",
        health_url="http://127.0.0.1:8010/health",
        rag_stats_url="http://127.0.0.1:8010/rag_stats",
        vl_rag_stats_url="http://127.0.0.1:8010/vl_rag_stats",
        audit_path=Path("/home/ubuntu/generate/qwen35a3b_finetune/runtime/gpu7_dpo_guardrails_proxy/audit.jsonl"),
    ),
]


RAG_DEFS: List[RagDef] = [
    RagDef(
        key="gpu5_rag",
        label="GPU5 RAG",
        env_path=ROOT / "rag_api.env",
        unit="tydata-rag-api.service",
        health_url="http://127.0.0.1:18080/health",
    ),
    RagDef(
        key="gpu7_rag",
        label="GPU7 RAG",
        env_path=ROOT / "rag_api_gpu7.env",
        unit="tydata-rag-api-gpu7.service",
        health_url="http://127.0.0.1:18081/health",
    ),
]


EXTERNAL_GATEWAY_AUDIT = Path("/home/ubuntu/generate/qwen35a3b_finetune/runtime/external_api_gateway/audit.jsonl")
GPU7_PROXY_UNIT_FILE = Path("/home/ubuntu/.config/systemd/user/dpo-guardrails-proxy.service")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate daily inference report for GPU5/GPU7 services.")
    parser.add_argument("--date", help="UTC date in YYYY-MM-DD. Default: today UTC.")
    parser.add_argument("--output-dir", default=str(REPORTS_DIR), help="Directory for markdown report output.")
    parser.add_argument("--stdout", action="store_true", help="Also print the generated report to stdout.")
    return parser.parse_args()


def utc_now() -> datetime:
    return datetime.now(tz=UTC)


def parse_report_date(raw: Optional[str]) -> date:
    if not raw:
        return utc_now().date()
    return date.fromisoformat(raw)


def make_window(report_day: date) -> tuple[datetime, datetime]:
    start = datetime.combine(report_day, time.min, tzinfo=UTC)
    end = start + timedelta(days=1)
    now = utc_now()
    if report_day == now.date() and now < end:
        end = now
    return start, end


def make_previous_compare_window(start: datetime, end: datetime) -> tuple[datetime, datetime]:
    duration = end - start
    prev_end = start
    prev_start = prev_end - duration
    return prev_start, prev_end


def run_cmd(args: List[str]) -> tuple[int, str, str]:
    proc = subprocess.run(args, capture_output=True, text=True)
    return proc.returncode, proc.stdout, proc.stderr


def systemctl_show(unit: str, props: Iterable[str]) -> Dict[str, str]:
    cmd = ["systemctl", "--user", "show", unit]
    for prop in props:
        cmd.extend(["--property", prop])
    code, out, err = run_cmd(cmd)
    data: Dict[str, str] = {}
    if code != 0:
        data["_error"] = (err or out or f"systemctl show failed for {unit}").strip()
        return data
    for line in out.splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        data[key] = value
    return data


def fetch_json(url: str, timeout_s: float = 4.0) -> Dict[str, Any]:
    req = request.Request(url, headers={"Accept": "application/json"})
    with request.urlopen(req, timeout=timeout_s) as resp:
        body = resp.read().decode("utf-8", errors="replace")
        return json.loads(body)


def safe_fetch_json(url: Optional[str], timeout_s: float = 4.0) -> Dict[str, Any]:
    if not url:
        return {"_ok": False, "_error": "health url not configured"}
    try:
        data = fetch_json(url, timeout_s=timeout_s)
        data["_ok"] = True
        return data
    except error.HTTPError as exc:
        return {"_ok": False, "_error": f"HTTP {exc.code}: {exc.reason}"}
    except Exception as exc:  # pragma: no cover - defensive only
        return {"_ok": False, "_error": str(exc)}


def parse_env_file(path: Path) -> Dict[str, str]:
    data: Dict[str, str] = {}
    if not path.exists():
        return data
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.strip().strip('"').strip("'")
        data[key.strip()] = value
    return data


def summarize_audit(path: Path, start: datetime, end: datetime) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "path": str(path),
        "exists": path.exists(),
        "events": 0,
        "accepted": 0,
        "ok": 0,
        "status_counts": Counter(),
        "route_counts": Counter(),
        "model_counts": Counter(),
        "key_counts": Counter(),
        "status_code_counts": Counter(),
        "upstream_status_code_counts": Counter(),
        "calc_degrade_bucket_counts": Counter(),
        "calc_missing_field_counts": Counter(),
        "stream_true": 0,
        "image_true": 0,
        "rag_true": 0,
        "vl_fallback_true": 0,
        "teacher_fallback_true": 0,
        "tool_trace_total": 0,
        "tool_trace_nonzero": 0,
        "elapsed_ms_samples": [],
        "max_tokens_samples": [],
        "latest_ts": None,
        "latest_request_ts": None,
        "latest_request_status_code": None,
        "latest_request_upstream_status_code": None,
        "latest_request_model": None,
        "last_bad_upstream_ts": None,
        "last_bad_upstream_status_code": None,
        "parse_errors": 0,
    }
    if not path.exists():
        return summary

    start_s = start.isoformat().replace("+00:00", "Z")
    end_s = end.isoformat().replace("+00:00", "Z")

    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            raw = line.strip()
            if not raw:
                continue
            try:
                item = json.loads(raw)
            except Exception:
                summary["parse_errors"] += 1
                continue
            ts_raw = str(item.get("ts") or "")
            if not ts_raw:
                continue
            if ts_raw < start_s:
                continue
            if ts_raw >= end_s:
                break
            summary["events"] += 1
            summary["latest_ts"] = ts_raw
            status = str(item.get("status") or "unknown")
            route = str(item.get("route") or "unknown")
            model = str(item.get("model") or item.get("detail") or "unknown")
            key_id = str(item.get("key_id") or "unknown")
            summary["status_counts"][status] += 1
            summary["route_counts"][route] += 1
            summary["model_counts"][model] += 1
            summary["key_counts"][key_id] += 1
            if status in {"accepted", "ok"}:
                summary["accepted"] += 1
            if status == "ok":
                summary["ok"] += 1
            if bool(item.get("stream")):
                summary["stream_true"] += 1
            if bool(item.get("has_image")):
                summary["image_true"] += 1
            if bool(item.get("rag_used")):
                summary["rag_true"] += 1
            if bool(item.get("vl_fallback_used")):
                summary["vl_fallback_true"] += 1
            if bool(item.get("teacher_fallback_used")):
                summary["teacher_fallback_true"] += 1
            tool_trace_count = item.get("tool_trace_count")
            if isinstance(tool_trace_count, int):
                summary["tool_trace_total"] += tool_trace_count
                if tool_trace_count > 0:
                    summary["tool_trace_nonzero"] += 1
            status_code = item.get("status_code")
            if isinstance(status_code, int):
                summary["status_code_counts"][str(status_code)] += 1
            upstream_status_code = item.get("upstream_status_code")
            if isinstance(upstream_status_code, int):
                summary["upstream_status_code_counts"][str(upstream_status_code)] += 1
            calc_degrade_bucket = item.get("calc_degrade_bucket")
            if isinstance(calc_degrade_bucket, str) and calc_degrade_bucket.strip():
                summary["calc_degrade_bucket_counts"][calc_degrade_bucket.strip()] += 1
            calc_missing_fields = item.get("calc_missing_fields")
            if isinstance(calc_missing_fields, list):
                for field in calc_missing_fields:
                    if isinstance(field, str) and field.strip():
                        summary["calc_missing_field_counts"][field.strip()] += 1
            if isinstance(status_code, int) or isinstance(upstream_status_code, int):
                summary["latest_request_ts"] = ts_raw
                summary["latest_request_status_code"] = status_code if isinstance(status_code, int) else None
                summary["latest_request_upstream_status_code"] = (
                    upstream_status_code if isinstance(upstream_status_code, int) else None
                )
                summary["latest_request_model"] = model
            if isinstance(upstream_status_code, int) and not str(upstream_status_code).startswith("2"):
                summary["last_bad_upstream_ts"] = ts_raw
                summary["last_bad_upstream_status_code"] = upstream_status_code
            elapsed_ms = item.get("elapsed_ms")
            if isinstance(elapsed_ms, (int, float)):
                summary["elapsed_ms_samples"].append(float(elapsed_ms))
            max_tokens = item.get("max_tokens")
            if isinstance(max_tokens, int):
                summary["max_tokens_samples"].append(max_tokens)
    return summary


def top_counter(counter: Counter[str], limit: int = 3) -> str:
    if not counter:
        return "-"
    return ", ".join(f"{k}:{v}" for k, v in counter.most_common(limit))


def merge_counters(counters: Iterable[Counter[str]]) -> Counter[str]:
    merged: Counter[str] = Counter()
    for counter in counters:
        merged.update(counter)
    return merged


def as_int(raw: Optional[str], default: int = 0) -> int:
    try:
        return int(str(raw or default).strip())
    except Exception:
        return default


def as_bool_label(raw: str) -> str:
    return "on" if str(raw).strip() not in {"", "0", "false", "False", "no"} else "off"


def proxy_auth_enabled(env: Dict[str, str]) -> bool:
    return bool(env.get("PROXY_API_KEY") or env.get("PROXY_API_KEYS"))


def proxy_rate_limit_enabled(env: Dict[str, str]) -> bool:
    return as_int(env.get("PROXY_RATE_LIMIT_RPM"), 0) > 0


def gpu7_dependency_configured() -> bool:
    if not GPU7_PROXY_UNIT_FILE.exists():
        return False
    text = GPU7_PROXY_UNIT_FILE.read_text(encoding="utf-8", errors="replace")
    return (
        "Requires=vllm-qwen35-gpu7-baseline.service" in text
        and "vllm-qwen35-gpu7-baseline.service" in text
        and "After=" in text
    )


def latest_request_recovered(summary: Dict[str, Any]) -> bool:
    code = summary.get("latest_request_upstream_status_code")
    if not isinstance(code, int):
        code = summary.get("latest_request_status_code")
    return isinstance(code, int) and str(code).startswith("2")


def fmt_delta(current: int, previous: int) -> str:
    delta = current - previous
    if delta > 0:
        return f"+{delta}"
    return str(delta)


def percentile(values: List[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * (pct / 100.0)
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    if low == high:
        return ordered[low]
    weight = rank - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def mean(values: List[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def human_bytes(raw: str) -> str:
    try:
        value = int(raw)
    except Exception:
        return "-"
    if value < 0:
        return "-"
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(value)
    unit = units[0]
    for unit in units:
        if size < 1024 or unit == units[-1]:
            break
        size /= 1024.0
    return f"{size:.1f} {unit}"


def human_seconds_from_ns(raw: str) -> str:
    try:
        ns = int(raw)
    except Exception:
        return "-"
    seconds = ns / 1_000_000_000
    if seconds < 60:
        return f"{seconds:.0f}s"
    minutes = seconds / 60
    if minutes < 60:
        return f"{minutes:.1f}m"
    hours = minutes / 60
    return f"{hours:.1f}h"


def finding_sort_key(level: str) -> int:
    order = {"高": 0, "中": 1, "低": 2}
    return order.get(level, 99)


def summarize_findings(findings: List[Finding]) -> Dict[str, int]:
    summary = {"高": 0, "中": 0, "低": 0}
    for item in findings:
        if item.level in summary:
            summary[item.level] += 1
    return summary


def summarize_mid_tags(findings: List[Finding]) -> Dict[str, int]:
    summary = {"影响收入": 0, "影响对外接入": 0, "影响稳定性": 0}
    for item in findings:
        if item.level == "中" and item.tag in summary:
            summary[item.tag] += 1
    return summary


def choose_decision(findings: List[Finding]) -> str:
    if not findings:
        return "维持现网配置，继续观察日常流量与失败率。"

    ordered = sorted(findings, key=lambda item: finding_sort_key(item.level))
    top = ordered[0]
    if top.level == "低":
        return "今日无高风险阻断，维持现网并继续观察流量、失败码和时延变化。"
    if top.tag == "影响对外接入":
        return "优先恢复对外接入链路和访问边界，避免影响外部调用与放量计划。"
    if top.tag == "影响稳定性":
        return "优先处理稳定性风险项，先降低重启回退和服务波动风险。"
    if top.tag == "影响收入":
        return "优先处理影响收入的风险项，避免交易、转化或客户服务能力受损。"
    if "gateway" in top.text.lower() or "8025" in top.text:
        return "优先恢复并稳定对外 gateway（8025）链路，再决定是否继续对外放量。"
    if "鉴权" in top.text or "限流" in top.text:
        return "优先启用 proxy 鉴权和 RPM 限流，先收口服务边界，再扩大接入范围。"
    if "依赖" in top.text or "重启顺序" in top.text:
        return "优先修正 systemd 依赖关系，避免服务重启后链路回退。"
    if top.level == "高":
        return "优先处理高风险项，先恢复核心链路稳定性，再推进优化动作。"
    if top.level == "中":
        return "优先处理最高优先级中风险项，避免治理缺口继续累积。"
    return "今日无高风险阻断，按低风险治理项逐步收口即可。"


def infer_risk_level(
    service_states: Dict[str, Dict[str, str]],
    service_health: Dict[str, Dict[str, Any]],
    proxy_envs: Dict[str, Dict[str, str]],
    proxy_audits: Optional[Dict[str, Dict[str, Any]]] = None,
) -> tuple[str, str]:
    core_units = [
        "vllm-qwen35-gpu5-agent-upstream.service",
        "dpo-guardrails-proxy-gpu5.service",
        "tydata-rag-api.service",
        "vllm-qwen35-gpu7-baseline.service",
        "dpo-guardrails-proxy.service",
        "tydata-rag-api-gpu7.service",
    ]
    if any(service_states.get(unit, {}).get("ActiveState") != "active" for unit in core_units):
        return "高", "核心推理链存在未在线服务。"

    critical_health_keys = ["gpu5_proxy", "gpu5_rag", "gpu7_proxy", "gpu7_rag", "mm_rag"]
    for key in critical_health_keys:
        health = service_health.get(key, {})
        if not health.get("_ok", False) or health.get("ok") is False:
            return "高", "核心接口健康检查存在失败。"

    if service_states.get("external-api-gateway.service", {}).get("ActiveState") != "active":
        return "中", "核心内链正常，但对外 gateway 当前未在线。"

    for env in proxy_envs.values():
        if not proxy_auth_enabled(env):
            return "中", "代理层仍未开启调用方鉴权。"
        if not proxy_rate_limit_enabled(env):
            return "中", "代理层仍未开启 RPM 限流。"

    if proxy_audits:
        for summary in proxy_audits.values():
            bad_codes = [
                code
                for code, count in summary.get("upstream_status_code_counts", {}).items()
                if count > 0 and not code.startswith("2")
            ]
            if bad_codes:
                if latest_request_recovered(summary):
                    return "低", f"核心链路在线；今日曾出现非 2xx 上游状态码，但最新真实请求已恢复 2xx。"
                return "中", f"核心链路在线，但今日最新真实请求仍出现非 2xx 上游状态码：{', '.join(sorted(bad_codes))}。"

    return "低", "核心链路在线，且未发现明显治理缺口。"


def build_exec_summary(
    report_day: date,
    start: datetime,
    end: datetime,
    risk_level: str,
    risk_reason: str,
    active_core: int,
    total_proxy_events: int,
    total_gateway_events: int,
    previous_proxy_events: int,
    previous_gateway_events: int,
    findings: List[Finding],
) -> List[str]:
    partial = end.date() == report_day and end.time() != time.max.replace(microsecond=0)
    compare_scope = "昨日同时间窗口" if partial else "昨日全天"
    finding_counts = summarize_findings(findings)
    mid_tag_counts = summarize_mid_tags(findings)
    summary = [
        f"- 结论：截至 `{end.isoformat().replace('+00:00', 'Z')}`，`gpu5/gpu7` 核心推理链在线 `#{active_core}/6`，整体风险等级为 `【{risk_level}】`。",
        f"- 风险说明：{risk_reason}",
        f"- 流量概况：内部 proxy 今日事件 `{total_proxy_events}` 条，较{compare_scope} `{fmt_delta(total_proxy_events, previous_proxy_events)}`；外部 gateway 今日事件 `{total_gateway_events}` 条，较{compare_scope} `{fmt_delta(total_gateway_events, previous_gateway_events)}`。",
    ]
    if findings:
        summary.append(
            f"- 管理关注点：当前自动发现 `{len(findings)}` 条事项，其中高风险 `{finding_counts['高']}`、中风险 `{finding_counts['中']}`、低风险 `{finding_counts['低']}`。"
        )
        if finding_counts["中"] > 0:
            summary.append(
                f"- 中风险结构：影响收入 `{mid_tag_counts['影响收入']}`、影响对外接入 `{mid_tag_counts['影响对外接入']}`、影响稳定性 `{mid_tag_counts['影响稳定性']}`。"
            )
    return summary


def build_findings(
    service_states: Dict[str, Dict[str, str]],
    service_health: Dict[str, Dict[str, Any]],
    proxy_envs: Dict[str, Dict[str, str]],
    gateway_audit: Dict[str, Any],
    proxy_audits: Optional[Dict[str, Dict[str, Any]]] = None,
) -> List[Finding]:
    findings: List[Finding] = []

    core_units = [
        "vllm-qwen35-gpu5-agent-upstream.service",
        "dpo-guardrails-proxy-gpu5.service",
        "tydata-rag-api.service",
        "vllm-qwen35-gpu7-baseline.service",
        "dpo-guardrails-proxy.service",
        "tydata-rag-api-gpu7.service",
    ]
    for unit in core_units:
        state = service_states.get(unit, {})
        if state.get("ActiveState") != "active":
            findings.append(Finding("高", f"核心服务 `{unit}` 当前不是 `active`，需要优先检查。", "影响稳定性"))

    for service in SERVICE_DEFS:
        health = service_health.get(service.key, {})
        if service.health_url and not health.get("_ok", False):
            level = "高" if service.key in {"gpu5_proxy", "gpu5_rag", "gpu7_proxy", "gpu7_rag", "mm_rag"} else "中"
            tag = "影响对外接入" if service.key == "gateway" else "影响稳定性"
            findings.append(Finding(level, f"`{service.label}` 健康检查失败：{health.get('_error', 'unknown error')}。", tag))
        elif service.health_url and health.get("ok") is False:
            level = "高" if service.key in {"gpu5_proxy", "gpu5_rag", "gpu7_proxy", "gpu7_rag", "mm_rag"} else "中"
            tag = "影响对外接入" if service.key == "gateway" else "影响稳定性"
            findings.append(Finding(level, f"`{service.label}` 健康接口可达，但返回 `ok=false`。", tag))

    for key, env in proxy_envs.items():
        if not proxy_auth_enabled(env):
            findings.append(Finding("中", f"`{key}` proxy 仍未启用调用方鉴权。", "影响对外接入"))
        if not proxy_rate_limit_enabled(env):
            findings.append(Finding("中", f"`{key}` proxy 仍未启用 RPM 限流。", "影响收入"))

    if not gpu7_dependency_configured():
        findings.append(Finding("中", "GPU7 proxy unit 尚未显式依赖 `vllm-qwen35-gpu7-baseline.service`，重启顺序仍有回退风险。", "影响稳定性"))

    if proxy_audits:
        for proxy in PROXY_DEFS:
            summary = proxy_audits.get(proxy.key, {})
            bad_upstream = {
                code: count
                for code, count in (summary.get("upstream_status_code_counts") or {}).items()
                if count > 0 and not str(code).startswith("2")
            }
            if bad_upstream:
                codes_text = ", ".join(f"{code}:{count}" for code, count in sorted(bad_upstream.items()))
                if latest_request_recovered(summary):
                    findings.append(
                        Finding(
                            "低",
                            f"`{proxy.label}` 今日曾出现非 2xx 上游状态码（{codes_text}），但最新真实请求已恢复 2xx，可继续观察。",
                            "影响稳定性",
                        )
                    )
                else:
                    findings.append(
                        Finding(
                            "中",
                            f"`{proxy.label}` 今日最新真实请求仍存在非 2xx 上游状态码：{codes_text}。",
                            "影响稳定性",
                        )
                    )

    gateway_state = service_states.get("external-api-gateway.service", {})
    if gateway_state.get("ActiveState") != "active":
        findings.append(Finding("中", "对外 `external-api-gateway.service` 当前不在线；如果准备开放外部流量，需要先恢复 8025 链路。", "影响对外接入"))
    elif gateway_audit.get("events", 0) == 0:
        findings.append(Finding("低", "对外 gateway 今日没有审计事件，需确认是否尚未接流量或审计未落盘。", "影响对外接入"))

    if not findings:
        findings.append(Finding("低", "当前未发现阻断级异常；建议继续把鉴权、限流和自动化观测补齐。"))
    return sorted(findings, key=lambda item: (finding_sort_key(item.level), item.text))


def build_next_steps(
    proxy_envs: Dict[str, Dict[str, str]],
    proxy_audits: Dict[str, Dict[str, Any]],
    gateway_audit: Dict[str, Any],
) -> List[str]:
    steps: List[str] = []

    if not all(proxy_auth_enabled(env) for env in proxy_envs.values()):
        steps.append("补齐 `gpu5/gpu7` proxy 的调用方鉴权，避免继续以匿名方式暴露内部推理链路。")
    if not all(proxy_rate_limit_enabled(env) for env in proxy_envs.values()):
        steps.append("补齐 `gpu5/gpu7` proxy 的 RPM 限流，先收口异常流量放大风险。")
    if not gpu7_dependency_configured():
        steps.append("给 GPU7 proxy unit 补回对 `vllm-qwen35-gpu7-baseline.service` 的显式依赖，降低重启顺序回退风险。")

    recovered_proxies = [
        proxy.label
        for proxy in PROXY_DEFS
        if proxy_audits.get(proxy.key, {}).get("last_bad_upstream_ts") and latest_request_recovered(proxy_audits.get(proxy.key, {}))
    ]
    if recovered_proxies:
        steps.append(
            f"继续观察 {', '.join(recovered_proxies)} 的上游状态码与时延，确认当天异常修复后不再复发。"
        )

    if gateway_audit.get("events", 0) > 0:
        steps.append("基于现有 gateway 审计日志，继续沉淀按模型/Key/状态码分层的对外接入看板，支持放量与风控决策。")
    else:
        steps.append("External API Gateway 已恢复在线，下一步补充真实外部流量验证并确认审计持续落盘。")

    if not steps:
        steps.append("维持当前配置，继续观察时延、失败码和 fallback 指标，必要时再推进容量优化。")

    return steps


def markdown_table(headers: List[str], rows: List[List[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        safe = [str(cell).replace("\n", "<br>") for cell in row]
        lines.append("| " + " | ".join(safe) + " |")
    return "\n".join(lines)


def generate_report(report_day: date, output_dir: Path) -> tuple[str, Path]:
    start, end = make_window(report_day)
    prev_start, prev_end = make_previous_compare_window(start, end)
    generated_at = utc_now()

    service_states: Dict[str, Dict[str, str]] = {}
    service_health: Dict[str, Dict[str, Any]] = {}
    for service in SERVICE_DEFS:
        service_states[service.unit] = systemctl_show(
            service.unit,
            [
                "ActiveState",
                "SubState",
                "UnitFileState",
                "MainPID",
                "NRestarts",
                "ActiveEnterTimestamp",
                "MemoryCurrent",
                "CPUUsageNSec",
                "FragmentPath",
                "Result",
            ],
        )
        service_health[service.key] = safe_fetch_json(service.health_url) if service.health_url else {"_ok": False, "_error": "n/a"}

    proxy_envs = {proxy.key: parse_env_file(proxy.env_path) for proxy in PROXY_DEFS}
    rag_envs = {rag.key: parse_env_file(rag.env_path) for rag in RAG_DEFS}

    proxy_rag_stats = {proxy.key: safe_fetch_json(proxy.rag_stats_url) for proxy in PROXY_DEFS}
    proxy_vl_stats = {proxy.key: safe_fetch_json(proxy.vl_rag_stats_url) for proxy in PROXY_DEFS}

    proxy_audits = {proxy.key: summarize_audit(proxy.audit_path, start, end) for proxy in PROXY_DEFS}
    proxy_audits_prev = {proxy.key: summarize_audit(proxy.audit_path, prev_start, prev_end) for proxy in PROXY_DEFS}
    gateway_audit = summarize_audit(EXTERNAL_GATEWAY_AUDIT, start, end)
    gateway_audit_prev = summarize_audit(EXTERNAL_GATEWAY_AUDIT, prev_start, prev_end)

    active_core = sum(
        1
        for unit in [
            "vllm-qwen35-gpu5-agent-upstream.service",
            "dpo-guardrails-proxy-gpu5.service",
            "tydata-rag-api.service",
            "vllm-qwen35-gpu7-baseline.service",
            "dpo-guardrails-proxy.service",
            "tydata-rag-api-gpu7.service",
        ]
        if service_states.get(unit, {}).get("ActiveState") == "active"
    )
    total_proxy_events = sum(item["events"] for item in proxy_audits.values())
    total_proxy_events_prev = sum(item["events"] for item in proxy_audits_prev.values())
    total_gateway_events = gateway_audit["events"]
    total_gateway_events_prev = gateway_audit_prev["events"]
    findings = build_findings(service_states, service_health, proxy_envs, gateway_audit, proxy_audits)
    finding_counts = summarize_findings(findings)
    mid_tag_counts = summarize_mid_tags(findings)
    risk_level, risk_reason = infer_risk_level(service_states, service_health, proxy_envs, proxy_audits)
    decision_line = choose_decision(findings)
    next_steps = build_next_steps(proxy_envs, proxy_audits, gateway_audit)
    exec_summary_lines = build_exec_summary(
        report_day=report_day,
        start=start,
        end=end,
        risk_level=risk_level,
        risk_reason=risk_reason,
        active_core=active_core,
        total_proxy_events=total_proxy_events,
        total_gateway_events=total_gateway_events,
        previous_proxy_events=total_proxy_events_prev,
        previous_gateway_events=total_gateway_events_prev,
        findings=findings,
    )

    summary_lines = [
        f"- 报告窗口：`{start.isoformat().replace('+00:00', 'Z')}` 到 `{end.isoformat().replace('+00:00', 'Z')}`。",
        f"- 对比窗口：`{prev_start.isoformat().replace('+00:00', 'Z')}` 到 `{prev_end.isoformat().replace('+00:00', 'Z')}`。",
        f"- 风险等级：`{risk_level}`。",
        f"- 风险分布：高风险 `{finding_counts['高']}` / 中风险 `{finding_counts['中']}` / 低风险 `{finding_counts['低']}`。",
        f"- 中风险标签：影响收入 `{mid_tag_counts['影响收入']}` / 影响对外接入 `{mid_tag_counts['影响对外接入']}` / 影响稳定性 `{mid_tag_counts['影响稳定性']}`。",
        f"- 核心链路在线数：`{active_core}/6`。",
        f"- 代理层今日审计事件：内部 `gpu5/gpu7` 合计 `{total_proxy_events}` 条；外部 gateway `{total_gateway_events}` 条。",
        f"- 自动发现事项：`{len(findings)}` 条。",
    ]

    service_rows: List[List[str]] = []
    for service in SERVICE_DEFS:
        state = service_states.get(service.unit, {})
        health = service_health.get(service.key, {})
        health_text = "-"
        if service.health_url:
            if not health.get("_ok", False):
                health_text = f"error: {health.get('_error', '-')}"
            else:
                ok = health.get("ok")
                if ok is None:
                    health_text = "reachable"
                else:
                    health_text = f"ok={ok}"
        service_rows.append(
            [
                service.label,
                service.role,
                state.get("ActiveState", "-") + "/" + state.get("SubState", "-"),
                health_text,
                state.get("NRestarts", "0") or "0",
                human_bytes(state.get("MemoryCurrent", "")),
                human_seconds_from_ns(state.get("CPUUsageNSec", "")),
                state.get("ActiveEnterTimestamp", "-") or "-",
            ]
        )

    proxy_config_rows: List[List[str]] = []
    for proxy in PROXY_DEFS:
        env = proxy_envs[proxy.key]
        health = service_health.get(f"{proxy.key}_proxy", {})
        rag_stats = proxy_rag_stats.get(proxy.key, {})
        vl_stats = proxy_vl_stats.get(proxy.key, {})
        proxy_config_rows.append(
            [
                proxy.label,
                health.get("default_model", env.get("DEFAULT_MODEL", "-")),
                env.get("UPSTREAM_BASE_URL", "-"),
                "on" if env.get("RAG_ENABLED", "0") not in {"0", "", "false", "False"} else "off",
                env.get("RAG_RATIO", "-"),
                "on" if env.get("MM_RAG_ENABLED", "0") not in {"0", "", "false", "False"} else "off",
                "required" if proxy_auth_enabled(env) else "off",
                env.get("PROXY_RATE_LIMIT_RPM", "0") or "0",
                str((rag_stats.get("stats") or {}).get("called", 0)) if rag_stats.get("_ok") else "-",
                str((vl_stats.get("stats") or {}).get("triggered", 0)) if vl_stats.get("_ok") else "-",
            ]
        )

    rag_rows: List[List[str]] = []
    for rag in RAG_DEFS:
        health = service_health.get(rag.key, {})
        env = rag_envs[rag.key]
        retrieve = health.get("retrieve") or {}
        rag_rows.append(
            [
                rag.label,
                health.get("index_dir", env.get("RAG_INDEX_DIR", "-")),
                str(health.get("num_chunks", "-")),
                str(health.get("dense_backend", "-")),
                str(health.get("default_top_k", "-")),
                f"{retrieve.get('cache_hit_rate', 0.0):.2%}" if retrieve else "-",
                f"{retrieve.get('latency_ms_p95', 0.0):.1f}" if retrieve else "-",
                health.get("llm_base", env.get("LLM_API_BASE", "-")),
            ]
        )

    audit_rows: List[List[str]] = []
    for proxy in PROXY_DEFS:
        summary = proxy_audits[proxy.key]
        audit_rows.append(
            [
                proxy.label,
                str(summary["events"]),
                str(summary["accepted"]),
                top_counter(summary["status_code_counts"]),
                top_counter(summary["upstream_status_code_counts"]),
                f"{percentile(summary['elapsed_ms_samples'], 95):.1f}/{mean(summary['elapsed_ms_samples']):.1f}" if summary["elapsed_ms_samples"] else "-",
                f"rag:{summary['rag_true']} vl:{summary['vl_fallback_true']} teacher:{summary['teacher_fallback_true']}",
                f"req>0:{summary['tool_trace_nonzero']} total:{summary['tool_trace_total']}",
                top_counter(summary["route_counts"]),
                top_counter(summary["model_counts"]),
                top_counter(summary["key_counts"]),
                f"{summary['stream_true']}/{summary['events']}" if summary["events"] else "0/0",
                f"{summary['image_true']}/{summary['events']}" if summary["events"] else "0/0",
                summary["latest_ts"] or "-",
            ]
        )
    audit_rows.append(
        [
            "External Gateway",
            str(gateway_audit["events"]),
            str(gateway_audit["accepted"]),
            top_counter(gateway_audit["status_code_counts"]),
            top_counter(gateway_audit["upstream_status_code_counts"]),
            f"{percentile(gateway_audit['elapsed_ms_samples'], 95):.1f}/{mean(gateway_audit['elapsed_ms_samples']):.1f}" if gateway_audit["elapsed_ms_samples"] else "-",
            f"rag:{gateway_audit['rag_true']} vl:{gateway_audit['vl_fallback_true']} teacher:{gateway_audit['teacher_fallback_true']}",
            f"req>0:{gateway_audit['tool_trace_nonzero']} total:{gateway_audit['tool_trace_total']}",
            top_counter(gateway_audit["route_counts"]),
            top_counter(gateway_audit["model_counts"]),
            top_counter(gateway_audit["key_counts"]),
            f"{gateway_audit['stream_true']}/{gateway_audit['events']}" if gateway_audit["events"] else "0/0",
            f"{gateway_audit['image_true']}/{gateway_audit['events']}" if gateway_audit["events"] else "0/0",
            gateway_audit["latest_ts"] or "-",
        ]
    )

    calc_bucket_rows: List[List[str]] = []
    for proxy in PROXY_DEFS:
        summary = proxy_audits[proxy.key]
        calc_bucket_rows.append(
            [
                proxy.label,
                top_counter(summary["calc_degrade_bucket_counts"], limit=10),
                top_counter(summary["calc_missing_field_counts"], limit=10),
            ]
        )
    merged_calc_bucket_counts = merge_counters(
        summary["calc_degrade_bucket_counts"] for summary in proxy_audits.values()
    )
    merged_calc_missing_field_counts = merge_counters(
        summary["calc_missing_field_counts"] for summary in proxy_audits.values()
    )
    calc_bucket_rows.append(
        [
            "GPU5+GPU7 合计",
            top_counter(merged_calc_bucket_counts, limit=10),
            top_counter(merged_calc_missing_field_counts, limit=10),
        ]
    )

    compare_rows: List[List[str]] = []
    for proxy in PROXY_DEFS:
        cur = proxy_audits[proxy.key]
        prev = proxy_audits_prev[proxy.key]
        compare_rows.append(
            [
                proxy.label,
                str(cur["events"]),
                str(prev["events"]),
                fmt_delta(cur["events"], prev["events"]),
                str(cur["accepted"]),
                str(prev["accepted"]),
                fmt_delta(cur["accepted"], prev["accepted"]),
                top_counter(cur["route_counts"]),
            ]
        )
    compare_rows.append(
        [
            "External Gateway",
            str(gateway_audit["events"]),
            str(gateway_audit_prev["events"]),
            fmt_delta(gateway_audit["events"], gateway_audit_prev["events"]),
            str(gateway_audit["accepted"]),
            str(gateway_audit_prev["accepted"]),
            fmt_delta(gateway_audit["accepted"], gateway_audit_prev["accepted"]),
            top_counter(gateway_audit["route_counts"]),
        ]
    )

    report_lines: List[str] = [
        f"# GPU5/GPU7 推理服务工作日报 {report_day.isoformat()}",
        "",
        f"- 生成时间：`{generated_at.isoformat().replace('+00:00', 'Z')}`",
        f"- 时区：`UTC`",
        "",
        "## 老板摘要",
        *exec_summary_lines,
        f"- 今日建议决策：{decision_line}",
        "",
        "## 今日概览",
        *summary_lines,
        "",
        "## 服务状态快照",
        markdown_table(
            ["服务", "角色", "systemd 状态", "健康检查", "重启计数", "内存", "CPU 累计", "最近激活时间"],
            service_rows,
        ),
        "",
        "## Proxy 配置与路由摘要",
        markdown_table(
            ["链路", "默认模型", "上游", "RAG", "RAG_RATIO", "MM_RAG", "鉴权", "RPM 限流", "RAG 调用累计", "VL/RAG 触发累计"],
            proxy_config_rows,
        ),
        "",
        "## RAG 状态快照",
        "> 说明：以下 `retrieve` 指标来自当前进程内累计，不是自然日独立计数。",
        "",
        markdown_table(
            ["链路", "索引目录", "chunks", "dense backend", "top_k", "cache hit rate", "retrieve p95(ms)", "上游 LLM"],
            rag_rows,
        ),
        "",
        "## 今日流量与审计摘要",
        markdown_table(
            ["入口", "事件数", "accepted/ok", "状态码", "上游码", "P95/均值ms", "Fallback", "Tool调用", "Top Routes", "Top Models", "Top Keys", "Stream", "Has Image", "最新事件"],
            audit_rows,
        ),
        "",
        "## Calculator 降级补参摘要",
        "> 说明：仅统计 `short_circuit=calculator_degrade` 相关审计事件，用于观察最常见的补参类型和缺失字段。",
        "",
        markdown_table(
            ["入口", "calc_degrade_bucket 次数", "calc_missing_fields Top N"],
            calc_bucket_rows,
        ),
        "",
        "## 昨日对比",
        "> 如果今天还没过完，默认比较“昨日同时间窗口”，避免把今日半天流量和昨日全天直接比较。",
        "",
        markdown_table(
            ["入口", "今日事件", "昨日事件", "事件增减", "今日 accepted/ok", "昨日 accepted/ok", "accepted 增减", "今日 Top Routes"],
            compare_rows,
        ),
        "",
        "## 今日建议决策",
        f"- {decision_line}",
        "",
        "## 自动发现事项",
    ]
    for level in ["高", "中", "低"]:
        report_lines.append(f"### {level}风险")
        level_items = [item for item in findings if item.level == level]
        if not level_items:
            report_lines.append("- 无")
        else:
            for item in level_items:
                if level == "中" and item.tag:
                    report_lines.append(f"- [{item.tag}] {item.text}")
                else:
                    report_lines.append(f"- {item.text}")
        report_lines.append("")

    report_lines.extend(
        [
            "## 建议下一步",
            *[f"- {step}" for step in next_steps],
            "",
            "## 数据来源",
            "- systemd: `systemctl --user show/status`",
            "- 健康接口: `8001/8010/18080/18081/18100/8025`",
            "- 审计日志: `gpu5_dpo_guardrails_proxy/audit.jsonl`, `gpu7_dpo_guardrails_proxy/audit.jsonl`, `external_api_gateway/audit.jsonl`",
        ]
    )

    report_text = "\n".join(report_lines).strip() + "\n"
    output_path = output_dir / f"inference_daily_report_{report_day.isoformat()}.md"
    return report_text, output_path


if __name__ == "__main__":
    args = parse_args()
    report_day = parse_report_date(args.date)
    report_text, output_path = generate_report(report_day, Path(args.output_dir))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report_text, encoding="utf-8")
    print(str(output_path))
    if args.stdout:
        print()
        print(report_text, end="")
