#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import requests


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def now_epoch() -> float:
    return time.time()


def percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    data = sorted(values)
    if len(data) == 1:
        return float(data[0])
    rank = max(0.0, min(100.0, p)) / 100.0 * (len(data) - 1)
    lo = int(math.floor(rank))
    hi = int(math.ceil(rank))
    if lo == hi:
        return float(data[lo])
    frac = rank - lo
    return float(data[lo] * (1.0 - frac) + data[hi] * frac)


def as_bool(raw: str, default: bool = False) -> bool:
    if raw is None:
        return default
    v = str(raw).strip().lower()
    if v in {"1", "true", "yes", "y", "on"}:
        return True
    if v in {"0", "false", "no", "n", "off"}:
        return False
    return default


def read_env_file(path: Path) -> Dict[str, str]:
    out: Dict[str, str] = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#") or "=" not in s:
            continue
        k, v = s.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def set_env_values(path: Path, updates: Dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        lines = path.read_text(encoding="utf-8").splitlines()
    else:
        lines = []

    out_lines: List[str] = []
    seen = set()
    for line in lines:
        s = line.strip()
        if not s or s.startswith("#") or "=" not in s:
            out_lines.append(line)
            continue
        k = s.split("=", 1)[0].strip()
        if k in updates:
            out_lines.append(f"{k}={updates[k]}")
            seen.add(k)
        else:
            out_lines.append(line)
    for k, v in updates.items():
        if k not in seen:
            out_lines.append(f"{k}={v}")
    path.write_text("\n".join(out_lines).rstrip() + "\n", encoding="utf-8")


def append_jsonl(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def load_queries(dataset: Path, query_field: str, limit: int) -> List[str]:
    out: List[str] = []
    with dataset.open("r", encoding="utf-8") as f:
        for line in f:
            if limit > 0 and len(out) >= limit:
                break
            s = line.strip()
            if not s:
                continue
            obj = json.loads(s)
            q = str(obj.get(query_field, "")).strip()
            if q:
                out.append(q)
    return out


def detect_route(api_base: str, check_route: str) -> str:
    route = (check_route or "auto").strip().lower()
    if route in {"stable", "canary"}:
        return route
    try:
        resp = requests.get(api_base.rstrip("/") + "/health", timeout=5.0)
        data = resp.json() if resp.status_code == 200 else {}
    except Exception:
        return "stable"
    if bool(data.get("gray_enabled")) and bool(data.get("gray_canary_loaded")) and float(data.get("gray_ratio", 0.0)) > 0.0:
        return "canary"
    return "stable"


def run_probe(
    api_base: str,
    api_key: str,
    queries: List[str],
    timeout_s: float,
    route: str,
    retrieval_payload: Dict[str, Any],
) -> Dict[str, Any]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    if route in {"stable", "canary"}:
        headers["X-Rag-Route"] = route

    url = api_base.rstrip("/") + "/retrieve"
    latencies: List[float] = []
    ok_count = 0
    non_empty = 0
    errors = 0
    for q in queries:
        payload = dict(retrieval_payload)
        payload["query"] = q
        started = time.perf_counter()
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=timeout_s)
            latency_ms = (time.perf_counter() - started) * 1000.0
            latencies.append(latency_ms)
            if r.status_code == 200:
                ok_count += 1
                data = r.json()
                if int(data.get("total_hits", 0)) > 0:
                    non_empty += 1
            else:
                errors += 1
        except Exception:
            latency_ms = (time.perf_counter() - started) * 1000.0
            latencies.append(latency_ms)
            errors += 1

    total = max(1, len(queries))
    return {
        "total": len(queries),
        "route": route,
        "request_ok_rate": ok_count / total,
        "non_empty_rate": non_empty / total,
        "latency_ms_p50": percentile(latencies, 50),
        "latency_ms_p95": percentile(latencies, 95),
        "latency_ms_mean": (sum(latencies) / len(latencies)) if latencies else 0.0,
        "errors": errors,
    }


def load_state(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {"breach_since": {}, "last_action_epoch": 0.0}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"breach_since": {}, "last_action_epoch": 0.0}


def save_state(path: Path, state: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    p = argparse.ArgumentParser(description="RAG guard with auto rollback")
    p.add_argument("--rag-env", default="/home/ubuntu/generate/tydata_rag/rag_api.env")
    p.add_argument("--state-dir", default="/home/ubuntu/generate/tydata_rag/runtime/rag_guard")
    p.add_argument("--dataset", default="/home/ubuntu/generate/qwen35a3b_finetune/datasets/eval_sports_customer_canary_120.jsonl")
    p.add_argument("--query-field", default="user_query")
    p.add_argument("--limit", type=int, default=120)
    p.add_argument("--api-base", default="http://127.0.0.1:18080")
    p.add_argument("--timeout-s", type=float, default=10.0)
    p.add_argument("--check-route", default="auto", help="auto|stable|canary")
    p.add_argument("--min-request-ok-rate", type=float, default=0.99)
    p.add_argument("--max-p95-ms", type=float, default=50.0)
    p.add_argument("--sustain-min", type=int, default=10)
    p.add_argument("--cooldown-min", type=int, default=30)
    p.add_argument("--rollback-index-dir", default="/home/ubuntu/generate/tydata_rag/index_svd_20260421T065404Z")
    p.add_argument("--restart-cmd", default="systemctl --user restart tydata-rag-api.service")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    state_dir = Path(args.state_dir)
    state_file = state_dir / "state.json"
    event_log = state_dir / "events.jsonl"
    state = load_state(state_file)

    rag_env_path = Path(args.rag_env)
    rag_env = read_env_file(rag_env_path)
    api_key = rag_env.get("RAG_API_KEY", "")

    queries = load_queries(Path(args.dataset), args.query_field, args.limit)
    if not queries:
        append_jsonl(
            event_log,
            {
                "ts": now_iso(),
                "action": "skip_no_queries",
                "dataset": args.dataset,
            },
        )
        return 1

    route = detect_route(args.api_base, args.check_route)
    retrieval_payload = {
        "top_k": 5,
        "min_score": 0.01,
        "min_score_vec": 0.0,
        "retrieval_mode": "hybrid",
        "fusion": "rrf",
        "candidate_k": 40,
        "rrf_k": 60,
        "weight_lex": 0.55,
        "weight_vec": 0.45,
        "enable_rerank": True,
    }
    metrics = run_probe(
        api_base=args.api_base,
        api_key=api_key,
        queries=queries,
        timeout_s=args.timeout_s,
        route=route,
        retrieval_payload=retrieval_payload,
    )

    breaches: Dict[str, float] = {}
    if metrics["request_ok_rate"] < args.min_request_ok_rate:
        breaches["request_ok_rate"] = metrics["request_ok_rate"]
    if metrics["latency_ms_p95"] > args.max_p95_ms:
        breaches["latency_ms_p95"] = metrics["latency_ms_p95"]

    current = now_epoch()
    breach_since: Dict[str, float] = dict(state.get("breach_since", {}))
    for key in list(breach_since.keys()):
        if key not in breaches:
            breach_since.pop(key, None)
    for key in breaches.keys():
        breach_since.setdefault(key, current)

    sustained_keys = [
        key
        for key, since_epoch in breach_since.items()
        if (current - float(since_epoch)) >= (args.sustain_min * 60)
    ]

    event: Dict[str, Any] = {
        "ts": now_iso(),
        "action": "probe",
        "route": route,
        "metrics": metrics,
        "breaches": breaches,
        "sustained_keys": sustained_keys,
        "thresholds": {
            "min_request_ok_rate": args.min_request_ok_rate,
            "max_p95_ms": args.max_p95_ms,
            "sustain_min": args.sustain_min,
            "cooldown_min": args.cooldown_min,
        },
    }

    if sustained_keys:
        last_action = float(state.get("last_action_epoch", 0.0) or 0.0)
        if (current - last_action) < (args.cooldown_min * 60):
            event["action"] = "skip_cooldown"
        else:
            updates: Dict[str, str] = {}
            gray_enabled = as_bool(rag_env.get("RAG_GRAY_ENABLED", "0"), False)
            gray_ratio = float(rag_env.get("RAG_GRAY_RATIO", "0.0") or 0.0)
            rollback_action = ""

            if gray_enabled and gray_ratio > 0.0:
                updates["RAG_GRAY_ENABLED"] = "0"
                updates["RAG_GRAY_RATIO"] = "0.0"
                rollback_action = "disable_gray"
            else:
                if args.rollback_index_dir:
                    updates["RAG_INDEX_DIR"] = args.rollback_index_dir
                    rollback_action = "switch_index"
                updates["RAG_GRAY_ENABLED"] = "0"
                updates["RAG_GRAY_RATIO"] = "0.0"

            event["action"] = "rollback"
            event["rollback_action"] = rollback_action
            event["env_updates"] = updates
            event["restart_cmd"] = args.restart_cmd

            if not args.dry_run and updates:
                set_env_values(rag_env_path, updates)
                proc = subprocess.run(args.restart_cmd, shell=True, capture_output=True, text=True)
                event["restart_rc"] = proc.returncode
                event["restart_stdout"] = proc.stdout[-4000:]
                event["restart_stderr"] = proc.stderr[-4000:]
                if proc.returncode == 0:
                    state["last_action_epoch"] = current
            elif args.dry_run:
                event["dry_run"] = True

    state["breach_since"] = breach_since
    save_state(state_file, state)
    append_jsonl(event_log, event)
    print(json.dumps(event, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
