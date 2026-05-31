#!/usr/bin/env python3
"""
Continuous canary monitor with automatic rollback trigger.

Workflow:
  1) Periodically run eval_sports_customer_service.py on a canary eval set
  2) Read summary.json and evaluate SLO/SLA thresholds
  3) If breach sustained for a configured window (default 10 min), trigger rollback command
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple


def now_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_intent_thresholds(raw: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for item in (raw or "").split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            continue
        k, v = item.split("=", 1)
        k = k.strip()
        try:
            out[k] = float(v.strip())
        except Exception:
            continue
    return out


def run_eval(args: argparse.Namespace, run_dir: Path) -> Tuple[int, Dict[str, Any], str]:
    run_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        args.python_bin,
        str(Path(args.eval_script).resolve()),
        "--input-jsonl",
        str(Path(args.input_jsonl).resolve()),
        "--output-dir",
        str(run_dir),
        "--base-url",
        args.base_url,
        "--model",
        args.model,
        "--workers",
        str(args.workers),
        "--timeout-sec",
        str(args.timeout_sec),
        "--max-retries",
        str(args.max_retries),
        "--max-tokens",
        str(args.max_tokens),
        "--http-backend",
        args.http_backend,
    ]
    if args.api_key:
        cmd.extend(["--api-key", args.api_key])
    if args.limit > 0:
        cmd.extend(["--limit", str(args.limit)])

    proc = subprocess.run(cmd, capture_output=True, text=True)
    summary_path = run_dir / "summary.json"
    summary: Dict[str, Any] = {}
    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            summary = {}
    stderr = (proc.stderr or "").strip()
    if not stderr and proc.returncode != 0:
        stderr = (proc.stdout or "").strip()[-800:]
    return proc.returncode, summary, stderr


def eval_breaches(summary: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    breaches: List[str] = []
    detail: Dict[str, Any] = {}

    req_ok = float(summary.get("request_ok_rate", 0.0) or 0.0)
    p95 = float(summary.get("latency_ms_p95", 0.0) or 0.0)
    detail["request_ok_rate"] = req_ok
    detail["latency_ms_p95"] = p95

    if req_ok < args.request_ok_min:
        breaches.append("request_ok_rate")
    if p95 > args.p95_max_ms:
        breaches.append("latency_ms_p95")

    by_intent = summary.get("by_intent") or {}
    intent_breaches: Dict[str, float] = {}
    for intent, th in args.intent_thresholds.items():
        item = by_intent.get(intent) or {}
        val = item.get("intent_acc")
        try:
            valf = float(val)
        except Exception:
            valf = -1.0
        detail[f"intent_acc:{intent}"] = valf
        if valf < th:
            intent_breaches[intent] = valf
            breaches.append(f"intent_acc:{intent}")

    return {
        "breaches": breaches,
        "detail": detail,
        "intent_breaches": intent_breaches,
    }


def load_history(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def sustained_breach(
    history: List[Dict[str, Any]],
    now_epoch: float,
    window_sec: int,
    min_points: int,
    breach_key: str,
) -> bool:
    recent = [
        x
        for x in history
        if isinstance(x.get("ts_epoch"), (int, float))
        and (now_epoch - float(x["ts_epoch"]) <= window_sec)
        and x.get("eval_ok") is True
    ]
    if len(recent) < min_points:
        return False
    for row in recent:
        b = row.get("breaches") or []
        if breach_key not in b:
            return False
    return True


def load_state(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_state(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def maybe_trigger_rollback(
    sustained_keys: List[str],
    args: argparse.Namespace,
    state_path: Path,
    log_path: Path,
) -> None:
    if not sustained_keys:
        return
    state = load_state(state_path)
    now_epoch = time.time()
    last_ts = float(state.get("last_rollback_epoch", 0.0) or 0.0)
    if (now_epoch - last_ts) < (args.rollback_cooldown_min * 60):
        append_jsonl(
            log_path,
            {
                "ts_utc": now_iso(),
                "action": "skip_rollback_cooldown",
                "sustained_breaches": sustained_keys,
                "cooldown_min": args.rollback_cooldown_min,
            },
        )
        return
    if not args.rollback_command.strip():
        append_jsonl(
            log_path,
            {
                "ts_utc": now_iso(),
                "action": "rollback_command_missing",
                "sustained_breaches": sustained_keys,
            },
        )
        return

    proc = subprocess.run(args.rollback_command, shell=True, capture_output=True, text=True)
    event = {
        "ts_utc": now_iso(),
        "action": "rollback_triggered",
        "sustained_breaches": sustained_keys,
        "command": args.rollback_command,
        "returncode": proc.returncode,
        "stdout_tail": (proc.stdout or "")[-800:],
        "stderr_tail": (proc.stderr or "")[-800:],
    }
    append_jsonl(log_path, event)
    if proc.returncode == 0:
        state["last_rollback_epoch"] = now_epoch
        state["last_rollback_utc"] = now_iso()
        state["last_rollback_breaches"] = sustained_keys
        save_state(state_path, state)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Canary monitor + auto rollback.")
    p.add_argument("--python-bin", default=sys.executable)
    p.add_argument("--eval-script", default="/home/ubuntu/qwen35a3b_finetune/scripts/eval_sports_customer_service.py")
    p.add_argument("--input-jsonl", default="/home/ubuntu/qwen35a3b_finetune/datasets/eval_sports_customer_canary_120.jsonl")
    p.add_argument("--output-root", default="/home/ubuntu/qwen35a3b_finetune/eval_outputs/canary_guard")
    p.add_argument("--state-dir", default="/home/ubuntu/qwen35a3b_finetune/runtime/canary_guard")

    p.add_argument("--base-url", default="http://10.128.203.5")
    p.add_argument("--model", default="qwen35a3b-dpo-latest")
    p.add_argument("--api-key", default="")
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--timeout-sec", type=int, default=60)
    p.add_argument("--max-retries", type=int, default=1)
    p.add_argument("--max-tokens", type=int, default=192)
    p.add_argument("--http-backend", default="curl", choices=["auto", "urllib", "curl"])
    p.add_argument("--limit", type=int, default=0)

    p.add_argument("--request-ok-min", type=float, default=0.95)
    p.add_argument("--p95-max-ms", type=float, default=15000.0)
    p.add_argument("--intent-thresholds", default="串关规则=0.5,滚球延迟=0.5,赔率异常=0.5,赛事变更=0.5")

    p.add_argument("--window-min", type=int, default=10)
    p.add_argument("--min-points", type=int, default=3)
    p.add_argument("--interval-sec", type=int, default=120)
    p.add_argument("--rollback-command", default="")
    p.add_argument("--rollback-cooldown-min", type=int, default=30)
    p.add_argument("--once", action="store_true")
    return p


def main() -> None:
    args = build_parser().parse_args()
    args.intent_thresholds = parse_intent_thresholds(args.intent_thresholds)

    output_root = Path(args.output_root).resolve()
    state_dir = Path(args.state_dir).resolve()
    history_path = state_dir / "history.jsonl"
    actions_path = state_dir / "actions.jsonl"
    state_path = state_dir / "state.json"
    state_dir.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)

    print(
        json.dumps(
            {
                "ts_utc": now_iso(),
                "monitor": "start",
                "input_jsonl": args.input_jsonl,
                "base_url": args.base_url,
                "model": args.model,
                "window_min": args.window_min,
                "min_points": args.min_points,
                "request_ok_min": args.request_ok_min,
                "p95_max_ms": args.p95_max_ms,
                "intent_thresholds": args.intent_thresholds,
                "once": bool(args.once),
            },
            ensure_ascii=False,
        )
    )

    while True:
        ts = now_ts()
        run_dir = output_root / f"canary_{ts}"
        t0 = time.time()
        returncode, summary, err = run_eval(args, run_dir)
        eval_ok = bool(returncode == 0 and isinstance(summary, dict) and summary)

        breach_eval = eval_breaches(summary if eval_ok else {}, args)
        breaches = breach_eval["breaches"] if eval_ok else ["eval_failed"]
        elapsed_ms = int((time.time() - t0) * 1000)

        record = {
            "ts_utc": now_iso(),
            "ts_epoch": time.time(),
            "run_dir": str(run_dir),
            "eval_ok": eval_ok,
            "returncode": returncode,
            "elapsed_ms": elapsed_ms,
            "breaches": breaches,
            "metrics": breach_eval["detail"] if eval_ok else {},
            "error": err,
            "summary_path": str(run_dir / "summary.json"),
        }
        append_jsonl(history_path, record)

        history = load_history(history_path)
        window_sec = args.window_min * 60
        sustained: List[str] = []
        for key in ["latency_ms_p95", "request_ok_rate"]:
            if sustained_breach(history, time.time(), window_sec, args.min_points, key):
                sustained.append(key)
        for intent in args.intent_thresholds:
            k = f"intent_acc:{intent}"
            if sustained_breach(history, time.time(), window_sec, args.min_points, k):
                sustained.append(k)

        status = {
            "ts_utc": now_iso(),
            "run_dir": str(run_dir),
            "eval_ok": eval_ok,
            "breaches": breaches,
            "sustained_breaches": sustained,
            "metrics": record["metrics"],
        }
        print(json.dumps(status, ensure_ascii=False))

        if sustained:
            maybe_trigger_rollback(
                sustained_keys=sustained,
                args=args,
                state_path=state_path,
                log_path=actions_path,
            )

        if args.once:
            break
        time.sleep(max(1, int(args.interval_sec)))


if __name__ == "__main__":
    main()
