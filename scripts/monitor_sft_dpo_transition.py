#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple


def now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def latest_step(logging_jsonl: Path) -> Tuple[Optional[int], Optional[int], Optional[dict]]:
    if not logging_jsonl.exists():
        return None, None, None
    last = None
    try:
        with logging_jsonl.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    last = line
    except Exception:
        return None, None, None
    if not last:
        return None, None, None
    try:
        row = json.loads(last)
    except Exception:
        return None, None, None
    gsm = row.get("global_step/max_steps")
    if not isinstance(gsm, str):
        return None, None, row
    m = re.match(r"^\s*(\d+)\s*/\s*(\d+)\s*$", gsm)
    if not m:
        return None, None, row
    return int(m.group(1)), int(m.group(2)), row


def dpo_started(dpo_dir: Path) -> bool:
    if not dpo_dir.exists():
        return False
    if (dpo_dir / "logging.jsonl").exists():
        return True
    if any(dpo_dir.glob("checkpoint-*")):
        return True
    return False


def dpo_done(dpo_dir: Path) -> bool:
    if not dpo_dir.exists():
        return False
    if any(dpo_dir.glob("checkpoint-*")):
        return True
    return False


def write_status(path: Path, status: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def main() -> None:
    p = argparse.ArgumentParser(description="Monitor SFT->DPO transition by filesystem signals.")
    p.add_argument("--sft-dir", required=True)
    p.add_argument("--dpo-dir", required=True)
    p.add_argument("--status-json", required=True)
    p.add_argument("--poll-seconds", type=int, default=30)
    p.add_argument("--max-hours", type=int, default=48)
    args = p.parse_args()

    sft_dir = Path(args.sft_dir)
    dpo_dir = Path(args.dpo_dir)
    status_json = Path(args.status_json)
    sft_log = sft_dir / "logging.jsonl"
    deadline = time.time() + args.max_hours * 3600

    while True:
        step, max_steps, raw = latest_step(sft_log)
        sft_finished = bool(step is not None and max_steps is not None and step >= max_steps)
        has_dpo_started = dpo_started(dpo_dir)
        has_dpo_done = dpo_done(dpo_dir) and has_dpo_started

        status = {
            "ts_utc": now_utc(),
            "sft_dir": str(sft_dir),
            "dpo_dir": str(dpo_dir),
            "sft_step": step,
            "sft_max_steps": max_steps,
            "sft_finished": sft_finished,
            "dpo_started": has_dpo_started,
            "dpo_done_signal": has_dpo_done,
            "latest_train_log": raw,
        }
        if sft_finished and has_dpo_started:
            status["terminal_state"] = "sft_done_dpo_started"
            write_status(status_json, status)
            return
        if time.time() > deadline:
            status["terminal_state"] = "timeout"
            write_status(status_json, status)
            return

        write_status(status_json, status)
        time.sleep(max(args.poll_seconds, 5))


if __name__ == "__main__":
    main()
