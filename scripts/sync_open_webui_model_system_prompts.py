#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


DEFAULT_DB_PATH = "/home/ubuntu/open_webui/data/webui.db"
DEFAULT_PROMPTS_FILE = "/home/ubuntu/qwen35a3b_finetune/services/model-system-prompts.json"


def load_prompt_map(path: str) -> Dict[str, str]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        raise ValueError("prompt file must be a JSON object")
    prompts: Dict[str, str] = {}
    for model_id, prompt in raw.items():
        key = str(model_id or "").strip()
        value = str(prompt or "").strip()
        if key and value:
            prompts[key] = value
    return prompts


def _load_params(params_text: str) -> Dict[str, Any]:
    if not params_text:
        return {}
    try:
        data = json.loads(params_text)
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def sync_open_webui_model_system_prompts(
    db_path: str,
    prompts: Dict[str, str],
    model_ids: Optional[Iterable[str]] = None,
    *,
    dry_run: bool = False,
) -> Dict[str, int]:
    selected_ids = [str(x).strip() for x in (model_ids or []) if str(x).strip()]
    target_ids = selected_ids or sorted(prompts.keys())
    db_file = Path(db_path)
    if not db_file.exists():
        raise FileNotFoundError(f"Open WebUI DB not found: {db_path}")

    conn = sqlite3.connect(str(db_file))
    try:
        cur = conn.cursor()
        updated = 0
        unchanged = 0
        missing = 0
        for model_id in target_ids:
            prompt = prompts.get(model_id)
            if not prompt:
                missing += 1
                continue
            row = cur.execute(
                "select params from model where id = ?",
                (model_id,),
            ).fetchone()
            if row is None:
                missing += 1
                continue
            params = _load_params(row[0] or "")
            if params.get("system") == prompt:
                unchanged += 1
                continue
            params["system"] = prompt
            if not dry_run:
                cur.execute(
                    "update model set params = ?, updated_at = ? where id = ?",
                    (
                        json.dumps(params, ensure_ascii=False),
                        int(time.time()),
                        model_id,
                    ),
                )
            updated += 1
        if not dry_run:
            conn.commit()
    finally:
        conn.close()

    return {
        "updated": updated,
        "unchanged": unchanged,
        "missing": missing,
        "targeted": len(target_ids),
        "dry_run": int(dry_run),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Sync shared model system prompts into Open WebUI model params.",
    )
    parser.add_argument("--db", default=DEFAULT_DB_PATH, help="Path to Open WebUI sqlite DB")
    parser.add_argument(
        "--prompts-file",
        default=DEFAULT_PROMPTS_FILE,
        help="Path to shared alias prompt JSON file",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Optional explicit model IDs to sync; defaults to all prompt keys",
    )
    parser.add_argument("--dry-run", action="store_true", help="Preview only, do not write DB")
    args = parser.parse_args()

    prompts = load_prompt_map(args.prompts_file)
    stats = sync_open_webui_model_system_prompts(
        args.db,
        prompts,
        args.models,
        dry_run=args.dry_run,
    )
    print(json.dumps(stats, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
