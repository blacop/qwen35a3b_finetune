from pathlib import Path
import importlib.util
import json
import sqlite3
import sys


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "sync_open_webui_model_system_prompts.py"


def load_module():
    name = "sync_open_webui_model_system_prompts_test"
    sys.modules.pop(name, None)
    spec = importlib.util.spec_from_file_location(name, SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def create_model_db(path: Path) -> None:
    conn = sqlite3.connect(path)
    cur = conn.cursor()
    cur.execute(
        """
        create table model (
            id text not null,
            user_id text not null,
            base_model_id text,
            name text not null,
            meta text not null,
            params text not null,
            created_at integer not null,
            updated_at integer not null,
            is_active boolean not null default 1
        )
        """
    )
    cur.execute(
        "insert into model (id, user_id, base_model_id, name, meta, params, created_at, updated_at, is_active) values (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            "gpu5-v5.1f-merged",
            "admin",
            None,
            "GPU5",
            "{}",
            json.dumps({"temperature": 0.2}, ensure_ascii=False),
            1,
            1,
            1,
        ),
    )
    cur.execute(
        "insert into model (id, user_id, base_model_id, name, meta, params, created_at, updated_at, is_active) values (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            "gpu7-v5-combined",
            "admin",
            None,
            "GPU7",
            "{}",
            json.dumps({"system": "旧提示词", "top_p": 0.9}, ensure_ascii=False),
            1,
            1,
            1,
        ),
    )
    conn.commit()
    conn.close()


def test_sync_updates_system_and_preserves_other_params(tmp_path):
    module = load_module()
    db_path = tmp_path / "webui.db"
    create_model_db(db_path)

    stats = module.sync_open_webui_model_system_prompts(
        str(db_path),
        {
            "gpu5-v5.1f-merged": "新客服提示词 A",
            "gpu7-v5-combined": "新客服提示词 B",
        },
    )

    assert stats["updated"] == 2
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    gpu5 = json.loads(cur.execute("select params from model where id = ?", ("gpu5-v5.1f-merged",)).fetchone()[0])
    gpu7 = json.loads(cur.execute("select params from model where id = ?", ("gpu7-v5-combined",)).fetchone()[0])
    conn.close()

    assert gpu5["system"] == "新客服提示词 A"
    assert gpu5["temperature"] == 0.2
    assert gpu7["system"] == "新客服提示词 B"
    assert gpu7["top_p"] == 0.9


def test_sync_reports_missing_and_unchanged_models(tmp_path):
    module = load_module()
    db_path = tmp_path / "webui.db"
    create_model_db(db_path)

    stats = module.sync_open_webui_model_system_prompts(
        str(db_path),
        {
            "gpu5-v5.1f-merged": "同一个提示词",
            "missing-model": "不会写入",
        },
        ["gpu5-v5.1f-merged", "missing-model"],
        dry_run=True,
    )

    assert stats["updated"] == 1
    assert stats["missing"] == 1
    assert stats["targeted"] == 2
