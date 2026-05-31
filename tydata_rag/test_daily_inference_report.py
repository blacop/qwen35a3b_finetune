from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from daily_inference_report import summarize_audit


UTC = timezone.utc


def test_summarize_audit_collects_calc_degrade_stats(tmp_path: Path) -> None:
    audit_path = tmp_path / "audit.jsonl"
    rows = [
        {
            "ts": "2026-05-30T09:04:22Z",
            "status": "ok",
            "route": "/v1/chat/completions",
            "model": "gpu5-model",
            "key_id": "gateway",
            "status_code": 200,
            "short_circuit": "calculator_degrade",
            "calc_degrade_bucket": "odds",
            "calc_missing_fields": ["plate_type", "stake", "odds"],
        },
        {
            "ts": "2026-05-30T09:04:41Z",
            "status": "ok",
            "route": "/v1/chat/completions",
            "model": "gpu5-model",
            "key_id": "gateway",
            "status_code": 200,
            "short_circuit": "calculator_degrade",
            "calc_degrade_bucket": "odds",
            "calc_missing_fields": ["stake", "odds"],
        },
        {
            "ts": "2026-05-30T09:05:00Z",
            "status": "ok",
            "route": "/v1/chat/completions",
            "model": "gpu5-model",
            "key_id": "gateway",
            "status_code": 200,
            "short_circuit": "calculator_degrade",
            "calc_degrade_bucket": "freeze",
            "calc_missing_fields": ["amount", "ratio"],
        },
    ]
    audit_path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")

    summary = summarize_audit(
        audit_path,
        datetime(2026, 5, 30, 0, 0, 0, tzinfo=UTC),
        datetime(2026, 5, 31, 0, 0, 0, tzinfo=UTC),
    )

    assert summary["calc_degrade_bucket_counts"]["odds"] == 2
    assert summary["calc_degrade_bucket_counts"]["freeze"] == 1
    assert summary["calc_missing_field_counts"]["stake"] == 2
    assert summary["calc_missing_field_counts"]["odds"] == 2
    assert summary["calc_missing_field_counts"]["plate_type"] == 1
    assert summary["calc_missing_field_counts"]["amount"] == 1
    assert summary["calc_missing_field_counts"]["ratio"] == 1
