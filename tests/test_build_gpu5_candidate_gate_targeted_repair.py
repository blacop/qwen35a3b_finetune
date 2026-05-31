from pathlib import Path
import json
import subprocess
import sys


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "build_gpu5_candidate_gate_targeted_repair.py"
KNOWLEDGE_RESULTS = Path(__file__).resolve().parents[1] / "eval_outputs" / "gpu5_candidate_gate_20260508_full_rerun" / "knowledge_results.json"


def test_build_gpu5_candidate_gate_targeted_repair_script_exists():
    assert SCRIPT_PATH.exists()


def test_build_gpu5_candidate_gate_targeted_repair_outputs_expected_rows(tmp_path):
    out_jsonl = tmp_path / "repair.jsonl"
    out_summary = tmp_path / "summary.json"
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--knowledge-results",
            str(KNOWLEDGE_RESULTS),
            "--output-jsonl",
            str(out_jsonl),
            "--summary-json",
            str(out_summary),
        ],
        check=True,
    )

    rows = [json.loads(line) for line in out_jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]
    summary = json.loads(out_summary.read_text(encoding="utf-8"))

    assert len(rows) == 27
    assert summary["total"] == 27
    assert "后台仪表盘" in summary["clusters"]
    assert "导入商户" in summary["clusters"]
    assert any(row["meta"]["sample_id"] == "kb_20260428_021" for row in rows)
    assert all(row["messages"][0]["role"] == "system" for row in rows)
    assert summary["variants_per_sample"]["kb_20260428_010"] == 3
    assert summary["clusters"]["后台仪表盘"] == 3
