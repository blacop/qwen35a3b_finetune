from pathlib import Path


def test_gpu7_v61_release_pipeline_exists_and_has_required_gates():
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_gpu7_v6_1_release_pipeline.sh"
    text = path.read_text(encoding="utf-8")
    assert "check_agent_trace_fields.py" in text
    assert "eval_tool_dialogue_gate.py" in text
    assert "eval_ops_feedback_16_20260513.jsonl" in text
    assert "eval_sports_baowang_knowledge_40_20260428.jsonl" in text
    assert "release_gate_summary_strict.json" in text
    assert "release_gate_report_strict.md" in text
    assert "strict release gate failed" in text


def test_gpu7_v61_workflow_exists():
    wf = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "gpu7-v6_1-release-gate.yml"
    text = wf.read_text(encoding="utf-8")
    assert "workflow_dispatch" in text
    assert "run_gpu7_v6_1_release_pipeline.sh" in text
    assert "actions/upload-artifact@v4" in text
    assert "run_merge" in text


def test_check_agent_trace_fields_script_contract():
    path = Path(__file__).resolve().parents[1] / "scripts" / "check_agent_trace_fields.py"
    text = path.read_text(encoding="utf-8")
    assert "trace_result_preview" in text
    assert "selected_group" in text
    assert "--min-pass-rate" in text
