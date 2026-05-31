from pathlib import Path


def test_proxy_release_gate_script_exists_and_has_required_checks():
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_proxy_release_gate_ci.sh"
    text = path.read_text(encoding="utf-8")

    assert "check_agent_trace_fields.py" in text
    assert "eval_tool_dialogue_gate.py" in text
    assert "eval_sports_customer_service.py" in text
    assert "REPEAT_RUNS" in text
    assert "MAX_KB40_LATENCY_P95_MS" in text
    assert "MAX_OPS16_LATENCY_P95_MS" in text
    assert "release_gate_summary.json" in text
    assert "release_gate_report.md" in text


def test_proxy_release_gate_workflow_exists_and_calls_script():
    wf = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "proxy-release-gate.yml"
    text = wf.read_text(encoding="utf-8")

    assert "workflow_dispatch" in text
    assert "repeat_runs" in text
    assert "run_proxy_release_gate_ci.sh" in text
    assert "actions/upload-artifact@v4" in text


def test_check_agent_trace_fields_supports_trace_args_switches():
    path = Path(__file__).resolve().parents[1] / "scripts" / "check_agent_trace_fields.py"
    text = path.read_text(encoding="utf-8")

    assert "require_trace_args" in text
    assert "require_trace_args_non_empty" in text
