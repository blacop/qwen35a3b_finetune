from pathlib import Path
import json
import subprocess
import sys


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_sports_knowledge_gate.sh"
CUSTOMER_SERVICE_GATE_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_customer_service_eval_gate.sh"
SHOW_RELEASE_LOGS_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "show_latest_release_logs.py"


def test_sports_knowledge_gate_script_exists():
    assert SCRIPT_PATH.exists()


def test_sports_knowledge_gate_script_mentions_both_eval_sets():
    text = SCRIPT_PATH.read_text(encoding="utf-8")
    assert "eval_sports_baowang_knowledge_40_20260428.jsonl" in text
    assert "eval_sports_baowang_knowledge_regression_20260506.jsonl" in text
    assert "eval_gpu7_ops_feedback_regression_8_20260509.jsonl" in text
    assert "MIN_MUST_INCLUDE_AVG" in text
    assert "MIN_REQUEST_OK_RATE" in text
    assert "MIN_INTENT_ACC_GPU7_OPS_20260509" in text
    assert "MIN_ESCALATION_ACC_GPU7_OPS_20260509" in text


def test_candidate_gate_invokes_sports_knowledge_gate():
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_gpu5_candidate_gate.sh"
    text = path.read_text(encoding="utf-8")
    assert "run_sports_knowledge_gate.sh" in text
    assert "KNOWLEDGE_GATE_OUT_DIR" in text
    assert "run_customer_service_eval_gate.sh" in text
    assert "SERVING_SANITY_OUT_DIR" in text
    assert "OPS_GATE_OUT_DIR" in text
    assert "eval_ops_high_freq_20.jsonl" in text
    assert "eval_serving_sanity_gpu5_20260508.jsonl" in text


def test_customer_service_gate_script_mentions_eval_and_thresholds():
    text = CUSTOMER_SERVICE_GATE_SCRIPT.read_text(encoding="utf-8")
    assert "eval_sports_customer_service.py" in text
    assert "INPUT_FILE" in text
    assert "MIN_REQUEST_OK_RATE" in text
    assert "MIN_INTENT_ACC" in text
    assert "MIN_OVERALL_AVG" in text


def test_ops_high_freq_dataset_is_labeled_for_release_gate():
    path = Path(__file__).resolve().parents[1] / "datasets" / "eval_ops_high_freq_20.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows) == 20
    assert all(row.get("gold_intent") for row in rows)
    assert all("gold_need_escalation" in row for row in rows)
    assert all("must_not_include" in row and "作为AI" in row["must_not_include"] for row in rows)


def test_gpu7_ops_feedback_regression_dataset_contract():
    path = Path(__file__).resolve().parents[1] / "datasets" / "eval_gpu7_ops_feedback_regression_8_20260509.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows) == 8
    assert all(row.get("gold_intent") for row in rows)
    assert all(row.get("gold_need_escalation") is True for row in rows)
    assert all("must_include" in row and row["must_include"] for row in rows)
    assert all("must_not_include" in row and "作为AI" in row["must_not_include"] for row in rows)


def test_serving_sanity_dataset_covers_three_core_cases():
    path = Path(__file__).resolve().parents[1] / "datasets" / "eval_serving_sanity_gpu5_20260508.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert [row["scenario"] for row in rows] == ["登录异常", "充值不到账", "提款不到账"]
    assert all(row.get("gold_need_escalation") is True for row in rows)


def test_post_deploy_scripts_invoke_optional_sports_knowledge_gate():
    root = Path(__file__).resolve().parents[1] / "scripts"
    for name in ["post_train_merge_deploy_gpu5.sh", "post_train_merge_deploy_text_vl_unified_gpu5.sh"]:
        text = (root / name).read_text(encoding="utf-8")
        assert "RUN_POST_DEPLOY_KNOWLEDGE_GATE" in text
        assert "run_sports_knowledge_gate.sh" in text


def test_orchestrator_records_knowledge_gate_output_paths():
    path = Path(__file__).resolve().parents[1] / "scripts" / "post_train_merge_deploy_and_ab_text_vl_unified_gpu5.sh"
    text = path.read_text(encoding="utf-8")
    assert "KNOWLEDGE_GATE_OUT_DIR" in text
    assert "release_gate_summary.json" in text
    assert "release_log_index.jsonl" in text
    assert "release_notifications.log" in text
    assert "gpu5_20260506_summary" in text
    assert "gpu7_20260506_summary" in text
    assert "kb40_20260428_summary" in text
    assert "gpu7_ops_feedback_8_20260509" in text


def test_all_release_orchestrators_use_shared_release_log_index_helper():
    root = Path(__file__).resolve().parents[1] / "scripts"
    for path in sorted(root.glob("post_train_merge_deploy*.sh")):
        text = path.read_text(encoding="utf-8")
        assert "append_release_log_index.py" in text, path.name
        assert "RELEASE_LOG_INDEX_FILE" in text, path.name


def test_append_release_log_index_helper_appends_jsonl(tmp_path):
    helper = Path(__file__).resolve().parents[1] / "scripts" / "append_release_log_index.py"
    out = tmp_path / "release_log_index.jsonl"
    subprocess.run(
        [
            sys.executable,
            str(helper),
            "--index-file",
            str(out),
            "--ts",
            "20260507T080000Z",
            "--pipeline",
            "unit_test_pipeline",
            "--model-dir",
            "/tmp/model",
            "--merged-model-dir",
            "/tmp/model",
            "--served-model-name",
            "demo-model",
            "--extra-json",
            '{"foo":"bar"}',
        ],
        check=True,
    )

    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows) == 1
    assert rows[0]["pipeline"] == "unit_test_pipeline"
    assert rows[0]["model_dir"] == "/tmp/model"
    assert rows[0]["served_model_name"] == "demo-model"
    assert rows[0]["foo"] == "bar"


def test_show_latest_release_logs_handles_missing_index(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            str(SHOW_RELEASE_LOGS_SCRIPT),
            "--index-file",
            str(tmp_path / "missing.jsonl"),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "no release records found" in result.stdout


def test_show_latest_release_logs_filters_and_limits(tmp_path):
    index = tmp_path / "release_log_index.jsonl"
    rows = [
        {
            "ts": "20260508T130000Z",
            "pipeline": "pipe_a",
            "release_gate_summary_path": "/tmp/a3.json",
        },
        {
            "ts": "20260508T120000Z",
            "pipeline": "pipe_b",
            "release_gate_summary_path": "/tmp/b2.json",
        },
        {
            "ts": "20260508T110000Z",
            "pipeline": "pipe_a",
            "release_gate_summary_path": "/tmp/a1.json",
        },
    ]
    index.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(SHOW_RELEASE_LOGS_SCRIPT),
            "--index-file",
            str(index),
            "--pipeline",
            "pipe_a",
            "--limit",
            "1",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "matched: 1" in result.stdout
    assert "pipeline: pipe_a" in result.stdout
    assert "/tmp/a3.json" in result.stdout
    assert "/tmp/a1.json" not in result.stdout
    assert "pipe_b" not in result.stdout


def test_show_latest_release_logs_filters_by_time(tmp_path):
    index = tmp_path / "release_log_index.jsonl"
    rows = [
        {"ts": "20260508T130000Z", "pipeline": "pipe_a", "release_gate_summary_path": "/tmp/a3.json"},
        {"ts": "20260508T120000Z", "pipeline": "pipe_a", "release_gate_summary_path": "/tmp/a2.json"},
        {"ts": "20260508T110000Z", "pipeline": "pipe_a", "release_gate_summary_path": "/tmp/a1.json"},
    ]
    index.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(SHOW_RELEASE_LOGS_SCRIPT),
            "--index-file",
            str(index),
            "--since",
            "20260508T115959Z",
            "--until",
            "20260508T125959Z",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "matched: 1" in result.stdout
    assert "/tmp/a2.json" in result.stdout
    assert "/tmp/a1.json" not in result.stdout
    assert "/tmp/a3.json" not in result.stdout


def test_show_latest_release_logs_json_output(tmp_path):
    index = tmp_path / "release_log_index.jsonl"
    index.write_text(
        json.dumps(
            {
                "ts": "20260508T130000Z",
                "pipeline": "pipe_a",
                "release_gate_summary_path": "/tmp/a3.json",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            str(SHOW_RELEASE_LOGS_SCRIPT),
            "--index-file",
            str(index),
            "--json",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(result.stdout)
    assert payload == [
        {
            "ts": "20260508T130000Z",
            "pipeline": "pipe_a",
            "release_gate_summary_path": "/tmp/a3.json",
        }
    ]
