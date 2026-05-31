from pathlib import Path
import importlib.util
import sys


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "build_sports_baowang_knowledge_regression.py"
spec = importlib.util.spec_from_file_location("build_sports_baowang_knowledge_regression_test", SCRIPT_PATH)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def test_regression_set_covers_required_focus_areas():
    rows = module.build_rows()
    joined_tags = {tag for row in rows for tag in row["tags"]}
    scenarios = {row["scenario"] for row in rows}

    assert len(rows) >= 30
    assert {"odds", "handicap", "parlay", "settlement", "void", "backend"}.issubset(joined_tags)
    assert {"作废规则", "后台知识", "串关规则", "赛果结算"}.issubset(scenarios)


def test_regression_rows_use_eval_contract_fields():
    rows = module.build_rows()
    ids = [row["id"] for row in rows]

    assert len(ids) == len(set(ids))
    for row in rows:
        assert row["user_query"]
        assert row["gold_intent"]
        assert isinstance(row["gold_need_escalation"], bool)
        assert row["must_include"]
        assert "洗钱" in row["must_not_include"]
        assert "sports_baowang_regression_20260506" in row["tags"]
