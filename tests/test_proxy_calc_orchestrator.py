from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.proxy_calc_orchestrator import extract_inputs, explain_calc_failure, infer_formula_id


def test_infer_formula_id_from_query_and_rag_text():
    rag_hits = [{"source_file": "rule.txt", "text": "欧洲盘：赢：本金*(赔率-1)"}]
    assert infer_formula_id("赔率怎么算", rag_hits) == "european_odds_net_profit"


def test_extract_inputs_with_labeled_fields():
    out = extract_inputs("欧洲盘 本金1000 赔率2.15 净赢多少", "european_odds_net_profit")
    assert out == {"stake": 1000.0, "odds": 2.15}


def test_extract_inputs_with_weak_freeze_hint():
    out = extract_inputs("金额500 冻结20% 需要冻结多少", "freeze_amount_by_ratio")
    assert out == {"amount": 500.0, "ratio": 0.2}


def test_explain_calc_failure_for_odds():
    out = explain_calc_failure("赔率怎么算", [], "european_odds_net_profit", "INPUTS_NOT_EXTRACTED")
    assert out["bucket"] == "odds"
    assert out["missing_fields"] == ["plate_type", "stake", "odds"]
    assert "请补充" in out["suggestion"]
    assert "赔率" in out["suggestion"]
    assert "这个我可以帮你算" in out["reply"]
    assert "盘型 + 本金 + 赔率" in out["reply"]


def test_explain_calc_failure_for_formula_not_inferred_is_conversational():
    out = explain_calc_failure("赔率怎么算", [], "", "FORMULA_NOT_INFERRED")
    assert out["bucket"] == "odds"
    assert out["missing_fields"] == ["plate_type", "stake", "odds"]
    assert "盘型" in out["suggestion"]
    assert "欧洲盘还是香港盘" in out["reply"]


def test_explain_calc_failure_for_formula_not_inferred_parlay_defaults_to_parlay_prompt():
    out = explain_calc_failure("串关怎么算", [], "", "FORMULA_NOT_INFERRED")
    assert out["bucket"] == "parlay"
    assert out["missing_fields"] == ["stake", "odds_list"]
    assert "本金" in out["suggestion"]
    assert "每一关的赔率列表" in out["reply"]


def test_explain_calc_failure_for_formula_not_inferred_freeze_defaults_to_freeze_prompt():
    out = explain_calc_failure("冻结比例怎么算", [], "", "FORMULA_NOT_INFERRED")
    assert out["bucket"] == "freeze"
    assert out["missing_fields"] == ["amount", "ratio"]
    assert "金额和冻结比例" in out["suggestion"]
    assert "冻结金额" in out["reply"]


def test_explain_calc_failure_for_freeze_is_conversational():
    out = explain_calc_failure("冻结怎么算", [], "freeze_amount_by_ratio", "INPUTS_NOT_EXTRACTED")
    assert out["bucket"] == "freeze"
    assert out["missing_fields"] == ["amount", "ratio"]
    assert "金额和冻结比例" in out["suggestion"]
    assert "冻结金额" in out["reply"]


def test_explain_calc_failure_for_parlay_is_conversational():
    out = explain_calc_failure("串关怎么算", [], "parlay_all_win_payout", "INPUTS_NOT_EXTRACTED")
    assert out["bucket"] == "parlay"
    assert out["missing_fields"] == ["stake", "odds_list"]
    assert "本金" in out["suggestion"]
    assert "每一关的赔率列表" in out["reply"]
