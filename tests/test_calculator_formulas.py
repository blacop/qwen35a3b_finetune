from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from calculator.engine import execute_formula


def test_european_odds_formula():
    out = execute_formula(
        formula_id="european_odds_net_profit",
        inputs={"stake": 1000, "odds": 2.15},
        precision=2,
    )
    assert out["result"]["gross_payout"] == 2150.0
    assert out["result"]["net_profit"] == 1150.0


def test_parlay_formula():
    out = execute_formula(
        formula_id="parlay_all_win_payout",
        inputs={"stake": 50, "odds_list": [1.62, 1.52, 1.76]},
        precision=3,
    )
    assert out["result"]["gross_payout"] > 0
