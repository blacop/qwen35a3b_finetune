from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from calculator.api_models import CalculateRequest
from scripts.calculator_service import calculate, health


def test_calculator_service_health():
    body = health()
    assert body["ok"] is True
    assert body["service"] == "calculator-service"


def test_calculator_service_calculate():
    resp = calculate(
        CalculateRequest(
            request_id="req-test",
            formula_id="freeze_amount_by_ratio",
            inputs={"amount": 500, "ratio": 0.2},
        ),
        authorization=None,
    )
    assert resp.ok is True
    assert resp.result["freeze_amount"] == 100.0
