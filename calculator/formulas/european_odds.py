from __future__ import annotations

from decimal import Decimal

from calculator.errors import InvalidInputError, MissingFieldError
from calculator.utils import as_decimal, dec_to_float, quantize_decimal


FORMULA_ID = "european_odds_net_profit"


def metadata() -> dict:
    return {
        "formula_id": FORMULA_ID,
        "category": "odds",
        "required_inputs": ["stake", "odds"],
        "description": "European odds payout and net profit",
    }


def calculate(inputs: dict, precision: int) -> dict:
    if "stake" not in inputs:
        raise MissingFieldError("stake is required")
    if "odds" not in inputs:
        raise MissingFieldError("odds is required")

    stake = as_decimal(inputs["stake"], "stake")
    odds = as_decimal(inputs["odds"], "odds")

    if stake <= 0:
        raise InvalidInputError("stake must be greater than 0")
    if odds <= Decimal("1"):
        raise InvalidInputError("odds must be greater than 1")

    gross_payout = quantize_decimal(stake * odds, precision)
    net_profit = quantize_decimal(stake * (odds - Decimal("1")), precision)
    loss_if_lose = quantize_decimal(stake, precision)

    return {
        "result": {
            "gross_payout": dec_to_float(gross_payout),
            "net_profit": dec_to_float(net_profit),
            "loss_if_lose": dec_to_float(loss_if_lose),
        },
        "formula_text": "gross_payout = stake * odds; net_profit = stake * (odds - 1)",
        "steps": [
            f"gross_payout = {stake} * {odds}",
            f"net_profit = {stake} * ({odds} - 1)",
        ],
        "warnings": [],
    }

