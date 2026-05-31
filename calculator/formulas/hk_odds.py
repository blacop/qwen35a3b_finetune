from __future__ import annotations

from decimal import Decimal

from calculator.errors import InvalidInputError, MissingFieldError
from calculator.utils import as_decimal, dec_to_float, quantize_decimal


FORMULA_ID = "hk_odds_profit"


def metadata() -> dict:
    return {
        "formula_id": FORMULA_ID,
        "category": "odds",
        "required_inputs": ["stake", "odds"],
        "description": "Hong Kong odds profit",
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
    if odds < Decimal("0"):
        raise InvalidInputError("hk odds must be >= 0")

    profit = quantize_decimal(stake * odds, precision)
    loss_if_lose = quantize_decimal(stake, precision)

    return {
        "result": {
            "profit_if_win": dec_to_float(profit),
            "loss_if_lose": dec_to_float(loss_if_lose),
        },
        "formula_text": "profit_if_win = stake * odds; loss_if_lose = stake",
        "steps": [f"profit_if_win = {stake} * {odds}"],
        "warnings": [],
    }

