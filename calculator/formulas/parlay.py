from __future__ import annotations

from decimal import Decimal

from calculator.errors import InvalidInputError, MissingFieldError
from calculator.utils import as_decimal, dec_to_float, quantize_decimal


FORMULA_ID = "parlay_all_win_payout"


def metadata() -> dict:
    return {
        "formula_id": FORMULA_ID,
        "category": "parlay",
        "required_inputs": ["stake", "odds_list"],
        "description": "All-win parlay payout",
    }


def calculate(inputs: dict, precision: int) -> dict:
    if "stake" not in inputs:
        raise MissingFieldError("stake is required")
    if "odds_list" not in inputs:
        raise MissingFieldError("odds_list is required")

    stake = as_decimal(inputs["stake"], "stake")
    odds_list_raw = inputs["odds_list"]
    if stake <= 0:
        raise InvalidInputError("stake must be greater than 0")
    if not isinstance(odds_list_raw, list) or not odds_list_raw:
        raise InvalidInputError("odds_list must be a non-empty list")

    combined_odds = Decimal("1")
    parsed_odds = []
    for i, raw in enumerate(odds_list_raw):
        odds = as_decimal(raw, f"odds_list[{i}]")
        if odds <= Decimal("1"):
            raise InvalidInputError("all parlay odds must be greater than 1")
        combined_odds *= odds
        parsed_odds.append(str(odds))

    combined_odds = quantize_decimal(combined_odds, precision)
    gross_payout = quantize_decimal(stake * combined_odds, precision)
    net_profit = quantize_decimal(gross_payout - stake, precision)

    return {
        "result": {
            "combined_odds": dec_to_float(combined_odds),
            "gross_payout": dec_to_float(gross_payout),
            "net_profit": dec_to_float(net_profit),
        },
        "formula_text": "gross_payout = stake * product(odds_list); net_profit = gross_payout - stake",
        "steps": [
            f"combined_odds = {' * '.join(parsed_odds)}",
            f"gross_payout = {stake} * {combined_odds}",
        ],
        "warnings": [],
    }

