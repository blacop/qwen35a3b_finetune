from __future__ import annotations

from decimal import Decimal

from calculator.errors import InvalidInputError, MissingFieldError
from calculator.utils import as_decimal, dec_to_float, quantize_decimal


FORMULA_ID = "freeze_amount_by_ratio"


def metadata() -> dict:
    return {
        "formula_id": FORMULA_ID,
        "category": "ratio",
        "required_inputs": ["amount", "ratio"],
        "description": "Freeze amount by ratio",
    }


def calculate(inputs: dict, precision: int) -> dict:
    if "amount" not in inputs:
        raise MissingFieldError("amount is required")
    if "ratio" not in inputs:
        raise MissingFieldError("ratio is required")

    amount = as_decimal(inputs["amount"], "amount")
    ratio = as_decimal(inputs["ratio"], "ratio")
    if amount < 0:
        raise InvalidInputError("amount must be >= 0")
    if ratio < 0 or ratio > Decimal("1"):
        raise InvalidInputError("ratio must be between 0 and 1")

    freeze_amount = quantize_decimal(amount * ratio, precision)
    return {
        "result": {"freeze_amount": dec_to_float(freeze_amount)},
        "formula_text": "freeze_amount = amount * ratio",
        "steps": [f"freeze_amount = {amount} * {ratio}"],
        "warnings": [],
    }

