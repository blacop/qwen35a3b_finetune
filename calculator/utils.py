from __future__ import annotations

from decimal import Decimal, ROUND_HALF_UP
from typing import Any

from calculator.errors import InvalidInputError


def as_decimal(value: Any, field_name: str) -> Decimal:
    try:
        return Decimal(str(value))
    except Exception as exc:
        raise InvalidInputError(f"{field_name} is not a valid number") from exc


def quantize_decimal(value: Decimal, precision: int) -> Decimal:
    scale = Decimal("1").scaleb(-precision)
    return value.quantize(scale, rounding=ROUND_HALF_UP)


def dec_to_float(value: Decimal) -> float:
    return float(value)

