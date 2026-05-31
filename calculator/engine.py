from __future__ import annotations

from calculator.errors import UnsupportedFormulaError
from calculator.registry import FORMULA_REGISTRY


def execute_formula(formula_id: str, inputs: dict, precision: int) -> dict:
    module = FORMULA_REGISTRY.get(formula_id)
    if module is None:
        raise UnsupportedFormulaError(f"unsupported formula_id: {formula_id}")
    return module.calculate(inputs=inputs, precision=precision)

