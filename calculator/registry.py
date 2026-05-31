from __future__ import annotations

from calculator.formulas import european_odds, hk_odds, parlay, ratio


FORMULA_REGISTRY = {
    european_odds.FORMULA_ID: european_odds,
    hk_odds.FORMULA_ID: hk_odds,
    parlay.FORMULA_ID: parlay,
    ratio.FORMULA_ID: ratio,
}


def list_formulas() -> list[dict]:
    return [module.metadata() for module in FORMULA_REGISTRY.values()]

