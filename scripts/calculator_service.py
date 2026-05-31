#!/usr/bin/env python3
from __future__ import annotations

import os
from typing import Any, Dict

from fastapi import FastAPI, Header

from calculator.api_models import CalculateRequest, CalculateResponse, ErrorPayload
from calculator.engine import execute_formula
from calculator.errors import CalculatorError
from calculator.registry import list_formulas


APP_VERSION = "1.0.0"
CALCULATOR_API_KEY = os.getenv("CALCULATOR_API_KEY", "").strip()

app = FastAPI(title="calculator-service", version=APP_VERSION)


def check_api_key(authorization: str | None) -> None:
    if not CALCULATOR_API_KEY:
        return
    if not authorization:
        raise PermissionError("Missing Authorization header")
    prefix = "Bearer "
    if not authorization.startswith(prefix):
        raise PermissionError("Invalid Authorization header")
    token = authorization[len(prefix):].strip()
    if token != CALCULATOR_API_KEY:
        raise PermissionError("Unauthorized")


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "ok": True,
        "service": "calculator-service",
        "version": APP_VERSION,
        "formulas": len(list_formulas()),
    }


@app.get("/v1/formulas")
def formulas(authorization: str | None = Header(default=None)) -> Dict[str, Any]:
    check_api_key(authorization)
    return {"object": "list", "data": list_formulas()}


@app.post("/v1/calculate", response_model=CalculateResponse)
def calculate(req: CalculateRequest, authorization: str | None = Header(default=None)) -> CalculateResponse:
    try:
        check_api_key(authorization)
    except PermissionError as exc:
        return CalculateResponse(
            ok=False,
            formula_id=req.formula_id,
            inputs=req.inputs,
            error=ErrorPayload(code="UNAUTHORIZED", message=str(exc)),
            trace={"version": APP_VERSION, "engine": "deterministic_rules_v1"},
        )
    try:
        payload = execute_formula(
            formula_id=req.formula_id,
            inputs=req.inputs,
            precision=req.options.precision,
        )
        return CalculateResponse(
            ok=True,
            formula_id=req.formula_id,
            inputs=req.inputs,
            result=payload.get("result", {}),
            formula_text=payload.get("formula_text", ""),
            steps=payload.get("steps", []),
            warnings=payload.get("warnings", []),
            trace={"version": APP_VERSION, "engine": "deterministic_rules_v1"},
        )
    except CalculatorError as exc:
        return CalculateResponse(
            ok=False,
            formula_id=req.formula_id,
            inputs=req.inputs,
            error=ErrorPayload(code=exc.code, message=exc.message),
            trace={"version": APP_VERSION, "engine": "deterministic_rules_v1"},
        )
    except Exception as exc:
        return CalculateResponse(
            ok=False,
            formula_id=req.formula_id,
            inputs=req.inputs,
            error=ErrorPayload(code="INTERNAL_ERROR", message=str(exc)),
            trace={"version": APP_VERSION, "engine": "deterministic_rules_v1"},
        )

