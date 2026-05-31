#!/usr/bin/env python3
from __future__ import annotations

import os
from typing import Any, Dict

import requests


CALCULATOR_BASE_URL = os.getenv("CALCULATOR_BASE_URL", "http://127.0.0.1:18200").rstrip("/")
CALCULATOR_API_KEY = os.getenv("CALCULATOR_API_KEY", "").strip()
CALCULATOR_TIMEOUT_S = float(os.getenv("CALCULATOR_TIMEOUT_S", "8"))


def call_calculator(
    request_id: str,
    formula_id: str,
    inputs: Dict[str, Any],
    context: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    headers = {"Content-Type": "application/json"}
    if CALCULATOR_API_KEY:
        headers["Authorization"] = f"Bearer {CALCULATOR_API_KEY}"
    payload = {
        "request_id": request_id,
        "formula_id": formula_id,
        "inputs": inputs,
        "context": context or {},
    }
    resp = requests.post(
        f"{CALCULATOR_BASE_URL}/v1/calculate",
        headers=headers,
        json=payload,
        timeout=CALCULATOR_TIMEOUT_S,
    )
    resp.raise_for_status()
    return resp.json()

