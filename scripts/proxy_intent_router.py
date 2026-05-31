#!/usr/bin/env python3
from __future__ import annotations

import os
import re
from typing import Dict


CALC_KEYWORDS = [x.strip() for x in os.getenv("CALC_ROUTE_KEYWORDS", "").split(",") if x.strip()]
REALTIME_KEYWORDS = [x.strip() for x in os.getenv("REALTIME_ROUTE_KEYWORDS", "").split(",") if x.strip()]
NUMBER_RE = re.compile(r"\d+(?:\.\d+)?%?")


def classify_query(query: str) -> Dict[str, str]:
    q = (query or "").strip()
    if not q:
        return {"route_type": "knowledge_only", "route_reason": "empty_query_default"}
    if any(kw in q for kw in REALTIME_KEYWORDS):
        return {"route_type": "realtime_query_required", "route_reason": "matched_realtime_keyword"}

    calc_hit = any(kw in q for kw in CALC_KEYWORDS)
    has_number = bool(NUMBER_RE.search(q))
    if calc_hit and has_number:
        return {"route_type": "knowledge_plus_calc", "route_reason": "matched_calc_keyword_and_number"}
    if calc_hit:
        return {"route_type": "knowledge_plus_calc", "route_reason": "matched_calc_keyword"}
    return {"route_type": "knowledge_only", "route_reason": "default_knowledge_only"}

