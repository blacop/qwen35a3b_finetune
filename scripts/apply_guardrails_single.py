#!/usr/bin/env python3
"""
Apply intent/escalation guardrails for a single request payload.

Example:
  python3 scripts/apply_guardrails_single.py \
    --user-query "我的注单被取消了" \
    --scenario "注单取消作废" \
    --pred-intent "充值" \
    --pred-need-escalation false \
    --answer "我帮您看下" \
    --next-action "" \
    --must-include "后台查询,联系运营,注单号"
"""

from __future__ import annotations

import argparse
import json
from typing import Any, List, Optional

from intent_escalation_guardrails import apply_guardrails, parse_bool


def to_list(v: str) -> List[str]:
    if not v:
        return []
    return [x.strip() for x in v.replace("；", ",").replace("，", ",").split(",") if x.strip()]


def main() -> None:
    p = argparse.ArgumentParser(description="Apply guardrails for one sample.")
    p.add_argument("--user-query", required=True)
    p.add_argument("--scenario", default="")
    p.add_argument("--pred-intent", default="其他")
    p.add_argument("--pred-need-escalation", default="")
    p.add_argument("--answer", default="")
    p.add_argument("--next-action", default="")
    p.add_argument("--must-include", default="")
    args = p.parse_args()

    pred_need_escalation: Optional[bool] = parse_bool(args.pred_need_escalation)
    result = apply_guardrails(
        user_query=args.user_query,
        scenario=args.scenario,
        pred_intent=args.pred_intent,
        pred_need_escalation=pred_need_escalation,
        answer=args.answer,
        next_action=args.next_action,
        must_include=to_list(args.must_include),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
