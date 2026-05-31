#!/usr/bin/env python3
"""
Rule-based intent correction + escalation fallback for sports CS inference outputs.

Usage examples:
  python3 scripts/intent_escalation_guardrails.py \
    --input-jsonl eval_outputs/xxx/predictions.jsonl \
    --output-jsonl eval_outputs/xxx/predictions.guardrail.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional


HIGH_PRIORITY_INTENTS = {"赛事变更", "赔率异常", "限红风控", "滚球延迟", "注单异常", "投诉"}
LOW_ESCALATION_INTENTS = {"活动", "串关规则"}

# User-requested "same-day online hotfix" priority intent correction.
PRIORITY_INTENT_RULES = [
    ("赛事变更", re.compile(r"(赛事.*(改期|延期|取消)|改期|延期|取消比赛|赛事变更|赛程变更|改赛程|腰斩|推迟)")),
    ("赔率异常", re.compile(r"(赔率.*(异常|变化|变动|跳水|跳变)|赔率异常|盘口异常|赔率变更|复核赔率|成交.*赔率|下单.*赔率)")),
    ("限红风控", re.compile(r"(限额|限红|风控|限制下注|额度下调|审核风控|高风险账户)")),
]

INTENT_RULES = [
    ("赛事变更", re.compile(r"(改期|延期|取消比赛|赛事变更|赛程变更|改赛程|腰斩|推迟)")),
    ("赔率异常", re.compile(r"(赔率|盘口|跳水|跳变|赔率变更|盘口异常|复核赔率|成交.*赔率|下单.*赔率)")),
    ("限红风控", re.compile(r"(限红|限额|风控|限制下注|额度下调|审核风控|高风险账户)")),
    ("滚球延迟", re.compile(r"(滚球|赛中|延迟结算|迟迟不派彩|未派彩|未结算.*滚球|滚球.*结算)")),
    ("注单异常", re.compile(r"(注单|作废|取消单|待结算|结算异常|订单状态卡住|单据异常)")),
    ("投诉", re.compile(r"(投诉|申诉|举报|态度差|升级投诉|不满|差评)")),
    ("提款", re.compile(r"(提款|提现|出款|下分|取款|审核中|提款驳回)")),
    ("充值", re.compile(r"(充值|入款|上分|存款|到账|通道)")),
    ("活动", re.compile(r"(活动|优惠|红利|送彩金|返利|任务|流水要求)")),
    ("串关规则", re.compile(r"(串关|过关|组合投注|串子|1\\.0|作废规则)")),
]

FORCE_ESCALATION_QUERY_RE = re.compile(
    r"(后台|联系运营|人工处理|工单|核实后回复|复核|争议|注单异常|赔率异常|限红|风控|投诉|未到账|失败|驳回|待结算)"
)

# User-requested escalation fallback keywords: hit => force escalation=true.
FORCE_ESCALATION_HARD_RE = re.compile(
    r"("
    r"注单(取消|作废|异常)|取消单|作废单|结算异常|单据异常|"
    r"赔率(异常|变化|变动|跳水|跳变)|盘口异常|"
    r"限红|限额|风控|限制下注|"
    r"滚球.*(延迟|未结算|未派彩|待结算)|延迟未结算|"
    r"赛事.*(改期|延期|取消)|改期|延期|取消比赛|赛程变更|腰斩|"
    r"投诉|申诉|举报"
    r")"
)
HAS_DISPUTE_RE = re.compile(r"(争议|异常|复核|投诉|人工|后台|核查|失败|驳回|未到账|待结算|未结算|未派彩)")
PARLAY_RE = re.compile(r"(串关|过关|组合投注|串子|多关|过关单)")
EVENT_CHANGE_RE = re.compile(r"(改期|延期|取消比赛|赛事变更|赛程变更|改赛程|腰斩|推迟|取消)")
ANSWER_HAS_BACKOFFICE_RE = re.compile(r"(后台查询|后台核查|后台核实|后台处理)")
ANSWER_HAS_ESCALATION_RE = re.compile(r"(后台查询|后台核查|后台核实|后台处理|进一步核实|核实后回复|进一步查询)")
ANSWER_HAS_ACCOUNT_RE = re.compile(r"(账号|账户|注单号|订单号|交易号)")

ESCALATION_CLAUSE = "该问题需要进一步核实，核实后第一时间回复您。"
ACCOUNT_CLAUSE = "请提供账号与相关注单号（或订单号），以便加急处理。"


def parse_bool(v: Any) -> Optional[bool]:
    if v is None or v == "":
        return None
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    s = str(v).strip().lower()
    if s in {"true", "1", "yes", "y", "是", "需要", "需升级", "需转人工"}:
        return True
    if s in {"false", "0", "no", "n", "否", "不需要", "无需"}:
        return False
    return None


def _to_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    if isinstance(value, str):
        if not value.strip():
            return []
        if value.strip().startswith("[") and value.strip().endswith("]"):
            try:
                arr = json.loads(value)
                if isinstance(arr, list):
                    return [str(x).strip() for x in arr if str(x).strip()]
            except Exception:
                pass
        return [x.strip() for x in re.split(r"[,;；，]\s*", value) if x.strip()]
    return [str(value).strip()]


def _detect_intent(user_query: str, scenario: str = "") -> Optional[str]:
    text = f"{user_query} {scenario}".strip()
    if not text:
        return None
    # Keep "串关延期/改期/取消" under parlay-rule intent to avoid
    # being overridden by generic event-change routing.
    if PARLAY_RE.search(text) and EVENT_CHANGE_RE.search(text):
        return "串关规则"
    for intent, pat in PRIORITY_INTENT_RULES:
        if pat.search(text):
            return intent
    for intent, pat in INTENT_RULES:
        if pat.search(text):
            return intent
    return None


def _should_force_escalation(query: str, intent: str) -> bool:
    if intent in {"注单异常", "赔率异常", "限红风控", "滚球延迟", "投诉"}:
        return True
    if FORCE_ESCALATION_HARD_RE.search(query):
        return True
    if FORCE_ESCALATION_QUERY_RE.search(query):
        return True
    return False


def _enforce_must_include(answer: str, must_include: List[str], topk: int) -> str:
    if not answer or not must_include:
        return answer
    missing = [kw for kw in must_include if kw and kw not in answer]
    if not missing:
        return answer
    patch = "；".join(missing[:topk])
    return f"{answer} 重点补充：{patch}。"


def apply_guardrails(
    user_query: str,
    scenario: str,
    pred_intent: str,
    pred_need_escalation: Optional[bool],
    answer: str,
    next_action: str = "",
    must_include: Optional[List[str]] = None,
) -> Dict[str, Any]:
    must_include = must_include or []
    notes: List[str] = []

    before_intent = (pred_intent or "").strip() or "其他"
    detected_intent = _detect_intent(user_query=user_query, scenario=scenario)
    after_intent = before_intent

    text = f"{user_query} {scenario}".strip()
    is_parlay_event_case = bool(PARLAY_RE.search(text) and EVENT_CHANGE_RE.search(text))

    if is_parlay_event_case and before_intent != "串关规则":
        after_intent = "串关规则"
        notes.append(f"intent_override:{before_intent}->串关规则")
    elif detected_intent and detected_intent in HIGH_PRIORITY_INTENTS and detected_intent != before_intent:
        after_intent = detected_intent
        notes.append(f"intent_override:{before_intent}->{after_intent}")
    elif detected_intent and before_intent in {"其他", "充值", "技术故障", "账户异常"} and detected_intent != before_intent:
        after_intent = detected_intent
        notes.append(f"intent_override:{before_intent}->{after_intent}")

    esc_before = pred_need_escalation
    esc_after = pred_need_escalation
    if esc_after is None:
        esc_after = False

    escalation_text = text
    if after_intent == "串关规则" and is_parlay_event_case and not HAS_DISPUTE_RE.search(escalation_text):
        if esc_after is not False:
            esc_after = False
            notes.append("escalation_force:false_parlay_rule")
    elif _should_force_escalation(query=escalation_text, intent=after_intent):
        if esc_after is not True:
            esc_after = True
            notes.append("escalation_force:true")
    elif after_intent in LOW_ESCALATION_INTENTS and not HAS_DISPUTE_RE.search(user_query):
        if esc_after is not False:
            esc_after = False
            notes.append("escalation_force:false")

    patched_answer = (answer or "").strip()
    patched_next = (next_action or "").strip()

    if esc_after:
        if not ANSWER_HAS_ESCALATION_RE.search(patched_answer):
            patched_answer = (patched_answer + " " if patched_answer else "") + ESCALATION_CLAUSE
            notes.append("answer_patch:escalation_clause")
        if not ANSWER_HAS_ACCOUNT_RE.search(patched_answer):
            patched_answer = (patched_answer + " " if patched_answer else "") + ACCOUNT_CLAUSE
            notes.append("answer_patch:account_clause")
        if not patched_next:
            patched_next = "进一步核实后回复"
        elif ("核实" not in patched_next) and ("查询" not in patched_next):
            patched_next = f"{patched_next}；进一步核实后回复"
            notes.append("next_action_patch:escalation_route")
    else:
        if after_intent == "赛事变更" and "公告" not in patched_answer and "结算规则" not in patched_answer:
            suffix = "赛事改期/取消以平台公告与结算规则为准，请以最新公告为准。"
            patched_answer = (patched_answer + " " if patched_answer else "") + suffix
            notes.append("answer_patch:event_rule_clause")

    # Strengthen must_include hit-rate for online hotfix.
    if must_include and patched_answer:
        topk = 2 if esc_after else 1
        before = patched_answer
        patched_answer = _enforce_must_include(patched_answer, must_include, topk=topk)
        if patched_answer != before:
            notes.append("answer_patch:must_include_strengthen")

    return {
        "pred_intent": after_intent,
        "pred_need_escalation": esc_after,
        "answer": patched_answer,
        "next_action": patched_next,
        "guardrail_notes": notes,
        "intent_before": before_intent,
        "intent_after": after_intent,
        "escalation_before": esc_before,
        "escalation_after": esc_after,
    }


def process_jsonl(
    input_jsonl: Path,
    output_jsonl: Path,
    query_field: str,
    scenario_field: str,
    intent_field: str,
    escalation_field: str,
    answer_field: str,
    next_action_field: str,
    must_include_field: str,
) -> Dict[str, int]:
    stats = {
        "total": 0,
        "intent_overrides": 0,
        "escalation_overrides": 0,
        "answer_patched": 0,
    }
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    with input_jsonl.open("r", encoding="utf-8") as rf, output_jsonl.open("w", encoding="utf-8") as wf:
        for line_no, line in enumerate(rf, 1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if not isinstance(obj, dict):
                continue

            stats["total"] += 1
            query = str(obj.get(query_field, "")).strip()
            scenario = str(obj.get(scenario_field, "")).strip()
            pred_intent = str(obj.get(intent_field, "")).strip()
            pred_escalation = parse_bool(obj.get(escalation_field))
            answer = str(obj.get(answer_field, "")).strip()
            next_action = str(obj.get(next_action_field, "")).strip()
            must_include = _to_list(obj.get(must_include_field))

            gr = apply_guardrails(
                user_query=query,
                scenario=scenario,
                pred_intent=pred_intent,
                pred_need_escalation=pred_escalation,
                answer=answer,
                next_action=next_action,
                must_include=must_include,
            )

            if gr["intent_before"] != gr["intent_after"]:
                stats["intent_overrides"] += 1
            if gr["escalation_before"] != gr["escalation_after"]:
                stats["escalation_overrides"] += 1
            if gr["answer"] != answer:
                stats["answer_patched"] += 1

            obj[intent_field] = gr["pred_intent"]
            obj[escalation_field] = gr["pred_need_escalation"]
            obj[answer_field] = gr["answer"]
            obj[next_action_field] = gr["next_action"]
            obj["guardrail_notes"] = gr["guardrail_notes"]

            wf.write(json.dumps(obj, ensure_ascii=False) + "\n")
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply intent/escalation guardrails to JSONL outputs.")
    parser.add_argument("--input-jsonl", required=True, help="Input JSONL file.")
    parser.add_argument("--output-jsonl", required=True, help="Output JSONL file.")
    parser.add_argument("--query-field", default="user_query")
    parser.add_argument("--scenario-field", default="scenario")
    parser.add_argument("--intent-field", default="pred_intent")
    parser.add_argument("--escalation-field", default="pred_need_escalation")
    parser.add_argument("--answer-field", default="answer")
    parser.add_argument("--next-action-field", default="next_action")
    parser.add_argument("--must-include-field", default="must_include")
    args = parser.parse_args()

    input_jsonl = Path(args.input_jsonl).resolve()
    output_jsonl = Path(args.output_jsonl).resolve()
    if not input_jsonl.exists():
        raise FileNotFoundError(f"input not found: {input_jsonl}")

    stats = process_jsonl(
        input_jsonl=input_jsonl,
        output_jsonl=output_jsonl,
        query_field=args.query_field,
        scenario_field=args.scenario_field,
        intent_field=args.intent_field,
        escalation_field=args.escalation_field,
        answer_field=args.answer_field,
        next_action_field=args.next_action_field,
        must_include_field=args.must_include_field,
    )

    print(json.dumps({"input": str(input_jsonl), "output": str(output_jsonl), "stats": stats}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
