#!/usr/bin/env python3
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from scripts.proxy_calculator_client import call_calculator


ODDS_RE = re.compile(r"(?<!\d)(\d+(?:\.\d+)?)(?!\d)")
PERCENT_RE = re.compile(r"(\d+(?:\.\d+)?)%")
MONEY_RE = re.compile(r"(本金|金额|下注)\s*[:：]?\s*(\d+(?:\.\d+)?)")
STAKE_RE = re.compile(r"(本金|金额|下注|投注|投|押)\s*[:：]?\s*(\d+(?:\.\d+)?)")
ODDS_LABEL_RE = re.compile(r"(赔率|水位|odds)\s*[:：=]?\s*(-?\d+(?:\.\d+)?)", re.I)
PLATE_HINT_RE = re.compile(r"(欧洲盘|欧赔|香港盘|港盘|串关|三串三|一串四|冻结比例|冻结)")
PARLAY_HINT_RE = re.compile(r"(串关|过关|三串三|一串四)")
FREEZE_HINT_RE = re.compile(r"(冻结|比例|百分比|%)")
ODDS_HINT_RE = re.compile(r"(赔率|派彩|净赢|净赚|赢多少|返多少|水位)")


def _joined_rag_text(rag_hits: List[Dict[str, Any]]) -> str:
    parts = []
    for hit in rag_hits[:5]:
        parts.append(str(hit.get("source_file", "")))
        parts.append(str(hit.get("text", ""))[:600])
    return "\n".join(parts)


def infer_formula_id(query: str, rag_hits: List[Dict[str, Any]]) -> Optional[str]:
    q = (query or "").strip()
    rag_text = _joined_rag_text(rag_hits)
    merged = f"{q}\n{rag_text}"
    if "欧洲盘" in merged or "欧赔" in merged:
        return "european_odds_net_profit"
    if "香港盘" in merged or "港盘" in merged:
        return "hk_odds_profit"
    if "串关" in merged or "三串三" in merged or "一串四" in merged:
        return "parlay_all_win_payout"
    if "冻结" in merged and ("比例" in merged or "%" in merged):
        return "freeze_amount_by_ratio"
    if ("赔率" in q or "派彩" in q or "净赢" in q) and ("欧洲" in rag_text or "欧赔" in rag_text):
        return "european_odds_net_profit"
    if ("赔率" in q or "派彩" in q or "净赢" in q) and ("香港盘" in rag_text or "港盘" in rag_text):
        return "hk_odds_profit"
    if "串关" in q or ("赔率相乘" in rag_text and ("派彩" in q or "怎么算" in q)):
        return "parlay_all_win_payout"
    if "比例" in q and "冻结" in q:
        return "freeze_amount_by_ratio"
    return None


def extract_inputs(query: str, formula_id: str) -> Dict[str, Any]:
    q = (query or "").strip()
    if formula_id in {"european_odds_net_profit", "hk_odds_profit"}:
        stake = None
        odds = None
        money_match = STAKE_RE.search(q)
        if money_match:
            stake = float(money_match.group(2))
        odds_match = ODDS_LABEL_RE.search(q)
        if odds_match:
            odds = float(odds_match.group(2))
        nums = [float(x.group(1)) for x in ODDS_RE.finditer(q)]
        if stake is None and nums:
            stake = nums[0]
        if odds is None and len(nums) >= 2:
            odds = nums[1]
        if stake is not None and odds is not None:
            return {"stake": stake, "odds": odds}
        return {}

    if formula_id == "freeze_amount_by_ratio":
        amount = None
        ratio = None
        money_match = STAKE_RE.search(q)
        if money_match:
            amount = float(money_match.group(2))
        pct_match = PERCENT_RE.search(q)
        if pct_match:
            ratio = float(pct_match.group(1)) / 100.0
        nums = [float(x.group(1)) for x in ODDS_RE.finditer(q)]
        if amount is None and nums:
            amount = nums[0]
        if ratio is None and len(nums) >= 2:
            ratio = nums[1]
            if ratio > 1:
                ratio = ratio / 100.0
        if amount is not None and ratio is not None:
            return {"amount": amount, "ratio": ratio}
        return {}

    if formula_id == "parlay_all_win_payout":
        nums = [float(x.group(1)) for x in ODDS_RE.finditer(q)]
        if len(nums) >= 3:
            return {"stake": nums[0], "odds_list": nums[1:]}
        return {}
    return {}


def explain_calc_failure(query: str, rag_hits: List[Dict[str, Any]], formula_id: str, error_code: str) -> Dict[str, Any]:
    q = (query or "").strip()
    hint_match = PLATE_HINT_RE.search(q)
    plate_hint = hint_match.group(1) if hint_match else ""
    if formula_id in {"european_odds_net_profit", "hk_odds_profit"}:
        missing = []
        reply_fields = []
        missing_fields = []
        numbers = [float(x.group(1)) for x in ODDS_RE.finditer(q)]
        if not plate_hint and "欧洲盘" not in q and "香港盘" not in q:
            missing.append("盘型（欧洲盘或香港盘）")
            reply_fields.append("盘型")
            missing_fields.append("plate_type")
        if not STAKE_RE.search(q) and "本金" not in q and "金额" not in q and "下注" not in q:
            missing.append("本金/下注金额")
            reply_fields.append("本金")
            missing_fields.append("stake")
        if not ODDS_LABEL_RE.search(q) and len(numbers) < 2:
            missing.append("赔率")
            reply_fields.append("赔率")
            missing_fields.append("odds")
        return {
            "bucket": "odds",
            "message": "已识别为赔率/派彩计算问题，但当前参数不足，无法可靠计算。",
            "suggestion": f"请补充：{'、'.join(missing) if missing else '本金、赔率、盘型'}。",
            "reply": f"这个我可以帮你算。请先告诉我{' + '.join(reply_fields) if reply_fields else '盘型 + 本金 + 赔率'}，我再按当前规则帮你计算。",
            "missing_fields": missing_fields or ["plate_type", "stake", "odds"],
        }
    if formula_id == "parlay_all_win_payout":
        return {
            "bucket": "parlay",
            "message": "已识别为串关计算问题，但当前参数不足，无法可靠计算。",
            "suggestion": "请补充：本金，以及每一关的赔率列表，例如 A@1.62、B@1.52、C@1.76。",
            "reply": "这个我可以帮你算串关派彩。请先告诉我本金，以及每一关的赔率列表，我再按当前规则帮你计算。",
            "missing_fields": ["stake", "odds_list"],
        }
    if formula_id == "freeze_amount_by_ratio":
        return {
            "bucket": "freeze",
            "message": "已识别为冻结比例计算问题，但当前参数不足，无法可靠计算。",
            "suggestion": "请补充：金额和冻结比例，例如 金额500、比例20%。",
            "reply": "这个我可以帮你算冻结金额。请先告诉我金额和冻结比例，例如金额500、比例20%。",
            "missing_fields": ["amount", "ratio"],
        }
    if error_code == "FORMULA_NOT_INFERRED":
        if PARLAY_HINT_RE.search(q):
            return {
                "bucket": "parlay",
                "message": "已识别为串关计算问题，但当前参数不足，无法可靠计算。",
                "suggestion": "请补充：本金，以及每一关的赔率列表，例如 A@1.62、B@1.52、C@1.76。",
                "reply": "这个我可以帮你算串关派彩。请先告诉我本金，以及每一关的赔率列表，我再按当前规则帮你计算。",
                "missing_fields": ["stake", "odds_list"],
            }
        if FREEZE_HINT_RE.search(q):
            return {
                "bucket": "freeze",
                "message": "已识别为冻结比例计算问题，但当前参数不足，无法可靠计算。",
                "suggestion": "请补充：金额和冻结比例，例如 金额500、比例20%。",
                "reply": "这个我可以帮你算冻结金额。请先告诉我金额和冻结比例，例如金额500、比例20%。",
                "missing_fields": ["amount", "ratio"],
            }
        if ODDS_HINT_RE.search(q):
            return {
                "bucket": "odds",
                "message": "已识别为赔率/派彩计算问题，但当前参数不足，无法可靠计算。",
                "suggestion": "请补充：盘型（欧洲盘或香港盘）、本金/下注金额、赔率。",
                "reply": "这个我可以帮你算。请先告诉我是欧洲盘还是香港盘，再给我本金和赔率。",
                "missing_fields": ["plate_type", "stake", "odds"],
            }
        return {
            "bucket": "generic",
            "message": "已识别为计算类问题，但当前无法确定具体计算类型。",
            "suggestion": "请补充盘型或规则类型，例如 欧洲盘、香港盘、串关、冻结比例。",
            "reply": "这个我可以帮你算。请先告诉我是欧洲盘、香港盘、串关还是冻结比例这类计算，我再继续帮你算。",
            "missing_fields": ["formula_type"],
        }
    return {
        "bucket": "generic",
        "message": "已识别为计算类问题，但当前无法可靠提取参数或完成计算。",
        "suggestion": "请补充更明确的计算条件，例如本金、赔率、盘型、比例等。",
        "reply": "这个我可以帮你算。请再补充更明确的计算条件，例如本金、赔率、盘型或比例。",
        "missing_fields": ["formula_type"],
    }


def build_calc_system_message(query: str, rag_hits: List[Dict[str, Any]], calc_result: Dict[str, Any]) -> str:
    refs = []
    for idx, hit in enumerate(rag_hits[:3], start=1):
        refs.append(
            f"[{idx}] {hit.get('source_file', 'unknown')} {hit.get('unit_type', 'chunk')}#{hit.get('unit_id', '?')} chunk#{hit.get('chunk_index', '?')}\n"
            f"{str(hit.get('text', ''))[:400]}"
        )
    result = calc_result.get("result", {})
    steps = calc_result.get("steps", [])
    formula_text = calc_result.get("formula_text", "")
    return (
        "你是一个计算型客服助手。必须优先依据提供的规则和计算结果回答，"
        "不要自己重新心算，也不要编造未提供的数字。\n\n"
        f"用户问题：\n{query}\n\n"
        f"规则依据：\n{chr(10).join(refs) if refs else '（无命中）'}\n\n"
        f"计算公式：\n{formula_text or '（未提供）'}\n\n"
        f"计算步骤：\n{chr(10).join(steps) if steps else '（未提供）'}\n\n"
        f"计算结果：\n{result}\n\n"
        "请使用中文给出简洁、可复核的答案，并明确说明结果基于当前规则和输入参数。"
    )


def run_calc_flow(
    request_id: str,
    query: str,
    rag_hits: List[Dict[str, Any]],
) -> Dict[str, Any]:
    formula_id = infer_formula_id(query, rag_hits)
    if not formula_id:
        return {"ok": False, "error": {"code": "FORMULA_NOT_INFERRED", "message": "unable to infer formula"}, "formula_id": ""}
    inputs = extract_inputs(query, formula_id)
    if not inputs:
        return {"ok": False, "error": {"code": "INPUTS_NOT_EXTRACTED", "message": "unable to extract calculator inputs"}, "formula_id": formula_id}
    top_hit = rag_hits[0] if rag_hits else {}
    context = {
        "source_file": top_hit.get("source_file", ""),
        "source_chunk_id": top_hit.get("chunk_id"),
        "source_rule": str(top_hit.get("text", ""))[:500],
    }
    return call_calculator(
        request_id=request_id,
        formula_id=formula_id,
        inputs=inputs,
        context=context,
    )
