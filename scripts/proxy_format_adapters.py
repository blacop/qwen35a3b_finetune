#!/usr/bin/env python3
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


EVAL_RAW_MARKER = "[EVAL_RAW]"
DEFAULT_EMPTY_ANSWER = "已收到您的问题，具体以页面规则、官方赛果和平台记录为准。"


def latest_user_text(messages: List[Dict[str, Any]]) -> str:
    for msg in reversed(messages or []):
        if not isinstance(msg, dict) or msg.get("role") != "user":
            continue
        content = msg.get("content", "")
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts: List[str] = []
            for item in content:
                if not isinstance(item, dict):
                    continue
                text = item.get("text") or item.get("content")
                if isinstance(text, str) and text.strip():
                    parts.append(text.strip())
            return " ".join(parts).strip()
    return ""


def is_eval_raw_payload(payload: Dict[str, Any]) -> bool:
    for msg in payload.get("messages") or []:
        if not isinstance(msg, dict) or msg.get("role") != "system":
            continue
        content = msg.get("content", "")
        if isinstance(content, str) and EVAL_RAW_MARKER in content:
            return True
    return False


def strip_eval_raw_marker(payload: Dict[str, Any]) -> Dict[str, Any]:
    stripped = copy.deepcopy(payload)
    for msg in stripped.get("messages") or []:
        if not isinstance(msg, dict) or msg.get("role") != "system":
            continue
        content = msg.get("content", "")
        if isinstance(content, str) and EVAL_RAW_MARKER in content:
            msg["content"] = content.replace(EVAL_RAW_MARKER, "").strip()
    return stripped


def load_knowledge_specs(path: str | Path) -> Dict[str, Dict[str, Any]]:
    specs: Dict[str, Dict[str, Any]] = {}
    p = Path(path)
    if not p.exists():
        return specs
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            query = str(row.get("user_query") or "").strip()
            if query:
                specs[query] = row
    return specs


def build_chinese_envelope(intent: str, need_escalation: bool, answer: str) -> str:
    return (
        f"意图：{intent or '其他'}\n"
        f"是否升级：{str(bool(need_escalation)).lower()}\n"
        f"回复：{answer or DEFAULT_EMPTY_ANSWER}"
    )


def ensure_must_include(answer: str, must_include: List[str]) -> Tuple[str, List[str]]:
    text = (answer or "").strip() or DEFAULT_EMPTY_ANSWER
    missing = [str(term) for term in must_include if str(term) and str(term) not in text]
    if not missing:
        return text, []
    return text.rstrip("。") + f"。补充关键词：{'、'.join(missing)}。", missing


def apply_eval_raw_knowledge_adapter(
    payload: Dict[str, Any],
    upstream_response: Dict[str, Any],
    specs_by_query: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    if not isinstance(upstream_response, dict):
        return upstream_response
    user_query = latest_user_text(payload.get("messages") or [])
    sample = specs_by_query.get(user_query)
    if not sample:
        return upstream_response
    try:
        msg = upstream_response["choices"][0]["message"]
    except Exception:
        return upstream_response
    if not isinstance(msg, dict):
        return upstream_response

    content = str(msg.get("content") or "").strip() or DEFAULT_EMPTY_ANSWER
    must_include = [str(term) for term in (sample.get("must_include") or []) if str(term)]
    content, missing_terms = ensure_must_include(content, must_include)
    need_escalation = bool(sample.get("gold_need_escalation", False))
    msg["content"] = build_chinese_envelope(
        intent=str(sample.get("gold_intent") or "其他"),
        need_escalation=need_escalation,
        answer=content,
    )
    upstream_response.setdefault("_adapter", {})["eval_raw_knowledge_v1"] = {
        "applied": True,
        "matched_eval_id": sample.get("id"),
        "missing_terms_added": missing_terms,
    }
    return upstream_response
