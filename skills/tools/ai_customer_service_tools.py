from __future__ import annotations

import asyncio
import json
import os
import random
import time
from typing import Any, Dict, Optional

import requests

from skills.registry import register_skill


SESSION = requests.Session()
AI_CS_BASE_URL = os.getenv("AI_CS_BASE_URL", "").strip().rstrip("/")
AI_CS_TIMEOUT_S = float(os.getenv("AI_CS_TIMEOUT_S", "15") or "15")
AI_CS_USE_BACKEND = os.getenv("AI_CS_USE_BACKEND", "0").strip().lower() in {"1", "true", "yes", "on"}
AI_CS_BACKEND_TOKEN = os.getenv("AI_CS_BACKEND_TOKEN", "").strip()


def _headers() -> Dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if AI_CS_BACKEND_TOKEN:
        headers["Authorization"] = f"Bearer {AI_CS_BACKEND_TOKEN}"
    return headers


def _call_backend(path: str, body: Dict[str, Any]) -> Dict[str, Any]:
    if not AI_CS_BASE_URL:
        return {"ok": False, "error": "AI_CS_BASE_URL is empty"}
    url = AI_CS_BASE_URL + path
    try:
        resp = SESSION.post(url, headers=_headers(), json=body, timeout=AI_CS_TIMEOUT_S)
    except Exception as exc:
        return {"ok": False, "error": f"backend request failed: {type(exc).__name__}: {exc}", "path": path}
    try:
        data = resp.json()
    except Exception:
        data = {"raw": (resp.text or "")[:2000]}
    return {
        "ok": resp.status_code < 400,
        "http_status": resp.status_code,
        "path": path,
        "payload": data,
    }


def _normalize_text(v: Any) -> str:
    return str(v or "").strip()


def _mock_intent(text: str) -> Dict[str, Any]:
    t = text.lower()
    intent = "other"
    if any(k in text for k in ["活动", "彩金", "领取"]):
        intent = "activity_claim"
    elif any(k in text for k in ["充值", "掉单", "入款"]):
        intent = "recharge_exception"
    elif any(k in text for k in ["提款", "提现", "出款"]):
        intent = "withdraw_exception"
    elif any(k in text for k in ["密码", "手机号", "邮箱", "绑定"]):
        intent = "account_management"
    elif any(k in text for k in ["登录", "卡顿", "风控", "封禁"]):
        intent = "platform_issue"
    elif any(k in text for k in ["派奖", "局号", "游戏结果", "奖金不对"]):
        intent = "game_issue"
    elif any(k in t for k in ["人工", "human", "客服"]):
        intent = "handover"
    return {
        "ok": True,
        "intent": intent,
        "confidence": 0.93,
        "_mock": True,
    }


@register_skill
class DetectIntentSkill:
    name = "detect_intent"
    description = "识别用户意图和槽位。对应草案接口 POST /nlp/intent。"
    parameters = {
        "type": "object",
        "properties": {
            "text": {"type": "string", "description": "用户原始输入文本"},
            "user_id": {"type": "string", "description": "用户ID（可选）"},
        },
        "required": ["text"],
    }

    @staticmethod
    async def execute(args: dict) -> dict:
        text = _normalize_text(args.get("text"))
        user_id = _normalize_text(args.get("user_id"))
        if not text:
            return {"ok": False, "error": "缺少必填参数 text"}
        body = {"text": text, "user_id": user_id}
        if AI_CS_USE_BACKEND:
            return await asyncio.to_thread(_call_backend, "/nlp/intent", body)
        out = _mock_intent(text)
        out["slots"] = {"user_id": user_id} if user_id else {}
        return out


@register_skill
class ClaimActivityBonusSkill:
    name = "claim_activity_bonus"
    description = "活动办理：查询资格并申请发放彩金。对应草案接口 POST /activity/claim。"
    parameters = {
        "type": "object",
        "properties": {
            "user_id": {"type": "string", "description": "会员ID"},
            "activity_name": {"type": "string", "description": "活动名称"},
            "profit_loss_amount": {"type": "number", "description": "活动周期盈亏值（可选，便于规则计算）"},
        },
        "required": ["user_id", "activity_name"],
    }

    @staticmethod
    async def execute(args: dict) -> dict:
        user_id = _normalize_text(args.get("user_id"))
        activity_name = _normalize_text(args.get("activity_name"))
        profit_loss_amount = args.get("profit_loss_amount")
        if not user_id or not activity_name:
            return {"ok": False, "error": "缺少必填参数 user_id 或 activity_name"}
        body = {
            "user_id": user_id,
            "activity_name": activity_name,
            "profit_loss_amount": profit_loss_amount,
        }
        if AI_CS_USE_BACKEND:
            return await asyncio.to_thread(_call_backend, "/activity/claim", body)

        seed = hash((user_id, activity_name)) & 0xFFFFFFFF
        rnd = random.Random(seed)
        eligible = rnd.random() > 0.35
        bonus = round(abs(float(profit_loss_amount or rnd.uniform(50, 1200))) * 0.1, 2) if eligible else 0
        return {
            "ok": True,
            "eligible": eligible,
            "bonus_amount": bonus,
            "status": "success" if eligible else "rejected",
            "message": f"活动 {activity_name} 已处理",
            "_mock": True,
        }


@register_skill
class HandleRechargeExceptionSkill:
    name = "handle_recharge_exception"
    description = "充值异常处理：掉单、回调失败、到账核对。"
    parameters = {
        "type": "object",
        "properties": {
            "order_no": {"type": "string", "description": "充值订单号"},
            "user_id": {"type": "string", "description": "会员ID"},
            "amount": {"type": "number", "description": "充值金额（可选）"},
            "occurred_at": {"type": "string", "description": "异常时间（可选）"},
        },
        "required": ["order_no"],
    }

    @staticmethod
    async def execute(args: dict) -> dict:
        order_no = _normalize_text(args.get("order_no"))
        if not order_no:
            return {"ok": False, "error": "缺少必填参数 order_no"}
        body = {
            "order_no": order_no,
            "user_id": _normalize_text(args.get("user_id")),
            "amount": args.get("amount"),
            "occurred_at": _normalize_text(args.get("occurred_at")),
        }
        if AI_CS_USE_BACKEND:
            return await asyncio.to_thread(_call_backend, "/payment/recharge/exception", body)
        fixed = random.Random(hash(order_no) & 0xFFFFFFFF).random() > 0.2
        return {
            "ok": True,
            "order_no": order_no,
            "status": "fixed" if fixed else "pending_finance_review",
            "action": "trigger_callback" if fixed else "create_finance_ticket",
            "_mock": True,
        }


@register_skill
class HandleWithdrawExceptionSkill:
    name = "handle_withdraw_exception"
    description = "提款异常处理：未到账核对、渠道状态检查、财务工单。"
    parameters = {
        "type": "object",
        "properties": {
            "order_no": {"type": "string", "description": "提款订单号"},
            "user_id": {"type": "string", "description": "会员ID"},
            "amount": {"type": "number", "description": "提款金额（可选）"},
            "occurred_at": {"type": "string", "description": "异常时间（可选）"},
        },
        "required": ["order_no"],
    }

    @staticmethod
    async def execute(args: dict) -> dict:
        order_no = _normalize_text(args.get("order_no"))
        if not order_no:
            return {"ok": False, "error": "缺少必填参数 order_no"}
        body = {
            "order_no": order_no,
            "user_id": _normalize_text(args.get("user_id")),
            "amount": args.get("amount"),
            "occurred_at": _normalize_text(args.get("occurred_at")),
        }
        if AI_CS_USE_BACKEND:
            return await asyncio.to_thread(_call_backend, "/payment/withdraw/exception", body)
        resolved = random.Random(hash(("wd", order_no)) & 0xFFFFFFFF).random() > 0.3
        return {
            "ok": True,
            "order_no": order_no,
            "status": "resolved" if resolved else "queued_for_manual_adjustment",
            "action": "notify_user" if resolved else "handover_finance",
            "_mock": True,
        }


@register_skill
class ManageAccountInfoSkill:
    name = "manage_account_info"
    description = "账户信息管理：密码重置、手机号/邮箱绑定修改、第三方账号绑定。"
    parameters = {
        "type": "object",
        "properties": {
            "user_id": {"type": "string", "description": "会员ID"},
            "operation": {
                "type": "string",
                "enum": ["reset_password", "update_phone", "update_email", "bind_third_party", "unbind_third_party"],
                "description": "账户操作类型",
            },
            "verify_channel": {
                "type": "string",
                "enum": ["sms", "email", "security_question"],
                "description": "验证方式",
            },
            "target_value": {"type": "string", "description": "新手机号/新邮箱/第三方账号ID（按操作类型传）"},
        },
        "required": ["user_id", "operation", "verify_channel"],
    }

    @staticmethod
    async def execute(args: dict) -> dict:
        user_id = _normalize_text(args.get("user_id"))
        operation = _normalize_text(args.get("operation"))
        verify_channel = _normalize_text(args.get("verify_channel"))
        if not user_id or not operation or not verify_channel:
            return {"ok": False, "error": "缺少必填参数 user_id / operation / verify_channel"}
        body = {
            "user_id": user_id,
            "operation": operation,
            "verify_channel": verify_channel,
            "target_value": _normalize_text(args.get("target_value")),
        }
        if AI_CS_USE_BACKEND:
            return await asyncio.to_thread(_call_backend, "/account/manage", body)
        return {
            "ok": True,
            "user_id": user_id,
            "operation": operation,
            "status": "verification_required",
            "message": f"已触发 {verify_channel} 验证流程",
            "_mock": True,
        }


@register_skill
class DiagnosePlatformIssueSkill:
    name = "diagnose_platform_issue"
    description = "平台问题处理：登录异常、卡顿、账号状态异常诊断。"
    parameters = {
        "type": "object",
        "properties": {
            "user_id": {"type": "string", "description": "会员ID（可选）"},
            "issue_type": {
                "type": "string",
                "enum": ["login_failed", "performance_lag", "account_status_abnormal"],
                "description": "问题类型",
            },
            "device_info": {"type": "string", "description": "设备信息（可选）"},
            "ip": {"type": "string", "description": "用户IP（可选）"},
            "description": {"type": "string", "description": "问题描述（可选）"},
        },
        "required": ["issue_type"],
    }

    @staticmethod
    async def execute(args: dict) -> dict:
        issue_type = _normalize_text(args.get("issue_type"))
        if not issue_type:
            return {"ok": False, "error": "缺少必填参数 issue_type"}
        body = {
            "user_id": _normalize_text(args.get("user_id")),
            "issue_type": issue_type,
            "device_info": _normalize_text(args.get("device_info")),
            "ip": _normalize_text(args.get("ip")),
            "description": _normalize_text(args.get("description")),
        }
        if AI_CS_USE_BACKEND:
            return await asyncio.to_thread(_call_backend, "/platform/issue/diagnose", body)
        mapping = {
            "login_failed": "建议检查账号状态和风控限制",
            "performance_lag": "建议清缓存、切网络并已提交性能工单",
            "account_status_abnormal": "检测到账号状态异常，建议人工复核",
        }
        return {
            "ok": True,
            "issue_type": issue_type,
            "diagnosis": mapping.get(issue_type, "已记录并待排查"),
            "need_ticket": issue_type != "login_failed",
            "_mock": True,
        }


@register_skill
class AuditGameIssueSkill:
    name = "audit_game_issue"
    description = "游戏问题处理：派奖异常、结果争议核查并发起复核。"
    parameters = {
        "type": "object",
        "properties": {
            "user_id": {"type": "string", "description": "会员ID（可选）"},
            "game_name": {"type": "string", "description": "游戏名称"},
            "round_id": {"type": "string", "description": "局号/回合号"},
            "occurred_at": {"type": "string", "description": "发生时间（可选）"},
            "description": {"type": "string", "description": "问题描述（可选）"},
        },
        "required": ["game_name", "round_id"],
    }

    @staticmethod
    async def execute(args: dict) -> dict:
        game_name = _normalize_text(args.get("game_name"))
        round_id = _normalize_text(args.get("round_id"))
        if not game_name or not round_id:
            return {"ok": False, "error": "缺少必填参数 game_name 或 round_id"}
        body = {
            "user_id": _normalize_text(args.get("user_id")),
            "game_name": game_name,
            "round_id": round_id,
            "occurred_at": _normalize_text(args.get("occurred_at")),
            "description": _normalize_text(args.get("description")),
        }
        if AI_CS_USE_BACKEND:
            return await asyncio.to_thread(_call_backend, "/game/issue/audit", body)
        consistent = random.Random(hash((game_name, round_id)) & 0xFFFFFFFF).random() > 0.4
        return {
            "ok": True,
            "game_name": game_name,
            "round_id": round_id,
            "result": "records_match" if consistent else "need_third_party_review",
            "message": "已发起复核流程" if not consistent else "后台记录一致",
            "_mock": True,
        }


@register_skill
class HandoverToHumanSkill:
    name = "handover_to_human"
    description = "转人工：将会话上下文与原因提交人工队列。对应草案接口 POST /human/handover。"
    parameters = {
        "type": "object",
        "properties": {
            "session_id": {"type": "string", "description": "会话ID"},
            "reason": {"type": "string", "description": "转人工原因"},
            "context": {"type": "object", "description": "附带上下文"},
            "priority": {"type": "string", "enum": ["low", "normal", "high"], "description": "优先级"},
        },
        "required": ["session_id", "reason"],
    }

    @staticmethod
    async def execute(args: dict) -> dict:
        session_id = _normalize_text(args.get("session_id"))
        reason = _normalize_text(args.get("reason"))
        if not session_id or not reason:
            return {"ok": False, "error": "缺少必填参数 session_id 或 reason"}
        body = {
            "session_id": session_id,
            "reason": reason,
            "context": args.get("context") if isinstance(args.get("context"), dict) else {},
            "priority": _normalize_text(args.get("priority")) or "normal",
        }
        if AI_CS_USE_BACKEND:
            return await asyncio.to_thread(_call_backend, "/human/handover", body)
        ticket_id = f"T{time.strftime('%Y%m%d%H%M%S')}{random.randint(100, 999)}"
        return {
            "ok": True,
            "status": "queued",
            "ticket_id": ticket_id,
            "session_id": session_id,
            "_mock": True,
        }

