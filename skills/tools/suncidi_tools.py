from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from urllib.parse import quote

import requests

from skills.context import get_tool_context, get_user_access_token
from skills.registry import get_skill, register_skill


SESSION = requests.Session()
DEFAULT_BASE_URL = os.getenv("SUNCIDI_API_BASE_URL", "https://m.suncidi.com/api/user").rstrip("/")
DEFAULT_TIMEOUT_S = float(os.getenv("SUNCIDI_API_TIMEOUT_S", "20") or "20")
DEFAULT_USER_ACCESS_TOKEN = os.getenv("SUNCIDI_USER_ACCESS_TOKEN", "").strip()
DEFAULT_SESSION_COOKIE = os.getenv("SUNCIDI_SESSION_COOKIE", "").strip()
ALLOW_RAW_PAYLOAD = os.getenv("SUNCIDI_ALLOW_RAW_PAYLOAD", "1").strip().lower() in {"1", "true", "yes", "on"}
SWAGGER_AUTO_TOOLS_ENABLED = os.getenv("SUNCIDI_SWAGGER_AUTO_TOOLS_ENABLED", "1").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
SWAGGER_DOC_URL = (os.getenv("SUNCIDI_SWAGGER_DOC_URL", "https://m.suncidi.com/api/user/swagger/doc.json") or "").strip()
SWAGGER_DOC_PATH = (os.getenv("SUNCIDI_SWAGGER_DOC_PATH", "") or "").strip()
SWAGGER_DOC_CACHE_PATH = (
    os.getenv("SUNCIDI_SWAGGER_DOC_CACHE_PATH", "/home/ubuntu/qwen35a3b_finetune/runtime/suncidi_swagger_doc_cache.json")
    or ""
).strip()
SWAGGER_TOOL_PREFIX = (os.getenv("SUNCIDI_SWAGGER_TOOL_PREFIX", "suncidi_auto_") or "suncidi_auto_").strip()
SWAGGER_MAX_TOOLS = int(os.getenv("SUNCIDI_SWAGGER_MAX_TOOLS", "300") or "300")


def _base_url() -> str:
    return (os.getenv("SUNCIDI_API_BASE_URL", DEFAULT_BASE_URL) or DEFAULT_BASE_URL).rstrip("/")


def _timeout_s() -> float:
    try:
        return float(os.getenv("SUNCIDI_API_TIMEOUT_S", str(DEFAULT_TIMEOUT_S)) or DEFAULT_TIMEOUT_S)
    except Exception:
        return DEFAULT_TIMEOUT_S


def _current_access_token() -> str:
    token = get_user_access_token()
    if token:
        return token
    return DEFAULT_USER_ACCESS_TOKEN


def _request_headers(require_auth: bool = False) -> Dict[str, str]:
    headers = {"Content-Type": "application/json"}
    token = _current_access_token()
    if require_auth:
        if not token:
            raise ValueError("缺少用户 access token，请通过 X-User-Access-Token 传入，或配置 SUNCIDI_USER_ACCESS_TOKEN")
        headers["Authorization"] = f"Bearer {token}"
    elif token:
        headers["Authorization"] = f"Bearer {token}"
    if DEFAULT_SESSION_COOKIE:
        headers["Cookie"] = DEFAULT_SESSION_COOKIE
    return headers


def _safe_jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, list):
        return [_safe_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(k): _safe_jsonable(v) for k, v in value.items()}
    return str(value)


def _maybe_decode_payload(payload: str) -> Optional[Dict[str, Any]]:
    raw = (payload or "").strip()
    if not raw:
        return None
    normalized = raw + "=" * (-len(raw) % 4)
    try:
        decoded = base64.b64decode(normalized, validate=False)
    except Exception:
        return None
    if not decoded:
        return None
    try:
        text = decoded.decode("utf-8")
    except Exception:
        return None
    text = text.strip()
    if not text:
        return None
    try:
        obj = json.loads(text)
        return {"kind": "json", "value": obj}
    except Exception:
        return {"kind": "text", "value": text}


def _summarize_response(data: Any) -> Dict[str, Any]:
    if not isinstance(data, dict):
        return {"ok": True, "response": _safe_jsonable(data)}

    out: Dict[str, Any] = {
        "ok": True,
        "code": data.get("code"),
        "count": data.get("count"),
        "message": data.get("message"),
        "page": data.get("page"),
        "totalPage": data.get("totalPage"),
    }
    payload = data.get("payload")
    if isinstance(payload, str):
        payload_preview = payload[:200]
        decoded = _maybe_decode_payload(payload) if ALLOW_RAW_PAYLOAD else None
        out["payload_opaque"] = True
        out["payload_length"] = len(payload)
        out["payload_preview"] = payload_preview
        if decoded is not None:
            out["payload_decoded"] = decoded
    else:
        out["payload_opaque"] = False
        out["payload"] = _safe_jsonable(payload)

    extra = {k: v for k, v in data.items() if k not in {"code", "count", "message", "page", "payload", "totalPage"}}
    if extra:
        out["extra"] = _safe_jsonable(extra)
    return out


def _call_api(
    method: str,
    path: str,
    *,
    params: Optional[Dict[str, Any]] = None,
    body: Optional[Dict[str, Any]] = None,
    require_auth: bool = False,
) -> Dict[str, Any]:
    url = _base_url().rstrip("/") + path
    try:
        headers = _request_headers(require_auth=require_auth)
    except ValueError as exc:
        return {"ok": False, "error": str(exc), "endpoint": path, "method": method}

    try:
        resp = SESSION.request(
            method=method.upper(),
            url=url,
            params=params,
            json=body,
            headers=headers,
            timeout=_timeout_s(),
        )
    except Exception as exc:
        return {"ok": False, "error": f"request failed: {type(exc).__name__}: {exc}", "endpoint": path, "method": method}

    raw_text = resp.text or ""
    try:
        data = resp.json()
    except Exception:
        data = {"raw_text": raw_text[:2000] if raw_text else "", "content_type": resp.headers.get("content-type", "")}

    out = _summarize_response(data)
    out.update(
        {
            "http_status": resp.status_code,
            "endpoint": path,
            "method": method.upper(),
            "request": {
                "params": _safe_jsonable(params or {}),
                "body": _safe_jsonable(body or {}),
                "require_auth": require_auth,
                "user_access_token_present": bool(_current_access_token()),
            },
        }
    )
    if resp.status_code >= 400:
        out["ok"] = False
        out["error"] = out.get("message") or raw_text[:500] or f"http {resp.status_code}"
    return out


def _introspect_call(name: str, args: Dict[str, Any], result: Dict[str, Any]) -> Dict[str, Any]:
    trace = dict(get_tool_context())
    trace["tool"] = name
    trace["args"] = args
    trace["result_ok"] = bool(result.get("ok"))
    return trace


def _normalize_swagger_path(path: str) -> str:
    if not path:
        return path
    if path == "/user":
        return "/"
    if path.startswith("/user/"):
        return path[len("/user") :]
    return path


def _extract_path_params(endpoint: str) -> Tuple[str, ...]:
    keys = re.findall(r"\{([A-Za-z_][A-Za-z0-9_]*)\}", endpoint or "")
    return tuple(keys)


def _consume_path_params(endpoint: str, payload: Dict[str, Any], path_params: Tuple[str, ...]) -> Tuple[Optional[str], Dict[str, Any], Optional[str]]:
    if not path_params:
        return endpoint, payload, None
    local = dict(payload)
    out_endpoint = endpoint
    for key in path_params:
        token = "{" + key + "}"
        if token not in out_endpoint:
            continue
        raw = local.pop(key, None)
        if raw is None or str(raw).strip() == "":
            return None, payload, f"缺少路径参数 {key}"
        out_endpoint = out_endpoint.replace(token, quote(str(raw), safe=""))
    return out_endpoint, local, None


class _SuncidiToolBase:
    endpoint: str = ""
    method: str = "GET"
    requires_auth: bool = False
    path_params: Tuple[str, ...] = ()

    @classmethod
    async def execute(cls, args: dict) -> dict:
        payload = dict(args or {})
        endpoint = str(getattr(cls, "endpoint", "") or "")
        method = str(getattr(cls, "method", "GET") or "GET").upper()
        path_params = tuple(getattr(cls, "path_params", ()) or ())
        endpoint, payload_for_api, path_err = _consume_path_params(endpoint, payload, path_params)
        if path_err:
            return {
                "ok": False,
                "error": path_err,
                "endpoint": getattr(cls, "endpoint", ""),
                "method": method,
                "tool_context": _introspect_call(cls.name, payload, {"ok": False}),
            }
        result = await asyncio.to_thread(
            _call_api,
            method,
            endpoint or getattr(cls, "endpoint", ""),
            params=payload_for_api if method == "GET" and payload_for_api else None,
            body=payload_for_api if method != "GET" else None,
            require_auth=cls.requires_auth,
        )
        result["tool_context"] = _introspect_call(cls.name, payload, result)
        return result


@register_skill
class GetGameTypeListSkill(_SuncidiToolBase):
    name = "get_game_type_list"
    endpoint = "/game/type/list"
    method = "GET"
    requires_auth = False
    description = "查询游戏类型列表。用户问“有哪些游戏类型”“查一下游戏分类”时使用。"
    parameters = {
        "type": "object",
        "properties": {},
    }


@register_skill
class GetGameVendorListSkill(_SuncidiToolBase):
    name = "get_game_vendor_list"
    endpoint = "/game/vendor/list"
    method = "GET"
    requires_auth = False
    description = "查询平台类型列表。用户问“有哪些平台”“查平台列表”时使用。"
    parameters = {
        "type": "object",
        "properties": {},
    }


@register_skill
class SearchGameVendorListSkill(_SuncidiToolBase):
    name = "search_game_vendor_list"
    endpoint = "/game/vendor/list"
    method = "POST"
    requires_auth = False
    description = "按货币查询平台类型列表。需要根据币种过滤平台时使用。"
    parameters = {
        "type": "object",
        "properties": {
            "currency": {
                "type": "string",
                "description": "货币代码，例如 USD、CNY、USDT",
            },
        },
    }


@register_skill
class ListBetOrdersSkill(_SuncidiToolBase):
    name = "list_bet_orders"
    endpoint = "/order/list"
    method = "POST"
    requires_auth = True
    description = "查询投注记录列表。需要按时间范围、平台、游戏类型、状态筛选用户投注记录时使用。"
    parameters = {
        "type": "object",
        "properties": {
            "createTimeFrom": {
                "type": "integer",
                "description": "开始时间戳，按业务端约定传入，通常为毫秒。",
            },
            "createTimeTo": {
                "type": "integer",
                "description": "结束时间戳，按业务端约定传入，通常为毫秒。",
            },
            "currency": {
                "type": "string",
                "description": "货币代码，按业务需要可选。",
            },
            "gameTypeId": {
                "type": "string",
                "description": "游戏类型 ID，可选。",
            },
            "page": {
                "type": "integer",
                "description": "页码，从 1 开始。",
            },
            "pageSize": {
                "type": "integer",
                "description": "每页数量。",
            },
            "status": {
                "type": "string",
                "description": "订单状态，空字符串表示全部，settled 表示已结算，unsettled 表示未结算。",
                "enum": ["", "settled", "unsettled"],
            },
            "vendorId": {
                "type": "string",
                "description": "平台 ID，可选。",
            },
        },
        "required": ["createTimeFrom", "createTimeTo", "page", "pageSize"],
    }


@register_skill
class GetBetOrderStatsSkill(_SuncidiToolBase):
    name = "get_bet_order_stats"
    endpoint = "/order/stats"
    method = "POST"
    requires_auth = True
    description = "查询投注记录统计。需要统计投注金额、有效投注、单量、输赢总额时使用。"
    parameters = ListBetOrdersSkill.parameters


@register_skill
class GetOrderStatusListSkill(_SuncidiToolBase):
    name = "get_order_status_list"
    endpoint = "/order/status/list"
    method = "GET"
    requires_auth = False
    description = "查询订单状态列表。用户问“有哪些订单状态”“状态码有哪些”时使用。"
    parameters = {
        "type": "object",
        "properties": {},
    }


@register_skill
class GetActivityMoneyTasksSkill(_SuncidiToolBase):
    name = "get_activity_money_tasks"
    endpoint = "/activity/money/task/list"
    method = "GET"
    requires_auth = False
    description = "查询活跃度彩金任务列表。用于活动办理模块展示可参与任务。"
    parameters = {
        "type": "object",
        "properties": {},
    }


@register_skill
class GetActivityMoneyTasksByCurrencySkill(_SuncidiToolBase):
    name = "get_activity_money_tasks_by_currency"
    endpoint = "/activity/money/task/list"
    method = "POST"
    requires_auth = False
    description = "按货币查询活跃度彩金任务列表。对应 Swagger 的 ActivityMoneyTaskListReq。"
    parameters = {
        "type": "object",
        "properties": {
            "currency": {
                "type": "string",
                "description": "货币代码",
            },
        },
        "required": ["currency"],
    }


@register_skill
class GetDepositTaskListSkill(_SuncidiToolBase):
    name = "get_deposit_task_list"
    endpoint = "/deposit/task/list"
    method = "POST"
    requires_auth = True
    description = "查询充值任务列表。用于充值活动、首充/累计充值任务办理。"
    parameters = {
        "type": "object",
        "properties": {
            "id": {
                "type": "string",
                "description": "活动ID",
            },
        },
        "required": ["id"],
    }


@register_skill
class GetUserBetTasksSkill(_SuncidiToolBase):
    name = "get_user_bet_tasks"
    endpoint = "/bet/task/list"
    method = "GET"
    requires_auth = False
    description = "查询用户投注任务概览（所需流水、未完成流水等）。"
    parameters = {
        "type": "object",
        "properties": {},
    }


@register_skill
class GetBetTaskListSkill(_SuncidiToolBase):
    name = "get_bet_task_list"
    endpoint = "/bet/task/list"
    method = "POST"
    requires_auth = True
    description = "按活动ID查询打码任务详情。"
    parameters = {
        "type": "object",
        "properties": {
            "id": {
                "type": "string",
                "description": "活动ID",
            },
        },
        "required": ["id"],
    }


@register_skill
class GetPromoMoneyStatusListSkill(_SuncidiToolBase):
    name = "get_promo_money_status_list"
    endpoint = "/money/status/list"
    method = "GET"
    requires_auth = False
    description = "查询优惠活动状态列表。用于对齐活动记录状态含义。"
    parameters = {
        "type": "object",
        "properties": {},
    }


@register_skill
class GetPromoMoneyTypeListSkill(_SuncidiToolBase):
    name = "get_promo_money_type_list"
    endpoint = "/money/type/list"
    method = "GET"
    requires_auth = False
    description = "查询优惠活动类型列表。用于筛选活动记录类别。"
    parameters = {
        "type": "object",
        "properties": {},
    }


@register_skill
class CheckCaptchaTokenSkill(_SuncidiToolBase):
    name = "check_captcha_token"
    endpoint = "/captcha/check"
    method = "POST"
    requires_auth = False
    description = "校验验证码并获取 captcha token。用于账户安全操作前验证。"
    parameters = {
        "type": "object",
        "properties": {
            "key": {"type": "string", "description": "验证码 key"},
            "x": {"type": "integer", "description": "X 轴坐标（滑块验证码）"},
            "y": {"type": "integer", "description": "Y 轴坐标（滑块验证码）"},
            "angle": {"type": "integer", "description": "旋转角度（旋转验证码）"},
            "dots": {
                "type": "array",
                "description": "点击验证码坐标数组",
                "items": {
                    "type": "object",
                    "properties": {
                        "x": {"type": "integer"},
                        "y": {"type": "integer"},
                    },
                },
            },
        },
    }


@register_skill
class SendBindSmsCaptchaSkill(_SuncidiToolBase):
    name = "send_bind_sms_captcha"
    endpoint = "/captcha/sms/bind"
    method = "POST"
    requires_auth = False
    description = "发送绑定手机验证码。用于账户信息修改流程。"
    parameters = {
        "type": "object",
        "properties": {
            "captchaToken": {"type": "string", "description": "验证码校验令牌"},
            "phoneNumber": {"type": "string", "description": "手机号"},
        },
    }


@register_skill
class CheckIpBlacklistSkill(_SuncidiToolBase):
    name = "check_ip_blacklist"
    endpoint = "/ip/blacklist/check"
    method = "POST"
    requires_auth = False
    description = "检查 IP 是否在黑名单。用于平台登录异常、风控问题排查。"
    parameters = {
        "type": "object",
        "properties": {
            "ip": {"type": "string", "description": "待检查IP"},
            "traceId": {"type": "string", "description": "跟踪ID（可选）"},
        },
        "required": ["ip"],
    }


@register_skill
class GetUserCardListSkill(_SuncidiToolBase):
    name = "get_user_card_list"
    endpoint = "/card/list"
    method = "GET"
    requires_auth = True
    description = "查询用户银行卡列表。用于提款异常与账户核验。"
    parameters = {
        "type": "object",
        "properties": {},
    }


@register_skill
class GetUserAccountListSkill(_SuncidiToolBase):
    name = "get_user_account_list"
    endpoint = "/account/list"
    method = "GET"
    requires_auth = True
    description = "查询用户账户余额列表。用于账户管理与余额核对。"
    parameters = {
        "type": "object",
        "properties": {},
    }


@register_skill
class GetVendorAccountBalanceSkill(_SuncidiToolBase):
    name = "get_vendor_account_balance"
    endpoint = "/vendor/account"
    method = "POST"
    requires_auth = True
    description = "查询指定游戏平台账户与主账户余额。用于转账与余额异常排查。"
    parameters = {
        "type": "object",
        "properties": {
            "vendorId": {"type": "string", "description": "平台ID"},
        },
        "required": ["vendorId"],
    }


@register_skill
class GetGameCategoryStatsSkill(_SuncidiToolBase):
    name = "get_game_category_stats"
    endpoint = "/game/category/stats/info"
    method = "POST"
    requires_auth = True
    description = "查询游戏分类统计信息。用于游戏问题与代理报表核查。"
    parameters = {
        "type": "object",
        "properties": {
            "dateFrom": {"type": "string", "description": "开始日期，示例 2026-05-01"},
            "dateTo": {"type": "string", "description": "结束日期，示例 2026-05-08"},
            "userId": {"type": "string", "description": "用户ID"},
        },
        "required": ["dateFrom", "dateTo", "userId"],
    }


@register_skill
class GetDepositCategoryStatsSkill(_SuncidiToolBase):
    name = "get_deposit_category_stats"
    endpoint = "/deposit/category/stats/info"
    method = "POST"
    requires_auth = True
    description = "查询充值分类统计信息。用于充值异常和活动充值条件核对。"
    parameters = {
        "type": "object",
        "properties": {
            "dateFrom": {"type": "string", "description": "开始日期，示例 2026-05-01"},
            "dateTo": {"type": "string", "description": "结束日期，示例 2026-05-08"},
            "userId": {"type": "string", "description": "用户ID"},
        },
        "required": ["dateFrom", "dateTo", "userId"],
    }


@register_skill
class GetEventTypeListSkill(_SuncidiToolBase):
    name = "get_event_type_list"
    endpoint = "/event/type/list"
    method = "GET"
    requires_auth = False
    description = "查询活动类型列表。用于活动办理前的类型选择。"
    parameters = {
        "type": "object",
        "properties": {},
    }


@register_skill
class GetEventListSkill(_SuncidiToolBase):
    name = "get_event_list"
    endpoint = "/event/list"
    method = "POST"
    requires_auth = False
    description = "查询活动列表。结合货币和活动类型过滤活动。"
    parameters = {
        "type": "object",
        "properties": {
            "currency": {"type": "string", "description": "货币代码"},
            "type": {"type": "string", "description": "活动类型"},
        },
        "required": ["currency", "type"],
    }


@register_skill
class GetHelpCenterListSkill(_SuncidiToolBase):
    name = "get_help_center_list"
    endpoint = "/help/center/list"
    method = "POST"
    requires_auth = False
    description = "查询帮助中心列表。用于回答常见问题、规则说明、FAQ 入口。"
    parameters = {
        "type": "object",
        "properties": {
            "currency": {"type": "string", "description": "货币代码（可选）。"},
        },
    }


@register_skill
class GetInboxListSkill(_SuncidiToolBase):
    name = "get_inbox_list"
    endpoint = "/inbox/list"
    method = "POST"
    requires_auth = True
    description = "查询站内信列表。用于查看公告/活动/通知消息。"
    parameters = {
        "type": "object",
        "properties": {
            "currency": {"type": "string", "description": "货币代码（可选）。"},
            "page": {"type": "integer", "description": "页码，从 1 开始。"},
            "pageSize": {"type": "integer", "description": "每页数量。"},
            "type": {
                "type": "string",
                "description": "消息类型：announcement(公告) / event(活动) / recommend(推荐) / notice(通知)。",
                "enum": ["announcement", "event", "recommend", "notice"],
            },
        },
        "required": ["page", "pageSize", "type"],
    }


@register_skill
class GetInboxDetailSkill(_SuncidiToolBase):
    name = "get_inbox_detail"
    endpoint = "/inbox/{id}"
    method = "GET"
    requires_auth = True
    description = "查询站内信详情。用于读取指定消息内容。"
    path_params = ("id",)
    parameters = {
        "type": "object",
        "properties": {
            "id": {"type": "string", "description": "站内信 ID。"},
        },
        "required": ["id"],
    }


@register_skill
class GetMessageListSkill(_SuncidiToolBase):
    name = "get_message_list"
    endpoint = "/message/list"
    method = "POST"
    requires_auth = True
    description = "查询消息中心列表。用于拉取用户消息列表并按类型分页。"
    parameters = GetInboxListSkill.parameters


@register_skill
class GetMessageTypeListSkill(_SuncidiToolBase):
    name = "get_message_type_list"
    endpoint = "/message/type/list"
    method = "GET"
    requires_auth = True
    description = "查询消息类型列表。用于将消息类型码映射为可读文案。"
    parameters = {
        "type": "object",
        "properties": {},
    }


@register_skill
class GetMessageDetailSkill(_SuncidiToolBase):
    name = "get_message_detail"
    endpoint = "/message/{id}"
    method = "GET"
    requires_auth = True
    description = "查询消息详情。用于读取指定消息正文。"
    path_params = ("id",)
    parameters = {
        "type": "object",
        "properties": {
            "id": {"type": "string", "description": "消息 ID。"},
        },
        "required": ["id"],
    }


@register_skill
class GetKfListSkill(_SuncidiToolBase):
    name = "get_kf_list"
    endpoint = "/message/kf/list"
    method = "GET"
    requires_auth = False
    description = "查询客服列表。用于获取在线客服入口或分流信息。"
    parameters = {
        "type": "object",
        "properties": {},
    }


@register_skill
class GetKfGroupListSkill(_SuncidiToolBase):
    name = "get_kf_group_list"
    endpoint = "/message/kf/group/list/{type}"
    method = "GET"
    requires_auth = False
    description = "按类型查询客服分组列表。用于路由到指定客服组。"
    path_params = ("type",)
    parameters = {
        "type": "object",
        "properties": {
            "type": {"type": "string", "description": "客服分组类型。"},
        },
        "required": ["type"],
    }


@register_skill
class GetPromoMoneyLogListSkill(_SuncidiToolBase):
    name = "get_promo_money_log_list"
    endpoint = "/money/log"
    method = "POST"
    requires_auth = True
    description = "查询优惠记录列表。用于按时间范围查看礼金/活动领取记录。"
    parameters = {
        "type": "object",
        "properties": {
            "createTimeFrom": {"type": "integer", "description": "开始时间（毫秒时间戳）。"},
            "createTimeTo": {"type": "integer", "description": "结束时间（毫秒时间戳）。"},
            "moneyType": {
                "type": "string",
                "description": "活动类型，例如 vip/birthday/week/month。",
            },
            "page": {"type": "integer", "description": "页码，从 1 开始。"},
            "pageSize": {"type": "integer", "description": "每页数量。"},
            "status": {
                "type": "integer",
                "description": "状态：-2 全部，-1 已过期，0 待领取，1 已领取，3 已取消。",
                "enum": [-2, -1, 0, 1, 3],
            },
        },
        "required": ["createTimeFrom", "createTimeTo", "moneyType", "page", "pageSize", "status"],
    }


@register_skill
class GetPromoMoneyStatsSkill(_SuncidiToolBase):
    name = "get_promo_money_stats"
    endpoint = "/money/stats"
    method = "POST"
    requires_auth = True
    description = "查询优惠记录统计。用于汇总优惠领取状态与金额。"
    parameters = GetPromoMoneyLogListSkill.parameters


@register_skill
class GetTransferLogListSkill(_SuncidiToolBase):
    name = "get_transfer_log_list"
    endpoint = "/transfer/log/list"
    method = "POST"
    requires_auth = True
    description = "查询转账记录。用于核对转入/转出流水及状态。"
    parameters = {
        "type": "object",
        "properties": {
            "page": {"type": "integer", "description": "页码，从 1 开始。"},
            "pageSize": {"type": "integer", "description": "每页数量。"},
            "status": {
                "type": "integer",
                "description": "状态：-1 全部，0 未支付，1 已支付。",
                "enum": [-1, 0, 1],
            },
            "timeFrom": {"type": "integer", "description": "开始时间（毫秒时间戳）。"},
            "timeTo": {"type": "integer", "description": "结束时间（毫秒时间戳）。"},
            "type": {
                "type": "integer",
                "description": "转账类型：-1 全部，0 转入，1 转出。",
                "enum": [-1, 0, 1],
            },
        },
        "required": ["page", "pageSize", "status", "timeFrom", "timeTo", "type"],
    }


@register_skill
class GetCashFlowListSkill(_SuncidiToolBase):
    name = "get_cash_flow_list"
    endpoint = "/cash/list"
    method = "POST"
    requires_auth = True
    description = "查询资金流水记录。用于按时间和流水类型排查账变。"
    parameters = {
        "type": "object",
        "properties": {
            "createTimeFrom": {"type": "integer", "description": "开始时间（毫秒时间戳）。"},
            "createTimeTo": {"type": "integer", "description": "结束时间（毫秒时间戳）。"},
            "page": {"type": "integer", "description": "页码，从 1 开始。"},
            "pageSize": {"type": "integer", "description": "每页数量。"},
            "type": {"type": "string", "description": "流水类型。"},
        },
        "required": ["createTimeFrom", "createTimeTo", "page", "pageSize", "type"],
    }


@register_skill
class GetCashTypeListSkill(_SuncidiToolBase):
    name = "get_cash_type_list"
    endpoint = "/cash/type/list"
    method = "GET"
    requires_auth = False
    description = "查询流水类型列表。用于将流水类型码映射为可读文案。"
    parameters = {
        "type": "object",
        "properties": {},
    }


@register_skill
class GetRebateCurrentStatsSkill(_SuncidiToolBase):
    name = "get_rebate_current_stats"
    endpoint = "/rebate/current"
    method = "GET"
    requires_auth = True
    description = "查询当前洗码统计。用于查看可领/累计返水概览。"
    parameters = {
        "type": "object",
        "properties": {},
    }


@register_skill
class GetRebateLogListSkill(_SuncidiToolBase):
    name = "get_rebate_log_list"
    endpoint = "/rebate/log/list"
    method = "POST"
    requires_auth = True
    description = "查询洗码记录列表。用于核对返水发放明细。"
    parameters = {
        "type": "object",
        "properties": {
            "createTimeFrom": {"type": "integer", "description": "开始时间（毫秒时间戳）。"},
            "createTimeTo": {"type": "integer", "description": "结束时间（毫秒时间戳）。"},
            "page": {"type": "integer", "description": "页码，从 1 开始。"},
            "pageSize": {"type": "integer", "description": "每页数量。"},
        },
        "required": ["createTimeFrom", "createTimeTo", "page", "pageSize"],
    }


@register_skill
class GetRebateStatsSkill(_SuncidiToolBase):
    name = "get_rebate_stats"
    endpoint = "/rebate/stats"
    method = "POST"
    requires_auth = True
    description = "查询洗码统计。用于汇总返水金额与状态。"
    parameters = GetRebateLogListSkill.parameters


@register_skill
class GetRebateDetailSkill(_SuncidiToolBase):
    name = "get_rebate_detail"
    endpoint = "/rebate/detail/{id}"
    method = "GET"
    requires_auth = True
    description = "查询洗码详情。用于定位单条返水记录详情。"
    path_params = ("id",)
    parameters = {
        "type": "object",
        "properties": {
            "id": {"type": "string", "description": "洗码记录 ID。"},
        },
        "required": ["id"],
    }


@register_skill
class GetVipDescInfoSkill(_SuncidiToolBase):
    name = "get_vip_desc_info"
    endpoint = "/vip/desc/info"
    method = "GET"
    requires_auth = False
    description = "查询 VIP 条款与规则。用于回答 VIP 升级条件与权益规则。"
    parameters = {
        "type": "object",
        "properties": {},
    }


@register_skill
class GetVipListSkill(_SuncidiToolBase):
    name = "get_vip_list"
    endpoint = "/vip/list"
    method = "GET"
    requires_auth = False
    description = "查询 VIP 特权信息列表。用于展示各等级权益。"
    parameters = {
        "type": "object",
        "properties": {},
    }


@register_skill
class GetEventDetailSkill(_SuncidiToolBase):
    name = "get_event_detail"
    endpoint = "/event/info/{id}"
    method = "GET"
    requires_auth = False
    description = "查询活动详情。用于回答某个活动的规则、时间、门槛。"
    path_params = ("id",)
    parameters = {
        "type": "object",
        "properties": {
            "id": {"type": "string", "description": "活动 ID。"},
        },
        "required": ["id"],
    }


@register_skill
class GetPrizeLogListSkill(_SuncidiToolBase):
    name = "get_prize_log_list"
    endpoint = "/prize/log"
    method = "GET"
    requires_auth = True
    description = "查询用户中奖记录。用于活动中奖核对。"
    parameters = {
        "type": "object",
        "properties": {},
    }


@register_skill
class GetFavoriteGameIdsSkill(_SuncidiToolBase):
    name = "get_favorite_game_ids"
    endpoint = "/game/favorite/game/list"
    method = "GET"
    requires_auth = True
    description = "查询用户收藏的游戏 ID 列表。用于个性化推荐或收藏排查。"
    parameters = {
        "type": "object",
        "properties": {},
    }


@register_skill
class GetFavoriteGameIdsByTypeSkill(_SuncidiToolBase):
    name = "get_favorite_game_ids_by_type"
    endpoint = "/game/favorite/game/list/{gameTypeId}"
    method = "GET"
    requires_auth = True
    description = "按游戏类型查询用户收藏的游戏 ID 列表。"
    path_params = ("gameTypeId",)
    parameters = {
        "type": "object",
        "properties": {
            "gameTypeId": {"type": "string", "description": "游戏类型 ID。"},
        },
        "required": ["gameTypeId"],
    }


def _swagger_slug(text: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9_]+", "_", (text or "").strip().lower())
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "api"


def _swagger_tool_name(method: str, path: str, operation_id: str = "") -> str:
    base = _swagger_slug(operation_id) if operation_id else ""
    if not base:
        parts = re.sub(r"\{([^}]+)\}", r"by_\1", path or "/").strip("/").split("/")
        parts = [_swagger_slug(p) for p in parts if p]
        base = "_".join([_swagger_slug(method)] + parts)
    name = f"{SWAGGER_TOOL_PREFIX}{base}"
    if len(name) <= 64:
        return name
    digest = hashlib.md5(name.encode("utf-8")).hexdigest()[:8]
    keep = max(8, 64 - len(SWAGGER_TOOL_PREFIX) - len(digest) - 1)
    return f"{SWAGGER_TOOL_PREFIX}{base[:keep]}_{digest}"


def _swagger_resolve_ref(ref: str, definitions: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(ref, str) or not ref.startswith("#/definitions/"):
        return {}
    key = ref.split("/")[-1]
    node = definitions.get(key)
    return node if isinstance(node, dict) else {}


def _swagger_schema_to_json(node: Dict[str, Any], definitions: Dict[str, Any], depth: int = 0, seen: Optional[Set[str]] = None) -> Dict[str, Any]:
    if not isinstance(node, dict):
        return {"type": "string"}
    if depth > 6:
        return {"type": "object"}
    seen = seen or set()

    if "$ref" in node:
        ref = str(node.get("$ref") or "")
        if ref in seen:
            return {"type": "object"}
        target = _swagger_resolve_ref(ref, definitions)
        if not target:
            return {"type": "object"}
        return _swagger_schema_to_json(target, definitions, depth + 1, seen | {ref})

    t = str(node.get("type") or "")
    out: Dict[str, Any] = {}

    if t in {"string", "integer", "number", "boolean"}:
        out["type"] = t
        if node.get("enum") is not None:
            out["enum"] = node.get("enum")
        if node.get("format"):
            out["format"] = node.get("format")
        if node.get("description"):
            out["description"] = str(node.get("description"))
        return out

    if t == "array":
        out["type"] = "array"
        out["items"] = _swagger_schema_to_json(node.get("items") or {}, definitions, depth + 1, seen)
        if node.get("description"):
            out["description"] = str(node.get("description"))
        return out

    # default object
    out["type"] = "object"
    props: Dict[str, Any] = {}
    required = node.get("required") or []
    if isinstance(required, list):
        out["required"] = [str(x) for x in required if str(x)]
    for key, val in (node.get("properties") or {}).items():
        props[str(key)] = _swagger_schema_to_json(val if isinstance(val, dict) else {}, definitions, depth + 1, seen)
    if props:
        out["properties"] = props
    if node.get("description"):
        out["description"] = str(node.get("description"))
    if node.get("enum") is not None:
        out["enum"] = node.get("enum")
    return out


def _swagger_parameter_to_json(param: Dict[str, Any], definitions: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(param, dict):
        return {"type": "string"}
    schema = param.get("schema")
    if isinstance(schema, dict):
        out = _swagger_schema_to_json(schema, definitions)
    else:
        typ = str(param.get("type") or "string")
        out = {"type": typ if typ in {"string", "integer", "number", "boolean", "array", "object"} else "string"}
        if out["type"] == "array":
            out["items"] = _swagger_schema_to_json(param.get("items") or {}, definitions)
        if param.get("enum") is not None:
            out["enum"] = param.get("enum")
        if param.get("format"):
            out["format"] = str(param.get("format"))
    if param.get("description") and "description" not in out:
        out["description"] = str(param.get("description"))
    return out


def _swagger_build_parameters_schema(path_item: Dict[str, Any], op: Dict[str, Any], definitions: Dict[str, Any]) -> Dict[str, Any]:
    schema: Dict[str, Any] = {"type": "object", "properties": {}}
    required: List[str] = []
    merged_params: List[Dict[str, Any]] = []
    for source in (path_item.get("parameters") or [], op.get("parameters") or []):
        if isinstance(source, dict):
            merged_params.append(source)
    for p in merged_params:
        pin = str(p.get("in") or "").lower()
        name = str(p.get("name") or "").strip()
        if pin == "body":
            body_schema = p.get("schema")
            body_json = _swagger_schema_to_json(body_schema if isinstance(body_schema, dict) else {}, definitions)
            if body_json.get("type") == "object":
                for k, v in (body_json.get("properties") or {}).items():
                    schema["properties"][str(k)] = v
                for k in body_json.get("required", []) or []:
                    if str(k) not in required:
                        required.append(str(k))
            else:
                schema["properties"]["body"] = body_json
                if p.get("required") is True and "body" not in required:
                    required.append("body")
            continue
        if pin not in {"query", "path", "header", "formData"}:
            continue
        if not name:
            continue
        schema["properties"][name] = _swagger_parameter_to_json(p, definitions)
        if p.get("required") is True and name not in required:
            required.append(name)
    if required:
        schema["required"] = required
    return schema


def _swagger_load_doc() -> Dict[str, Any]:
    def _load_json_file(path: str) -> Dict[str, Any]:
        if not path:
            return {}
        p = Path(path).expanduser()
        if not p.exists():
            return {}
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def _save_json_file(path: str, data: Dict[str, Any]) -> None:
        if not path or not isinstance(data, dict):
            return
        p = Path(path).expanduser()
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        except Exception:
            pass

    if SWAGGER_DOC_PATH:
        data = _load_json_file(SWAGGER_DOC_PATH)
        if data:
            return data
    if not SWAGGER_DOC_URL:
        return _load_json_file(SWAGGER_DOC_CACHE_PATH)
    try:
        resp = SESSION.get(SWAGGER_DOC_URL, timeout=_timeout_s())
        if resp.status_code >= 400:
            return _load_json_file(SWAGGER_DOC_CACHE_PATH)
        data = resp.json()
        if isinstance(data, dict):
            _save_json_file(SWAGGER_DOC_CACHE_PATH, data)
            return data
    except Exception:
        return _load_json_file(SWAGGER_DOC_CACHE_PATH)
    return _load_json_file(SWAGGER_DOC_CACHE_PATH)


def _existing_endpoint_methods() -> Set[Tuple[str, str]]:
    pairs: Set[Tuple[str, str]] = set()
    for val in list(globals().values()):
        if isinstance(val, type) and issubclass(val, _SuncidiToolBase):
            endpoint = str(getattr(val, "endpoint", "") or "")
            method = str(getattr(val, "method", "GET") or "GET").upper()
            if endpoint:
                pairs.add((method, endpoint))
    return pairs


def _should_require_auth(endpoint: str) -> bool:
    # Keep default strict auth for user-facing business APIs.
    if endpoint in {"/ping"}:
        return False
    if endpoint.startswith("/overlay/") or endpoint.startswith("/sponsorship/"):
        return False
    return True


def _register_swagger_tools_from_doc() -> int:
    if not SWAGGER_AUTO_TOOLS_ENABLED:
        return 0
    doc = _swagger_load_doc()
    if not doc:
        return 0

    paths = doc.get("paths") or {}
    definitions = doc.get("definitions") or {}
    if not isinstance(paths, dict) or not isinstance(definitions, dict):
        return 0

    existing = _existing_endpoint_methods()
    added = 0

    for raw_path, path_item in paths.items():
        if not isinstance(path_item, dict):
            continue
        normalized_path = _normalize_swagger_path(str(raw_path))
        for method in ("get", "post", "put", "delete", "patch"):
            op = path_item.get(method)
            if not isinstance(op, dict):
                continue
            method_u = method.upper()
            if (method_u, normalized_path) in existing:
                continue
            operation_id = str(op.get("operationId") or "")
            tool_name = _swagger_tool_name(method_u, normalized_path, operation_id=operation_id)
            if get_skill(tool_name) is not None:
                continue
            summary = str(op.get("summary") or op.get("description") or "").strip()
            description = summary or f"{method_u} {normalized_path}"
            parameters = _swagger_build_parameters_schema(path_item, op, definitions)
            attrs = {
                "name": tool_name,
                "endpoint": normalized_path,
                "method": method_u,
                "requires_auth": _should_require_auth(normalized_path),
                "description": f"[Swagger] {description}",
                "parameters": parameters,
                "path_params": _extract_path_params(normalized_path),
            }
            cls_name = "SuncidiAuto" + "".join(x.capitalize() for x in _swagger_slug(tool_name).split("_"))
            cls = type(cls_name, (_SuncidiToolBase,), attrs)
            register_skill(cls)
            existing.add((method_u, normalized_path))
            added += 1
            if added >= SWAGGER_MAX_TOOLS:
                return added
    return added


try:
    _added_swagger_tools = _register_swagger_tools_from_doc()
    if _added_swagger_tools:
        print(f"[SUNCIDI] swagger auto-tools loaded: {_added_swagger_tools}", flush=True)
except Exception as _swagger_register_err:
    print(f"[SUNCIDI] swagger auto-tools load failed: {_swagger_register_err}", flush=True)
