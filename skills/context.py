from __future__ import annotations

from contextvars import ContextVar, Token
from typing import Any, Dict, Optional


_TOOL_CONTEXT: ContextVar[Dict[str, Any]] = ContextVar("tool_context", default={})


def set_tool_context(ctx: Optional[Dict[str, Any]]) -> Token:
    return _TOOL_CONTEXT.set(dict(ctx or {}))


def reset_tool_context(token: Token) -> None:
    _TOOL_CONTEXT.reset(token)


def get_tool_context() -> Dict[str, Any]:
    return dict(_TOOL_CONTEXT.get() or {})


def get_user_access_token() -> str:
    ctx = _TOOL_CONTEXT.get() or {}
    return str(ctx.get("user_access_token") or "").strip()
