#!/usr/bin/env python3
"""External OpenAI-compatible API gateway for GPU5/GPU7 public access.

The gateway is intentionally small:
- authenticate external API keys;
- enforce per-key model permissions and RPM limits;
- expose stable model aliases;
- route requests to internal GPU5/GPU7 guardrails proxies;
- write metadata-only audit logs.
"""

from __future__ import annotations

import json
import os
import threading
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse


DEFAULT_MODEL_ROUTES = {
    "baowang-gpu5": {
        "base_url": "http://127.0.0.1:8001/v1",
        "upstream_model": "gpu5-v5.1f-merged",
        "display_name": "Baowang GPU5",
        "allow_agent": True,
    },
    "baowang-gpu7": {
        "base_url": "http://127.0.0.1:8010/v1",
        "upstream_model": "gpu7-v5-combined",
        "display_name": "Baowang GPU7",
        "allow_agent": True,
    },
}

APP_VERSION = "1.0.0"
GATEWAY_MODEL_ROUTES_JSON = os.getenv("GATEWAY_MODEL_ROUTES_JSON", "").strip()
GATEWAY_API_KEYS_JSON = os.getenv("GATEWAY_API_KEYS_JSON", "").strip()
GATEWAY_API_KEYS_FILE = os.getenv("GATEWAY_API_KEYS_FILE", "").strip()
GATEWAY_DEFAULT_RPM = int(os.getenv("GATEWAY_DEFAULT_RPM", "60") or "60")
GATEWAY_TIMEOUT_S = float(os.getenv("GATEWAY_TIMEOUT_S", "180") or "180")
GATEWAY_MODEL_SYSTEM_PROMPTS_JSON = os.getenv("GATEWAY_MODEL_SYSTEM_PROMPTS_JSON", "").strip()
GATEWAY_MODEL_SYSTEM_PROMPTS_FILE = os.getenv("GATEWAY_MODEL_SYSTEM_PROMPTS_FILE", "").strip()
GATEWAY_AUDIT_LOG_JSONL = os.getenv(
    "GATEWAY_AUDIT_LOG_JSONL",
    "/video-storage/ai-customer/qwen35a3b_finetune/runtime/external_api_gateway/audit.jsonl",
).strip()


app = FastAPI(title="qwen-external-api-gateway", version=APP_VERSION)
SESSION = requests.Session()
_RATE_LIMIT_LOCK = threading.Lock()
_RATE_LIMIT_WINDOWS: Dict[str, deque] = defaultdict(deque)
_AUDIT_LOCK = threading.Lock()


def _load_json_from_env_or_file(env_json: str, file_path: str, default: Any) -> Any:
    if env_json:
        try:
            return json.loads(env_json)
        except Exception as exc:
            print(f"[gateway] invalid JSON env: {exc}", flush=True)
    if file_path:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"[gateway] config file not found: {file_path}", flush=True)
        except Exception as exc:
            print(f"[gateway] invalid config file {file_path}: {exc}", flush=True)
    return default


def _load_model_routes() -> Dict[str, Dict[str, Any]]:
    raw = _load_json_from_env_or_file(GATEWAY_MODEL_ROUTES_JSON, "", DEFAULT_MODEL_ROUTES)
    if not isinstance(raw, dict):
        return dict(DEFAULT_MODEL_ROUTES)
    routes: Dict[str, Dict[str, Any]] = {}
    for alias, route in raw.items():
        if not isinstance(route, dict):
            continue
        base_url = str(route.get("base_url") or "").rstrip("/")
        upstream_model = str(route.get("upstream_model") or "").strip()
        if not alias or not base_url or not upstream_model:
            continue
        routes[str(alias)] = {
            "base_url": base_url,
            "upstream_model": upstream_model,
            "display_name": str(route.get("display_name") or alias),
            "allow_agent": bool(route.get("allow_agent", True)),
            "backend_api_key": str(route.get("backend_api_key") or ""),
        }
    return routes or dict(DEFAULT_MODEL_ROUTES)


def _normalize_key_rows(raw: Any) -> List[Dict[str, Any]]:
    if isinstance(raw, dict) and isinstance(raw.get("keys"), list):
        return [x for x in raw["keys"] if isinstance(x, dict)]
    if isinstance(raw, list):
        return [x for x in raw if isinstance(x, dict)]
    if isinstance(raw, dict):
        rows = []
        for key_id, spec in raw.items():
            if isinstance(spec, str):
                rows.append({"id": str(key_id), "key": spec})
            elif isinstance(spec, dict):
                rows.append({"id": str(key_id), **spec})
        return rows
    return []


def _load_api_keys() -> Dict[str, Dict[str, Any]]:
    raw = _load_json_from_env_or_file(GATEWAY_API_KEYS_JSON, GATEWAY_API_KEYS_FILE, {"keys": []})
    registry: Dict[str, Dict[str, Any]] = {}
    for row in _normalize_key_rows(raw):
        token = str(row.get("key") or row.get("token") or "").strip()
        if not token:
            continue
        key_id = str(row.get("id") or f"key_{len(registry) + 1}").strip()
        models = row.get("models") or list(MODEL_ROUTES.keys())
        routes = row.get("routes") or ["chat", "agent", "models"]
        registry[token] = {
            "id": key_id,
            "models": set(str(x) for x in models),
            "routes": set(str(x) for x in routes),
            "rpm": int(row.get("rpm") or GATEWAY_DEFAULT_RPM),
        }
    return registry


MODEL_ROUTES = _load_model_routes()
API_KEYS = _load_api_keys()


def _load_model_system_prompts() -> Dict[str, str]:
    raw = _load_json_from_env_or_file(
        GATEWAY_MODEL_SYSTEM_PROMPTS_JSON,
        GATEWAY_MODEL_SYSTEM_PROMPTS_FILE,
        {},
    )
    if not isinstance(raw, dict):
        return {}
    prompts: Dict[str, str] = {}
    for model_name, prompt in raw.items():
        key = str(model_name or "").strip()
        value = str(prompt or "").strip()
        if key and value:
            prompts[key] = value
    if prompts:
        print(f"[gateway] loaded {len(prompts)} alias prompts", flush=True)
    return prompts


MODEL_SYSTEM_PROMPTS = _load_model_system_prompts()


def _extract_token(authorization: Optional[str]) -> str:
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing Authorization header")
    prefix = "Bearer "
    if not authorization.startswith(prefix):
        raise HTTPException(status_code=401, detail="Invalid Authorization header")
    return authorization[len(prefix) :].strip()


def _auth_context(authorization: Optional[str], route: str) -> Dict[str, Any]:
    if not API_KEYS:
        raise HTTPException(status_code=503, detail="Gateway API keys are not configured")
    token = _extract_token(authorization)
    ctx = API_KEYS.get(token)
    if not ctx:
        raise HTTPException(status_code=401, detail="Unauthorized")
    if route not in ctx["routes"]:
        raise HTTPException(status_code=403, detail="API key is not allowed to use this route")
    _rate_limit(ctx)
    return ctx


def _rate_limit(ctx: Dict[str, Any]) -> None:
    rpm = int(ctx.get("rpm") or 0)
    if rpm <= 0:
        return
    now = time.time()
    key_id = str(ctx["id"])
    with _RATE_LIMIT_LOCK:
        window = _RATE_LIMIT_WINDOWS[key_id]
        while window and now - window[0] >= 60:
            window.popleft()
        if len(window) >= rpm:
            raise HTTPException(status_code=429, detail="Rate limit exceeded")
        window.append(now)


def _payload_has_image(payload: Dict[str, Any]) -> bool:
    for message in payload.get("messages") or []:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type") == "image_url":
                    return True
    return False


def _audit(route: str, ctx: Optional[Dict[str, Any]], payload: Optional[Dict[str, Any]], status: str, detail: str = "") -> None:
    if not GATEWAY_AUDIT_LOG_JSONL:
        return
    event = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "route": route,
        "key_id": (ctx or {}).get("id", "unknown"),
        "status": status,
        "detail": detail,
    }
    if isinstance(payload, dict):
        messages = payload.get("messages") or []
        event.update({
            "model": payload.get("model"),
            "message_count": len(messages) if isinstance(messages, list) else 0,
            "has_image": _payload_has_image(payload),
            "stream": bool(payload.get("stream")),
            "max_tokens": payload.get("max_tokens"),
        })
    try:
        path = Path(GATEWAY_AUDIT_LOG_JSONL)
        path.parent.mkdir(parents=True, exist_ok=True)
        with _AUDIT_LOCK:
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(event, ensure_ascii=False) + "\n")
    except Exception as exc:
        print(f"[gateway] audit write failed: {exc}", flush=True)


def _resolve_route(payload: Dict[str, Any], ctx: Dict[str, Any], route_type: str) -> Tuple[str, Dict[str, Any]]:
    model = str(payload.get("model") or "").strip()
    if not model:
        raise HTTPException(status_code=400, detail="Missing model")
    route = MODEL_ROUTES.get(model)
    if not route:
        raise HTTPException(status_code=404, detail=f"Unknown model alias: {model}")
    if model not in ctx["models"]:
        raise HTTPException(status_code=403, detail=f"API key is not allowed to use model: {model}")
    if route_type == "agent" and not route.get("allow_agent", True):
        raise HTTPException(status_code=403, detail=f"Model does not allow agent route: {model}")
    return model, route


def _backend_headers(route: Dict[str, Any]) -> Dict[str, str]:
    headers = {"Content-Type": "application/json"}
    backend_key = str(route.get("backend_api_key") or "")
    if backend_key:
        headers["Authorization"] = f"Bearer {backend_key}"
    return headers


def _inject_model_system_prompt(payload: Dict[str, Any], public_model: str) -> Tuple[Dict[str, Any], bool]:
    prompt = MODEL_SYSTEM_PROMPTS.get(str(public_model or "").strip(), "")
    messages = list(payload.get("messages") or [])
    if not prompt or not messages:
        return payload, False
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        if msg.get("role") != "system":
            continue
        content = msg.get("content")
        if isinstance(content, str) and content.strip():
            return payload, False
        if content:
            return payload, False
    new_payload = {**payload, "messages": [{"role": "system", "content": prompt}] + messages}
    print(f"[gateway] injected system prompt for alias={public_model}", flush=True)
    return new_payload, True


def _rewrite_completion_response_model(data: Any, public_model: str) -> Any:
    if not public_model or not isinstance(data, dict):
        return data
    rewritten = dict(data)
    rewritten["model"] = public_model
    return rewritten


def _rewrite_sse_model_line(line: bytes, public_model: str) -> bytes:
    if not public_model or not line.startswith(b"data: "):
        return line
    raw = line[6:]
    if raw.strip() == b"[DONE]":
        return line
    try:
        data = json.loads(raw.decode("utf-8"))
    except Exception:
        return line
    if not isinstance(data, dict):
        return line
    data["model"] = public_model
    return b"data: " + json.dumps(data, ensure_ascii=False).encode("utf-8")


def _forward(payload: Dict[str, Any], route: Dict[str, Any], endpoint: str, public_model: str) -> Response:
    backend_payload = dict(payload)
    backend_payload["model"] = route["upstream_model"]
    stream = bool(payload.get("stream"))
    try:
        resp = SESSION.post(
            f"{route['base_url']}/{endpoint.lstrip('/')}",
            json=backend_payload,
            headers=_backend_headers(route),
            timeout=GATEWAY_TIMEOUT_S,
            stream=stream,
        )
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"Backend request failed: {exc}") from exc
    content_type = resp.headers.get("content-type", "application/json")
    if stream and resp.status_code < 400 and "text/event-stream" in content_type.lower():
        def _stream_generator():
            for line in resp.iter_lines():
                if line:
                    yield _rewrite_sse_model_line(line, public_model) + b"\n"
        return StreamingResponse(
            _stream_generator(),
            status_code=resp.status_code,
            media_type=content_type,
        )
    if "application/json" in content_type.lower():
        try:
            data = resp.json()
            data = _rewrite_completion_response_model(data, public_model)
            return JSONResponse(status_code=resp.status_code, content=data)
        except Exception:
            pass
    return Response(content=resp.content, status_code=resp.status_code, media_type=content_type)


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "ok": True,
        "version": APP_VERSION,
        "models": list(MODEL_ROUTES.keys()),
        "api_keys_configured": bool(API_KEYS),
        "audit_log": GATEWAY_AUDIT_LOG_JSONL,
    }


@app.get("/v1/models")
def models(authorization: Optional[str] = Header(default=None)) -> JSONResponse:
    ctx = _auth_context(authorization, "models")
    rows = []
    now = int(time.time())
    for alias, route in MODEL_ROUTES.items():
        if alias not in ctx["models"]:
            continue
        rows.append({
            "id": alias,
            "object": "model",
            "created": now,
            "owned_by": "baowang-api-gateway",
            "root": alias,
            "parent": None,
            "name": route["display_name"],
        })
    _audit("/v1/models", ctx, None, "ok")
    return JSONResponse({"object": "list", "data": rows})


@app.post("/v1/chat/completions")
async def chat_completions(request: Request, authorization: Optional[str] = Header(default=None)) -> Response:
    ctx = _auth_context(authorization, "chat")
    try:
        payload = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON payload")
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Payload must be a JSON object")
    alias, route = _resolve_route(payload, ctx, "chat")
    _audit("/v1/chat/completions", ctx, payload, "accepted", alias)
    payload, _system_prompt_injected = _inject_model_system_prompt(payload, alias)
    return _forward(payload, route, "chat/completions", alias)


@app.post("/v1/chat/completions/agent")
async def chat_completions_agent(request: Request, authorization: Optional[str] = Header(default=None)) -> Response:
    ctx = _auth_context(authorization, "agent")
    try:
        payload = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON payload")
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Payload must be a JSON object")
    alias, route = _resolve_route(payload, ctx, "agent")
    _audit("/v1/chat/completions/agent", ctx, payload, "accepted", alias)
    payload, _system_prompt_injected = _inject_model_system_prompt(payload, alias)
    return _forward(payload, route, "chat/completions/agent", alias)
