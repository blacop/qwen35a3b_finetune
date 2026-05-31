#!/usr/bin/env python3
import hashlib
import json
import os
import time
from collections import OrderedDict, deque
from copy import deepcopy
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple

import requests
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from retrieval_hybrid import HybridRetriever, build_context_block


def _as_bool(raw: Optional[str], default: bool) -> bool:
    if raw is None:
        return default
    v = str(raw).strip().lower()
    if v in {"1", "true", "yes", "y", "on"}:
        return True
    if v in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _as_int(raw: Optional[str], default: int, min_value: int, max_value: int) -> int:
    try:
        v = int(str(raw).strip()) if raw is not None else default
    except Exception:
        v = default
    if v < min_value:
        return min_value
    if v > max_value:
        return max_value
    return v


def _as_float(raw: Optional[str], default: float, min_value: float, max_value: float) -> float:
    try:
        v = float(str(raw).strip()) if raw is not None else default
    except Exception:
        v = default
    if v < min_value:
        return min_value
    if v > max_value:
        return max_value
    return v


def _message_to_text(message: Dict[str, Any]) -> str:
    content = message.get("content")
    if isinstance(content, str) and content.strip():
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                txt = item.get("text")
                if isinstance(txt, str) and txt.strip():
                    parts.append(txt.strip())
        if parts:
            return "\n".join(parts)
    reasoning = message.get("reasoning")
    if isinstance(reasoning, str) and reasoning.strip():
        return reasoning
    return ""


def call_chat_completion(
    base_url: str,
    model: str,
    messages: List[Dict[str, str]],
    temperature: float,
    max_tokens: Optional[int],
    api_key: Optional[str],
    timeout_s: float,
) -> Dict[str, Any]:
    url = base_url.rstrip("/") + "/chat/completions"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "stream": False,
    }
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    if not LLM_ENABLE_THINKING:
        payload.setdefault("chat_template_kwargs", {})["enable_thinking"] = False

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=timeout_s)
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"LLM request failed: {exc}") from exc

    if resp.status_code >= 400:
        raise HTTPException(status_code=502, detail=f"LLM error {resp.status_code}: {resp.text[:500]}")

    data = resp.json()
    choices = data.get("choices") or []
    if not choices:
        raise HTTPException(status_code=502, detail="LLM response has no choices")

    message = choices[0].get("message") or {}
    content = _message_to_text(message)
    return {
        "raw": data,
        "content": content,
        "finish_reason": choices[0].get("finish_reason"),
        "usage": data.get("usage", {}),
    }


class RetrieveRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)
    min_score: float = Field(default=0.01, ge=0.0, le=1.0)
    min_score_vec: float = Field(default=0.0, ge=-1.0, le=1.0)
    retrieval_mode: str = Field(default="hybrid")
    fusion: str = Field(default="rrf")
    candidate_k: int = Field(default=40, ge=1, le=300)
    rrf_k: int = Field(default=60, ge=1, le=2000)
    weight_lex: float = Field(default=0.55, ge=0.0, le=1.0)
    weight_vec: float = Field(default=0.45, ge=0.0, le=1.0)
    enable_rerank: bool = True


class AnswerRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)
    min_score: float = Field(default=0.01, ge=0.0, le=1.0)
    min_score_vec: float = Field(default=0.0, ge=-1.0, le=1.0)
    retrieval_mode: str = Field(default="hybrid")
    fusion: str = Field(default="rrf")
    candidate_k: int = Field(default=40, ge=1, le=300)
    rrf_k: int = Field(default=60, ge=1, le=2000)
    weight_lex: float = Field(default=0.55, ge=0.0, le=1.0)
    weight_vec: float = Field(default=0.45, ge=0.0, le=1.0)
    enable_rerank: bool = True
    max_context_chars: int = Field(default=600, ge=100, le=3000)
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(default=512, ge=1, le=4096)
    model: Optional[str] = None
    llm_base_url: Optional[str] = None
    api_key: Optional[str] = None
    include_context: bool = False
    system_prompt: Optional[str] = None


class RagChatCompletionRequest(BaseModel):
    model: Optional[str] = None
    messages: List[Dict[str, Any]] = Field(..., min_length=1)
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(default=512, ge=1, le=4096)
    top_k: int = Field(default=5, ge=1, le=20)
    min_score: float = Field(default=0.01, ge=0.0, le=1.0)
    min_score_vec: float = Field(default=0.0, ge=-1.0, le=1.0)
    retrieval_mode: str = Field(default="hybrid")
    fusion: str = Field(default="rrf")
    candidate_k: int = Field(default=40, ge=1, le=300)
    rrf_k: int = Field(default=60, ge=1, le=2000)
    weight_lex: float = Field(default=0.55, ge=0.0, le=1.0)
    weight_vec: float = Field(default=0.45, ge=0.0, le=1.0)
    enable_rerank: bool = True
    max_context_chars: int = Field(default=600, ge=100, le=3000)
    include_context: bool = False
    llm_base_url: Optional[str] = None
    api_key: Optional[str] = None
    system_prompt: Optional[str] = None


INDEX_DIR = os.getenv("RAG_INDEX_DIR", "/home/ubuntu/generate/tydata_rag/index")
DEFAULT_LLM_BASE = os.getenv("LLM_API_BASE", "http://127.0.0.1:8014/v1")
DEFAULT_LLM_MODEL = os.getenv("LLM_MODEL", "")
DEFAULT_LLM_API_KEY = os.getenv("LLM_API_KEY", "")
DEFAULT_RETRIEVAL_MODE = os.getenv("RAG_DEFAULT_MODE", "hybrid")
DEFAULT_FUSION = os.getenv("RAG_DEFAULT_FUSION", "rrf")
DEFAULT_ENABLE_RERANK = _as_bool(os.getenv("RAG_DEFAULT_ENABLE_RERANK", "1"), True)
DEFAULT_TOP_K = _as_int(os.getenv("RAG_DEFAULT_TOP_K", "5"), 5, 1, 100)
DEFAULT_MIN_SCORE = _as_float(os.getenv("RAG_DEFAULT_MIN_SCORE", "0.01"), 0.01, 0.0, 1.0)
DEFAULT_MIN_SCORE_VEC = _as_float(os.getenv("RAG_DEFAULT_MIN_SCORE_VEC", "0.0"), 0.0, -1.0, 1.0)
DEFAULT_CANDIDATE_K = _as_int(os.getenv("RAG_DEFAULT_CANDIDATE_K", "40"), 40, 1, 300)
DEFAULT_RRF_K = _as_int(os.getenv("RAG_DEFAULT_RRF_K", "60"), 60, 1, 2000)
DEFAULT_WEIGHT_LEX = _as_float(os.getenv("RAG_DEFAULT_WEIGHT_LEX", "0.55"), 0.55, 0.0, 1.0)
DEFAULT_WEIGHT_VEC = _as_float(os.getenv("RAG_DEFAULT_WEIGHT_VEC", "0.45"), 0.45, 0.0, 1.0)
DEFAULT_ENABLE_DIVERSITY = _as_bool(os.getenv("RAG_DEFAULT_ENABLE_DIVERSITY", "1"), True)
DEFAULT_MAX_PER_SOURCE = _as_int(os.getenv("RAG_DEFAULT_MAX_PER_SOURCE", "1"), 1, 1, 20)
DEFAULT_MAX_PER_FAMILY = _as_int(os.getenv("RAG_DEFAULT_MAX_PER_FAMILY", "3"), 3, 1, 50)
DEFAULT_DIVERSITY_LAMBDA = _as_float(os.getenv("RAG_DEFAULT_DIVERSITY_LAMBDA", "0.7"), 0.7, 0.0, 1.0)
DEFAULT_SOURCE_MATCH_WEIGHT = _as_float(os.getenv("RAG_DEFAULT_SOURCE_MATCH_WEIGHT", "0.0"), 0.0, 0.0, 1.0)
AGG_ENABLE_DIVERSITY = _as_bool(os.getenv("RAG_AGG_ENABLE_DIVERSITY", "1"), True)
AGG_MAX_PER_SOURCE = _as_int(os.getenv("RAG_AGG_MAX_PER_SOURCE", "1"), 1, 1, 20)
AGG_MAX_PER_FAMILY = _as_int(os.getenv("RAG_AGG_MAX_PER_FAMILY", "4"), 4, 1, 50)
AGG_DIVERSITY_LAMBDA = _as_float(os.getenv("RAG_AGG_DIVERSITY_LAMBDA", "0.5"), 0.5, 0.0, 1.0)
AGG_SOURCE_MATCH_WEIGHT = _as_float(os.getenv("RAG_AGG_SOURCE_MATCH_WEIGHT", "0.12"), 0.12, 0.0, 1.0)
LOCK_RETRIEVAL_PARAMS = _as_bool(os.getenv("RAG_LOCK_RETRIEVAL_PARAMS", "1"), True)
RETRIEVE_CACHE_ENABLED = _as_bool(os.getenv("RAG_RETRIEVE_CACHE_ENABLED", "1"), True)
RETRIEVE_CACHE_TTL_S = _as_float(os.getenv("RAG_RETRIEVE_CACHE_TTL_S", "30"), 30.0, 0.0, 3600.0)
RETRIEVE_CACHE_MAX_ITEMS = _as_int(os.getenv("RAG_RETRIEVE_CACHE_MAX_ITEMS", "2048"), 2048, 0, 200000)

GRAY_ENABLED = _as_bool(os.getenv("RAG_GRAY_ENABLED", "0"), False)
GRAY_RATIO = _as_float(os.getenv("RAG_GRAY_RATIO", "0.0"), 0.0, 0.0, 1.0)
GRAY_CANARY_INDEX_DIR = os.getenv("RAG_GRAY_CANARY_INDEX_DIR", "").strip()
GRAY_STABLE_LABEL = os.getenv("RAG_GRAY_STABLE_LABEL", "stable").strip() or "stable"
GRAY_CANARY_LABEL = os.getenv("RAG_GRAY_CANARY_LABEL", "canary").strip() or "canary"

RAG_API_KEY = os.getenv("RAG_API_KEY", "")
LLM_TIMEOUT_S = float(os.getenv("LLM_TIMEOUT_S", "120"))
LLM_ENABLE_THINKING = _as_bool(os.getenv("LLM_ENABLE_THINKING", "0"), False)


_CACHE_LOCK = Lock()
_RETRIEVE_CACHE: "OrderedDict[str, Tuple[float, Dict[str, Any]]]" = OrderedDict()
_RETRIEVE_LATENCIES_MS = deque(maxlen=1000)
_RETRIEVE_METRICS: Dict[str, Any] = {
    "requests": 0,
    "cache_hits": 0,
    "cache_misses": 0,
    "cache_expired": 0,
    "empty_hits": 0,
    "total_hits": 0,
    "latency_ms_total": 0.0,
}


app = FastAPI(title="tydata-rag-api", version="1.2.0")
retriever = HybridRetriever(INDEX_DIR)
gray_retriever: Optional[HybridRetriever] = None
gray_canary_error = ""

if GRAY_ENABLED and GRAY_RATIO > 0.0:
    if GRAY_CANARY_INDEX_DIR and GRAY_CANARY_INDEX_DIR != INDEX_DIR:
        try:
            gray_retriever = HybridRetriever(GRAY_CANARY_INDEX_DIR)
        except Exception as exc:
            gray_canary_error = str(exc)
            gray_retriever = None
    else:
        gray_canary_error = "RAG_GRAY_CANARY_INDEX_DIR is empty or equals RAG_INDEX_DIR"


def _query_bucket(query: str) -> float:
    digest = hashlib.md5((query or "").encode("utf-8")).hexdigest()
    # Use first 8 hex chars for stable routing bucket [0,1).
    value = int(digest[:8], 16)
    return value / float(0xFFFFFFFF)


def _pick_retriever(query: str, forced_route: Optional[str]) -> Tuple[HybridRetriever, str]:
    route = GRAY_STABLE_LABEL
    forced = (forced_route or "").strip().lower()
    if forced in {GRAY_CANARY_LABEL.lower(), "canary", "gray"} and gray_retriever is not None:
        return gray_retriever, GRAY_CANARY_LABEL
    if forced in {GRAY_STABLE_LABEL.lower(), "stable", "primary"}:
        return retriever, GRAY_STABLE_LABEL

    if not (GRAY_ENABLED and GRAY_RATIO > 0.0 and gray_retriever is not None):
        return retriever, route
    if _query_bucket(query) < GRAY_RATIO:
        return gray_retriever, GRAY_CANARY_LABEL
    return retriever, route


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "ok": True,
        "index_dir": INDEX_DIR,
        "num_chunks": len(retriever.chunks),
        "dense_available": bool(retriever.dense_available),
        "dense_backend": str(retriever.dense_backend),
        "retrieval_params_locked": LOCK_RETRIEVAL_PARAMS,
        "default_top_k": DEFAULT_TOP_K,
        "default_min_score": DEFAULT_MIN_SCORE,
        "default_min_score_vec": DEFAULT_MIN_SCORE_VEC,
        "default_candidate_k": DEFAULT_CANDIDATE_K,
        "default_rrf_k": DEFAULT_RRF_K,
        "default_weight_lex": DEFAULT_WEIGHT_LEX,
        "default_weight_vec": DEFAULT_WEIGHT_VEC,
        "default_retrieval_mode": DEFAULT_RETRIEVAL_MODE,
        "default_fusion": DEFAULT_FUSION,
        "default_enable_rerank": DEFAULT_ENABLE_RERANK,
        "default_enable_diversity": DEFAULT_ENABLE_DIVERSITY,
        "default_max_per_source": DEFAULT_MAX_PER_SOURCE,
        "default_max_per_family": DEFAULT_MAX_PER_FAMILY,
        "default_diversity_lambda": DEFAULT_DIVERSITY_LAMBDA,
        "default_source_match_weight": DEFAULT_SOURCE_MATCH_WEIGHT,
        "agg_enable_diversity": AGG_ENABLE_DIVERSITY,
        "agg_max_per_source": AGG_MAX_PER_SOURCE,
        "agg_max_per_family": AGG_MAX_PER_FAMILY,
        "agg_diversity_lambda": AGG_DIVERSITY_LAMBDA,
        "agg_source_match_weight": AGG_SOURCE_MATCH_WEIGHT,
        "gray_enabled": GRAY_ENABLED,
        "gray_ratio": GRAY_RATIO,
        "gray_stable_label": GRAY_STABLE_LABEL,
        "gray_canary_label": GRAY_CANARY_LABEL,
        "gray_canary_index_dir": GRAY_CANARY_INDEX_DIR or None,
        "gray_canary_loaded": bool(gray_retriever is not None),
        "gray_canary_dense_backend": str(gray_retriever.dense_backend) if gray_retriever else None,
        "gray_canary_error": gray_canary_error or None,
        "llm_base": DEFAULT_LLM_BASE,
        "llm_model": DEFAULT_LLM_MODEL,
        "rag_api_key_required": bool(RAG_API_KEY),
        "retrieve": _retrieve_stats_snapshot(),
    }


def check_rag_api_key(authorization: Optional[str]) -> None:
    if not RAG_API_KEY:
        return
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing Authorization header")
    prefix = "Bearer "
    if not authorization.startswith(prefix):
        raise HTTPException(status_code=401, detail="Invalid Authorization header")
    token = authorization[len(prefix) :].strip()
    if token != RAG_API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")


def _latest_user_query(messages: List[Dict[str, Any]]) -> str:
    for m in reversed(messages):
        if str(m.get("role", "")).lower() != "user":
            continue
        text = _message_to_text(m)
        if text:
            return text
    return ""


def _is_multi_evidence_query(query: str) -> bool:
    q = (query or "").strip()
    if not q:
        return False
    patterns = [
        "是否都",
        "这四份",
        "这三份",
        "这几份",
        "是否一致",
        "都使用",
        "都属于",
        "对比",
    ]
    return any(p in q for p in patterns)


def _search_kwargs_from_req(req: Any) -> Dict[str, Any]:
    if LOCK_RETRIEVAL_PARAMS:
        return {
            "top_k": DEFAULT_TOP_K,
            "min_score": DEFAULT_MIN_SCORE,
            "min_score_vec": DEFAULT_MIN_SCORE_VEC,
            "mode": DEFAULT_RETRIEVAL_MODE.strip().lower(),
            "fusion": DEFAULT_FUSION.strip().lower(),
            "candidate_k": DEFAULT_CANDIDATE_K,
            "rrf_k": DEFAULT_RRF_K,
            "weight_lex": DEFAULT_WEIGHT_LEX,
            "weight_vec": DEFAULT_WEIGHT_VEC,
            "enable_rerank": DEFAULT_ENABLE_RERANK,
            "enable_diversity": DEFAULT_ENABLE_DIVERSITY,
            "max_per_source": DEFAULT_MAX_PER_SOURCE,
            "max_per_family": DEFAULT_MAX_PER_FAMILY,
            "diversity_lambda": DEFAULT_DIVERSITY_LAMBDA,
            "source_match_weight": DEFAULT_SOURCE_MATCH_WEIGHT,
        }

    mode = (getattr(req, "retrieval_mode", None) or DEFAULT_RETRIEVAL_MODE or "hybrid").strip().lower()
    fusion = (getattr(req, "fusion", None) or DEFAULT_FUSION or "rrf").strip().lower()
    enable_rerank = getattr(req, "enable_rerank", DEFAULT_ENABLE_RERANK)
    if enable_rerank is None:
        enable_rerank = DEFAULT_ENABLE_RERANK

    return {
        "top_k": int(getattr(req, "top_k", DEFAULT_TOP_K) or DEFAULT_TOP_K),
        "min_score": float(getattr(req, "min_score", DEFAULT_MIN_SCORE) or DEFAULT_MIN_SCORE),
        "min_score_vec": float(getattr(req, "min_score_vec", DEFAULT_MIN_SCORE_VEC) or DEFAULT_MIN_SCORE_VEC),
        "mode": mode,
        "fusion": fusion,
        "candidate_k": int(getattr(req, "candidate_k", DEFAULT_CANDIDATE_K) or DEFAULT_CANDIDATE_K),
        "rrf_k": int(getattr(req, "rrf_k", DEFAULT_RRF_K) or DEFAULT_RRF_K),
        "weight_lex": float(getattr(req, "weight_lex", DEFAULT_WEIGHT_LEX) or DEFAULT_WEIGHT_LEX),
        "weight_vec": float(getattr(req, "weight_vec", DEFAULT_WEIGHT_VEC) or DEFAULT_WEIGHT_VEC),
        "enable_rerank": bool(enable_rerank),
        "enable_diversity": DEFAULT_ENABLE_DIVERSITY,
        "max_per_source": DEFAULT_MAX_PER_SOURCE,
        "max_per_family": DEFAULT_MAX_PER_FAMILY,
        "diversity_lambda": DEFAULT_DIVERSITY_LAMBDA,
        "source_match_weight": DEFAULT_SOURCE_MATCH_WEIGHT,
    }


def _maybe_apply_aggregation_profile(query: str, search_kwargs: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(search_kwargs)
    if not _is_multi_evidence_query(query):
        return out
    out["enable_diversity"] = AGG_ENABLE_DIVERSITY
    out["max_per_source"] = AGG_MAX_PER_SOURCE
    out["max_per_family"] = AGG_MAX_PER_FAMILY
    out["diversity_lambda"] = AGG_DIVERSITY_LAMBDA
    out["source_match_weight"] = AGG_SOURCE_MATCH_WEIGHT
    out["candidate_k"] = max(int(out.get("candidate_k", DEFAULT_CANDIDATE_K)), 80)
    out["top_k"] = max(int(out.get("top_k", DEFAULT_TOP_K)), 8)
    return out


def _effective_retrieval_mode(hits: List[Dict[str, Any]], requested_mode: str) -> str:
    if hits:
        mode = str(hits[0].get("retrieval_mode", "")).strip()
        if mode:
            return mode
    return requested_mode


def _retrieve_cache_key(
    query: str,
    route: str,
    index_dir: str,
    search_kwargs: Dict[str, Any],
) -> str:
    payload = {
        "query": (query or "").strip(),
        "route": route,
        "index_dir": index_dir,
        "search": search_kwargs,
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _retrieve_cache_get(key: str) -> Optional[Dict[str, Any]]:
    if not (RETRIEVE_CACHE_ENABLED and RETRIEVE_CACHE_TTL_S > 0.0 and RETRIEVE_CACHE_MAX_ITEMS > 0):
        return None
    now = time.monotonic()
    with _CACHE_LOCK:
        item = _RETRIEVE_CACHE.get(key)
        if item is None:
            return None
        expires_at, value = item
        if expires_at <= now:
            _RETRIEVE_CACHE.pop(key, None)
            _RETRIEVE_METRICS["cache_expired"] += 1
            return None
        _RETRIEVE_CACHE.move_to_end(key)
        return deepcopy(value)


def _retrieve_cache_set(key: str, value: Dict[str, Any]) -> None:
    if not (RETRIEVE_CACHE_ENABLED and RETRIEVE_CACHE_TTL_S > 0.0 and RETRIEVE_CACHE_MAX_ITEMS > 0):
        return
    expires_at = time.monotonic() + RETRIEVE_CACHE_TTL_S
    with _CACHE_LOCK:
        _RETRIEVE_CACHE[key] = (expires_at, deepcopy(value))
        _RETRIEVE_CACHE.move_to_end(key)
        while len(_RETRIEVE_CACHE) > RETRIEVE_CACHE_MAX_ITEMS:
            _RETRIEVE_CACHE.popitem(last=False)


def _record_retrieve_metrics(latency_ms: float, total_hits: int, cache_hit: bool) -> None:
    with _CACHE_LOCK:
        _RETRIEVE_METRICS["requests"] += 1
        _RETRIEVE_METRICS["cache_hits" if cache_hit else "cache_misses"] += 1
        if total_hits <= 0:
            _RETRIEVE_METRICS["empty_hits"] += 1
        _RETRIEVE_METRICS["total_hits"] += max(0, int(total_hits))
        _RETRIEVE_METRICS["latency_ms_total"] += float(latency_ms)
        _RETRIEVE_LATENCIES_MS.append(float(latency_ms))


def _percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    data = sorted(values)
    if len(data) == 1:
        return float(data[0])
    rank = max(0.0, min(100.0, p)) / 100.0 * (len(data) - 1)
    lo = int(rank)
    hi = min(len(data) - 1, lo + 1)
    frac = rank - lo
    return float(data[lo] * (1.0 - frac) + data[hi] * frac)


def _retrieve_stats_snapshot() -> Dict[str, Any]:
    with _CACHE_LOCK:
        metrics = dict(_RETRIEVE_METRICS)
        samples = list(_RETRIEVE_LATENCIES_MS)
        cache_size = len(_RETRIEVE_CACHE)
    requests_count = max(1, int(metrics.get("requests", 0)))
    return {
        "requests": int(metrics.get("requests", 0)),
        "cache_hits": int(metrics.get("cache_hits", 0)),
        "cache_misses": int(metrics.get("cache_misses", 0)),
        "cache_expired": int(metrics.get("cache_expired", 0)),
        "cache_hit_rate": float(metrics.get("cache_hits", 0)) / float(requests_count),
        "empty_hits": int(metrics.get("empty_hits", 0)),
        "empty_hit_rate": float(metrics.get("empty_hits", 0)) / float(requests_count),
        "avg_hits": float(metrics.get("total_hits", 0)) / float(requests_count),
        "latency_ms_avg": float(metrics.get("latency_ms_total", 0.0)) / float(requests_count),
        "latency_ms_p50": _percentile(samples, 50),
        "latency_ms_p95": _percentile(samples, 95),
        "latency_sample_size": len(samples),
        "cache_size": cache_size,
        "cache_enabled": RETRIEVE_CACHE_ENABLED,
        "cache_ttl_s": RETRIEVE_CACHE_TTL_S,
        "cache_max_items": RETRIEVE_CACHE_MAX_ITEMS,
    }


@app.post("/retrieve")
def retrieve(
    req: RetrieveRequest,
    authorization: Optional[str] = Header(default=None),
    x_rag_route: Optional[str] = Header(default=None, alias="X-Rag-Route"),
) -> Dict[str, Any]:
    check_rag_api_key(authorization)
    started = time.perf_counter()
    search_kwargs = _search_kwargs_from_req(req)
    search_kwargs = _maybe_apply_aggregation_profile(req.query, search_kwargs)
    active_retriever, route = _pick_retriever(req.query, x_rag_route)
    cache_key = _retrieve_cache_key(
        query=req.query,
        route=route,
        index_dir=str(active_retriever.index_dir),
        search_kwargs=search_kwargs,
    )
    cached = _retrieve_cache_get(cache_key)
    if cached is not None:
        latency_ms = (time.perf_counter() - started) * 1000.0
        _record_retrieve_metrics(latency_ms=latency_ms, total_hits=int(cached.get("total_hits", 0)), cache_hit=True)
        cached["cache"] = {"hit": True, "ttl_s": RETRIEVE_CACHE_TTL_S}
        cached["latency_ms"] = latency_ms
        return cached

    hits = active_retriever.search(query=req.query, **search_kwargs)
    requested_mode = str(search_kwargs.get("mode", "hybrid"))
    effective_mode = _effective_retrieval_mode(hits, requested_mode)
    out = {
        "query": req.query,
        "top_k": search_kwargs.get("top_k"),
        "min_score": search_kwargs.get("min_score"),
        "requested_retrieval_mode": requested_mode,
        "effective_retrieval_mode": effective_mode,
        "fusion": search_kwargs.get("fusion"),
        "enable_rerank": bool(search_kwargs.get("enable_rerank", True)),
        "retrieval_params_locked": LOCK_RETRIEVAL_PARAMS,
        "rag_route": route,
        "dense_available": bool(active_retriever.dense_available),
        "dense_backend": str(active_retriever.dense_backend),
        "total_hits": len(hits),
        "hits": hits,
    }
    _retrieve_cache_set(cache_key, out)
    latency_ms = (time.perf_counter() - started) * 1000.0
    _record_retrieve_metrics(latency_ms=latency_ms, total_hits=len(hits), cache_hit=False)
    out["cache"] = {"hit": False, "ttl_s": RETRIEVE_CACHE_TTL_S}
    out["latency_ms"] = latency_ms
    return out


@app.get("/stats")
def stats(authorization: Optional[str] = Header(default=None)) -> Dict[str, Any]:
    check_rag_api_key(authorization)
    return {
        "retrieve": _retrieve_stats_snapshot(),
        "index_dir": INDEX_DIR,
        "num_chunks": len(retriever.chunks),
        "dense_backend": str(retriever.dense_backend),
        "gray_enabled": GRAY_ENABLED,
        "gray_ratio": GRAY_RATIO,
        "gray_canary_loaded": bool(gray_retriever is not None),
    }


@app.post("/answer")
def answer(
    req: AnswerRequest,
    authorization: Optional[str] = Header(default=None),
    x_rag_route: Optional[str] = Header(default=None, alias="X-Rag-Route"),
) -> Dict[str, Any]:
    check_rag_api_key(authorization)
    search_kwargs = _search_kwargs_from_req(req)
    search_kwargs = _maybe_apply_aggregation_profile(req.query, search_kwargs)
    active_retriever, route = _pick_retriever(req.query, x_rag_route)
    hits = active_retriever.search(query=req.query, **search_kwargs)
    requested_mode = str(search_kwargs.get("mode", "hybrid"))
    effective_mode = _effective_retrieval_mode(hits, requested_mode)
    if not hits:
        return {
            "query": req.query,
            "answer": "未检索到可用资料，无法基于当前知识库给出可靠答案。",
            "requested_retrieval_mode": requested_mode,
            "effective_retrieval_mode": effective_mode,
            "fusion": search_kwargs.get("fusion"),
            "enable_rerank": bool(search_kwargs.get("enable_rerank", True)),
            "retrieval_params_locked": LOCK_RETRIEVAL_PARAMS,
            "rag_route": route,
            "total_hits": 0,
            "contexts": [] if req.include_context else None,
        }

    context_block = build_context_block(hits, max_context_chars=req.max_context_chars)

    sys_prompt = req.system_prompt or (
        "你是一个RAG问答助手。必须优先依据提供的参考资料回答；"
        "如果资料不足，请明确说“根据现有资料无法确定”。"
        "回答使用中文，简洁，并在结尾用[1][2]这种形式给出引用编号。"
    )
    user_prompt = (
        f"问题：\n{req.query}\n\n"
        f"参考资料：\n{context_block}\n\n"
        "请只基于参考资料回答。"
    )

    llm_base = (req.llm_base_url or DEFAULT_LLM_BASE).strip()
    model = (req.model or DEFAULT_LLM_MODEL).strip()
    api_key = req.api_key if req.api_key is not None else DEFAULT_LLM_API_KEY

    if not llm_base:
        raise HTTPException(status_code=400, detail="LLM base URL is empty. Set llm_base_url or LLM_API_BASE.")
    if not model:
        raise HTTPException(status_code=400, detail="LLM model is empty. Set model or LLM_MODEL.")

    llm = call_chat_completion(
        base_url=llm_base,
        model=model,
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=req.temperature,
        max_tokens=req.max_tokens,
        api_key=api_key,
        timeout_s=LLM_TIMEOUT_S,
    )

    out: Dict[str, Any] = {
        "query": req.query,
        "answer": llm["content"],
        "total_hits": len(hits),
        "requested_retrieval_mode": requested_mode,
        "effective_retrieval_mode": effective_mode,
        "fusion": search_kwargs.get("fusion"),
        "enable_rerank": bool(search_kwargs.get("enable_rerank", True)),
        "retrieval_params_locked": LOCK_RETRIEVAL_PARAMS,
        "rag_route": route,
        "model": model,
        "llm_base_url": llm_base,
        "finish_reason": llm.get("finish_reason"),
        "usage": llm.get("usage", {}),
    }
    if req.include_context:
        out["contexts"] = hits
    return out


@app.post("/v1/chat/completions")
def rag_chat_completions(
    req: RagChatCompletionRequest,
    authorization: Optional[str] = Header(default=None),
    x_rag_route: Optional[str] = Header(default=None, alias="X-Rag-Route"),
) -> Dict[str, Any]:
    check_rag_api_key(authorization)

    query = _latest_user_query(req.messages)
    if not query:
        raise HTTPException(status_code=400, detail="No user message found in messages")

    search_kwargs = _search_kwargs_from_req(req)
    search_kwargs = _maybe_apply_aggregation_profile(query, search_kwargs)
    active_retriever, route = _pick_retriever(query, x_rag_route)
    hits = active_retriever.search(query=query, **search_kwargs)
    requested_mode = str(search_kwargs.get("mode", "hybrid"))
    effective_mode = _effective_retrieval_mode(hits, requested_mode)
    context_block = build_context_block(hits, max_context_chars=req.max_context_chars) if hits else ""
    rag_sys = req.system_prompt or (
        "你是一个RAG问答助手。优先依据参考资料回答；"
        "如果资料不足，请明确说明“根据现有资料无法确定”。"
        "不要输出思考过程，直接给结论。"
    )
    injected_sys = (
        f"{rag_sys}\n\n"
        f"参考资料如下（可为空）：\n{context_block if context_block else '（无命中）'}\n\n"
        "请优先基于以上资料作答。"
    )
    messages = [{"role": "system", "content": injected_sys}] + req.messages

    llm_base = (req.llm_base_url or DEFAULT_LLM_BASE).strip()
    model = (req.model or DEFAULT_LLM_MODEL).strip()
    api_key = req.api_key if req.api_key is not None else DEFAULT_LLM_API_KEY
    if not llm_base:
        raise HTTPException(status_code=400, detail="LLM base URL is empty. Set llm_base_url or LLM_API_BASE.")
    if not model:
        raise HTTPException(status_code=400, detail="LLM model is empty. Set model or LLM_MODEL.")

    llm = call_chat_completion(
        base_url=llm_base,
        model=model,
        messages=messages,
        temperature=req.temperature,
        max_tokens=req.max_tokens,
        api_key=api_key,
        timeout_s=LLM_TIMEOUT_S,
    )
    out = llm["raw"]
    # Normalize qwen reasoning-style output for OpenAI-compatible clients:
    # if content is null but reasoning exists, copy reasoning into content.
    try:
        first_choice = (out.get("choices") or [])[0]
        msg = first_choice.get("message") or {}
        if msg.get("content") is None:
            reasoning = msg.get("reasoning")
            if isinstance(reasoning, str) and reasoning.strip():
                msg["content"] = reasoning
    except Exception:
        pass
    out["rag_retrieval"] = {
        "requested_retrieval_mode": requested_mode,
        "effective_retrieval_mode": effective_mode,
        "fusion": search_kwargs.get("fusion"),
        "enable_rerank": bool(search_kwargs.get("enable_rerank", True)),
        "retrieval_params_locked": LOCK_RETRIEVAL_PARAMS,
        "rag_route": route,
        "dense_backend": str(active_retriever.dense_backend),
        "total_hits": len(hits),
    }
    if req.include_context:
        out["rag_contexts"] = hits
    return out
