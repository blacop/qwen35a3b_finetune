#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

import requests
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field, model_validator

from mm_retrieval import MultiModalRetriever, build_context_block


def as_bool(raw: Optional[str], default: bool = False) -> bool:
    if raw is None:
        return default
    v = str(raw).strip().lower()
    if v in {"1", "true", "yes", "y", "on"}:
        return True
    if v in {"0", "false", "no", "n", "off"}:
        return False
    return default


def message_to_text(message: Dict[str, Any]) -> str:
    content = message.get("content")
    if isinstance(content, str) and content.strip():
        return content.strip()
    if isinstance(content, list):
        chunks: List[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                txt = item.get("text")
                if isinstance(txt, str) and txt.strip():
                    chunks.append(txt.strip())
        if chunks:
            return "\n".join(chunks)
    return ""


def call_chat_completion(
    base_url: str,
    model: str,
    messages: List[Dict[str, str]],
    temperature: float,
    max_tokens: Optional[int],
    api_key: str,
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
    msg = choices[0].get("message") or {}
    text = message_to_text(msg)
    return {
        "raw": data,
        "content": text,
        "finish_reason": choices[0].get("finish_reason"),
        "usage": data.get("usage", {}),
    }


class RetrieveRequest(BaseModel):
    query: str = ""
    query_image_path: str = ""
    top_k: int = Field(default=8, ge=1, le=30)
    top_k_text: int = Field(default=20, ge=1, le=100)
    top_k_image: int = Field(default=20, ge=1, le=100)
    text_weight: float = Field(default=0.65, ge=0.0, le=1.0)
    image_weight: float = Field(default=0.35, ge=0.0, le=1.0)
    min_score_text: float = Field(default=0.01, ge=0.0, le=1.0)
    min_score_image: float = Field(default=0.05, ge=-1.0, le=1.0)
    prefer_text_for_text_query: bool = True
    enable_rerank: bool = True
    enable_dedup: bool = True
    dedup_text_signature_len: int = Field(default=100, ge=20, le=1000)
    image_dedup_sim_threshold: float = Field(default=0.985, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_inputs(self) -> "RetrieveRequest":
        if not (self.query or self.query_image_path):
            raise ValueError("query and query_image_path cannot both be empty")
        return self


class AnswerRequest(RetrieveRequest):
    include_hits: bool = False
    max_context_chars: int = Field(default=600, ge=100, le=3000)
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(default=512, ge=1, le=4096)
    llm_base_url: Optional[str] = None
    model: Optional[str] = None
    api_key: Optional[str] = None
    system_prompt: Optional[str] = None


INDEX_DIR = os.getenv("MM_INDEX_DIR", "/home/ubuntu/generate/tydata_rag/mm_rag/index")
MM_RAG_API_KEY = os.getenv("MM_RAG_API_KEY", "")
DEFAULT_LLM_BASE = os.getenv("LLM_API_BASE", "http://127.0.0.1:8014/v1")
DEFAULT_LLM_MODEL = os.getenv("LLM_MODEL", "")
DEFAULT_LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_TIMEOUT_S = float(os.getenv("LLM_TIMEOUT_S", "120"))
LLM_ENABLE_THINKING = as_bool(os.getenv("LLM_ENABLE_THINKING", "0"), False)
ALLOW_NO_LLM = as_bool(os.getenv("MM_ALLOW_NO_LLM", "1"), True)

app = FastAPI(title="tydata-mm-rag-api", version="1.0.0")
retriever = MultiModalRetriever(INDEX_DIR)


def check_api_key(authorization: Optional[str]) -> None:
    if not MM_RAG_API_KEY:
        return
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing Authorization header")
    prefix = "Bearer "
    if not authorization.startswith(prefix):
        raise HTTPException(status_code=401, detail="Invalid Authorization header")
    token = authorization[len(prefix) :].strip()
    if token != MM_RAG_API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "ok": True,
        "index_dir": INDEX_DIR,
        "meta": retriever.meta,
        "mm_rag_api_key_required": bool(MM_RAG_API_KEY),
        "llm_base": DEFAULT_LLM_BASE,
        "llm_model": DEFAULT_LLM_MODEL,
        "allow_no_llm": ALLOW_NO_LLM,
    }


@app.post("/retrieve")
def retrieve(req: RetrieveRequest, authorization: Optional[str] = Header(default=None)) -> Dict[str, Any]:
    check_api_key(authorization)
    result = retriever.search(
        query=req.query,
        query_image_path=req.query_image_path,
        top_k=req.top_k,
        top_k_text=req.top_k_text,
        top_k_image=req.top_k_image,
        text_weight=req.text_weight,
        image_weight=req.image_weight,
        min_score_text=req.min_score_text,
        min_score_image=req.min_score_image,
        prefer_text_for_text_query=req.prefer_text_for_text_query,
        enable_rerank=req.enable_rerank,
        enable_dedup=req.enable_dedup,
        dedup_text_signature_len=req.dedup_text_signature_len,
        image_dedup_sim_threshold=req.image_dedup_sim_threshold,
    )
    return result


@app.post("/answer")
def answer(req: AnswerRequest, authorization: Optional[str] = Header(default=None)) -> Dict[str, Any]:
    check_api_key(authorization)
    result = retriever.search(
        query=req.query,
        query_image_path=req.query_image_path,
        top_k=req.top_k,
        top_k_text=req.top_k_text,
        top_k_image=req.top_k_image,
        text_weight=req.text_weight,
        image_weight=req.image_weight,
        min_score_text=req.min_score_text,
        min_score_image=req.min_score_image,
        prefer_text_for_text_query=req.prefer_text_for_text_query,
        enable_rerank=req.enable_rerank,
        enable_dedup=req.enable_dedup,
        dedup_text_signature_len=req.dedup_text_signature_len,
        image_dedup_sim_threshold=req.image_dedup_sim_threshold,
    )
    fused_hits = result.get("fused_hits", [])
    context_block = build_context_block(fused_hits, max_chars=req.max_context_chars) if fused_hits else ""

    llm_base = (req.llm_base_url or DEFAULT_LLM_BASE).strip()
    model = (req.model or DEFAULT_LLM_MODEL).strip()
    api_key = req.api_key if req.api_key is not None else DEFAULT_LLM_API_KEY

    if not llm_base or not model:
        if ALLOW_NO_LLM:
            out: Dict[str, Any] = {
                "query": req.query,
                "query_image_path": req.query_image_path,
                "answer": "",
                "context_block": context_block,
                "total_hits": len(fused_hits),
                "notice": "LLM_API_BASE/LLM_MODEL not set, return contexts only.",
            }
            if req.include_hits:
                out["hits"] = fused_hits
            return out
        raise HTTPException(status_code=400, detail="LLM base/model missing and MM_ALLOW_NO_LLM=0")

    sys_prompt = req.system_prompt or (
        "You are a multimodal customer-support assistant. "
        "Answer only from the provided contexts. "
        "If context is insufficient, state uncertainty explicitly."
    )
    user_prompt = (
        f"User question:\n{req.query or '(empty text query)'}\n"
        f"User image path:\n{req.query_image_path or '(none)'}\n\n"
        f"Retrieved contexts:\n{context_block if context_block else '(no hits)'}\n\n"
        "Provide a concise Chinese answer and cite context numbers like [1][2]."
    )
    llm = call_chat_completion(
        base_url=llm_base,
        model=model,
        messages=[{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_prompt}],
        temperature=req.temperature,
        max_tokens=req.max_tokens,
        api_key=api_key,
        timeout_s=LLM_TIMEOUT_S,
    )
    out = {
        "query": req.query,
        "query_image_path": req.query_image_path,
        "answer": llm["content"],
        "total_hits": len(fused_hits),
        "finish_reason": llm.get("finish_reason"),
        "usage": llm.get("usage", {}),
    }
    if req.include_hits:
        out["hits"] = fused_hits
    return out
