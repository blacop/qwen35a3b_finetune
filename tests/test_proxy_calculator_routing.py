from pathlib import Path
import asyncio
import importlib.util
import json
import sys


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "dpo_guardrails_proxy.py"


def load_module(monkeypatch, **env):
    keys = [
        "CALCULATOR_ENABLED",
        "CALCULATOR_STRICT_MODE",
        "CALC_ROUTE_KEYWORDS",
        "REALTIME_ROUTE_KEYWORDS",
        "RAG_SERVER_URL",
        "RAG_SERVER_API_KEY",
        "PROXY_API_KEYS",
    ]
    for key in keys:
        monkeypatch.delenv(key, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    sys.path.insert(0, str(SCRIPT_PATH.parents[1]))
    name = "dpo_guardrails_proxy_calc_routing_test"
    sys.modules.pop(name, None)
    sys.modules.pop("scripts.proxy_intent_router", None)
    sys.modules.pop("scripts.proxy_calc_orchestrator", None)
    sys.modules.pop("scripts.proxy_calculator_client", None)
    sys.modules.pop("proxy_intent_router", None)
    sys.modules.pop("proxy_calc_orchestrator", None)
    sys.modules.pop("proxy_calculator_client", None)
    spec = importlib.util.spec_from_file_location(name, SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class FakeRequest:
    def __init__(self, payload):
        self.payload = payload

    async def json(self):
        return self.payload


def test_realtime_query_is_rejected(monkeypatch):
    module = load_module(
        monkeypatch,
        PROXY_API_KEYS='{"qa":"sk-qa"}',
        REALTIME_ROUTE_KEYWORDS="订单号,当前状态",
    )
    payload = {"model": "gpu5-v5.1f-merged", "messages": [{"role": "user", "content": "订单号A123当前状态是什么"}]}
    resp = asyncio.run(module.chat_completions(FakeRequest(payload), authorization="Bearer sk-qa", x_user_access_token=None))
    assert resp.status_code == 400
    body = json.loads(resp.body.decode("utf-8"))
    assert body["error"]["type"] == "RealtimeQueryRequired"


def test_calc_query_strict_mode_degrades_to_normal_chat_response(monkeypatch):
    module = load_module(
        monkeypatch,
        PROXY_API_KEYS='{"qa":"sk-qa"}',
        CALCULATOR_ENABLED="1",
        CALCULATOR_STRICT_MODE="1",
        CALC_ROUTE_KEYWORDS="怎么算,赔率,本金",
    )
    monkeypatch.setattr(module, "_retrieve_rag_hits_for_calc", lambda query: [])
    monkeypatch.setattr(module, "run_calc_flow", lambda request_id, query, rag_hits: {
        "ok": False,
        "formula_id": "",
        "error": {"code": "FORMULA_NOT_INFERRED", "message": "unable to infer formula"},
    })
    audit_rows = []
    monkeypatch.setattr(module, "_audit_proxy_with_elapsed", lambda *args, **kwargs: audit_rows.append(kwargs.get("extra") if "extra" in kwargs else args[5]))
    payload = {"model": "gpu5-v5.1f-merged", "messages": [{"role": "user", "content": "赔率怎么算"}]}
    resp = asyncio.run(module.chat_completions(FakeRequest(payload), authorization="Bearer sk-qa", x_user_access_token=None))
    assert resp.status_code == 200
    body = json.loads(resp.body.decode("utf-8"))
    assert body["object"] == "chat.completion"
    assert body["choices"][0]["message"]["role"] == "assistant"
    assert "这个我可以帮你算" in body["choices"][0]["message"]["content"]
    assert "欧洲盘还是香港盘" in body["choices"][0]["message"]["content"]
    assert "本金和赔率" in body["choices"][0]["message"]["content"]
    assert audit_rows[-1]["calc_degrade_bucket"] == "odds"
    assert audit_rows[-1]["calc_missing_fields"] == ["plate_type", "stake", "odds"]


def test_calc_query_strict_mode_returns_explainable_degrade(monkeypatch):
    module = load_module(
        monkeypatch,
        PROXY_API_KEYS='{"qa":"sk-qa"}',
        CALCULATOR_ENABLED="1",
        CALCULATOR_STRICT_MODE="1",
        CALC_ROUTE_KEYWORDS="怎么算,赔率,本金",
    )
    monkeypatch.setattr(module, "_retrieve_rag_hits_for_calc", lambda query: [{"source_file": "rule.txt", "text": "欧洲盘：赢：本金*(赔率-1)"}])
    monkeypatch.setattr(module, "run_calc_flow", lambda request_id, query, rag_hits: {
        "ok": False,
        "formula_id": "european_odds_net_profit",
        "error": {"code": "INPUTS_NOT_EXTRACTED", "message": "unable to extract"},
    })
    payload = {"model": "gpu5-v5.1f-merged", "messages": [{"role": "user", "content": "欧洲盘赔率怎么算"}]}
    resp = asyncio.run(
        module.chat_completions(
            FakeRequest(payload),
            authorization="Bearer sk-qa",
            x_user_access_token=None,
        )
    )
    assert resp.status_code == 200
    body = json.loads(resp.body.decode("utf-8"))
    content = body["choices"][0]["message"]["content"]
    assert "这个我可以帮你算" in content
    assert "本金 + 赔率" in content


def test_calc_success_merges_into_single_leading_system_message(monkeypatch):
    module = load_module(
        monkeypatch,
        PROXY_API_KEYS='{"qa":"sk-qa"}',
        CALCULATOR_ENABLED="1",
        CALCULATOR_STRICT_MODE="1",
        CALC_ROUTE_KEYWORDS="怎么算,赔率,本金,欧洲盘",
    )
    merged_payloads = []

    monkeypatch.setattr(module, "_retrieve_rag_hits_for_calc", lambda query: [{"source_file": "rule.txt", "text": "欧洲盘：净赢=本金*(赔率-1)"}])
    monkeypatch.setattr(module, "run_calc_flow", lambda request_id, query, rag_hits: {
        "ok": True,
        "formula_id": "european_odds_net_profit",
        "result": {"net_profit": 1150.0},
        "steps": ["1000 x (2.15 - 1) = 1150"],
        "formula_text": "本金 x (赔率 - 1)",
    })

    class FakeResp:
        status_code = 200
        text = ""

        def json(self):
            return {
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "created": 0,
                "model": "gpu5-v5.1f-merged",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "净赢 1150"}, "finish_reason": "stop"}],
            }

    def fake_post(url, headers=None, json=None, timeout=None, stream=False):
        merged_payloads.append(json)
        return FakeResp()

    monkeypatch.setattr(module.SESSION, "post", fake_post)
    payload = {"model": "gpu5-v5.1f-merged", "messages": [{"role": "user", "content": "欧洲盘 本金1000 赔率2.15 净赢多少？"}]}
    resp = asyncio.run(
        module.chat_completions(
            FakeRequest(payload),
            authorization="Bearer sk-qa",
            x_user_access_token=None,
        )
    )

    assert resp.status_code == 200
    assert len(merged_payloads) == 1
    messages = merged_payloads[0]["messages"]
    assert messages[0]["role"] == "system"
    assert sum(1 for msg in messages if msg.get("role") == "system") == 1
    assert "本金 x (赔率 - 1)" in messages[0]["content"]
    assert messages[1]["role"] == "user"
