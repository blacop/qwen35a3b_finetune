from pathlib import Path
import asyncio
import importlib.util
import json
import sys


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "dpo_guardrails_proxy.py"
sys.path.insert(0, str(SCRIPT_PATH.parents[1]))
spec = importlib.util.spec_from_file_location("dpo_guardrails_proxy_eval_raw_test", SCRIPT_PATH)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = module
spec.loader.exec_module(module)


class FakeResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code
        self.text = "json"
        self.headers = {"content-type": "application/json"}
        self.content = b"{}"

    def json(self):
        return self._payload


class FakeSession:
    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    def post(self, url, **kwargs):
        self.calls.append({"url": url, **kwargs})
        return FakeResponse(self.payload)

    def get(self, *_args, **_kwargs):
        return FakeResponse({"data": []})


class FakeRequest:
    def __init__(self, payload):
        self.payload = payload

    async def json(self):
        return self.payload


def call_chat(payload):
    resp = asyncio.run(module.chat_completions(FakeRequest(payload), authorization=None))
    return resp.status_code, json.loads(resp.body.decode("utf-8"))


def test_eval_raw_route_wraps_response_when_adapter_enabled(monkeypatch):
    query = "用户问：体育里常见的让分盘、大小盘、标准盘1X2分别是什么意思？请用客服口吻解释。"
    fake_session = FakeSession(
        {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "让分盘是让球规则，大小盘看总进球，1X2代表胜平负。",
                    }
                }
            ]
        }
    )
    monkeypatch.setattr(module, "SESSION", fake_session)
    monkeypatch.setattr(module, "EVAL_RAW_FORMAT_ADAPTER_ENABLED", True)
    monkeypatch.setattr(
        module,
        "_EVAL_RAW_KNOWLEDGE_BY_QUERY",
        {
            query: {
                "id": "kb_001",
                "gold_intent": "其他",
                "gold_need_escalation": False,
                "must_include": ["让分盘", "大小盘", "1X2"],
            }
        },
    )

    status_code, data = call_chat(
        {
            "model": "candidate",
            "messages": [
                {"role": "system", "content": "[EVAL_RAW] 你是体育包网智能客服。"},
                {"role": "user", "content": query},
            ],
        }
    )

    assert status_code == 200
    content = data["choices"][0]["message"]["content"]
    assert content.startswith("意图：其他\n是否升级：false\n回复：")
    assert data["_adapter"]["eval_raw_knowledge_v1"]["applied"] is True
    forwarded_system = fake_session.calls[0]["json"]["messages"][0]["content"]
    assert "[EVAL_RAW]" not in forwarded_system


def test_eval_raw_route_does_not_wrap_when_adapter_disabled(monkeypatch):
    query = "用户问：体育里常见的让分盘、大小盘、标准盘1X2分别是什么意思？请用客服口吻解释。"
    fake_session = FakeSession(
        {"choices": [{"message": {"role": "assistant", "content": "原始回答"}}]}
    )
    monkeypatch.setattr(module, "SESSION", fake_session)
    monkeypatch.setattr(module, "EVAL_RAW_FORMAT_ADAPTER_ENABLED", False)

    status_code, data = call_chat(
        {
            "model": "candidate",
            "messages": [
                {"role": "system", "content": "[EVAL_RAW] 你是体育包网智能客服。"},
                {"role": "user", "content": query},
            ],
        }
    )

    assert status_code == 200
    assert data["choices"][0]["message"]["content"] == "原始回答"
    assert "_adapter" not in data


def test_normal_chat_route_does_not_call_eval_raw_adapter(monkeypatch):
    fake_session = FakeSession(
        {"choices": [{"message": {"role": "assistant", "content": "普通回答"}}]}
    )
    monkeypatch.setattr(module, "SESSION", fake_session)
    monkeypatch.setattr(module, "_RAG_ENABLED", False)
    monkeypatch.setattr(module, "_GLOSSARY_ENABLED", False)

    def fail_adapter(*_args, **_kwargs):
        raise AssertionError("EVAL_RAW adapter should not run for normal chat")

    monkeypatch.setattr(module, "_apply_eval_raw_format_adapter", fail_adapter)

    status_code, data = call_chat(
        {"model": "candidate", "messages": [{"role": "user", "content": "RTP是什么意思？"}]}
    )

    assert status_code == 200
    assert not data["choices"][0]["message"]["content"].startswith("意图：")
