from pathlib import Path
import asyncio
import importlib.util
import json
import sys


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "dpo_guardrails_proxy.py"
sys.path.insert(0, str(SCRIPT_PATH.parents[1]))
spec = importlib.util.spec_from_file_location("dpo_guardrails_proxy_agent_test", SCRIPT_PATH)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = module
spec.loader.exec_module(module)


class FakeResponse:
    status_code = 200
    text = "json"
    headers = {"content-type": "application/json"}
    content = b"{}"

    def json(self):
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "agent 正常回答",
                    }
                }
            ]
        }


class FakeSession:
    def post(self, *_args, **_kwargs):
        return FakeResponse()

    def get(self, *_args, **_kwargs):
        return FakeResponse()


class FakeRequest:
    def __init__(self, payload):
        self.payload = payload

    async def json(self):
        return self.payload


def call_agent(payload):
    resp = asyncio.run(module.chat_completions_agent(FakeRequest(payload), authorization=None))
    return resp.status_code, json.loads(resp.body.decode("utf-8"))


def test_agent_route_does_not_call_eval_raw_adapter(monkeypatch):
    monkeypatch.setattr(module, "SESSION", FakeSession())

    def fail_adapter(*_args, **_kwargs):
        raise AssertionError("EVAL_RAW adapter should not run for agent route")

    monkeypatch.setattr(module, "_apply_eval_raw_format_adapter", fail_adapter)

    status_code, data = call_agent(
        {
            "model": "candidate",
            "messages": [{"role": "user", "content": "[EVAL_RAW] 你好，不需要查工具"}],
        }
    )

    assert status_code == 200
    assert data["choices"][0]["message"]["content"] == "agent 正常回答"
    assert "_adapter" not in data
