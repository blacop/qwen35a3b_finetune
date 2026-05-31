from pathlib import Path
import importlib.util
import os
import sys


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "dpo_guardrails_proxy.py"


def load_module(monkeypatch, **env):
    for key in [
        "PROXY_API_KEY",
        "PROXY_API_KEYS",
        "PROXY_RATE_LIMIT_RPM",
        "PROXY_AUDIT_LOG_JSONL",
        "PUBLIC_MODEL_ID",
        "PUBLIC_MODEL_UPSTREAM_ID",
        "MODEL_ALIAS_JSON",
        "MODEL_SYSTEM_PROMPTS_JSON",
        "MODEL_SYSTEM_PROMPTS_FILE",
        "UPSTREAM_MODEL_OVERRIDE",
        "DEFAULT_MODEL",
        "MIN_OUTPUT_TOKENS",
        "MAX_OUTPUT_TOKENS",
        "EVAL_MAX_OUTPUT_TOKENS",
    ]:
        monkeypatch.delenv(key, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    sys.path.insert(0, str(SCRIPT_PATH.parents[1]))
    name = "dpo_guardrails_proxy_api_governance_test"
    sys.modules.pop(name, None)
    spec = importlib.util.spec_from_file_location(name, SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_proxy_key_registry_supports_multiple_named_keys(monkeypatch):
    module = load_module(
        monkeypatch,
        PROXY_API_KEYS='{"agent-prod":"sk-prod","qa":"sk-qa"}',
        PROXY_RATE_LIMIT_RPM="0",
    )

    assert module._check_proxy_api_key("Bearer sk-prod") == "agent-prod"
    assert module._check_proxy_api_key("Bearer sk-qa") == "qa"


def test_proxy_rate_limit_is_per_key(monkeypatch):
    module = load_module(
        monkeypatch,
        PROXY_API_KEYS='{"qa":"sk-qa"}',
        PROXY_RATE_LIMIT_RPM="1",
    )

    assert module._check_proxy_api_key("Bearer sk-qa") == "qa"
    try:
        module._check_proxy_api_key("Bearer sk-qa")
    except module.HTTPException as exc:
        assert exc.status_code == 429
    else:
        raise AssertionError("second request should be rate-limited")


def test_model_alias_maps_public_id_to_upstream(monkeypatch):
    module = load_module(
        monkeypatch,
        DEFAULT_MODEL="gpu7-v5-combined",
        PUBLIC_MODEL_ID="gpu7-v5-combined",
        PUBLIC_MODEL_UPSTREAM_ID="qwen35a3b-sft-v5-combined",
    )

    payload = {}
    requested, upstream = module._apply_model_alias(payload)

    assert requested == "gpu7-v5-combined"
    assert upstream == "qwen35a3b-sft-v5-combined"
    assert payload["model"] == "qwen35a3b-sft-v5-combined"


def test_completion_response_model_is_rewritten_to_public_alias(monkeypatch):
    module = load_module(monkeypatch)

    out = module._rewrite_completion_response_model(
        {"model": "qwen35a3b-sft-v5-combined", "choices": []},
        "gpu7-v5-combined",
    )

    assert out["model"] == "gpu7-v5-combined"


def test_model_system_prompt_is_injected_when_missing(monkeypatch):
    module = load_module(
        monkeypatch,
        MODEL_SYSTEM_PROMPTS_JSON='{"gpu5-v5.1f-merged":"客服默认提示词"}',
    )

    payload, injected = module._inject_model_system_prompt(
        {"messages": [{"role": "user", "content": "提款不到账怎么办？"}]},
        "gpu5-v5.1f-merged",
    )

    assert injected is True
    assert payload["messages"][0]["role"] == "system"
    assert payload["messages"][0]["content"] == "客服默认提示词"


def test_model_system_prompt_does_not_override_existing_system(monkeypatch):
    module = load_module(
        monkeypatch,
        MODEL_SYSTEM_PROMPTS_JSON='{"gpu5-v5.1f-merged":"客服默认提示词"}',
    )
    original = {
        "messages": [
            {"role": "system", "content": "调用方自定义 system"},
            {"role": "user", "content": "提款不到账怎么办？"},
        ]
    }

    payload, injected = module._inject_model_system_prompt(original, "gpu5-v5.1f-merged")

    assert injected is True
    assert payload["messages"][0]["role"] == "system"
    assert "客服默认提示词" in payload["messages"][0]["content"]
    assert "调用方自定义 system" in payload["messages"][0]["content"]


def test_public_models_response_hides_raw_parent(monkeypatch):
    module = load_module(
        monkeypatch,
        PUBLIC_MODEL_ID="gpu7-v5-combined",
        PUBLIC_MODEL_NAME="GPU7 v5 combined merged",
    )

    class FakeResp:
        status_code = 200
        headers = {"content-type": "application/json"}

        @staticmethod
        def json():
            return {
                "object": "list",
                "data": [
                    {
                        "id": "qwen35a3b-sft-v5-combined",
                        "root": "qwen35a3b-sft-v5-combined",
                        "parent": "Qwen/Qwen3.5-35B-A3B",
                    }
                ],
            }

    resp = module._public_models_response(FakeResp())
    body = resp.body.decode("utf-8")

    assert "qwen35a3b-sft-v5-combined" not in body
    assert '"parent":null' in body
    assert "gpu7-v5-combined" in body


def test_audit_event_avoids_prompt_content(monkeypatch, tmp_path):
    audit_path = tmp_path / "audit.jsonl"
    module = load_module(monkeypatch, PROXY_AUDIT_LOG_JSONL=str(audit_path))
    payload = {
        "model": "gpu7-v5-combined",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
                    {"type": "text", "text": "不要把这段用户原文写入审计日志"},
                ],
            }
        ],
    }

    module._audit_proxy_event("/v1/chat/completions", "qa", payload)
    text = audit_path.read_text(encoding="utf-8")

    assert '"has_image": true' in text
    assert "不要把这段用户原文写入审计日志" not in text


def test_normal_chat_unwraps_internal_json_answer(monkeypatch):
    module = load_module(monkeypatch)
    payload = {
        "messages": [
            {"role": "user", "content": "代付通道一直处理中，提款不到账怎么办？"}
        ]
    }
    out = {
        "choices": [
            {
                "message": {
                    "content": '{"intent":"提款","need_escalation":true,"answer":"请提供订单号，这边提交财务核实。","risk_flag":[],"next_action":"提交财务核实"}'
                }
            }
        ]
    }

    patched, _notes = module._apply_guardrails_to_output(payload, out)

    content = patched["choices"][0]["message"]["content"]
    assert "请先提供账号" in content
    assert "风控状态" in content
    assert '"intent"' not in content


def test_withdrawal_query_is_not_misclassified_as_eval(monkeypatch):
    module = load_module(monkeypatch)

    assert module._is_eval_request([
        {"role": "user", "content": "代付通道一直处理中，提款不到账怎么办？"}
    ]) is False


def test_eval_chat_preserves_json_answer(monkeypatch):
    module = load_module(monkeypatch)
    payload = {
        "messages": [
            {
                "role": "user",
                "content": '用户问题：提款不到账怎么办？\n\n请基于当前对话，输出且仅输出一个JSON对象，字段如下：{"intent":"充值|提款"}',
            }
        ]
    }
    out = {
        "choices": [
            {
                "message": {
                    "content": '{"intent":"提款","need_escalation":true,"answer":"请提供订单号，这边提交财务核实。","risk_flag":[],"next_action":"提交财务核实"}'
                }
            }
        ]
    }

    patched, _notes = module._apply_guardrails_to_output(payload, out)

    content = patched["choices"][0]["message"]["content"]
    assert '"intent"' in content
    assert '"answer"' in content


def test_normal_chat_raises_tiny_max_tokens_to_floor(monkeypatch):
    module = load_module(
        monkeypatch,
        MIN_OUTPUT_TOKENS="256",
        MAX_OUTPUT_TOKENS="512",
        EVAL_MAX_OUTPUT_TOKENS="192",
    )

    assert module._resolve_max_tokens(20, is_eval=False, rag_boost=False) == 256
    assert module._resolve_max_tokens(128, is_eval=False, rag_boost=False) == 256


def test_eval_chat_keeps_requested_small_max_tokens(monkeypatch):
    module = load_module(
        monkeypatch,
        MIN_OUTPUT_TOKENS="256",
        MAX_OUTPUT_TOKENS="512",
        EVAL_MAX_OUTPUT_TOKENS="192",
    )

    assert module._resolve_max_tokens(20, is_eval=True, rag_boost=False) == 20


def test_sanitize_answer_removes_markdown_bold_markers(monkeypatch):
    module = load_module(monkeypatch)

    out, changed = module._sanitize_answer("**处理步骤**：请先刷新页面，再重登账号。")

    assert changed is True
    assert "**" not in out
    assert "处理步骤" in out


def test_non_escalation_template_for_complaint_is_short(monkeypatch):
    module = load_module(monkeypatch)

    out = module._build_non_escalation_template("投诉")

    assert "不能按普通退款处理" in out
    assert "工作人员" in out
