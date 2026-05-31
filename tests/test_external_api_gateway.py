from pathlib import Path
import importlib.util
import json
import sys


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "external_api_gateway.py"


def load_module(monkeypatch, tmp_path, **env):
    key_file = tmp_path / "api_keys.json"
    key_file.write_text(
        json.dumps(
            {
                "keys": [
                    {
                        "id": "client-a",
                        "key": "sk-client-a",
                        "models": ["baowang-gpu5"],
                        "routes": ["models", "chat"],
                        "rpm": 2,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("GATEWAY_API_KEYS_FILE", str(key_file))
    monkeypatch.setenv("GATEWAY_DEFAULT_RPM", "2")
    monkeypatch.delenv("GATEWAY_API_KEYS_JSON", raising=False)
    monkeypatch.delenv("GATEWAY_MODEL_ROUTES_JSON", raising=False)
    monkeypatch.delenv("GATEWAY_MODEL_SYSTEM_PROMPTS_JSON", raising=False)
    monkeypatch.delenv("GATEWAY_MODEL_SYSTEM_PROMPTS_FILE", raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    name = "external_api_gateway_test"
    sys.modules.pop(name, None)
    spec = importlib.util.spec_from_file_location(name, SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_auth_context_loads_key_and_model_permissions(monkeypatch, tmp_path):
    module = load_module(monkeypatch, tmp_path)

    ctx = module._auth_context("Bearer sk-client-a", "chat")

    assert ctx["id"] == "client-a"
    assert "baowang-gpu5" in ctx["models"]
    assert "baowang-gpu7" not in ctx["models"]


def test_resolve_route_rejects_unpermitted_model(monkeypatch, tmp_path):
    module = load_module(monkeypatch, tmp_path)
    ctx = module._auth_context("Bearer sk-client-a", "chat")

    try:
        module._resolve_route({"model": "baowang-gpu7"}, ctx, "chat")
    except module.HTTPException as exc:
        assert exc.status_code == 403
    else:
        raise AssertionError("unpermitted model should be rejected")


def test_resolve_route_rewrites_stable_alias(monkeypatch, tmp_path):
    module = load_module(monkeypatch, tmp_path)
    ctx = module._auth_context("Bearer sk-client-a", "chat")

    alias, route = module._resolve_route({"model": "baowang-gpu5"}, ctx, "chat")

    assert alias == "baowang-gpu5"
    assert route["upstream_model"] == "gpu5-v5.1f-merged"


def test_gateway_injects_model_system_prompt_when_missing(monkeypatch, tmp_path):
    module = load_module(
        monkeypatch,
        tmp_path,
        GATEWAY_MODEL_SYSTEM_PROMPTS_JSON='{"baowang-gpu5":"外部客服提示词"}',
    )

    payload, injected = module._inject_model_system_prompt(
        {"messages": [{"role": "user", "content": "串关作废怎么算？"}]},
        "baowang-gpu5",
    )

    assert injected is True
    assert payload["messages"][0]["role"] == "system"
    assert payload["messages"][0]["content"] == "外部客服提示词"


def test_gateway_keeps_existing_system_prompt(monkeypatch, tmp_path):
    module = load_module(
        monkeypatch,
        tmp_path,
        GATEWAY_MODEL_SYSTEM_PROMPTS_JSON='{"baowang-gpu5":"外部客服提示词"}',
    )
    original = {
        "messages": [
            {"role": "system", "content": "调用方自定义 system"},
            {"role": "user", "content": "串关作废怎么算？"},
        ]
    }

    payload, injected = module._inject_model_system_prompt(original, "baowang-gpu5")

    assert injected is False
    assert payload["messages"][0]["content"] == "调用方自定义 system"


def test_gateway_rewrites_completion_response_model(monkeypatch, tmp_path):
    module = load_module(monkeypatch, tmp_path)

    out = module._rewrite_completion_response_model(
        {"model": "gpu5-v5.1f-merged", "choices": []},
        "baowang-gpu5",
    )

    assert out["model"] == "baowang-gpu5"


def test_models_endpoint_hides_upstream_parent(monkeypatch, tmp_path):
    module = load_module(monkeypatch, tmp_path)

    resp = module.models("Bearer sk-client-a")
    body = resp.body.decode("utf-8")

    assert '"id":"baowang-gpu5"' in body
    assert '"parent":null' in body
    assert "gpu5-v5.1f-merged" not in body
