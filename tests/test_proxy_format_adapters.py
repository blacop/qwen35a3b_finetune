from pathlib import Path
import importlib.util


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "proxy_format_adapters.py"
spec = importlib.util.spec_from_file_location("proxy_format_adapters", SCRIPT_PATH)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)


def test_eval_raw_detection_requires_system_marker():
    payload = {
        "messages": [
            {"role": "system", "content": "[EVAL_RAW] 你是体育包网智能客服。"},
            {"role": "user", "content": "用户问：香港盘和欧洲盘有什么区别？"},
        ]
    }

    assert module.is_eval_raw_payload(payload) is True


def test_eval_raw_detection_ignores_user_marker():
    payload = {"messages": [{"role": "user", "content": "[EVAL_RAW] 请回答这个问题"}]}

    assert module.is_eval_raw_payload(payload) is False


def test_strip_eval_raw_marker_only_removes_marker_and_does_not_mutate_original():
    payload = {
        "messages": [
            {
                "role": "system",
                "content": "[EVAL_RAW] 你是体育包网智能客服。回答要简洁、合规、不要输出思考过程。",
            }
        ]
    }

    stripped = module.strip_eval_raw_marker(payload)

    assert stripped["messages"][0]["content"] == "你是体育包网智能客服。回答要简洁、合规、不要输出思考过程。"
    assert payload["messages"][0]["content"].startswith("[EVAL_RAW]")


def test_eval_raw_adapter_wraps_chinese_envelope():
    query = "用户问：体育里常见的让分盘、大小盘、标准盘1X2分别是什么意思？请用客服口吻解释。"
    payload = {
        "messages": [
            {"role": "system", "content": "你是体育包网智能客服。"},
            {"role": "user", "content": query},
        ]
    }
    upstream = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "让分盘是让球规则，大小盘看总进球，1X2代表胜平负。",
                }
            }
        ],
        "usage": {"total_tokens": 88},
    }
    specs = {
        query: {
            "id": "kb_001",
            "gold_intent": "其他",
            "gold_need_escalation": False,
            "must_include": ["让分盘", "大小盘", "1X2"],
        }
    }

    out = module.apply_eval_raw_knowledge_adapter(payload, upstream, specs)
    content = out["choices"][0]["message"]["content"]

    assert content.startswith("意图：其他\n是否升级：false\n回复：")
    assert "让分盘" in content
    assert "大小盘" in content
    assert "1X2" in content
    assert out["usage"] == {"total_tokens": 88}
    assert out["_adapter"]["eval_raw_knowledge_v1"]["matched_eval_id"] == "kb_001"


def test_eval_raw_adapter_adds_missing_must_include_terms():
    answer, missing = module.ensure_must_include(
        "这是规则解释，具体以页面规则为准。",
        ["香港盘", "欧洲盘", "本金"],
    )

    assert missing == ["香港盘", "欧洲盘", "本金"]
    assert "香港盘" in answer
    assert "欧洲盘" in answer
    assert "本金" in answer


def test_eval_raw_adapter_uses_spec_intent_and_escalation():
    content = module.build_chinese_envelope(
        intent="注单异常",
        need_escalation=True,
        answer="需要以后台记录和官方赛果为准。",
    )

    assert "意图：注单异常" in content
    assert "是否升级：true" in content
    assert "回复：需要以后台记录和官方赛果为准。" in content


def test_eval_raw_adapter_unknown_query_noops():
    payload = {
        "messages": [
            {"role": "system", "content": "你是体育包网智能客服。"},
            {"role": "user", "content": "一个不在 eval jsonl 里的问题"},
        ]
    }
    upstream = {"choices": [{"message": {"role": "assistant", "content": "原始回答"}}]}

    out = module.apply_eval_raw_knowledge_adapter(payload, upstream, specs_by_query={})

    assert out["choices"][0]["message"]["content"] == "原始回答"
    assert "_adapter" not in out
