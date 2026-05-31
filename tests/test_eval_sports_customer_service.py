from pathlib import Path
import importlib.util
import csv
import sys


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "eval_sports_customer_service.py"
spec = importlib.util.spec_from_file_location("eval_sports_customer_service", SCRIPT_PATH)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def test_extract_json_from_reasoning_text_with_leading_chain_of_thought():
    reasoning = (
        "我先分析一下用户需求，然后再输出结果。\n"
        "{\"intent\":\"活动\",\"need_escalation\":false,"
        "\"answer\":\"当前首充活动仍可参与。\","
        "\"risk_flag\":[],\"next_action\":\"引导查看活动规则\"}"
    )

    parsed = module.extract_json_obj(reasoning)

    assert parsed is not None
    assert parsed["intent"] == "活动"
    assert parsed["need_escalation"] is False


def test_strip_think_blocks_removes_qwen_think_tags_only():
    text = "<think>internal reasoning</think>{\"answer\":\"ok\"}"

    cleaned = module.strip_think_blocks(text)

    assert cleaned == '{"answer":"ok"}'


def test_evaluate_one_accepts_structured_reasoning_chunks(monkeypatch):
    def fake_call_chat_completion(**_kwargs):
        return {
            "choices": [
                {
                    "message": {
                        "content": None,
                        "reasoning": [
                            {"type": "reasoning_text", "text": "先分析问题。"},
                            {
                                "type": "output_text",
                                "text": '{"intent":"活动","need_escalation":false,"answer":"当前首充活动仍可参与。","risk_flag":[],"next_action":"引导查看活动规则"}',
                            },
                        ],
                    }
                }
            ]
        }

    monkeypatch.setattr(module, "call_chat_completion", fake_call_chat_completion)

    result = module.evaluate_one(
        sample={
            "id": "case_reasoning_list",
            "scenario": "活动咨询",
            "user_query": "首充活动还有吗？",
            "gold_intent": "活动",
            "gold_need_escalation": False,
        },
        base_url="http://127.0.0.1:8001",
        model="qwen35a3b-dpo-latest",
        timeout_sec=30,
        max_retries=0,
        temperature=0.0,
        max_tokens=128,
        top_p=0.95,
        api_key="dummy",
        system_prompt=module.DEFAULT_SYSTEM_PROMPT,
        http_backend="urllib",
        use_guardrails=False,
    )

    assert result.request_ok == 1
    assert result.pred_intent == "活动"
    assert result.pred_need_escalation is False
    assert result.answer == "当前首充活动仍可参与。"


def test_read_csv_samples_supports_flat_export_schema(tmp_path):
    csv_path = tmp_path / "regression.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "id",
                "scenario",
                "query",
                "answer",
                "must_include_json",
                "must_not_include_json",
                "meta_json",
                "messages_json",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "id": "ops_policy_abc_a_01",
                "scenario": "模式确认-A",
                "query": "平台A到底开的是哪种代理模式？",
                "answer": "平台A只开流水代理，不开净盈利代理。",
                "must_include_json": "[\"流水代理\",\"自动结算\"]",
                "must_not_include_json": "[\"净盈利代理都开放\"]",
                "meta_json": "{\"platform_pattern\":\"A\",\"needs_backend\":false}",
                "messages_json": "[{\"role\":\"user\",\"content\":\"平台A到底开的是哪种代理模式？\"}]",
            }
        )

    rows = module.read_csv(csv_path)

    assert len(rows) == 1
    row = rows[0]
    assert row["user_query"] == "平台A到底开的是哪种代理模式？"
    assert row["gold_answer"] == "平台A只开流水代理，不开净盈利代理。"
    assert row["must_include"] == ["流水代理", "自动结算"]
    assert row["must_not_include"] == ["净盈利代理都开放"]
    assert row["meta"]["platform_pattern"] == "A"
    assert "messages" not in row


def test_read_csv_samples_can_opt_in_to_messages(tmp_path):
    csv_path = tmp_path / "regression.csv"
    csv_path.write_text(
        "id,query,messages_json\n"
        "ops_policy_abc_a_02,平台A是流水代理还是净盈利代理？,\"[{\"\"role\"\":\"\"user\"\",\"\"content\"\":\"\"平台A是流水代理还是净盈利代理？\"\"}]\"\n",
        encoding="utf-8",
    )

    rows = module.read_csv(csv_path, use_messages=True)

    assert len(rows) == 1
    assert rows[0]["messages"][0]["role"] == "user"
