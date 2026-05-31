from pathlib import Path
import importlib.util
import sys


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "eval_tool_dialogue_gate.py"
spec = importlib.util.spec_from_file_location("eval_tool_dialogue_gate", SCRIPT_PATH)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def test_evaluate_knowledge_sample_uses_structured_intent_and_escalation():
    sample = {
        "user_query": "比赛结束了但是我的单子还没结算",
        "gold_intent": "注单异常",
        "gold_need_escalation": True,
        "must_include": ["结算", "后台"],
        "must_not_include": ["作为AI"],
    }
    resp = {
        "choices": [
            {
                "message": {
                    "content": '{"intent":"注单异常","need_escalation":true,"answer":"您好，这边看到是结算延迟，需后台进一步核实处理。"}'
                }
            }
        ]
    }

    score = module.evaluate_knowledge_sample(sample, resp)

    assert score["intent_hit"] == 1
    assert score["escalation_hit"] == 1
    assert score["pred_intent"] == "注单异常"
    assert score["pred_need_escalation"] is True
    assert score["include_score"] == 1.0


def test_evaluate_knowledge_sample_falls_back_to_plain_text_intent_and_escalation():
    sample = {
        "user_query": "充值没到账",
        "gold_intent": "充值",
        "gold_need_escalation": True,
        "must_include": ["订单", "核实"],
        "must_not_include": ["作为AI"],
    }
    resp = {
        "choices": [
            {
                "message": {
                    "content": "您好，充值订单还未入账，这边建议提供订单信息给后台核实处理。"
                }
            }
        ]
    }

    score = module.evaluate_knowledge_sample(sample, resp)

    assert score["intent_hit"] == 1
    assert score["escalation_hit"] == 1
    assert score["pred_intent"] == "充值"
    assert score["pred_need_escalation"] is True
    assert score["include_score"] == 1.0


def test_evaluate_knowledge_sample_handles_reasoning_only_json():
    sample = {
        "user_query": "串关有一场取消了怎么算",
        "gold_intent": "串关规则",
        "gold_need_escalation": False,
        "must_include": ["取消", "结算"],
        "must_not_include": ["作为AI"],
    }
    resp = {
        "choices": [
            {
                "message": {
                    "content": None,
                    "reasoning": [
                        {"type": "reasoning_text", "text": "先分析。"},
                        {
                            "type": "output_text",
                            "text": '{"intent":"串关规则","need_escalation":false,"answer":"您好，串关里若有一场取消，通常会按该场赔率1处理，其他场次继续结算。"}',
                        },
                    ],
                }
            }
        ]
    }

    score = module.evaluate_knowledge_sample(sample, resp)

    assert score["intent_hit"] == 1
    assert score["escalation_hit"] == 1
    assert score["pred_intent"] == "串关规则"
    assert score["pred_need_escalation"] is False
