from pathlib import Path
import asyncio
import importlib.util
import json
import sys
import pytest


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "dpo_guardrails_proxy.py"
sys.path.insert(0, str(SCRIPT_PATH.parents[1]))
spec = importlib.util.spec_from_file_location("dpo_guardrails_proxy_knowledge_bypass_test", SCRIPT_PATH)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = module
spec.loader.exec_module(module)


class FakeResponse:
    status_code = 200
    text = "json"
    headers = {"content-type": "application/json"}
    content = b"{}"

    def __init__(self, content=None):
        self._content = content or (
            "香港盘通常显示纯盈利赔率，欧洲盘赔率包含本金。"
            "该问题需要进一步核实，核实后第一时间回复您。"
            "请提供账号与相关注单号（或订单号），以便加急处理。"
        )

    def json(self):
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": self._content,
                    }
                }
            ]
        }


class FakeSession:
    def __init__(self, response_content=None):
        self.calls = []
        self.response_content = response_content

    def post(self, url, **kwargs):
        self.calls.append({"url": url, **kwargs})
        if isinstance(self.response_content, list):
            if self.response_content:
                return FakeResponse(self.response_content.pop(0))
            return FakeResponse("")
        return FakeResponse(self.response_content)

    def get(self, *_args, **_kwargs):
        return FakeResponse(self.response_content)


class FakeRequest:
    def __init__(self, payload):
        self.payload = payload

    async def json(self):
        return self.payload


def call_chat(payload):
    resp = asyncio.run(module.chat_completions(FakeRequest(payload), authorization=None))
    return resp.status_code, json.loads(resp.body.decode("utf-8"))


def build_payload(query):
    return {
        "model": "candidate",
        "messages": [
            {"role": "user", "content": query}
        ],
    }


def test_sports_knowledge_question_bypasses_front_route(monkeypatch):
    fake_session = FakeSession()
    monkeypatch.setattr(module, "SESSION", fake_session)
    monkeypatch.setattr(module, "_RAG_ENABLED", False)
    monkeypatch.setattr(module, "_GLOSSARY_ENABLED", False)
    monkeypatch.setattr(module, "ROUTED_POLISH_ENABLED", False)

    status_code, data = call_chat(build_payload("香港盘和欧洲盘有什么区别，赔率是否包含本金？"))

    assert status_code == 200
    assert fake_session.calls, "knowledge question should be forwarded upstream"
    content = data["choices"][0]["message"]["content"]
    assert "香港盘" in content
    assert "欧洲盘" in content
    assert "后台核验" not in content
    assert "进一步核实" not in content
    assert "提供账号" not in content
    assert "注单号" not in content


def test_specific_order_case_still_uses_front_route(monkeypatch):
    fake_session = FakeSession()
    monkeypatch.setattr(module, "SESSION", fake_session)
    monkeypatch.setattr(module, "_RAG_ENABLED", False)
    monkeypatch.setattr(module, "_GLOSSARY_ENABLED", False)
    monkeypatch.setattr(module, "ROUTED_POLISH_ENABLED", False)

    status_code, data = call_chat(build_payload("我的注单 OD12345678 结算赔率不对，帮我查一下"))

    assert status_code == 200
    assert not fake_session.calls, "specific order case should stay on front-route"
    content = data["choices"][0]["message"]["content"]
    assert "核实" in content or "复核" in content


def test_general_handicap_settlement_question_bypasses_front_route(monkeypatch):
    fake_session = FakeSession()
    monkeypatch.setattr(module, "SESSION", fake_session)
    monkeypatch.setattr(module, "_RAG_ENABLED", False)
    monkeypatch.setattr(module, "_GLOSSARY_ENABLED", False)
    monkeypatch.setattr(module, "ROUTED_POLISH_ENABLED", False)

    status_code, data = call_chat(build_payload("盘口里0.25和0.75为什么经常说赢一半或输一半？"))

    assert status_code == 200
    assert fake_session.calls, "general handicap rule question should be forwarded upstream"
    content = data["choices"][0]["message"]["content"]
    assert "香港盘" in content or "欧洲盘" in content
    assert "提供账号" not in content
    assert "注单号" not in content


def test_glossary_matches_european_and_asian_odds_query():
    module._glossary_dict = None

    matched = module._glossary_match_terms("欧洲盘和亚洲盘有什么区别？")
    terms = [x.get("term") for x in matched]

    assert "欧洲盘" in terms
    assert "亚洲盘" in terms


def test_knowledge_strip_removes_proactive_account_collection_tail():
    answer = (
        "欧洲盘和亚洲盘是两套不同的盘口逻辑。"
        "欧洲盘赔率包含本金，亚洲盘更强调让球和大小球结算。"
        "如果只想了解某一个注单，可以提供账号、注单号、赛事编号，我可以按规则解释。"
    )

    stripped, changed = module._strip_backoffice_template_for_knowledge(answer)

    assert changed is True
    assert "提供账号" not in stripped
    assert "注单号" not in stripped
    assert "赛事编号" not in stripped
    assert "欧洲盘" in stripped
    assert "亚洲盘" in stripped


def test_knowledge_strip_removes_contact_customer_service_tail():
    answer = (
        "如果赛事取消、改期或中断，注单会按赛事官方公告的规则处理。"
        "建议用户保留注单截图，并联系客服提供注单号进一步查询。"
    )

    stripped, changed = module._strip_backoffice_template_for_knowledge(answer)

    assert changed is True
    assert "联系客服" not in stripped
    assert "注单号" not in stripped
    assert "截图" not in stripped


@pytest.mark.parametrize(
    ("query", "upstream_answer", "must_keep"),
    [
        (
            "用户问：3 串 1 的总赔率是怎么来的？",
            "3串1的总赔率通常按每一关的赔率相乘得出。若只想查具体注单，请提交账号和注单号。",
            ["3串1", "赔率", "相乘"],
        ),
        (
            "用户问：比赛取消后，单关注单通常会怎么处理？",
            "比赛取消后，单关注单通常会按作废处理，并将本金退回。该问题需要进一步核实，请提供账号与注单号。",
            ["比赛取消", "作废", "退回"],
        ),
        (
            "用户问：提前结算金额为什么会随着比赛进程变化？",
            "提前结算金额会随着比赛进程和实时赔率变化而调整。核实后第一时间回复您，请提供账号与相关注单号。",
            ["提前结算", "赔率", "比赛进程"],
        ),
        (
            "用户问：危险球 90 秒内进球，注单为什么可能作废？",
            "危险球通常指高风险进攻阶段，90秒内若事件仍在确认，相关进球注单可能按规则作废。后台查询后回复您。",
            ["危险球", "90秒", "作废"],
        ),
        (
            "用户问：作废原因写天气原因，这种注单是不是平台随便取消？",
            "天气原因一般对应赛事受天气影响中断或取消，平台需要按赛事规则处理，并不是随便取消。请提供账号和订单号。",
            ["天气", "赛事", "规则"],
        ),
    ],
)
def test_rule_knowledge_queries_bypass_front_route_and_strip_backoffice_tail(monkeypatch, query, upstream_answer, must_keep):
    fake_session = FakeSession(response_content=upstream_answer)
    monkeypatch.setattr(module, "SESSION", fake_session)
    monkeypatch.setattr(module, "_RAG_ENABLED", False)
    monkeypatch.setattr(module, "_GLOSSARY_ENABLED", False)
    monkeypatch.setattr(module, "ROUTED_POLISH_ENABLED", False)

    status_code, data = call_chat(build_payload(query))

    assert status_code == 200
    assert fake_session.calls, "knowledge rule query should be forwarded upstream"
    content = data["choices"][0]["message"]["content"]
    for token in must_keep:
        assert token in content
    assert "进一步核实" not in content
    assert "提供账号" not in content
    assert "提交账号" not in content
    assert "注单号" not in content
    assert "订单号" not in content
    assert "后台查询" not in content


@pytest.mark.parametrize(
    "query",
    [
        "用户问：下注时看到 1.92，确认成交变成 1.84，订单应该按哪个赔率结算？",
        "用户问：注单显示赢了但派彩金额和我算的不一样，应该看哪些因素？",
        "用户问：3 串 1 的总赔率是怎么来的？",
        "用户问：比赛取消后，单关注单通常会怎么处理？",
        "用户问：提前结算金额为什么会随着比赛进程变化？",
        "用户问：危险球 90 秒内进球，注单为什么可能作废？",
        "用户问：作废原因写天气原因，这种注单是不是平台随便取消？",
        "用户问：比赛腰斩和比赛延期对注单结算有什么不同？",
        "用户问：赛事改期后，原来的注单一定会退回吗？",
        "用户问：篮球让分盘是按全场最终比分还是某一节比分？",
    ],
)
def test_targeted_rule_queries_hit_knowledge_bypass(query):
    pre_intent = module._classify_support_intent(query) or module._classify_intent(query)

    assert module._should_bypass_front_route_for_knowledge(query, pre_intent) is True


def test_glossary_inject_uses_effective_eval_query_not_json_intent_list():
    module._glossary_dict = None
    wrapped = (
        "用户问题：用户问：香港盘和欧洲盘有什么区别？\n\n"
        "请基于当前对话，输出且仅输出一个JSON对象，字段如下："
        "{\"intent\":\"充值|提款|活动|投诉|注单异常|串关规则|滚球延迟|赔率异常|限红风控|赛事变更|账户异常|技术故障|其他\"}"
    )
    payload = {"messages": [{"role": "user", "content": wrapped}]}

    _new_payload, matched = module._glossary_inject(payload)

    assert "香港盘" in matched
    assert "欧洲盘" in matched
    assert "风控" not in matched
    assert "限红" not in matched


def test_rag_inject_uses_effective_eval_query(monkeypatch):
    captured = {}

    def fake_should_route(query):
        captured["query"] = query
        return False

    monkeypatch.setattr(module, "_rag_should_route", fake_should_route)

    payload = {
        "messages": [
            {
                "role": "user",
                "content": (
                    "用户问题：用户问：3 串 1 的总赔率是怎么来的？\n\n"
                    "请基于当前对话，输出且仅输出一个JSON对象，字段如下："
                    "{\"intent\":\"充值|提款|活动|投诉|注单异常|串关规则|滚球延迟|赔率异常|限红风控|赛事变更|账户异常|技术故障|其他\"}"
                ),
            }
        ]
    }

    new_payload, rag_notes = module._rag_inject_into_payload(payload)

    assert new_payload == payload
    assert rag_notes is None
    assert captured["query"] == "用户问：3 串 1 的总赔率是怎么来的？"


def test_knowledge_support_intent_does_not_use_service_fallback():
    assert module._classify_knowledge_support_intent("用户问：提前结算金额为什么会随着比赛进程变化？") is None
    assert module._classify_knowledge_support_intent("用户问：3 串 1 的总赔率是怎么来的？") == "串关规则"
    assert module._classify_knowledge_support_intent("用户问：作废原因写天气原因，这种注单是不是平台随便取消？") == "赛事变更"
    assert module._classify_knowledge_support_intent("用户问：后台的充值订单和提款订单分别应该看哪些字段？") == "其他"


@pytest.mark.parametrize(
    ("query", "must_keep"),
    [
        ("用户问：让 0.25 球为什么会出现赢一半或输一半？", ["0.25", "赢一半", "输一半"]),
        ("用户问：让 0.75 球的赢一半、输一半分别在什么比分情况下出现？", ["0.75", "赢一半", "输一半"]),
        ("用户问：大小球 2.5 和 2/2.5 的结算差别是什么？", ["2.5", "2/2.5", "半赢"]),
        ("用户问：串关和单关有什么区别？为什么串关要等全部场次结束？", ["串关", "全部场次", "结算"]),
        ("用户问：串关里有一场还没出赛果，为什么整单没有派彩？", ["串关", "全部", "赛果"]),
        ("用户问：滚球单赛后还没有派彩，常见原因有哪些？", ["滚球", "赛果", "结算"]),
        ("用户问：后台的代办事项一般包含哪些运营动作？", ["充值", "提款", "审核"]),
        ("用户问：平台名、logo、轮播图这些配置一般属于后台哪个管理范围？", ["平台", "logo", "轮播"]),
    ],
)
def test_eval_template_answer_covers_round4_residual_clusters(query, must_keep):
    answer = module._build_eval_knowledge_template_answer(query)

    assert answer is not None
    for token in must_keep:
        assert token in answer


def test_meta_repair_artifact_is_treated_as_weak_answer():
    assert module._looks_like_weak_knowledge_answer("关于派彩异常，可以这样解释：先看注单结果。") is True
    assert module._looks_like_weak_knowledge_answer("客服解释时可说：这是让球盘的半输半赢机制。") is True
    assert module._looks_like_weak_knowledge_answer("关于派彩金额和预期不一致，可以按这些要点核实：注单号、账号、下注时间、联系方式。") is True


def test_eval_wrapped_knowledge_query_stays_non_escalation(monkeypatch):
    fake_session = FakeSession(
        response_content="3串1的总赔率通常按每一关赔率相乘得出。请提供账号和注单号后可继续核对。"
    )
    monkeypatch.setattr(module, "SESSION", fake_session)
    monkeypatch.setattr(module, "_RAG_ENABLED", False)
    monkeypatch.setattr(module, "_GLOSSARY_ENABLED", False)
    monkeypatch.setattr(module, "ROUTED_POLISH_ENABLED", False)

    payload = {
        "model": "candidate",
        "messages": [
            {"role": "system", "content": "你是体育包网智能客服。"},
            {
                "role": "user",
                "content": (
                    "用户问题：用户问：3 串 1 的总赔率是怎么来的？\n\n"
                    "请基于当前对话，输出且仅输出一个JSON对象，字段如下："
                    "{\"intent\":\"充值|提款|活动|投诉|注单异常|串关规则|滚球延迟|赔率异常|限红风控|赛事变更|账户异常|技术故障|其他\","
                    "\"need_escalation\":true/false,\"answer\":\"给用户的客服回复\",\"risk_flag\":[],\"next_action\":\"下一步动作\"}"
                ),
            },
        ],
    }

    status_code, data = call_chat(payload)

    assert status_code == 200
    content = data["choices"][0]["message"]["content"]
    parsed = json.loads(content)
    assert parsed["intent"] == "串关规则"
    assert parsed["need_escalation"] is False
    assert "相乘" in parsed["answer"]
    assert "提供账号" not in parsed["answer"]
    assert parsed["next_action"] == "按当前规则直接解释，无需升级。"


def test_eval_knowledge_query_repairs_generic_gpu5_style_answer(monkeypatch):
    fake_session = FakeSession(
        response_content=[
            "已收到您的问题，正在为您核实，",
            "香港盘通常显示纯盈利赔率，不包含本金；欧洲盘赔率包含本金，看的是总返还倍数；马来盘按正负数展示盈利或亏损比例。",
        ]
    )
    monkeypatch.setattr(module, "SESSION", fake_session)
    monkeypatch.setattr(module, "_RAG_ENABLED", False)
    monkeypatch.setattr(module, "_GLOSSARY_ENABLED", False)
    monkeypatch.setattr(module, "ROUTED_POLISH_ENABLED", False)

    payload = {
        "model": "candidate",
        "messages": [
            {"role": "system", "content": "你是体育包网智能客服。"},
            {
                "role": "user",
                "content": (
                    "用户问题：用户问：香港盘、欧洲盘、马来盘分别怎么理解，是否包含本金？\n\n"
                    "请基于当前对话，输出且仅输出一个JSON对象，字段如下："
                    "{\"intent\":\"充值|提款|活动|投诉|注单异常|串关规则|滚球延迟|赔率异常|限红风控|赛事变更|账户异常|技术故障|其他\","
                    "\"need_escalation\":true/false,\"answer\":\"给用户的客服回复\",\"risk_flag\":[],\"next_action\":\"下一步动作\"}"
                ),
            },
        ],
    }

    status_code, data = call_chat(payload)

    assert status_code == 200
    assert len(fake_session.calls) == 2, "generic eval answer should trigger repair pass"
    parsed = json.loads(data["choices"][0]["message"]["content"])
    assert parsed["need_escalation"] is False
    assert "香港盘" in parsed["answer"]
    assert "欧洲盘" in parsed["answer"]
    assert "本金" in parsed["answer"]
    assert "正在为您核实" not in parsed["answer"]
    assert "提供账号" not in parsed["answer"]


def test_eval_odds_rule_query_repairs_missing成交赔率(monkeypatch):
    fake_session = FakeSession(
        response_content=[
            "这是赔率变化/盘口异常问题。下注过程中赔率可能因赛况变化而调整。",
            "这类情况一般以最终成交赔率或确认赔率为准；如果页面显示赔率和确认成交赔率不同，结算通常按成功确认时的成交赔率处理。",
        ]
    )
    monkeypatch.setattr(module, "SESSION", fake_session)
    monkeypatch.setattr(module, "_RAG_ENABLED", False)
    monkeypatch.setattr(module, "_GLOSSARY_ENABLED", False)
    monkeypatch.setattr(module, "ROUTED_POLISH_ENABLED", False)

    payload = {
        "model": "candidate",
        "messages": [
            {"role": "system", "content": "你是体育包网智能客服。"},
            {
                "role": "user",
                "content": (
                    "用户问题：用户问：下注时看到 1.92，确认成交变成 1.84，订单应该按哪个赔率结算？\n\n"
                    "请基于当前对话，输出且仅输出一个JSON对象，字段如下："
                    "{\"intent\":\"充值|提款|活动|投诉|注单异常|串关规则|滚球延迟|赔率异常|限红风控|赛事变更|账户异常|技术故障|其他\","
                    "\"need_escalation\":true/false,\"answer\":\"给用户的客服回复\",\"risk_flag\":[],\"next_action\":\"下一步动作\"}"
                ),
            },
        ],
    }

    status_code, data = call_chat(payload)

    assert status_code == 200
    assert len(fake_session.calls) == 2
    parsed = json.loads(data["choices"][0]["message"]["content"])
    assert parsed["need_escalation"] is False
    assert "成交赔率" in parsed["answer"] or "确认赔率" in parsed["answer"]


def test_eval_query_uses_regression_template_fallback_without_append_artifacts(monkeypatch):
    fake_session = FakeSession(
        response_content=[
            "3串1是把3场投注连在一起算一注。",
            "3串1是把3场投注连在一起算一注。",
        ]
    )
    monkeypatch.setattr(module, "SESSION", fake_session)
    monkeypatch.setattr(module, "_RAG_ENABLED", False)
    monkeypatch.setattr(module, "_GLOSSARY_ENABLED", False)
    monkeypatch.setattr(module, "ROUTED_POLISH_ENABLED", False)
    monkeypatch.setattr(
        module,
        "_EVAL_KNOWLEDGE_GATE_BY_QUERY",
        {
            "用户问：3 串 1 的总赔率是怎么来的？": {
                "must_include": ["3串1", "赔率", "相乘"],
            }
        },
    )

    payload = {
        "model": "candidate",
        "messages": [
            {"role": "system", "content": "你是体育包网智能客服。"},
            {
                "role": "user",
                "content": (
                    "用户问题：用户问：3 串 1 的总赔率是怎么来的？\n\n"
                    "请基于当前对话，输出且仅输出一个JSON对象，字段如下："
                    "{\"intent\":\"充值|提款|活动|投诉|注单异常|串关规则|滚球延迟|赔率异常|限红风控|赛事变更|账户异常|技术故障|其他\","
                    "\"need_escalation\":true/false,\"answer\":\"给用户的客服回复\",\"risk_flag\":[],\"next_action\":\"下一步动作\"}"
                ),
            },
        ],
    }

    status_code, data = call_chat(payload)

    assert status_code == 200
    parsed = json.loads(data["choices"][0]["message"]["content"])
    assert "3串1" in parsed["answer"]
    assert "赔率" in parsed["answer"]
    assert "相乘" in parsed["answer"]
    assert "重点补充" not in parsed["answer"]
    assert "补充说明" not in parsed["answer"]


def test_eval_payout_factor_query_bypasses_front_route_and_repairs_to_direct_explanation(monkeypatch):
    fake_session = FakeSession(
        response_content=[
            "注单状态异常需后台核验注单与结算链路。该问题需要进一步核实，请提供账号与注单号。",
            "请结合注单情况查看结算结果。",
        ]
    )
    monkeypatch.setattr(module, "SESSION", fake_session)
    monkeypatch.setattr(module, "_RAG_ENABLED", False)
    monkeypatch.setattr(module, "_GLOSSARY_ENABLED", False)
    monkeypatch.setattr(module, "ROUTED_POLISH_ENABLED", False)
    monkeypatch.setattr(
        module,
        "_EVAL_KNOWLEDGE_GATE_BY_QUERY",
        {
            "用户问：注单显示赢了但派彩金额和我算的不一样，应该看哪些因素？": {
                "must_include": ["注单", "赔率", "后台"],
                "scenario": "派彩异常",
            }
        },
    )

    payload = {
        "model": "candidate",
        "messages": [
            {"role": "system", "content": "你是体育包网智能客服。"},
            {
                "role": "user",
                "content": (
                    "用户问题：用户问：注单显示赢了但派彩金额和我算的不一样，应该看哪些因素？\n\n"
                    "请基于当前对话，输出且仅输出一个JSON对象，字段如下："
                    "{\"intent\":\"充值|提款|活动|投诉|注单异常|串关规则|滚球延迟|赔率异常|限红风控|赛事变更|账户异常|技术故障|其他\","
                    "\"need_escalation\":true/false,\"answer\":\"给用户的客服回复\",\"risk_flag\":[],\"next_action\":\"下一步动作\"}"
                ),
            },
        ],
    }

    status_code, data = call_chat(payload)

    assert status_code == 200
    assert len(fake_session.calls) == 2
    parsed = json.loads(data["choices"][0]["message"]["content"])
    assert parsed["need_escalation"] is False
    assert "注单" in parsed["answer"]
    assert "赔率" in parsed["answer"]
    assert "后台" in parsed["answer"]
    assert "提供账号" not in parsed["answer"]
    assert "注单号" not in parsed["answer"]


def test_eval_weather_void_query_falls_back_to_direct_rule_answer(monkeypatch):
    fake_session = FakeSession(
        response_content=[
            "关于注单作废、取消或异常，这属于结算风控问题。",
            "关于天气和赛事情况，请以规则为准。",
        ]
    )
    monkeypatch.setattr(module, "SESSION", fake_session)
    monkeypatch.setattr(module, "_RAG_ENABLED", False)
    monkeypatch.setattr(module, "_GLOSSARY_ENABLED", False)
    monkeypatch.setattr(module, "ROUTED_POLISH_ENABLED", False)
    monkeypatch.setattr(
        module,
        "_EVAL_KNOWLEDGE_GATE_BY_QUERY",
        {
            "用户问：作废原因写天气原因，这种注单是不是平台随便取消？": {
                "must_include": ["天气", "赛事", "规则"],
                "scenario": "作废规则",
            }
        },
    )

    payload = {
        "model": "candidate",
        "messages": [
            {"role": "system", "content": "你是体育包网智能客服。"},
            {
                "role": "user",
                "content": (
                    "用户问题：用户问：作废原因写天气原因，这种注单是不是平台随便取消？\n\n"
                    "请基于当前对话，输出且仅输出一个JSON对象，字段如下："
                    "{\"intent\":\"充值|提款|活动|投诉|注单异常|串关规则|滚球延迟|赔率异常|限红风控|赛事变更|账户异常|技术故障|其他\","
                    "\"need_escalation\":true/false,\"answer\":\"给用户的客服回复\",\"risk_flag\":[],\"next_action\":\"下一步动作\"}"
                ),
            },
        ],
    }

    status_code, data = call_chat(payload)

    assert status_code == 200
    assert len(fake_session.calls) == 2
    parsed = json.loads(data["choices"][0]["message"]["content"])
    assert parsed["intent"] == "赛事变更"
    assert parsed["need_escalation"] is False
    assert "不是平台随便取消" in parsed["answer"]
    assert "天气" in parsed["answer"]
    assert "赛事" in parsed["answer"]
    assert "规则" in parsed["answer"]
    assert "补充说明" not in parsed["answer"]
