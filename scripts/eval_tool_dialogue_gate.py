#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional


EVAL_RAW_SYSTEM_PROMPT = "[EVAL_RAW] 你是体育包网智能客服。回答要简洁、合规、不要输出思考过程。"
EVAL_JSON_INSTRUCTION = """请基于当前对话，输出且仅输出一个JSON对象，字段如下：
{
  "intent": "充值|提款|活动|投诉|注单异常|串关规则|滚球延迟|赔率异常|限红风控|赛事变更|账户异常|技术故障|其他",
  "need_escalation": true/false,
  "answer": "给用户的客服回复"
}
不要输出Markdown代码块，不要输出额外解释。"""

INTENT_ALIASES: Dict[str, List[str]] = {
    "充值": ["充值", "入款", "上分", "存款", "转入"],
    "提款": ["提款", "提现", "出款", "下分", "取款"],
    "活动": ["活动", "优惠", "红利", "送彩金", "奖励", "返利", "任务"],
    "投诉": ["投诉", "申诉", "不满", "差评", "举报", "客服态度"],
    "注单异常": ["注单异常", "注单查询", "注单取消", "注单作废", "注单", "订单异常", "单据异常"],
    "串关规则": ["串关", "过关", "多关", "组合投注", "串子"],
    "滚球延迟": ["滚球延迟", "滚球", "延迟结算", "赛中延迟", "inplay"],
    "赔率异常": ["赔率异常", "赔率", "盘口异常", "跳水", "赔率变更", "赔率跳变"],
    "限红风控": ["限红", "限额", "风控", "限制下注", "账户审核", "高风险"],
    "赛事变更": ["赛事变更", "改期", "延期", "取消比赛", "腰斩", "赛程变更"],
    "账户异常": ["账户异常", "登录异常", "冻结", "封禁", "账号异常", "无法登录"],
    "技术故障": ["技术故障", "系统错误", "接口报错", "页面打不开", "卡顿", "闪退", "bug"],
    "其他": [],
}


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def post_json(url: str, payload: Dict[str, Any], api_key: str = "", timeout: float = 60) -> Dict[str, Any]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(url, data=json.dumps(payload, ensure_ascii=False).encode(), headers=headers)
    t0 = time.time()
    try:
        raw = urllib.request.urlopen(req, timeout=timeout).read().decode()
        data = json.loads(raw)
        data["_latency_ms"] = round((time.time() - t0) * 1000, 2)
        return data
    except Exception as e:
        return {"error": str(e), "_latency_ms": round((time.time() - t0) * 1000, 2)}


def extract_message(resp: Dict[str, Any]) -> Dict[str, Any]:
    try:
        return resp["choices"][0]["message"] or {}
    except Exception:
        return {}


def flatten_message_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        if value.get("text"):
            return str(value["text"])
        if value.get("content"):
            return flatten_message_text(value["content"])
        return ""
    if isinstance(value, list):
        parts: List[str] = []
        for item in value:
            text = flatten_message_text(item)
            if text:
                parts.append(text)
        return "".join(parts)
    return str(value)


def parse_args(raw_args: Any) -> Dict[str, Any]:
    if isinstance(raw_args, dict):
        return raw_args
    if isinstance(raw_args, str):
        try:
            data = json.loads(raw_args)
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}
    return {}


def _coerce_numeric_str(v: Any) -> Any:
    if isinstance(v, bool) or v is None:
        return v
    if isinstance(v, (int, float)):
        return v
    if not isinstance(v, str):
        return v
    s = v.strip()
    if not s:
        return v
    if re.fullmatch(r"[+-]?\d+", s):
        try:
            return int(s)
        except Exception:
            return v
    if re.fullmatch(r"[+-]?(?:\d+\.\d*|\.\d+)", s):
        try:
            return float(s)
        except Exception:
            return v
    return v


def _args_value_equal(actual: Any, expected: Any) -> bool:
    if expected == "":
        return actual == expected
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            return False
        return all(_args_value_equal(actual.get(k), v) for k, v in expected.items())
    if isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            return False
        return all(_args_value_equal(a, b) for a, b in zip(actual, expected))
    a = _coerce_numeric_str(actual)
    b = _coerce_numeric_str(expected)
    return a == b


def args_match_expected(actual_args: Dict[str, Any], expected_args: Dict[str, Any]) -> bool:
    return all(_args_value_equal(actual_args.get(k), v) for k, v in expected_args.items())


def normalize_intent(intent: str) -> str:
    s = (intent or "").strip().lower()
    if not s:
        return ""
    s = re.sub(r"[\s_/|]+", "", s)
    for canonical, aliases in INTENT_ALIASES.items():
        for alias in [canonical] + aliases:
            a = re.sub(r"[\s_/|]+", "", alias.lower())
            if a and (a in s or s in a):
                return canonical
    return "其他"


def parse_bool(v: Any) -> Optional[bool]:
    if v is None or v == "":
        return None
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    s = str(v).strip().lower()
    if s in {"true", "1", "yes", "y", "是", "需要", "需转", "需升级", "需要升级"}:
        return True
    if s in {"false", "0", "no", "n", "否", "不需要", "无需", "不需升级"}:
        return False
    return None


def extract_json_obj(text: str) -> Optional[Dict[str, Any]]:
    text = text.strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    fence = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, flags=re.S)
    if fence:
        try:
            obj = json.loads(fence.group(1))
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    obj = json.loads(text[start : i + 1])
                    if isinstance(obj, dict):
                        return obj
                except Exception:
                    return None
    return None


def strip_think_blocks(text: str) -> str:
    if not text:
        return text
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S | re.I)
    text = re.sub(r"```think.*?```", "", text, flags=re.S | re.I)
    return text.strip()


def evaluate_tool_sample(sample: Dict[str, Any], resp: Dict[str, Any]) -> Dict[str, Any]:
    expected_tool = sample["expected_tool"]
    expected_args = sample.get("expected_args") or {}
    must_include = sample.get("must_include") or []
    msg = extract_message(resp)
    tool_calls = msg.get("tool_calls") or []
    trace = ((resp.get("_agent") or {}).get("trace") or [])

    tool_called = False
    args_match = False
    schema_valid = 1

    for tc in tool_calls:
        fn = tc.get("function") or {}
        fn_name = fn.get("name")
        args = parse_args(fn.get("arguments") or {})
        if not fn_name or not isinstance(args, dict):
            schema_valid = 0
        if fn_name == expected_tool:
            tool_called = True
            if args_match_expected(args, expected_args):
                args_match = True

    if not tool_called:
        for item in trace:
            if item.get("tool") == expected_tool:
                tool_called = True
                args = item.get("args") or {}
                if args_match_expected(args, expected_args):
                    args_match = True

    content = msg.get("content") or ""
    must_score = sum(1 for term in must_include if term in content) / max(len(must_include), 1)
    return {
        "tool_called": int(tool_called),
        "args_match": int(args_match),
        "schema_valid": int(schema_valid),
        "must_include": must_score,
        "overall": int(tool_called) * 0.4 + int(args_match) * 0.35 + int(schema_valid) * 0.15 + must_score * 0.10,
    }


def evaluate_knowledge_sample(sample: Dict[str, Any], resp: Dict[str, Any]) -> Dict[str, Any]:
    msg = extract_message(resp)
    content = flatten_message_text(msg.get("content") or "")
    reasoning = flatten_message_text(msg.get("reasoning", "") or msg.get("reasoning_content", ""))
    if not str(content).strip() and reasoning:
        content = reasoning
    parsed = extract_json_obj(str(content))
    answer = strip_think_blocks(str((parsed or {}).get("answer") or content or ""))
    must_include = sample.get("must_include") or []
    must_not = sample.get("must_not_include") or []
    include_score = sum(1 for term in must_include if term in answer) / max(len(must_include), 1)
    must_not_ok = int(not any(term in answer for term in must_not))
    gold_intent = normalize_intent(str(sample.get("gold_intent") or ""))
    gold_escalation = parse_bool(sample.get("gold_need_escalation"))
    pred_intent_raw = str((parsed or {}).get("intent") or "")
    if not pred_intent_raw:
        pred_intent_raw = answer + " " + str(sample.get("user_query") or "")
    pred_intent = normalize_intent(pred_intent_raw)
    pred_escalation = parse_bool((parsed or {}).get("need_escalation"))
    if pred_escalation is None:
        pred_escalation = bool(re.search(r"(后台|运营|人工|升级处理|提交工单|核实后回复|查询后回复|审核)", answer))

    intent_hit = 1
    if gold_intent:
        intent_hit = int(pred_intent == gold_intent)
    escalation_hit = 1
    if gold_escalation is not None and pred_escalation is not None:
        escalation_hit = int(pred_escalation == gold_escalation)

    overall = include_score * 0.5 + must_not_ok * 0.2 + intent_hit * 0.15 + escalation_hit * 0.15
    return {
        "include_score": include_score,
        "must_not_ok": must_not_ok,
        "intent_hit": intent_hit,
        "escalation_hit": escalation_hit,
        "pred_intent": pred_intent,
        "pred_need_escalation": pred_escalation,
        "answer": answer,
        "overall": overall,
    }


def avg(rows: List[Dict[str, Any]], key: str) -> float:
    vals = [float(row.get(key, 0)) for row in rows]
    return sum(vals) / max(len(vals), 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Gate v5.1 tool-dialogue migration candidate.")
    parser.add_argument("--base-url", required=True, help="OpenAI-compatible base URL, for example http://127.0.0.1:8001/v1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-key", default="")
    parser.add_argument("--tool-eval-jsonl", default="/home/ubuntu/qwen35a3b_finetune/datasets/tool_use/eval_tool_use_30.jsonl")
    parser.add_argument("--knowledge-eval-jsonl", default="/home/ubuntu/qwen35a3b_finetune/datasets/eval_sports_baowang_knowledge_40_20260428.jsonl")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=60)
    parser.add_argument("--tool-max-tokens", type=int, default=512)
    parser.add_argument("--knowledge-max-tokens", type=int, default=512)
    parser.add_argument("--min-tool-overall", type=float, default=0.98)
    parser.add_argument("--min-tool-called", type=float, default=0.98)
    parser.add_argument("--min-tool-schema", type=float, default=0.99)
    parser.add_argument("--min-knowledge-overall", type=float, default=0.92)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base = args.base_url.rstrip("/")
    tool_url = base + "/chat/completions/agent"
    chat_url = base + "/chat/completions"

    tool_samples = read_jsonl(Path(args.tool_eval_jsonl))
    knowledge_samples = read_jsonl(Path(args.knowledge_eval_jsonl))

    def run_tool(sample: Dict[str, Any]) -> Dict[str, Any]:
        payload = {
            "model": args.model,
            "messages": [{"role": "user", "content": sample["user_query"]}],
            "temperature": 0,
            "max_tokens": args.tool_max_tokens,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        resp = post_json(tool_url, payload, args.api_key, args.timeout)
        score = evaluate_tool_sample(sample, resp)
        return {**sample, "response": resp, "score": score}

    def run_knowledge(sample: Dict[str, Any]) -> Dict[str, Any]:
        payload = {
            "model": args.model,
            "messages": [
                {"role": "system", "content": EVAL_RAW_SYSTEM_PROMPT},
                {"role": "user", "content": f"用户问题：{sample['user_query']}\n\n{EVAL_JSON_INSTRUCTION}"},
            ],
            "temperature": 0,
            "max_tokens": args.knowledge_max_tokens,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        resp = post_json(chat_url, payload, args.api_key, args.timeout)
        score = evaluate_knowledge_sample(sample, resp)
        return {**sample, "response": resp, "score": score}

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        tool_results = list(ex.map(run_tool, tool_samples))
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        knowledge_results = list(ex.map(run_knowledge, knowledge_samples))

    tool_scores = [r["score"] for r in tool_results]
    knowledge_scores = [r["score"] for r in knowledge_results]
    summary = {
        "tool": {
            "samples": len(tool_results),
            "overall": avg(tool_scores, "overall"),
            "tool_called": avg(tool_scores, "tool_called"),
            "args_match": avg(tool_scores, "args_match"),
            "schema_valid": avg(tool_scores, "schema_valid"),
        },
        "knowledge": {
            "samples": len(knowledge_results),
            "overall": avg(knowledge_scores, "overall"),
            "include_score": avg(knowledge_scores, "include_score"),
            "must_not_ok": avg(knowledge_scores, "must_not_ok"),
            "intent_hit": avg(knowledge_scores, "intent_hit"),
            "escalation_hit": avg(knowledge_scores, "escalation_hit"),
        },
    }
    summary["passed"] = (
        summary["tool"]["overall"] >= args.min_tool_overall
        and summary["tool"]["tool_called"] >= args.min_tool_called
        and summary["tool"]["schema_valid"] >= args.min_tool_schema
        and summary["knowledge"]["overall"] >= args.min_knowledge_overall
    )
    summary["thresholds"] = {
        "min_tool_overall": args.min_tool_overall,
        "min_tool_called": args.min_tool_called,
        "min_tool_schema": args.min_tool_schema,
        "min_knowledge_overall": args.min_knowledge_overall,
    }

    (out_dir / "tool_results.json").write_text(json.dumps(tool_results, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "knowledge_results.json").write_text(json.dumps(knowledge_results, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not summary["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
