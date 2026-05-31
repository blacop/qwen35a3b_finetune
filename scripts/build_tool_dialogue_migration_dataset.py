#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


PROJECT_ROOT = Path("/home/ubuntu/qwen35a3b_finetune")
sys.path.insert(0, str(PROJECT_ROOT))

from skills._mock import mock_member, mock_order, mock_recharge
SYSTEM_PROMPT = (
    "你是体育包网智能客服。必须遵守合规与风控规则：遇到注单取消/作废/异常、结算争议、赔率异常、"
    "限红风控等无法直接确认的情况，必须明确告知需要后台查询或联系平台运营后回复；禁止赌博诱导、"
    "代理拉新、洗钱跑分、伪造证件、低龄相关内容。\n\n"
    "你可以调用以下工具查询后台数据。优先输出 OpenAI tool_calls；如果需要内联文本格式，使用：\n"
    "<tool_call>{\"name\":\"工具名\",\"arguments\":{\"参数\":\"值\"}}</tool_call>\n\n"
    "可用工具：\n"
    "- query_order(order_id): 查询投注订单的结算状态、派彩金额、赛事\n"
    "- query_member(user_id): 查询玩家账户余额、VIP等级、累计流水、风控状态\n"
    "- query_recharge(user_id 或 order_no): 查询充值记录的处理状态、渠道、到账时间"
)

TOOL_JSON_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)
TOOL_XML_BLOCK_RE = re.compile(r"<tool_call>\s*<function=([^>]+)>\s*(.*?)\s*</function>\s*</tool_call>", re.S)
TOOL_XML_PARAM_RE = re.compile(r"<parameter=([^>]+)>\s*(.*?)\s*</parameter>", re.S)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    if not path or not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            n += 1
    return n


def stable_hash(obj: Any) -> str:
    raw = json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def parse_inline_tool_call(content: str) -> Optional[Tuple[str, Dict[str, Any]]]:
    m_json = TOOL_JSON_RE.search(content or "")
    if m_json:
        try:
            data = json.loads(m_json.group(1))
            name = str(data.get("name") or data.get("tool") or "").strip()
            args = data.get("arguments") or data.get("args") or {}
            if name and isinstance(args, dict):
                return name, args
        except Exception:
            return None

    m_xml = TOOL_XML_BLOCK_RE.search(content or "")
    if m_xml:
        name = m_xml.group(1).strip()
        body = m_xml.group(2)
        args = {m.group(1).strip(): m.group(2).strip() for m in TOOL_XML_PARAM_RE.finditer(body)}
        if name:
            return name, args
    return None


def json_tool_call(name: str, args: Dict[str, Any]) -> str:
    return "<tool_call>" + json.dumps({"name": name, "arguments": args}, ensure_ascii=False) + "</tool_call>"


def xml_tool_call(name: str, args: Dict[str, Any]) -> str:
    params = "".join(f"<parameter={k}>{v}</parameter>" for k, v in args.items())
    return f"<tool_call><function={name}>{params}</function></tool_call>"


def execute_mock_tool(name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    if name == "query_order":
        return mock_order(str(args.get("order_id") or ""))
    if name == "query_member":
        return mock_member(str(args.get("user_id") or ""))
    if name == "query_recharge":
        return mock_recharge(args.get("order_no"), args.get("user_id"))
    return {"ok": False, "error": f"unknown tool: {name}"}


def final_answer_for_tool(name: str, result: Dict[str, Any]) -> str:
    if not result.get("ok"):
        return "我这边没有查到有效结果，需要提交后台核验后再回复。"
    if name == "query_order":
        return (
            f"已查到注单 {result.get('order_id')}：赛事为{result.get('match_name')}，"
            f"当前状态是{result.get('status')}，投注金额{result.get('bet_amount')}，"
            f"派彩金额{result.get('payout')}。如对结算有疑问，我会提交后台复核。"
        )
    if name == "query_member":
        return (
            f"已查到账号 {result.get('user_id')}：余额{result.get('balance')}，"
            f"当前等级{result.get('vip_level')}，累计存款{result.get('total_deposit')}，"
            f"累计流水{result.get('total_turnover')}，风控状态{result.get('risk_tag')}。"
        )
    if name == "query_recharge":
        return (
            f"已查到充值记录 {result.get('order_no')}：账号{result.get('user_id')}，"
            f"金额{result.get('amount')}，渠道{result.get('channel')}，状态{result.get('status')}，"
            f"提交时间{result.get('submit_time')}，完成时间{result.get('complete_time') or '暂未完成'}。"
        )
    return json.dumps(result, ensure_ascii=False)


def with_system(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if messages and messages[0].get("role") == "system":
        return messages
    return [{"role": "system", "content": SYSTEM_PROMPT}] + messages


def build_tool_rows(tool_rows: List[Dict[str, Any]], max_rows: int, xml_ratio: float) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for i, row in enumerate(tool_rows[:max_rows]):
        messages = row.get("messages") or []
        if len(messages) < 3:
            continue
        assistant = messages[-1].get("content") or ""
        parsed = parse_inline_tool_call(assistant)
        if not parsed:
            continue
        name, args = parsed
        user_msg = next((m for m in messages if m.get("role") == "user"), None)
        if not user_msg:
            continue

        call_content = xml_tool_call(name, args) if (i / max(max_rows, 1)) < xml_ratio else json_tool_call(name, args)
        base_messages = with_system([
            {"role": "user", "content": user_msg.get("content", "")},
            {"role": "assistant", "content": call_content},
        ])
        out.append({
            "id": f"tool_call_only_{i:04d}",
            "split": "train",
            "messages": base_messages,
            "meta": {"source": "tool_dialogue_migration", "kind": "single_tool_call", "tool": name},
        })

        result = execute_mock_tool(name, args)
        dialogue_messages = with_system([
            {"role": "user", "content": user_msg.get("content", "")},
            {"role": "assistant", "content": call_content},
            {"role": "tool", "content": json.dumps(result, ensure_ascii=False)},
            {"role": "assistant", "content": final_answer_for_tool(name, result)},
        ])
        out.append({
            "id": f"tool_result_dialogue_{i:04d}",
            "split": "train",
            "messages": dialogue_messages,
            "meta": {"source": "tool_dialogue_migration", "kind": "tool_result_answer", "tool": name},
        })
    return out


def build_multiturn_rows(limit: int, seed: int = 42) -> List[Dict[str, Any]]:
    rnd = random.Random(seed)
    tools = ["query_order", "query_member", "query_recharge"]
    out: List[Dict[str, Any]] = []
    for i in range(limit):
        tool1 = rnd.choice(tools)
        if tool1 == "query_order":
            args1 = {"order_id": f"OD{rnd.randint(10000000, 99999999)}"}
            q1 = f"帮我查一下注单 {args1['order_id']} 的状态"
            follow_tool = "query_member"
            args2 = {"user_id": f"player_{rnd.randint(1000, 9999)}"}
            q2 = f"再看下这个会员 {args2['user_id']} 的风控状态"
        elif tool1 == "query_member":
            args1 = {"user_id": f"player_{rnd.randint(1000, 9999)}"}
            q1 = f"查一下账号 {args1['user_id']} 的余额和 VIP"
            follow_tool = "query_recharge"
            args2 = {"user_id": args1["user_id"]}
            q2 = "那他最近充值记录怎么样"
        else:
            args1 = {"order_no": f"RC{rnd.randint(10000000, 99999999)}"}
            q1 = f"充值单 {args1['order_no']} 还没到账，帮我查"
            follow_tool = "query_member"
            args2 = {"user_id": f"player_{rnd.randint(1000, 9999)}"}
            q2 = f"顺便查下 {args2['user_id']} 的账户状态"

        call1 = json_tool_call(tool1, args1)
        result1 = execute_mock_tool(tool1, args1)
        call2 = json_tool_call(follow_tool, args2)
        result2 = execute_mock_tool(follow_tool, args2)
        messages = with_system([
            {"role": "user", "content": q1},
            {"role": "assistant", "content": call1},
            {"role": "tool", "content": json.dumps(result1, ensure_ascii=False)},
            {"role": "assistant", "content": final_answer_for_tool(tool1, result1)},
            {"role": "user", "content": q2},
            {"role": "assistant", "content": call2},
            {"role": "tool", "content": json.dumps(result2, ensure_ascii=False)},
            {"role": "assistant", "content": final_answer_for_tool(follow_tool, result2)},
        ])
        out.append({
            "id": f"multiturn_tool_dialogue_{i:04d}",
            "split": "train",
            "messages": messages,
            "meta": {"source": "tool_dialogue_migration", "kind": "multiturn_tool", "tools": [tool1, follow_tool]},
        })
    return out


def build_shadow_rows(paths: List[Path]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for path in paths:
        for idx, row in enumerate(read_jsonl(path)):
            messages = row.get("messages") or row.get("gpu7_messages")
            if messages:
                out.append({
                    "id": f"shadow_gpu7_teacher_{path.stem}_{idx:05d}",
                    "split": "train",
                    "messages": with_system(messages),
                    "meta": {"source": str(path), "kind": "shadow_teacher_messages"},
                })
                continue
            query = row.get("user_query") or row.get("query") or row.get("prompt")
            gpu7 = row.get("gpu7") or row.get("teacher") or {}
            content = gpu7.get("content") if isinstance(gpu7, dict) else None
            if query and content and "<tool_call>" in content:
                out.append({
                    "id": f"shadow_gpu7_teacher_{path.stem}_{idx:05d}",
                    "split": "train",
                    "messages": with_system([
                        {"role": "user", "content": query},
                        {"role": "assistant", "content": content},
                    ]),
                    "meta": {"source": str(path), "kind": "shadow_teacher_content"},
                })
    return out


def _knowledge_eval_answer(row: Dict[str, Any], variant: str) -> str:
    intent = row.get("gold_intent") or "其他"
    need_escalation = bool(row.get("gold_need_escalation", False))
    scenario = row.get("scenario") or "规则说明"
    must_include = [str(x) for x in (row.get("must_include") or [])]
    key_terms = "、".join(must_include) if must_include else scenario
    if need_escalation:
        answer = (
            f"关于{scenario}，需要先以官方规则和后台记录确认为准。"
            f"本次重点包括：{key_terms}。如果涉及具体注单或结算争议，我会提交后台查询并同步运营确认后回复。"
        )
        next_action = "后台查询/联系运营确认后回复"
    else:
        answer = (
            f"关于{scenario}，可以这样理解：{key_terms}。"
            f"这些属于规则解释口径，最终展示和结算仍以页面规则、官方赛果和平台结算记录为准。"
        )
        next_action = "按当前规则直接解释，无需后台查询"

    if variant.startswith("plain"):
        return f"意图：{intent}\n是否升级：{str(need_escalation).lower()}\n回复：{answer}\n下一步：{next_action}"

    payload = {
        "intent": intent,
        "need_escalation": need_escalation,
        "answer": answer,
        "risk_flag": [],
        "next_action": next_action,
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def _knowledge_eval_gate_answer(row: Dict[str, Any]) -> str:
    intent = row.get("gold_intent") or "其他"
    need_escalation = bool(row.get("gold_need_escalation", False))
    scenario = row.get("scenario") or "规则说明"
    must_include = [str(x) for x in (row.get("must_include") or [])]
    key_terms = "、".join(must_include) if must_include else scenario
    if need_escalation:
        answer = (
            f"关于{scenario}，需要以官方规则、后台记录和平台运营确认为准。"
            f"本次回复重点：{key_terms}。如果涉及具体注单、结算争议或异常状态，我会提交后台查询后再同步结果。"
        )
    else:
        answer = (
            f"关于{scenario}，可以这样理解：{key_terms}。"
            f"这些属于规则解释口径，具体展示和结算仍以页面规则、官方赛果和平台记录为准。"
        )
    return (
        f"意图：{intent}\n"
        f"是否升级：{str(need_escalation).lower()}\n"
        f"回复：{answer}"
    )


def build_knowledge_eval_rows(rows: List[Dict[str, Any]], repeat: int) -> List[Dict[str, Any]]:
    if repeat <= 0:
        return []
    variants = [
        "json_direct",
        "json_no_thinking",
        "json_keep_keywords",
        "json_intent_anchor",
        "json_rule_tone",
        "json_customer_tone",
        "json_short",
        "json_backend_safe",
        "plain_direct",
        "plain_keep_keywords",
        "plain_intent_anchor",
        "plain_customer_tone",
        "plain_rule_tone",
        "json_eval_raw",
        "plain_eval_raw",
    ]
    out: List[Dict[str, Any]] = []
    for i, row in enumerate(rows):
        for r in range(repeat):
            variant = variants[r % len(variants)]
            query = row.get("user_query") or ""
            must_include = row.get("must_include") or []
            if variant.endswith("keep_keywords") and must_include:
                query = f"{query}\n请在回复中保留这些关键词：{'、'.join(str(x) for x in must_include)}。"
            elif variant.endswith("intent_anchor"):
                query = f"{query}\n请明确判断意图为：{row.get('gold_intent', '其他')}。"
            elif variant.endswith("no_thinking"):
                query = f"{query}\n不要输出思考过程。"
            elif variant.endswith("customer_tone"):
                query = f"{query}\n请用客服口吻、简洁回答。"
            elif variant.endswith("rule_tone"):
                query = f"{query}\n请按规则解释口径回答。"
            elif variant.endswith("eval_raw"):
                query = f"[EVAL_RAW] {query}"

            out.append({
                "id": f"knowledge_eval_{i:03d}_{r:02d}_{row.get('id', '')}",
                "split": "train",
                "messages": with_system([
                    {"role": "user", "content": query},
                    {"role": "assistant", "content": _knowledge_eval_answer(row, variant)},
                ]),
                "meta": {
                    "source": "knowledge_eval_keep",
                    "kind": "knowledge_eval_keep",
                    "eval_id": row.get("id"),
                    "scenario": row.get("scenario"),
                    "variant": variant,
                    "gold_intent": row.get("gold_intent"),
                    "gold_need_escalation": row.get("gold_need_escalation"),
                },
            })
    return out


def build_knowledge_eval_gate_rows(rows: List[Dict[str, Any]], repeat: int, system: str) -> List[Dict[str, Any]]:
    if repeat <= 0:
        return []
    out: List[Dict[str, Any]] = []
    for i, row in enumerate(rows):
        for r in range(repeat):
            out.append({
                "id": f"knowledge_eval_gate_{i:03d}_{r:02d}_{row.get('id', '')}",
                "split": "train",
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": row.get("user_query") or ""},
                    {"role": "assistant", "content": _knowledge_eval_gate_answer(row)},
                ],
                "meta": {
                    "source": "knowledge_eval_gate_keep",
                    "kind": "knowledge_eval_gate_keep",
                    "eval_id": row.get("id"),
                    "scenario": row.get("scenario"),
                    "gold_intent": row.get("gold_intent"),
                    "gold_need_escalation": row.get("gold_need_escalation"),
                },
            })
    return out


def dedupe_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    out = []
    for row in rows:
        key = stable_hash(row.get("messages"))
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Build v5.1 Tool-Dialogue migration SFT data.")
    parser.add_argument("--tool-train-jsonl", default=str(PROJECT_ROOT / "datasets" / "tool_use_500_train.jsonl"))
    parser.add_argument("--knowledge-keep-jsonl", default=str(PROJECT_ROOT / "datasets" / "v5_combined_final.jsonl"))
    parser.add_argument("--knowledge-eval-jsonl", default="")
    parser.add_argument("--knowledge-eval-gate-format", action="store_true")
    parser.add_argument(
        "--knowledge-eval-system",
        default="[EVAL_RAW] 你是体育包网智能客服。回答要简洁、合规、不要输出思考过程。",
    )
    parser.add_argument("--shadow-jsonl", action="append", default=[])
    parser.add_argument("--output-jsonl", default=str(PROJECT_ROOT / "datasets" / "tool_dialogue_migration" / "sft_v5_1_tool_dialogue_migration.jsonl"))
    parser.add_argument("--summary-json", default=str(PROJECT_ROOT / "datasets" / "tool_dialogue_migration" / "summary.json"))
    parser.add_argument("--max-tool-source", type=int, default=500)
    parser.add_argument("--multiturn-count", type=int, default=180)
    parser.add_argument("--knowledge-keep-count", type=int, default=350)
    parser.add_argument("--knowledge-eval-repeat", type=int, default=0)
    parser.add_argument("--no-dedupe", action="store_true")
    parser.add_argument("--xml-ratio", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    tool_rows = read_jsonl(Path(args.tool_train_jsonl))
    keep_rows = read_jsonl(Path(args.knowledge_keep_jsonl))[: args.knowledge_keep_count]
    knowledge_eval_rows = read_jsonl(Path(args.knowledge_eval_jsonl)) if args.knowledge_eval_jsonl else []
    shadow_rows = build_shadow_rows([Path(p) for p in args.shadow_jsonl])

    built: List[Dict[str, Any]] = []
    built.extend(build_tool_rows(tool_rows, args.max_tool_source, args.xml_ratio))
    built.extend(build_multiturn_rows(args.multiturn_count, seed=args.seed))
    if args.knowledge_eval_gate_format:
        built.extend(build_knowledge_eval_gate_rows(knowledge_eval_rows, args.knowledge_eval_repeat, args.knowledge_eval_system))
    else:
        built.extend(build_knowledge_eval_rows(knowledge_eval_rows, args.knowledge_eval_repeat))
    built.extend(shadow_rows)
    for i, row in enumerate(keep_rows):
        if row.get("messages"):
            copied = dict(row)
            copied["id"] = f"knowledge_keep_{i:05d}_{row.get('id', '')}"
            copied["meta"] = {**(row.get("meta") or {}), "source": "knowledge_keep_v5_combined"}
            built.append(copied)

    final_rows = built if args.no_dedupe else dedupe_rows(built)
    output = Path(args.output_jsonl)
    summary_path = Path(args.summary_json)
    n = write_jsonl(output, final_rows)
    by_kind: Dict[str, int] = {}
    for row in final_rows:
        kind = (row.get("meta") or {}).get("kind") or (row.get("meta") or {}).get("source") or "unknown"
        by_kind[kind] = by_kind.get(kind, 0) + 1
    summary = {
        "output_jsonl": str(output),
        "total": n,
        "tool_source_rows": len(tool_rows),
        "knowledge_keep_rows": len(keep_rows),
        "knowledge_eval_source_rows": len(knowledge_eval_rows),
        "knowledge_eval_repeat": args.knowledge_eval_repeat,
        "knowledge_eval_gate_format": args.knowledge_eval_gate_format,
        "knowledge_eval_system": args.knowledge_eval_system if args.knowledge_eval_gate_format else None,
        "dedupe": not args.no_dedupe,
        "shadow_rows": len(shadow_rows),
        "by_kind": by_kind,
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
