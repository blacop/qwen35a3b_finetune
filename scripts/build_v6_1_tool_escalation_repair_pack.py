#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


SYSTEM_PROMPT = (
    "你是体育包网平台的一线客服助手，只按体育包网平台客服语境回答。"
    "回复短句、口语化、像真人客服。涉及后台数据、风控、财务时不要猜测，明确说需要后台核实。"
)

TOOL_SYSTEM_PROMPT = """你是体育包网智能客服。必须遵守合规与风控规则：
- 涉及订单明细、注单记录、转账流水等后台数据时，先调用工具查询再回复。
- 不要编造工具返回结果；没有工具结果时，不输出最终结论。

你可以调用以下工具查询后台数据。优先输出 OpenAI tool_calls；如果需要内联格式，可输出：
<tool_call>{"name":"工具名","arguments":{"参数":"值"}}</tool_call>

可用工具：
- list_bet_orders(createTimeFrom, createTimeTo, page, pageSize, status?, vendorId?, gameTypeId?, currency?)
- get_bet_order_stats(createTimeFrom, createTimeTo, page, pageSize, status?, vendorId?, gameTypeId?, currency?)
- get_order_status_list()
- get_transfer_log_list(page, pageSize, status, timeFrom, timeTo, type)
""".strip()


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


@dataclass
class ToolSpec:
    sample_id: str
    user_query: str
    tool: str
    args: Dict[str, Any]
    tool_result: Dict[str, Any]
    final_answer: str


def _tool_specs() -> List[ToolSpec]:
    specs: List[ToolSpec] = []
    base_from = 1746403200000
    for i in range(1, 9):
        start = base_from + (i - 1) * 86400000
        end = start + 86400000
        specs.append(
            ToolSpec(
                sample_id=f"suncidi_transfer_{i:03d}",
                user_query=(
                    "请先调用 get_transfer_log_list。"
                    f"查询最近一天转账记录：page=1,pageSize=10,status=-1,type=-1,"
                    f"timeFrom={start},timeTo={end}。只做查询。"
                ),
                tool="get_transfer_log_list",
                args={
                    "page": 1,
                    "pageSize": 10,
                    "status": -1,
                    "timeFrom": start,
                    "timeTo": end,
                    "type": -1,
                },
                tool_result={
                    "ok": True,
                    "total": 2,
                    "items": [
                        {"id": f"TR{i}A", "status": "success", "amount": 500},
                        {"id": f"TR{i}B", "status": "pending", "amount": 200},
                    ],
                    "_mock": True,
                },
                final_answer=(
                    f"已查到最近一天转账2笔，时间范围{start}到{end}。"
                    "其中1笔成功、1笔处理中。"
                ),
            )
        )

    for i in range(1, 9):
        start = base_from + (i - 1) * 43200000
        end = start + 43200000
        specs.append(
            ToolSpec(
                sample_id=f"suncidi_order_list_{i:03d}",
                user_query=(
                    "先调用 list_bet_orders，别直接总结。"
                    f"参数：createTimeFrom={start},createTimeTo={end},page=1,pageSize=20,status=settled。"
                ),
                tool="list_bet_orders",
                args={
                    "createTimeFrom": start,
                    "createTimeTo": end,
                    "page": 1,
                    "pageSize": 20,
                    "status": "settled",
                },
                tool_result={
                    "ok": True,
                    "total": 3,
                    "items": [
                        {"orderId": f"OD{i}01", "status": "settled", "winLoss": 120},
                        {"orderId": f"OD{i}02", "status": "settled", "winLoss": -50},
                        {"orderId": f"OD{i}03", "status": "settled", "winLoss": 30},
                    ],
                    "_mock": True,
                },
                final_answer="已查到已结算注单3笔，净输赢为+100。",
            )
        )

    for i in range(1, 7):
        start = base_from + (i - 1) * 86400000
        end = start + 86400000
        specs.append(
            ToolSpec(
                sample_id=f"suncidi_order_stats_{i:03d}",
                user_query=(
                    "请先调用 get_bet_order_stats。"
                    f"统计参数 createTimeFrom={start},createTimeTo={end},page=1,pageSize=50,status=。"
                ),
                tool="get_bet_order_stats",
                args={
                    "createTimeFrom": start,
                    "createTimeTo": end,
                    "page": 1,
                    "pageSize": 50,
                    "status": "",
                },
                tool_result={
                    "ok": True,
                    "betAmount": 12800 + i * 10,
                    "validBetAmount": 11100 + i * 10,
                    "orderCount": 45 + i,
                    "winLoss": -350 + i * 5,
                    "_mock": True,
                },
                final_answer="统计已查到：投注额、有效投注、单量和输赢汇总都已返回。",
            )
        )

    for i in range(1, 9):
        specs.append(
            ToolSpec(
                sample_id=f"suncidi_status_list_{i:03d}",
                user_query="请调用 get_order_status_list，返回可用订单状态列表。",
                tool="get_order_status_list",
                args={},
                tool_result={
                    "ok": True,
                    "items": ["settled", "unsettled", "cancelled"],
                    "_mock": True,
                },
                final_answer="订单状态列表已查到：settled、unsettled、cancelled。",
            )
        )
    return specs


def _tool_call_markup(spec: ToolSpec) -> str:
    return "<tool_call>" + json.dumps({"name": spec.tool, "arguments": spec.args}, ensure_ascii=False) + "</tool_call>"


def _alt_tool_query(spec: ToolSpec) -> str:
    return (
        f"请务必先调用 {spec.tool}，"
        "按我给的参数原样查询，不要先输出结论。"
        f"参数如下：{json.dumps(spec.args, ensure_ascii=False)}"
    )


def build_tool_train_rows() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for spec in _tool_specs():
        queries = [spec.user_query, _alt_tool_query(spec)]
        for variant_idx, query in enumerate(queries, 1):
            rows.append(
                {
                    "id": f"v61_tool_call_only_{spec.sample_id}_v{variant_idx}",
                    "split": "train",
                    "messages": [
                        {"role": "system", "content": TOOL_SYSTEM_PROMPT},
                        {"role": "user", "content": query},
                        {"role": "assistant", "content": _tool_call_markup(spec)},
                    ],
                    "meta": {
                        "source": "v6_1_tool_escalation_repair",
                        "kind": "tool_call_only",
                        "tool": spec.tool,
                        "variant": variant_idx,
                    },
                }
            )
            rows.append(
                {
                    "id": f"v61_tool_result_dialogue_{spec.sample_id}_v{variant_idx}",
                    "split": "train",
                    "messages": [
                        {"role": "system", "content": TOOL_SYSTEM_PROMPT},
                        {"role": "user", "content": query},
                        {"role": "assistant", "content": _tool_call_markup(spec)},
                        {"role": "tool", "content": json.dumps(spec.tool_result, ensure_ascii=False)},
                        {"role": "assistant", "content": spec.final_answer},
                    ],
                    "meta": {
                        "source": "v6_1_tool_escalation_repair",
                        "kind": "tool_result_answer",
                        "tool": spec.tool,
                        "variant": variant_idx,
                    },
                }
            )
    return rows


def build_tool_eval_rows() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for spec in _tool_specs():
        rows.append(
            {
                "id": spec.sample_id,
                "scenario": "suncidi_tool_gate",
                "user_query": spec.user_query,
                "expected_tool": spec.tool,
                "expected_args": spec.args,
                "must_include": [spec.tool],
            }
        )
    return rows


def _safe_terms(sample: Dict[str, Any], key: str, limit: int) -> List[str]:
    val = sample.get(key) or []
    if not isinstance(val, list):
        return []
    out: List[str] = []
    for term in val:
        text = str(term).strip()
        if text and len(text) <= 30:
            out.append(text)
    return out[:limit]


def _build_escalation_answer(sample: Dict[str, Any]) -> Tuple[bool, str, str]:
    need_escalation = bool(sample.get("gold_need_escalation"))
    must_terms = _safe_terms(sample, "must_include", 4)
    if not must_terms:
        must_terms = ["账号", "订单"]
    intent = str(sample.get("gold_intent") or "其他")
    if need_escalation:
        answer = (
            f"您这个情况我先帮您处理。先确认{ '、'.join(must_terms) }。"
            "这类问题需要后台核实，我这边立即提交查询，稍后给您回覆。"
        )
        next_action = "后台查询/联系运营核实"
    else:
        answer = (
            f"先按规则给您说明：{ '、'.join(must_terms) }。"
            "按当前口径可以先这样处理，不需要后台介入。"
        )
        next_action = "直接按规则说明"
    return need_escalation, intent, json.dumps(
        {
            "intent": intent,
            "need_escalation": need_escalation,
            "answer": answer,
            "risk_flag": [],
            "next_action": next_action,
        },
        ensure_ascii=False,
    )


def build_escalation_rows(eval_rows: List[Dict[str, Any]], tag: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for idx, sample in enumerate(eval_rows, 1):
        need_esc, intent, assistant_json = _build_escalation_answer(sample)
        query = str(sample.get("user_query") or "").strip()
        rows.append(
            {
                "id": f"v61_escalation_{tag}_{idx:03d}_a",
                "split": "train",
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": query},
                    {"role": "assistant", "content": assistant_json},
                ],
                "meta": {
                    "source": "v6_1_tool_escalation_repair",
                    "kind": "escalation_phrasing",
                    "from": tag,
                    "gold_intent": intent,
                    "gold_need_escalation": need_esc,
                },
            }
        )

        rows.append(
            {
                "id": f"v61_escalation_{tag}_{idx:03d}_b",
                "split": "train",
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": (
                            f"{query}\n\n请只输出JSON对象，字段为 intent/need_escalation/answer/risk_flag/next_action。"
                        ),
                    },
                    {"role": "assistant", "content": assistant_json},
                ],
                "meta": {
                    "source": "v6_1_tool_escalation_repair",
                    "kind": "escalation_phrasing_json_format",
                    "from": tag,
                    "gold_intent": intent,
                    "gold_need_escalation": need_esc,
                },
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Build v6.1 repair pack for tool use + escalation phrasing.")
    parser.add_argument("--base-train", required=True)
    parser.add_argument("--ops16-eval", required=True)
    parser.add_argument("--kb40-eval", required=True)
    parser.add_argument("--out-train-repair", required=True)
    parser.add_argument("--out-tool-eval", required=True)
    parser.add_argument("--out-train-merged", required=True)
    parser.add_argument("--out-summary", required=True)
    args = parser.parse_args()

    base_rows = read_jsonl(Path(args.base_train))
    ops_rows = read_jsonl(Path(args.ops16_eval))
    kb_rows = read_jsonl(Path(args.kb40_eval))

    tool_train_rows = build_tool_train_rows()
    esc_rows = build_escalation_rows(ops_rows, "ops16") + build_escalation_rows(kb_rows, "kb40")
    repair_rows = tool_train_rows + esc_rows
    merged_rows = base_rows + repair_rows
    tool_eval_rows = build_tool_eval_rows()

    repair_count = write_jsonl(Path(args.out_train_repair), repair_rows)
    merged_count = write_jsonl(Path(args.out_train_merged), merged_rows)
    tool_eval_count = write_jsonl(Path(args.out_tool_eval), tool_eval_rows)

    summary = {
        "base_train": args.base_train,
        "base_count": len(base_rows),
        "repair_count": repair_count,
        "tool_train_count": len(tool_train_rows),
        "escalation_train_count": len(esc_rows),
        "merged_train_count": merged_count,
        "tool_eval_count": tool_eval_count,
        "out_train_repair": args.out_train_repair,
        "out_tool_eval": args.out_tool_eval,
        "out_train_merged": args.out_train_merged,
    }
    Path(args.out_summary).write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
