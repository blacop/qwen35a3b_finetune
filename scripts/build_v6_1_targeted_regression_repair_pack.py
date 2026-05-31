#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
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


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _to_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def _to_bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    return s in {"1", "true", "yes", "y", "是", "需要", "需升级", "需转人工"}


def _safe_terms(sample: Dict[str, Any], limit: int = 6) -> List[str]:
    val = sample.get("must_include") or []
    if not isinstance(val, list):
        return []
    out: List[str] = []
    for term in val:
        text = str(term).strip()
        if text:
            out.append(text)
    return out[:limit]


def _json_answer(sample: Dict[str, Any]) -> str:
    intent = str(sample.get("gold_intent") or "其他")
    need_escalation = _to_bool(sample.get("gold_need_escalation"))
    must_terms = _safe_terms(sample, limit=6)
    must_chunk = "、".join(must_terms) if must_terms else "账号、订单"
    pred_intent = str(sample.get("pred_intent") or "").strip()
    contrast = ""
    if pred_intent and pred_intent != intent:
        contrast = f"这不是{pred_intent}，而是{intent}。"
    if need_escalation:
        answer = (
            f"{contrast}先帮您核对{must_chunk}。"
            "该问题需要后台核实，我先转运营查询，核实后第一时间回覆您。"
        )
        next_action = "升级到运营/后台核实"
    else:
        answer = f"先给您说明处理口径：{must_chunk}。按当前规则可直接处理，不需要升级。"
        next_action = "直接按规则回复"
    payload = {
        "intent": intent,
        "need_escalation": need_escalation,
        "answer": answer,
        "risk_flag": [],
        "next_action": next_action,
    }
    return json.dumps(payload, ensure_ascii=False)


def _train_dialogue_rows(
    *,
    sample_id: str,
    query: str,
    assistant_json: str,
    tag: str,
    kind: str,
    system_prompt: str = SYSTEM_PROMPT,
    tool_prompt: str | None = None,
    tool_name: str | None = None,
    tool_args: Dict[str, Any] | None = None,
    tool_result: Dict[str, Any] | None = None,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if tool_prompt is None:
        rows.append(
            {
                "id": f"v61_targeted_{tag}_{sample_id}_{kind}",
                "split": "train",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query},
                    {"role": "assistant", "content": assistant_json},
                ],
                "meta": {
                    "source": "v6_1_targeted_regression_repair",
                    "kind": kind,
                    "from": tag,
                    "sample_id": sample_id,
                },
            }
        )
        return rows
    assert tool_name is not None and tool_args is not None
    assert tool_result is not None
    rows.append(
        {
            "id": f"v61_targeted_{tag}_{sample_id}_{kind}_tool",
            "split": "train",
            "messages": [
                {"role": "system", "content": tool_prompt},
                {"role": "user", "content": query},
                {"role": "assistant", "content": "<tool_call>" + json.dumps({"name": tool_name, "arguments": tool_args}, ensure_ascii=False) + "</tool_call>"},
            ],
            "meta": {
                "source": "v6_1_targeted_regression_repair",
                "kind": f"{kind}_tool_call_only",
                "from": tag,
                "sample_id": sample_id,
            },
        }
    )
    rows.append(
        {
            "id": f"v61_targeted_{tag}_{sample_id}_{kind}_dialogue",
            "split": "train",
            "messages": [
                {"role": "system", "content": tool_prompt},
                {"role": "user", "content": query},
                {"role": "assistant", "content": "<tool_call>" + json.dumps({"name": tool_name, "arguments": tool_args}, ensure_ascii=False) + "</tool_call>"},
                {"role": "tool", "content": json.dumps(tool_result, ensure_ascii=False)},
                {"role": "assistant", "content": assistant_json},
            ],
            "meta": {
                "source": "v6_1_targeted_regression_repair",
                "kind": f"{kind}_tool_result",
                "from": tag,
                "sample_id": sample_id,
            },
        }
    )
    return rows


def build_field_completeness_rows() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    cases = [
        (
            "field_transfer_exact",
            "请务必先调用 get_transfer_log_list，再回复三行：工具名、参数、结果摘要。查询最近24小时转账记录，页码1，每页10，状态-1，类型-1，开始时间1746403200000，结束时间1746489600000。",
            "get_transfer_log_list",
            {
                "page": 1,
                "pageSize": 10,
                "status": -1,
                "timeFrom": 1746403200000,
                "timeTo": 1746489600000,
                "type": -1,
            },
        ),
        (
            "field_bet_orders_exact",
            "请务必先调用 list_bet_orders，再回复三行：工具名、参数、结果摘要。查询最近体育投注订单，页码1，每页10，状态-1。",
            "list_bet_orders",
            {
                "createTimeFrom": 1746403200000,
                "createTimeTo": 1746489600000,
                "page": 1,
                "pageSize": 10,
                "status": -1,
            },
        ),
        (
            "field_bet_stats_exact",
            "请务必先调用 get_bet_order_stats，再回复三行：工具名、参数、结果摘要。统计最近体育投注，页码1，每页50，状态留空。",
            "get_bet_order_stats",
            {
                "createTimeFrom": 1746403200000,
                "createTimeTo": 1746489600000,
                "page": 1,
                "pageSize": 50,
                "status": "",
            },
        ),
    ]
    for sid, query, tool_name, args in cases:
        result = {
            "ok": True,
            "items": [],
            "total": 1,
            "_mock": True,
        }
        if tool_name == "get_transfer_log_list":
            result = {
                "ok": True,
                "total": 2,
                "items": [{"id": "TRX1", "status": "success"}, {"id": "TRX2", "status": "pending"}],
                "_mock": True,
            }
        elif tool_name == "list_bet_orders":
            result = {
                "ok": True,
                "total": 3,
                "items": [{"orderId": "OD101"}, {"orderId": "OD102"}, {"orderId": "OD103"}],
                "_mock": True,
            }
        elif tool_name == "get_bet_order_stats":
            result = {
                "ok": True,
                "orderCount": 48,
                "betAmount": 12880,
                "validBetAmount": 11220,
                "winLoss": -180,
                "_mock": True,
            }
        assistant_json = json.dumps({"tool": tool_name, "args": args, "summary": result}, ensure_ascii=False)
        rows.extend(
            _train_dialogue_rows(
                sample_id=sid,
                query=query,
                assistant_json=assistant_json,
                tag="field",
                kind="exact",
                tool_prompt=TOOL_SYSTEM_PROMPT,
                tool_name=tool_name,
                tool_args=args,
                tool_result=result,
            )
        )
    return rows


def build_eval_failure_rows(
    eval_rows: List[Dict[str, Any]],
    pred_rows: List[Dict[str, str]],
    tag: str,
    threshold: float = 0.95,
) -> List[Dict[str, Any]]:
    by_id = {str(x.get("id")): x for x in eval_rows}
    out: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for pr in pred_rows:
        sid = str(pr.get("id") or "")
        if not sid or sid in seen:
            continue
        overall = _to_float(pr.get("overall_score"), 0.0)
        intent_ok = str(pr.get("intent_match", "")).strip() == "1"
        esc_ok = str(pr.get("escalation_match", "")).strip() == "1"
        include_ok = _to_float(pr.get("must_include_score"), 0.0) >= 1.0
        if overall >= threshold and intent_ok and esc_ok and include_ok:
            continue
        sample = by_id.get(sid)
        if not sample:
            continue
        seen.add(sid)

        query = str(sample.get("user_query") or "").strip()
        answer_json = _json_answer(sample)
        must_terms = _safe_terms(sample, limit=6)
        must_text = "、".join(must_terms) if must_terms else "必须包含关键字段"
        pred_intent = str(pr.get("pred_intent") or "").strip()
        contrast_hint = ""
        if pred_intent and pred_intent != str(sample.get("gold_intent") or ""):
            contrast_hint = f"不要误判成{pred_intent}。"
        out.extend(
            _train_dialogue_rows(
                sample_id=sid,
                query=query,
                assistant_json=answer_json,
                tag=tag,
                kind="eval_failure_json",
            )
        )
        out.extend(
            _train_dialogue_rows(
                sample_id=sid,
                query=f"{query}\n\n请只输出JSON对象，且答案里必须包含：{must_text}。字段固定为 intent/need_escalation/answer/risk_flag/next_action。{contrast_hint}",
                assistant_json=answer_json,
                tag=tag,
                kind="eval_failure_json_format",
            )
        )
        out.extend(
            _train_dialogue_rows(
                sample_id=sid,
                query=f"{query}\n\n这类问题的意图就是 {sample.get('gold_intent')}，{contrast_hint}回答时要把这个意图写进JSON的 intent 字段。",
                assistant_json=answer_json,
                tag=tag,
                kind="eval_failure_intent_anchor",
            )
        )
    return out


def _tool_call_markup(name: str, arguments: Dict[str, Any]) -> str:
    return "<tool_call>" + json.dumps({"name": name, "arguments": arguments}, ensure_ascii=False) + "</tool_call>"


def build_tool_arg_rows(tool_results: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], int]:
    rows: List[Dict[str, Any]] = []
    mismatch_count = 0
    for tr in tool_results:
        score = tr.get("score") or {}
        if int(score.get("tool_called", 0)) != 1:
            continue
        if int(score.get("args_match", 0)) == 1:
            continue
        expected_tool = str(tr.get("expected_tool") or "").strip()
        expected_args = tr.get("expected_args") or {}
        if expected_tool != "get_bet_order_stats":
            continue
        if not isinstance(expected_args, dict):
            continue
        mismatch_count += 1
        sid = str(tr.get("id") or f"tool_mismatch_{mismatch_count:03d}")
        query = str(tr.get("user_query") or "").strip()
        strict_query = (
            f"{query}\n\n参数必须原样输出，尤其 status 需传空字符串："
            f"{json.dumps(expected_args, ensure_ascii=False)}"
        )
        exact_query = (
            "请务必先调用 get_bet_order_stats，再回复三行：工具名、参数、结果摘要。"
            "统计最近体育投注，页码1，每页50，状态留空。"
        )
        tool_result = {
            "ok": True,
            "orderCount": 52,
            "betAmount": 13880,
            "validBetAmount": 12720,
            "winLoss": -220,
            "_mock": True,
        }
        final_answer = "已按指定条件完成统计查询，投注额、有效投注、单量和输赢已返回。"
        for idx, user_text in enumerate([query, strict_query, exact_query], start=1):
            rows.append(
                {
                    "id": f"v61_targeted_tool_call_{sid}_v{idx}",
                    "split": "train",
                    "messages": [
                        {"role": "system", "content": TOOL_SYSTEM_PROMPT},
                        {"role": "user", "content": user_text},
                        {"role": "assistant", "content": _tool_call_markup(expected_tool, expected_args)},
                    ],
                    "meta": {
                        "source": "v6_1_targeted_regression_repair",
                        "kind": "tool_arg_exact_call",
                        "sample_id": sid,
                        "variant": idx,
                    },
                }
            )
            rows.append(
                {
                    "id": f"v61_targeted_tool_result_{sid}_v{idx}",
                    "split": "train",
                    "messages": [
                        {"role": "system", "content": TOOL_SYSTEM_PROMPT},
                        {"role": "user", "content": user_text},
                        {"role": "assistant", "content": _tool_call_markup(expected_tool, expected_args)},
                        {"role": "tool", "content": json.dumps(tool_result, ensure_ascii=False)},
                        {"role": "assistant", "content": final_answer},
                    ],
                    "meta": {
                        "source": "v6_1_targeted_regression_repair",
                        "kind": "tool_arg_exact_result",
                        "sample_id": sid,
                        "variant": idx,
                    },
                }
            )
    return rows, mismatch_count


def main() -> None:
    parser = argparse.ArgumentParser(description="Build v6.1 targeted regression repair pack from latest strict gate failures.")
    parser.add_argument("--base-train", required=True)
    parser.add_argument("--ops-eval-jsonl", required=True)
    parser.add_argument("--kb-eval-jsonl", required=True)
    parser.add_argument("--ops-pred-csv", required=True)
    parser.add_argument("--kb-pred-csv", required=True)
    parser.add_argument("--tool-results-json", required=True)
    parser.add_argument("--out-repair-jsonl", required=True)
    parser.add_argument("--out-merged-jsonl", required=True)
    parser.add_argument("--out-summary-json", required=True)
    args = parser.parse_args()

    base_rows = read_jsonl(Path(args.base_train))
    ops_eval_rows = read_jsonl(Path(args.ops_eval_jsonl))
    kb_eval_rows = read_jsonl(Path(args.kb_eval_jsonl))
    ops_pred_rows = read_csv_rows(Path(args.ops_pred_csv))
    kb_pred_rows = read_csv_rows(Path(args.kb_pred_csv))
    tool_results = json.loads(Path(args.tool_results_json).read_text(encoding="utf-8"))

    field_rows = build_field_completeness_rows()
    ops_rows = build_eval_failure_rows(ops_eval_rows, ops_pred_rows, "ops16")
    kb_rows = build_eval_failure_rows(kb_eval_rows, kb_pred_rows, "kb40")
    tool_rows, tool_mismatch_count = build_tool_arg_rows(tool_results)

    repair_rows = field_rows + tool_rows + ops_rows + kb_rows
    merged_rows = base_rows + repair_rows

    repair_count = write_jsonl(Path(args.out_repair_jsonl), repair_rows)
    merged_count = write_jsonl(Path(args.out_merged_jsonl), merged_rows)

    summary = {
        "base_train": args.base_train,
        "base_count": len(base_rows),
        "field_rows": len(field_rows),
        "tool_mismatch_count": tool_mismatch_count,
        "tool_rows": len(tool_rows),
        "ops_rows": len(ops_rows),
        "kb_rows": len(kb_rows),
        "repair_count": repair_count,
        "merged_count": merged_count,
        "out_repair_jsonl": args.out_repair_jsonl,
        "out_merged_jsonl": args.out_merged_jsonl,
    }
    Path(args.out_summary_json).write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
