#!/usr/bin/env python3
"""Compare two eval runs produced by eval_sports_customer_service.py and emit REPORT.md + diff_cases.csv."""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
from typing import Any, Dict, List


def load_jsonl(p: Path) -> List[Dict[str, Any]]:
    out = []
    with open(p, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def fmt(v, nd=4):
    try:
        return f"{float(v):.{nd}f}"
    except Exception:
        return "-"


def fdelta(a, b, nd=4):
    try:
        d = float(a) - float(b)
        sign = "+" if d > 0 else ""
        arrow = " ↑" if d > 1e-9 else (" ↓" if d < -1e-9 else " ·")
        return f"{sign}{d:.{nd}f}{arrow}"
    except Exception:
        return "-"


def sample_correct(p: Dict[str, Any]) -> bool:
    """A sample is 'correct' when both intent and escalation match AND no risk violation."""
    return (
        p.get("intent_match", 0) == 1
        and p.get("escalation_match", 0) == 1
        and p.get("risk_violation", 0) == 0
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a-dir", required=True)
    ap.add_argument("--a-name", default="A")
    ap.add_argument("--b-dir", required=True)
    ap.add_argument("--b-name", default="B")
    ap.add_argument("--output", required=True, help="Output REPORT.md path")
    ap.add_argument("--bad-case-limit", type=int, default=15)
    args = ap.parse_args()

    a_dir, b_dir = Path(args.a_dir), Path(args.b_dir)
    a_sum = json.loads((a_dir / "summary.json").read_text(encoding='utf-8'))
    b_sum = json.loads((b_dir / "summary.json").read_text(encoding='utf-8'))
    a_pred = {p["id"]: p for p in load_jsonl(a_dir / "predictions.jsonl")}
    b_pred = {p["id"]: p for p in load_jsonl(b_dir / "predictions.jsonl")}

    # Alignment
    common = sorted(set(a_pred) & set(b_pred))
    only_a = [qid for qid in common if sample_correct(a_pred[qid]) and not sample_correct(b_pred[qid])]
    only_b = [qid for qid in common if sample_correct(b_pred[qid]) and not sample_correct(a_pred[qid])]
    both_right = [qid for qid in common if sample_correct(a_pred[qid]) and sample_correct(b_pred[qid])]
    both_wrong = [qid for qid in common if not sample_correct(a_pred[qid]) and not sample_correct(b_pred[qid])]

    # Per intent breakdown from by_intent
    a_int, b_int = a_sum.get("by_intent", {}), b_sum.get("by_intent", {})
    intents = sorted(set(a_int) | set(b_int))

    metric_rows = [
        ("request_ok_rate", "请求成功率", "↑"),
        ("intent_acc", "意图准确率", "↑"),
        ("escalation_acc", "升级判断准确率", "↑"),
        ("must_include_avg", "必含要素命中率", "↑"),
        ("must_not_avg", "禁用词合规率", "↑"),
        ("risk_violation_rate", "风险违规率", "↓"),
        ("overall_avg", "综合分", "↑"),
        ("latency_ms_p50", "延迟 P50 (ms)", "↓"),
        ("latency_ms_p95", "延迟 P95 (ms)", "↓"),
    ]

    L = []
    L.append(f"# SFT 对比评测报告：{args.a_name} vs {args.b_name}\n")
    L.append(f"- **A = {args.a_name}** (`{args.a_dir}`)")
    L.append(f"- **B = {args.b_name}** (`{args.b_dir}`)")
    L.append(f"- A 样本：{a_sum.get('total_samples')}，B 样本：{b_sum.get('total_samples')}")
    L.append(f"- 对齐样本：{len(common)}\n")

    L.append("## 一、总体指标对比\n")
    L.append(f"| 指标 | 方向 | {args.a_name} | {args.b_name} | Δ (A − B) |")
    L.append("|---|:---:|---:|---:|---:|")
    for k, label, direction in metric_rows:
        L.append(f"| {label} | {direction} | {fmt(a_sum.get(k))} | {fmt(b_sum.get(k))} | {fdelta(a_sum.get(k), b_sum.get(k))} |")

    L.append("\n## 二、分意图 intent_acc 对比\n")
    L.append(f"| 意图 | count | {args.a_name} | {args.b_name} | Δ |")
    L.append("|---|---:|---:|---:|---:|")
    for it in intents:
        ac = a_int.get(it) or {}
        bc = b_int.get(it) or {}
        cnt = ac.get("count", bc.get("count", "-"))
        L.append(f"| {it} | {cnt} | {fmt(ac.get('intent_acc'))} | {fmt(bc.get('intent_acc'))} | {fdelta(ac.get('intent_acc'), bc.get('intent_acc'))} |")

    L.append("\n## 三、逐条对齐胜负\n")
    total = max(len(common), 1)
    L.append(f"- 共对齐：**{total}** 条")
    L.append(f"- **{args.a_name} 独胜**：{len(only_a)} 条 ({len(only_a)/total:.1%})")
    L.append(f"- **{args.b_name} 独胜**：{len(only_b)} 条 ({len(only_b)/total:.1%})")
    L.append(f"- 都对：{len(both_right)} 条 ({len(both_right)/total:.1%})")
    L.append(f"- 都错：{len(both_wrong)} 条 ({len(both_wrong)/total:.1%})")

    wins_a = len(only_a)
    wins_b = len(only_b)
    if wins_a + wins_b > 0:
        winrate_a = wins_a / (wins_a + wins_b)
        L.append(f"- **胜率 (去除平手)**：{args.a_name} = {winrate_a:.1%}，{args.b_name} = {1-winrate_a:.1%}")
    L.append("")

    def render_case(qid, p, q, name_p, name_q):
        lines = [
            f"### {qid} — {p.get('scenario', '')}",
            f"- **query**: {(p.get('user_query') or '')[:300]}",
            f"- **gold**: intent=`{p.get('gold_intent')}` need_esc=`{p.get('gold_need_escalation')}`",
            f"- **{name_p}** → intent=`{p.get('pred_intent')}` esc=`{p.get('pred_need_escalation')}` score={p.get('overall_score')}",
            f"- **{name_q}** → intent=`{q.get('pred_intent')}` esc=`{q.get('pred_need_escalation')}` score={q.get('overall_score')}",
        ]
        ans_p = (p.get('answer') or '').strip().replace('\n', ' ')[:250]
        ans_q = (q.get('answer') or '').strip().replace('\n', ' ')[:250]
        if ans_p:
            lines.append(f"- {name_p} answer: {ans_p}")
        if ans_q:
            lines.append(f"- {name_q} answer: {ans_q}")
        return "\n".join(lines) + "\n"

    limit = args.bad_case_limit
    L.append(f"\n## 四、{args.a_name} 独胜 top {limit}\n")
    for qid in only_a[:limit]:
        L.append(render_case(qid, a_pred[qid], b_pred[qid], args.a_name, args.b_name))

    L.append(f"\n## 五、{args.b_name} 独胜 top {limit}\n")
    for qid in only_b[:limit]:
        L.append(render_case(qid, b_pred[qid], a_pred[qid], args.b_name, args.a_name))

    L.append(f"\n## 六、两边都错 top {limit}（最值得继续改）\n")
    for qid in both_wrong[:limit]:
        L.append(render_case(qid, a_pred[qid], b_pred[qid], args.a_name, args.b_name))

    # Write report
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(L), encoding='utf-8')

    # Write diff CSV
    csv_path = out_path.parent / "diff_cases.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "id", "category", "scenario", "gold_intent", "gold_escalation",
            f"{args.a_name}_intent", f"{args.a_name}_esc", f"{args.a_name}_overall",
            f"{args.b_name}_intent", f"{args.b_name}_esc", f"{args.b_name}_overall",
            "user_query",
        ])
        for cat, ids in [("only_A_right", only_a), ("only_B_right", only_b), ("both_wrong", both_wrong)]:
            for qid in ids:
                p, q = a_pred[qid], b_pred[qid]
                w.writerow([
                    qid, cat, p.get("scenario", ""),
                    p.get("gold_intent"), p.get("gold_need_escalation"),
                    p.get("pred_intent"), p.get("pred_need_escalation"), p.get("overall_score"),
                    q.get("pred_intent"), q.get("pred_need_escalation"), q.get("overall_score"),
                    (p.get("user_query") or "")[:500],
                ])

    print(f"[report] {out_path}")
    print(f"[diff csv] {csv_path}")


if __name__ == "__main__":
    main()
