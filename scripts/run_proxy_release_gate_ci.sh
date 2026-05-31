#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/ubuntu/qwen35a3b_finetune}"
cd "$PROJECT_ROOT"

BASE_URL="${BASE_URL:-http://127.0.0.1:8001}"
MODEL="${MODEL:-gpu5-v5.1f-merged}"
API_KEY="${API_KEY:-}"

REPEAT_RUNS="${REPEAT_RUNS:-3}"
WORKERS="${WORKERS:-4}"
TIMEOUT_SEC="${TIMEOUT_SEC:-120}"
MAX_RETRIES="${MAX_RETRIES:-1}"
MAX_TOKENS="${MAX_TOKENS:-192}"
TEMPERATURE="${TEMPERATURE:-0}"
TOP_P="${TOP_P:-0.95}"

TS="${TS:-$(date -u +%Y%m%dT%H%M%SZ)}"
OUT_DIR="${OUT_DIR:-$PROJECT_ROOT/eval_outputs/proxy_release_gate_ci_${TS}}"

FIELD_OUTPUT_JSON="$OUT_DIR/field_completeness_summary.json"
TOOL_OUTPUT_DIR="$OUT_DIR/tool_gate"
KB40_ROOT="$OUT_DIR/kb40"
OPS16_ROOT="$OUT_DIR/ops16"
SUMMARY_JSON="$OUT_DIR/release_gate_summary.json"
REPORT_MD="$OUT_DIR/release_gate_report.md"

mkdir -p "$OUT_DIR" "$KB40_ROOT" "$OPS16_ROOT"

# Gate thresholds (defaults are calibrated from 2026-05-20 stability reruns).
MIN_FIELD_PASS_RATE="${MIN_FIELD_PASS_RATE:-1.0}"
MIN_TOOL_OVERALL="${MIN_TOOL_OVERALL:-0.86}"
MIN_TOOL_CALLED="${MIN_TOOL_CALLED:-0.95}"
MIN_TOOL_SCHEMA="${MIN_TOOL_SCHEMA:-0.99}"
MIN_KNOWLEDGE_OVERALL="${MIN_KNOWLEDGE_OVERALL:-0.75}"

MIN_KB40_REQUEST_OK_RATE="${MIN_KB40_REQUEST_OK_RATE:-1.0}"
MIN_KB40_MUST_INCLUDE_AVG="${MIN_KB40_MUST_INCLUDE_AVG:-0.50}"
MIN_KB40_MUST_NOT_AVG="${MIN_KB40_MUST_NOT_AVG:-1.0}"
MAX_KB40_RISK_VIOLATION_RATE="${MAX_KB40_RISK_VIOLATION_RATE:-0.0}"
MIN_KB40_INTENT_ACC="${MIN_KB40_INTENT_ACC:-0.85}"
MIN_KB40_ESCALATION_ACC="${MIN_KB40_ESCALATION_ACC:-0.95}"
MIN_KB40_OVERALL_AVG="${MIN_KB40_OVERALL_AVG:-0.84}"
MAX_KB40_LATENCY_P95_MS="${MAX_KB40_LATENCY_P95_MS:-25000}"

MIN_OPS16_REQUEST_OK_RATE="${MIN_OPS16_REQUEST_OK_RATE:-1.0}"
MIN_OPS16_MUST_INCLUDE_AVG="${MIN_OPS16_MUST_INCLUDE_AVG:-0.78}"
MIN_OPS16_MUST_NOT_AVG="${MIN_OPS16_MUST_NOT_AVG:-1.0}"
MAX_OPS16_RISK_VIOLATION_RATE="${MAX_OPS16_RISK_VIOLATION_RATE:-0.0}"
MIN_OPS16_INTENT_ACC="${MIN_OPS16_INTENT_ACC:-0.98}"
MIN_OPS16_ESCALATION_ACC="${MIN_OPS16_ESCALATION_ACC:-0.93}"
MIN_OPS16_OVERALL_AVG="${MIN_OPS16_OVERALL_AVG:-0.93}"
MAX_OPS16_LATENCY_P95_MS="${MAX_OPS16_LATENCY_P95_MS:-15000}"

FIELD_QUERIES_JSON='[
  {
    "query":"请务必先调用 get_transfer_log_list，再回复三行：工具名、参数、结果摘要。查询最近24小时转账记录，页码1，每页10，状态-1，类型-1，开始时间1746403200000，结束时间1746489600000。",
    "tool_name":"get_transfer_log_list",
    "require_trace":true,
    "require_content":false,
    "require_trace_args":true,
    "require_trace_args_non_empty":true
  },
  {
    "query":"请务必先调用 list_bet_orders，再回复三行：工具名、参数、结果摘要。查询最近体育投注订单，页码1，每页10，状态settled，开始时间1746403200000，结束时间1746489600000。",
    "tool_name":"list_bet_orders",
    "require_trace":true,
    "require_content":false,
    "require_trace_args":true,
    "require_trace_args_non_empty":true
  },
  {
    "query":"请用一句话回复：充值未到账一般先核对什么信息？",
    "require_trace":false,
    "require_content":true
  }
]'

echo "[proxy-gate] out_dir=$OUT_DIR"
echo "[proxy-gate] base_url=$BASE_URL"
echo "[proxy-gate] model=$MODEL"

set +e
python3 "$PROJECT_ROOT/scripts/check_agent_trace_fields.py" \
  --base-url "${BASE_URL%/}/v1" \
  --model "$MODEL" \
  --api-key "$API_KEY" \
  --queries-json "$FIELD_QUERIES_JSON" \
  --output-json "$FIELD_OUTPUT_JSON" \
  --min-pass-rate "$MIN_FIELD_PASS_RATE"
FIELD_RC=$?
set -e

echo "[proxy-gate] field_rc=$FIELD_RC"

set +e
python3 "$PROJECT_ROOT/scripts/eval_tool_dialogue_gate.py" \
  --base-url "${BASE_URL%/}/v1" \
  --model "$MODEL" \
  --api-key "$API_KEY" \
  --tool-eval-jsonl "$PROJECT_ROOT/datasets/tool_use/eval_tool_use_suncidi_30_20260514.jsonl" \
  --knowledge-eval-jsonl "$PROJECT_ROOT/datasets/eval_sports_baowang_knowledge_40_20260428.jsonl" \
  --output-dir "$TOOL_OUTPUT_DIR" \
  --workers "$WORKERS" \
  --timeout "$TIMEOUT_SEC" \
  --min-tool-overall "$MIN_TOOL_OVERALL" \
  --min-tool-called "$MIN_TOOL_CALLED" \
  --min-tool-schema "$MIN_TOOL_SCHEMA" \
  --min-knowledge-overall "$MIN_KNOWLEDGE_OVERALL"
TOOL_RC=$?
set -e

echo "[proxy-gate] tool_rc=$TOOL_RC"

for i in $(seq 1 "$REPEAT_RUNS"); do
  python3 "$PROJECT_ROOT/scripts/eval_sports_customer_service.py" \
    --input-file "$PROJECT_ROOT/datasets/eval_sports_baowang_knowledge_40_20260428.jsonl" \
    --output-dir "$KB40_ROOT/run${i}" \
    --base-url "$BASE_URL" \
    --model "$MODEL" \
    --api-key "$API_KEY" \
    --workers "$WORKERS" \
    --timeout-sec "$TIMEOUT_SEC" \
    --max-retries "$MAX_RETRIES" \
    --temperature "$TEMPERATURE" \
    --top-p "$TOP_P" \
    --max-tokens "$MAX_TOKENS" \
    --disable-thinking

done

for i in $(seq 1 "$REPEAT_RUNS"); do
  python3 "$PROJECT_ROOT/scripts/eval_sports_customer_service.py" \
    --input-file "$PROJECT_ROOT/datasets/eval_ops_feedback_16_20260513.jsonl" \
    --output-dir "$OPS16_ROOT/run${i}" \
    --base-url "$BASE_URL" \
    --model "$MODEL" \
    --api-key "$API_KEY" \
    --workers "$WORKERS" \
    --timeout-sec "$TIMEOUT_SEC" \
    --max-retries "$MAX_RETRIES" \
    --temperature "$TEMPERATURE" \
    --top-p "$TOP_P" \
    --max-tokens "$MAX_TOKENS" \
    --disable-thinking

done

export OUT_DIR FIELD_OUTPUT_JSON TOOL_OUTPUT_DIR KB40_ROOT OPS16_ROOT SUMMARY_JSON REPORT_MD
export MIN_FIELD_PASS_RATE MIN_TOOL_OVERALL MIN_TOOL_CALLED MIN_TOOL_SCHEMA MIN_KNOWLEDGE_OVERALL
export MIN_KB40_REQUEST_OK_RATE MIN_KB40_MUST_INCLUDE_AVG MIN_KB40_MUST_NOT_AVG MAX_KB40_RISK_VIOLATION_RATE MIN_KB40_INTENT_ACC MIN_KB40_ESCALATION_ACC MIN_KB40_OVERALL_AVG MAX_KB40_LATENCY_P95_MS
export MIN_OPS16_REQUEST_OK_RATE MIN_OPS16_MUST_INCLUDE_AVG MIN_OPS16_MUST_NOT_AVG MAX_OPS16_RISK_VIOLATION_RATE MIN_OPS16_INTENT_ACC MIN_OPS16_ESCALATION_ACC MIN_OPS16_OVERALL_AVG MAX_OPS16_LATENCY_P95_MS
export FIELD_RC TOOL_RC REPEAT_RUNS

python3 - <<'PY'
import glob
import json
import os
import statistics
from pathlib import Path


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def to_f(v):
    try:
        return float(v)
    except Exception:
        return 0.0


def summarize_runs(root: Path, pattern: str) -> list[dict]:
    runs = []
    for p in sorted(glob.glob(str(root / pattern / "summary.json"))):
        s = load_json(Path(p))
        runs.append(
            {
                "summary": p,
                "request_ok_rate": to_f(s.get("request_ok_rate")),
                "must_include_avg": to_f(s.get("must_include_avg")),
                "must_not_avg": to_f(s.get("must_not_avg")),
                "risk_violation_rate": to_f(s.get("risk_violation_rate")),
                "intent_acc": to_f(s.get("intent_acc")),
                "escalation_acc": to_f(s.get("escalation_acc")),
                "overall_avg": to_f(s.get("overall_avg")),
                "latency_ms_p50": to_f(s.get("latency_ms_p50")),
                "latency_ms_p95": to_f(s.get("latency_ms_p95")),
            }
        )
    return runs


def aggregate(runs: list[dict]) -> dict:
    if not runs:
        return {"runs": 0}
    keys = ["overall_avg", "latency_ms_p50", "latency_ms_p95"]
    out = {"runs": len(runs)}
    for key in keys:
        vals = [to_f(r[key]) for r in runs]
        out[f"{key}_min"] = min(vals)
        out[f"{key}_max"] = max(vals)
        out[f"{key}_mean"] = statistics.mean(vals)
    return out


def gate_run(row: dict, cfg: dict) -> tuple[bool, list[str]]:
    errs = []
    if row["request_ok_rate"] < cfg["request_ok_rate"]:
        errs.append(f"request_ok_rate={row['request_ok_rate']:.4f} < {cfg['request_ok_rate']:.4f}")
    if row["must_include_avg"] < cfg["must_include_avg"]:
        errs.append(f"must_include_avg={row['must_include_avg']:.4f} < {cfg['must_include_avg']:.4f}")
    if row["must_not_avg"] < cfg["must_not_avg"]:
        errs.append(f"must_not_avg={row['must_not_avg']:.4f} < {cfg['must_not_avg']:.4f}")
    if row["risk_violation_rate"] > cfg["risk_violation_rate_max"]:
        errs.append(f"risk_violation_rate={row['risk_violation_rate']:.4f} > {cfg['risk_violation_rate_max']:.4f}")
    if row["intent_acc"] < cfg["intent_acc"]:
        errs.append(f"intent_acc={row['intent_acc']:.4f} < {cfg['intent_acc']:.4f}")
    if row["escalation_acc"] < cfg["escalation_acc"]:
        errs.append(f"escalation_acc={row['escalation_acc']:.4f} < {cfg['escalation_acc']:.4f}")
    if row["overall_avg"] < cfg["overall_avg"]:
        errs.append(f"overall_avg={row['overall_avg']:.4f} < {cfg['overall_avg']:.4f}")
    if row["latency_ms_p95"] > cfg["latency_p95_ms_max"]:
        errs.append(f"latency_ms_p95={row['latency_ms_p95']:.1f} > {cfg['latency_p95_ms_max']:.1f}")
    return (len(errs) == 0, errs)


out_dir = Path(os.environ["OUT_DIR"])
field_summary = load_json(Path(os.environ["FIELD_OUTPUT_JSON"]))
tool_summary = load_json(Path(os.environ["TOOL_OUTPUT_DIR"]) / "summary.json")
kb40_runs = summarize_runs(Path(os.environ["KB40_ROOT"]), "run*")
ops16_runs = summarize_runs(Path(os.environ["OPS16_ROOT"]), "run*")

thresholds = {
    "field": {"min_pass_rate": to_f(os.environ["MIN_FIELD_PASS_RATE"])},
    "tool": {
        "min_tool_overall": to_f(os.environ["MIN_TOOL_OVERALL"]),
        "min_tool_called": to_f(os.environ["MIN_TOOL_CALLED"]),
        "min_tool_schema": to_f(os.environ["MIN_TOOL_SCHEMA"]),
        "min_knowledge_overall": to_f(os.environ["MIN_KNOWLEDGE_OVERALL"]),
    },
    "kb40": {
        "request_ok_rate": to_f(os.environ["MIN_KB40_REQUEST_OK_RATE"]),
        "must_include_avg": to_f(os.environ["MIN_KB40_MUST_INCLUDE_AVG"]),
        "must_not_avg": to_f(os.environ["MIN_KB40_MUST_NOT_AVG"]),
        "risk_violation_rate_max": to_f(os.environ["MAX_KB40_RISK_VIOLATION_RATE"]),
        "intent_acc": to_f(os.environ["MIN_KB40_INTENT_ACC"]),
        "escalation_acc": to_f(os.environ["MIN_KB40_ESCALATION_ACC"]),
        "overall_avg": to_f(os.environ["MIN_KB40_OVERALL_AVG"]),
        "latency_p95_ms_max": to_f(os.environ["MAX_KB40_LATENCY_P95_MS"]),
    },
    "ops16": {
        "request_ok_rate": to_f(os.environ["MIN_OPS16_REQUEST_OK_RATE"]),
        "must_include_avg": to_f(os.environ["MIN_OPS16_MUST_INCLUDE_AVG"]),
        "must_not_avg": to_f(os.environ["MIN_OPS16_MUST_NOT_AVG"]),
        "risk_violation_rate_max": to_f(os.environ["MAX_OPS16_RISK_VIOLATION_RATE"]),
        "intent_acc": to_f(os.environ["MIN_OPS16_INTENT_ACC"]),
        "escalation_acc": to_f(os.environ["MIN_OPS16_ESCALATION_ACC"]),
        "overall_avg": to_f(os.environ["MIN_OPS16_OVERALL_AVG"]),
        "latency_p95_ms_max": to_f(os.environ["MAX_OPS16_LATENCY_P95_MS"]),
    },
}

field_pass = bool(field_summary.get("passed")) and int(os.environ.get("FIELD_RC", "1")) == 0
tool_pass = bool(tool_summary.get("passed")) and int(os.environ.get("TOOL_RC", "1")) == 0

kb40_checks = []
for idx, row in enumerate(kb40_runs, 1):
    ok, errors = gate_run(row, thresholds["kb40"])
    kb40_checks.append({"run": idx, "passed": ok, "errors": errors})
ops16_checks = []
for idx, row in enumerate(ops16_runs, 1):
    ok, errors = gate_run(row, thresholds["ops16"])
    ops16_checks.append({"run": idx, "passed": ok, "errors": errors})

kb40_pass = len(kb40_runs) == int(os.environ["REPEAT_RUNS"]) and all(x["passed"] for x in kb40_checks)
ops16_pass = len(ops16_runs) == int(os.environ["REPEAT_RUNS"]) and all(x["passed"] for x in ops16_checks)

overall_pass = field_pass and tool_pass and kb40_pass and ops16_pass

summary = {
    "base_url": os.environ.get("BASE_URL", ""),
    "model": os.environ.get("MODEL", ""),
    "repeat_runs": int(os.environ["REPEAT_RUNS"]),
    "field": field_summary,
    "tool": tool_summary,
    "kb40_runs": kb40_runs,
    "ops16_runs": ops16_runs,
    "kb40_aggregate": aggregate(kb40_runs),
    "ops16_aggregate": aggregate(ops16_runs),
    "thresholds": thresholds,
    "checks": {
        "field_pass": field_pass,
        "tool_pass": tool_pass,
        "kb40_runs_pass": kb40_pass,
        "ops16_runs_pass": ops16_pass,
        "kb40_run_checks": kb40_checks,
        "ops16_run_checks": ops16_checks,
    },
    "passed": overall_pass,
}

Path(os.environ["SUMMARY_JSON"]).write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

lines = [
    "# Proxy Release Gate Report",
    "",
    f"- output_dir: `{out_dir}`",
    f"- repeat_runs: `{summary['repeat_runs']}`",
    "",
    "## Gate Status",
    f"- field_pass: `{field_pass}`",
    f"- tool_pass: `{tool_pass}`",
    f"- kb40_runs_pass: `{kb40_pass}`",
    f"- ops16_runs_pass: `{ops16_pass}`",
    f"- overall_passed: `{overall_pass}`",
    "",
    "## Field Completeness",
    f"- pass_rate: `{field_summary.get('pass_rate')}`",
    "",
    "## Tool Hit",
    f"- tool.overall: `{(tool_summary.get('tool') or {}).get('overall')}`",
    f"- tool.called: `{(tool_summary.get('tool') or {}).get('tool_called')}`",
    f"- tool.schema_valid: `{(tool_summary.get('tool') or {}).get('schema_valid')}`",
    f"- knowledge.overall: `{(tool_summary.get('knowledge') or {}).get('overall')}`",
    "",
    "## KB40 Stability",
]
for i, row in enumerate(kb40_runs, 1):
    lines.append(
        f"- run{i}: overall={row['overall_avg']:.4f}, intent={row['intent_acc']:.4f}, esc={row['escalation_acc']:.4f}, must_include={row['must_include_avg']:.4f}, p95={row['latency_ms_p95']:.1f}ms"
    )
lines.extend(
    [
        f"- aggregate overall[min,max,mean]: `{summary['kb40_aggregate'].get('overall_avg_min')}` / `{summary['kb40_aggregate'].get('overall_avg_max')}` / `{summary['kb40_aggregate'].get('overall_avg_mean')}`",
        f"- aggregate latency p95[min,max,mean]: `{summary['kb40_aggregate'].get('latency_ms_p95_min')}` / `{summary['kb40_aggregate'].get('latency_ms_p95_max')}` / `{summary['kb40_aggregate'].get('latency_ms_p95_mean')}`",
        "",
        "## OPS16 Stability",
    ]
)
for i, row in enumerate(ops16_runs, 1):
    lines.append(
        f"- run{i}: overall={row['overall_avg']:.4f}, intent={row['intent_acc']:.4f}, esc={row['escalation_acc']:.4f}, must_include={row['must_include_avg']:.4f}, p95={row['latency_ms_p95']:.1f}ms"
    )
lines.extend(
    [
        f"- aggregate overall[min,max,mean]: `{summary['ops16_aggregate'].get('overall_avg_min')}` / `{summary['ops16_aggregate'].get('overall_avg_max')}` / `{summary['ops16_aggregate'].get('overall_avg_mean')}`",
        f"- aggregate latency p95[min,max,mean]: `{summary['ops16_aggregate'].get('latency_ms_p95_min')}` / `{summary['ops16_aggregate'].get('latency_ms_p95_max')}` / `{summary['ops16_aggregate'].get('latency_ms_p95_mean')}`",
    ]
)
Path(os.environ["REPORT_MD"]).write_text("\n".join(lines) + "\n", encoding="utf-8")

print(json.dumps(summary, ensure_ascii=False, indent=2))
print(f"OVERALL_PASSED={'true' if overall_pass else 'false'}")
PY

if ! python3 - <<'PY'
import json
import os
from pathlib import Path
summary = json.loads(Path(os.environ["SUMMARY_JSON"]).read_text(encoding="utf-8"))
raise SystemExit(0 if summary.get("passed") else 2)
PY
then
  echo "[proxy-gate] failed" >&2
  exit 2
fi

echo "[proxy-gate] passed"
echo "[proxy-gate] summary: $SUMMARY_JSON"
echo "[proxy-gate] report: $REPORT_MD"
