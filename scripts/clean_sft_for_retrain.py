#!/usr/bin/env python3
"""Clean SFT jsonl for customer-service retraining.

Targets:
- de-noise and de-duplicate
- remove/mark risky marketing & illegal content
- mask PII placeholders
- filter low-value / no-reply / empty-template conversations
- reduce repetitive text degeneration in assistant replies
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple


PHONE_RE = re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)")
BANK_RE = re.compile(r"(?<!\d)\d{15,19}(?!\d)")
ID_RE = re.compile(
    r"(?<![0-9A-Za-z])[1-9]\d{5}(?:19|20)\d{2}(?:0[1-9]|1[0-2])"
    r"(?:0[1-9]|[12]\d|3[01])\d{3}[0-9Xx](?![0-9A-Za-z])"
)
ORDER_RE = re.compile(r"\b[a-fA-F0-9]{16,32}\b")
ACCOUNT_CTX_RE = re.compile(
    r"(账号|会员号|会员账号|用户名|游戏账号|ID|id|帐号)\s*[:：]?\s*([A-Za-z0-9_\-]{4,32})"
)
POSSIBLE_ACCOUNT_RE = re.compile(r"(?<!\d)\d{6,12}(?!\d)")
HTML_TAG_RE = re.compile(r"<[^>]+>")
URL_RE = re.compile(r"https?://\S+", flags=re.I)
SPACE_RE = re.compile(r"\s+")
SENTENCE_SPLIT_RE = re.compile(r"[。！？!?；;|\n]+")
THINK_TAG_RE = re.compile(r"</?think>", flags=re.I)
INTERNAL_TICKET_RE = re.compile(r"(?:工单号|受理编号|受理号)\s*[:：]?\s*[A-Z]{0,4}\d{4,}", flags=re.I)
INTERNAL_FLOW_ID_RE = re.compile(r"\b(?:TK|CS|SP|SB|TP|RA)\d{5,}\b", flags=re.I)

LOG_NOISE_PATTERNS = [
    re.compile(p, flags=re.I)
    for p in [
        r"traceback",
        r"exception",
        r"http request",
        r"interface log",
        r"api error",
        r"stack",
        r"^\s*at\s+.*\(.+\)\s*$",
        r"<html",
        r"</html>",
        r"<body",
        r"</body>",
        r"^\s*/res/image\.html\?id=.*$",
    ]
]

RISK_PATTERNS = {
    "gambling_inducement": [
        r"包赢",
        r"稳赚",
        r"必赚",
        r"带你赢钱",
        r"跟单稳赢",
        r"回血方案",
        r"试试手气",
        r"爆大奖",
        r"盈利多多",
        r"以小博大",
    ],
    "agent_recruitment": [
        r"招代理",
        r"代理返佣",
        r"拉新返利",
        r"发展下线",
        r"推广专员",
    ],
    "illegal_fund_flow": [
        r"洗钱",
        r"跑分",
        r"代付",
        r"代收",
        r"过账",
        r"通道费",
        r"包司法",
        r"包冻结",
        r"无视风控",
    ],
    "forged_identity": [
        r"伪造银行卡",
        r"伪造身份",
        r"伪造证件",
        r"假证",
        r"套证",
    ],
    "underage": [
        r"未成年",
        r"低龄",
        r"学生兼职博彩",
    ],
}

LOW_VALUE_SET = {
    "好",
    "好的",
    "谢谢",
    "ok",
    "OK",
    "嗯",
    "哦",
    "收到",
    "在吗",
    "1",
    "？",
}

EMPTY_TEMPLATE_PATTERNS = [
    re.compile(p)
    for p in [
        r"转人工",
        r"稍后联系",
        r"请稍后",
        r"马上为您查询",
        r"稍等",
    ]
]

MARKETING_STYLE_PATTERNS = [
    re.compile(p, flags=re.I)
    for p in [
        r"试试手气",
        r"爆大奖",
        r"冲鸭",
        r"福利来袭",
        r"把握机会",
        r"盈利多多",
        r"以小博大",
        r"快来(参与|体验|充值|存款)",
        r"建议您现在进行存款",
        r"咨询这么多天.*上分",
        r"不上分",
        r"惊喜福利",
        # Contact-diversion / salesperson style phrases that should not appear
        # in clean sports-knowledge or support-answer SFT data.
        r"(添加|联系|咨询|对接).{0,8}(客服|专员|经理|顾问|POP)",
        r"(客服|专员|经理|顾问|POP).{0,8}(微信|v[x信]|QQ|Telegram|tg|WhatsApp)",
        r"(添加|加).{0,6}(微信|v[x信]|QQ|Telegram|tg|WhatsApp)",
        r"(欢迎|可以|可).{0,8}(咨询|联系).{0,8}(客服|专员|经理|顾问|POP)",
        r"(私聊|私信|拉群|建群|进群|对接群|专属群)",
        r"(客服POP|POP客服|联系POP|添加POP|对接POP)",
        r"(商务合作|代理合作|包网合作).{0,8}(联系|咨询|对接)",
        r"(开.?链接|注册链接|专属链接|开户链接)",
    ]
]

STRUCTURED_LEAK_PATTERNS = [
    re.compile(p, flags=re.I)
    for p in [
        r"</think>",
        r"<think>",
        r'"intent"\s*:',
        r'"need_escalation"\s*:',
        r'"risk_flag"\s*:',
        r'"next_action"\s*:',
        r'"answer"\s*:',
        r"\bintent\s*:\s*",
        r"\bneed_escalation\s*:\s*",
        r"\bnext_action\s*:\s*",
        r"^\s*\{[\s\S]*\}\s*$",
        r"用户问题[:：]",
    ]
]

OPS_TEMPLATE_PATTERNS = [
    re.compile(p, flags=re.I)
    for p in [
        r"(已提交|提交至|转交|升级至|同步给).{0,10}(后台|运营|专员|主管|财务团队|技术团队)",
        r"(运营专员|运营同事|运营主管|财务团队|技术团队).{0,24}(核查|核验|核实|跟进|处理|复核)",
        r"(预计|大约).{0,6}(3-5分钟|10-30分钟|15-30分钟|30分钟|1小时|24小时|48小时).{0,8}(反馈|处理完毕|给您反馈|内反馈|内处理完毕|内受理|内初审)",
        r"若超时未更新.{0,12}(回复本窗口|继续跟进)",
        r"(受理编号|受理号|工单号)",
        r"(提交运营后台|运营后台|联系运营|转运营|运营处理)",
    ]
]


def normalize_text(text: str) -> str:
    text = str(text or "")
    text = text.replace("\r", " ")
    text = HTML_TAG_RE.sub(" ", text)
    text = URL_RE.sub(" <URL> ", text)
    text = text.replace("&nbsp;", " ")
    text = SPACE_RE.sub(" ", text)
    return text.strip()


def mask_pii(text: str) -> str:
    text = PHONE_RE.sub("<PHONE>", text)
    text = ID_RE.sub("<ID_NO>", text)
    text = ORDER_RE.sub("<ORDER_ID>", text)
    text = BANK_RE.sub("<BANK_CARD>", text)

    def _mask_ctx(m: re.Match[str]) -> str:
        return f"{m.group(1)} <ACCOUNT>"

    text = ACCOUNT_CTX_RE.sub(_mask_ctx, text)
    text = POSSIBLE_ACCOUNT_RE.sub("<ACCOUNT>", text)
    return text


def is_noise_line(line: str) -> bool:
    if not line:
        return True
    s = line.strip()
    if not s:
        return True
    if len(s) > 2500:
        return True
    for pat in LOG_NOISE_PATTERNS:
        if pat.search(s):
            return True
    return False


def dedupe_lines(content: str) -> Tuple[str, int, int]:
    """Return cleaned content + repeat stats.

    repeated_line_hits: number of lines removed due to exact duplication.
    max_same_line_run: max consecutive repeat count in original content.
    """
    raw_lines = [ln.strip() for ln in str(content).split("\n")]
    raw_lines = [ln for ln in raw_lines if ln]
    if not raw_lines:
        return "", 0, 0

    deduped: List[str] = []
    repeated_line_hits = 0
    max_same_line_run = 1
    run = 1
    prev = None

    for ln in raw_lines:
        if prev is not None and ln == prev:
            run += 1
            max_same_line_run = max(max_same_line_run, run)
            repeated_line_hits += 1
        else:
            run = 1
        prev = ln
        if not deduped or deduped[-1] != ln:
            deduped.append(ln)

    return "\n".join(deduped).strip(), repeated_line_hits, max_same_line_run


def has_heavy_repetition(text: str) -> bool:
    compact = re.sub(r"\s+", "", text)
    if len(compact) < 80:
        return False
    n = 12
    if len(compact) < n * 6:
        return False
    cnt = Counter(compact[i : i + n] for i in range(0, len(compact) - n + 1))
    if not cnt:
        return False
    _, max_count = cnt.most_common(1)[0]
    repeat_mass = max_count * n
    return max_count >= 6 and (repeat_mass / len(compact)) >= 0.35


def detect_risk(text: str) -> List[str]:
    flags: List[str] = []
    for key, pats in RISK_PATTERNS.items():
        if any(re.search(p, text, flags=re.I) for p in pats):
            flags.append(key)
    return sorted(flags)


def has_marketing_style(text: str) -> bool:
    return any(p.search(text) for p in MARKETING_STYLE_PATTERNS)


def looks_like_empty_template(text: str) -> bool:
    t = text.strip()
    if len(t) < 4:
        return True
    return any(p.search(t) for p in EMPTY_TEMPLATE_PATTERNS)


def has_structured_leak(text: str) -> bool:
    return any(p.search(text) for p in STRUCTURED_LEAK_PATTERNS)


def has_ops_ticket_template(text: str) -> bool:
    if INTERNAL_TICKET_RE.search(text):
        return True
    ops_hits = sum(1 for p in OPS_TEMPLATE_PATTERNS if p.search(text))
    if ops_hits >= 2:
        return True
    return ops_hits >= 1 and bool(INTERNAL_FLOW_ID_RE.search(text))


def has_duplicate_sentences(text: str, min_len: int = 8, min_hits: int = 3) -> bool:
    parts = [normalize_text(x) for x in SENTENCE_SPLIT_RE.split(text or "")]
    parts = [x for x in parts if len(x) >= min_len]
    if not parts:
        return False
    cnt = Counter(parts)
    return any(v >= min_hits for v in cnt.values())


def normalize_for_hash(messages: List[Dict[str, str]]) -> str:
    parts = []
    for m in messages:
        c = m["content"].lower().strip()
        c = re.sub(r"\d+", "<NUM>", c)
        c = SPACE_RE.sub(" ", c)
        parts.append(f'{m["role"]}:{c}')
    return "\n".join(parts)


def stable_hash(text: str, length: int = 16) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:length]


def clean_messages(messages: Any) -> Tuple[List[Dict[str, str]], Counter]:
    stats = Counter()
    out: List[Dict[str, str]] = []

    if not isinstance(messages, list):
        return out, stats

    for m in messages:
        if not isinstance(m, dict):
            continue
        role = str(m.get("role", "")).strip().lower()
        if role not in {"user", "assistant", "system"}:
            continue
        if role == "system":
            continue

        raw_content = str(m.get("content", "")).replace("\r", "\n")
        if not raw_content.strip():
            continue

        lines = []
        for ln in raw_content.split("\n"):
            ln = normalize_text(ln)
            if is_noise_line(ln):
                stats["noise_lines_removed"] += 1
                continue
            lines.append(ln)
        content = "\n".join(lines).strip()
        if not content:
            continue

        content = mask_pii(content)
        content, repeat_removed, max_run = dedupe_lines(content)
        stats["repeat_lines_removed"] += repeat_removed
        stats["max_same_line_run"] = max(stats["max_same_line_run"], max_run)
        if not content:
            continue

        if len(content) > 1200:
            stats["oversize_truncate"] += 1
            content = content[:1200].strip()

        if role == "assistant" and has_heavy_repetition(content):
            stats["assistant_heavy_repetition"] += 1
            continue

        if out and out[-1]["role"] == role:
            out[-1]["content"] = (out[-1]["content"] + "\n" + content).strip()
            stats["merged_consecutive_role"] += 1
        else:
            out.append({"role": role, "content": content})
    return out, stats


def should_drop_record(messages: List[Dict[str, str]]) -> Tuple[bool, str]:
    if not messages:
        return True, "empty_after_clean"

    roles = {m["role"] for m in messages}
    if "user" not in roles:
        return True, "no_user"
    if "assistant" not in roles:
        return True, "question_without_reply"

    user_msgs = [m["content"].strip() for m in messages if m["role"] == "user"]
    assistant_msgs = [m["content"].strip() for m in messages if m["role"] == "assistant"]
    full_text = "\n".join(m["content"] for m in messages)

    if len(full_text) < 12:
        return True, "too_short_total"
    if len(full_text) > 6000:
        return True, "too_long_total"

    if all((len(x) < 4 or x in LOW_VALUE_SET) for x in user_msgs):
        return True, "low_value_user_only"

    if not any(len(x) >= 4 for x in assistant_msgs):
        return True, "assistant_too_short"

    # Drop conversations where assistant is only placeholder templates.
    if all(looks_like_empty_template(x) for x in assistant_msgs):
        return True, "assistant_empty_template_only"

    return False, ""


def clean_dataset(input_path: Path, output_path: Path, report_dir: Path) -> Dict[str, Any]:
    seen_hash = set()
    stats = Counter()
    max_same_line_run_global = 1
    reject_reasons = Counter()
    risk_counter = Counter()
    rejected_examples: List[Dict[str, Any]] = []
    cleaned_rows: List[Dict[str, Any]] = []

    with input_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                stats["empty_lines"] += 1
                continue
            stats["rows_total"] += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                reject_reasons["invalid_json"] += 1
                if len(rejected_examples) < 200:
                    rejected_examples.append({"line": line_no, "reason": "invalid_json"})
                continue

            messages, msg_stats = clean_messages(row.get("messages"))
            stats.update(msg_stats)
            max_same_line_run_global = max(
                max_same_line_run_global, int(msg_stats.get("max_same_line_run", 1))
            )
            drop, reason = should_drop_record(messages)
            if drop:
                reject_reasons[reason] += 1
                if len(rejected_examples) < 200:
                    rejected_examples.append({"line": line_no, "id": row.get("id", ""), "reason": reason})
                continue

            joined = "\n".join(m["content"] for m in messages)
            assistant_joined = "\n".join(m["content"] for m in messages if m["role"] == "assistant")

            if has_structured_leak(assistant_joined):
                reject_reasons["structured_intermediate_leak"] += 1
                if len(rejected_examples) < 200:
                    rejected_examples.append(
                        {
                            "line": line_no,
                            "id": row.get("id", ""),
                            "reason": "structured_intermediate_leak",
                        }
                    )
                continue

            if has_ops_ticket_template(assistant_joined):
                reject_reasons["ops_ticket_template_filtered"] += 1
                if len(rejected_examples) < 200:
                    rejected_examples.append(
                        {
                            "line": line_no,
                            "id": row.get("id", ""),
                            "reason": "ops_ticket_template_filtered",
                        }
                    )
                continue

            if has_duplicate_sentences(assistant_joined):
                reject_reasons["assistant_sentence_repetition"] += 1
                if len(rejected_examples) < 200:
                    rejected_examples.append(
                        {
                            "line": line_no,
                            "id": row.get("id", ""),
                            "reason": "assistant_sentence_repetition",
                        }
                    )
                continue

            if has_marketing_style(joined):
                reject_reasons["marketing_style_filtered"] += 1
                if len(rejected_examples) < 200:
                    rejected_examples.append(
                        {
                            "line": line_no,
                            "id": row.get("id", ""),
                            "reason": "marketing_style_filtered",
                        }
                    )
                continue

            risk_flags = detect_risk(joined)
            if risk_flags:
                reject_reasons["risk_filtered"] += 1
                for rf in risk_flags:
                    risk_counter[rf] += 1
                if len(rejected_examples) < 200:
                    rejected_examples.append(
                        {
                            "line": line_no,
                            "id": row.get("id", ""),
                            "reason": "risk_filtered",
                            "risk_flags": risk_flags,
                        }
                    )
                continue

            d_hash = stable_hash(normalize_for_hash(messages), 20)
            if d_hash in seen_hash:
                reject_reasons["duplicate_by_hash"] += 1
                continue
            seen_hash.add(d_hash)

            row_out = dict(row)
            row_out["messages"] = messages
            row_out["dialogue_hash"] = d_hash
            cleaned_rows.append(row_out)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for r in cleaned_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    report_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "rows_total": stats["rows_total"],
        "rows_kept": len(cleaned_rows),
        "rows_removed": stats["rows_total"] - len(cleaned_rows),
        "keep_rate": round(len(cleaned_rows) / stats["rows_total"], 4) if stats["rows_total"] else 0.0,
        "unique_dialogues": len(seen_hash),
        "noise_lines_removed": stats["noise_lines_removed"],
        "repeat_lines_removed": stats["repeat_lines_removed"],
        "merged_consecutive_role": stats["merged_consecutive_role"],
        "assistant_heavy_repetition_removed": stats["assistant_heavy_repetition"],
        "max_same_line_run_seen": max_same_line_run_global,
        "risk_distribution": dict(risk_counter),
        "reject_reasons": dict(reject_reasons),
    }

    (report_dir / "sft_cleaning_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    with (report_dir / "sft_cleaning_reject_reasons.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["reason", "count"])
        for k, v in sorted(reject_reasons.items(), key=lambda x: (-x[1], x[0])):
            w.writerow([k, v])

    with (report_dir / "sft_cleaning_risk_distribution.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["risk_flag", "count"])
        for k, v in sorted(risk_counter.items(), key=lambda x: (-x[1], x[0])):
            w.writerow([k, v])

    with (report_dir / "sft_cleaning_rejected_examples.jsonl").open("w", encoding="utf-8") as f:
        for r in rejected_examples:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean SFT dataset for retraining.")
    parser.add_argument(
        "--input",
        default="/home/ubuntu/qwen35a3b_finetune/datasets/sft_openai_messages.jsonl",
        help="Input JSONL path.",
    )
    parser.add_argument(
        "--output",
        default="/home/ubuntu/qwen35a3b_finetune/datasets/sft_openai_messages.cleaned.v2.jsonl",
        help="Output JSONL path.",
    )
    parser.add_argument(
        "--report-dir",
        default="/home/ubuntu/qwen35a3b_finetune/datasets/clean_reports",
        help="Report directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = clean_dataset(Path(args.input), Path(args.output), Path(args.report_dir))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
