#!/usr/bin/env python3
"""Validate hard sports-rule answers in eval predictions.

The normal customer-service evaluator scores intent/escalation and keyword
coverage. This checker reads optional `rule_checks` from the eval JSONL and
flags hard-rule mistakes such as wrong over/under settlement or wrong odds
calculation.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
            if not isinstance(obj, dict):
                raise ValueError(f"Expected object at {path}:{line_no}")
            rows.append(obj)
    return rows


def read_predictions(path: Path) -> dict[str, dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        rows = read_jsonl(path)
    elif path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
    else:
        raise ValueError(f"Unsupported predictions file extension: {path}")

    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        sample_id = str(row.get("id") or row.get("sample_id") or "").strip()
        if sample_id:
            out[sample_id] = row
    return out


def normalize(text: Any) -> str:
    return re.sub(r"\s+", "", str(text or "")).lower()


def digit_boundary_present(answer_norm: str, term_norm: str) -> bool:
    """Avoid matching numeric terms inside a larger number, e.g. 170元 in 1170元."""
    if not term_norm:
        return True
    if not any(ch.isdigit() for ch in term_norm):
        return term_norm in answer_norm
    pattern = re.escape(term_norm)
    return re.search(rf"(?<!\d){pattern}(?!\d)", answer_norm) is not None


def numeric_phrase_present(answer_norm: str, term_norm: str) -> bool:
    """Match equivalent short numeric rule phrases with intervening formula text.

    The rule file intentionally uses compact phrases such as "亏损880" or
    "输1170". Model answers often say "亏损1000*0.88=880元", which is the same
    rule but fails a raw substring check. Keep this conservative: only support
    known settlement verbs and require the target number nearby after the verb.
    """
    m = re.fullmatch(r"(亏损本金|损失本金|盈利|亏损|亏|输|赢)(\d+(?:\.\d+)?)", term_norm)
    if not m:
        return False
    verb, number = m.groups()
    verb_alternatives = {
        "亏损本金": ["亏损本金", "亏损为本金", "亏损"],
        "损失本金": ["损失本金", "损失为本金", "亏损"],
        "亏损": ["亏损"],
        "亏": ["亏", "亏损"],
        "输": ["输", "亏损", "未中奖"],
        "盈利": ["盈利", "赢"],
        "赢": ["赢", "盈利"],
    }.get(verb, [verb])
    for alt in verb_alternatives:
        # Allow formula text between the verb and the checked number, but keep
        # the window short enough that unrelated numbers in the same answer do
        # not satisfy the check.
        if re.search(rf"{re.escape(alt)}[^\d]{{0,8}}(?:\d+(?:\.\d+)?[*x×+\-/=])*{re.escape(number)}(?!\d)", answer_norm):
            return True
    return False


def term_present(answer_norm: str, term: str) -> bool:
    term_norm = normalize(term)
    return digit_boundary_present(answer_norm, term_norm) or numeric_phrase_present(answer_norm, term_norm)


def to_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x) for x in value if str(x)]
    return [str(value)]


def any_present(answer_norm: str, terms: list[str]) -> bool:
    if not terms:
        return True
    return any(term_present(answer_norm, term) for term in terms)


def any_absent(answer_norm: str, terms: list[str]) -> bool:
    return not any(digit_boundary_present(answer_norm, normalize(term)) for term in terms)


def evaluate_check(answer: str, check: dict[str, Any]) -> tuple[bool, str]:
    answer_norm = normalize(answer)
    must_include_any = to_list(check.get("must_include_any"))
    must_include_all = to_list(check.get("must_include_all"))
    must_not_include_any = to_list(check.get("must_not_include_any"))

    if must_include_any and not any_present(answer_norm, must_include_any):
        return False, f"missing_any={must_include_any}"
    missing_all = [term for term in must_include_all if normalize(term) not in answer_norm]
    if missing_all:
        return False, f"missing_all={missing_all}"
    if must_not_include_any and not any_absent(answer_norm, must_not_include_any):
        hits = [term for term in must_not_include_any if digit_boundary_present(answer_norm, normalize(term))]
        return False, f"forbidden={hits}"
    return True, ""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-jsonl", required=True, help="Eval set containing optional rule_checks")
    parser.add_argument("--predictions", required=True, help="predictions.csv or predictions.jsonl")
    parser.add_argument("--output-dir", default="", help="Output directory; defaults to predictions parent / rule_check")
    parser.add_argument("--answer-field", default="answer", help="Prediction field containing final answer")
    args = parser.parse_args()

    eval_path = Path(args.eval_jsonl).expanduser().resolve()
    pred_path = Path(args.predictions).expanduser().resolve()
    out_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else pred_path.parent / "rule_check"
    out_dir.mkdir(parents=True, exist_ok=True)

    eval_rows = read_jsonl(eval_path)
    predictions = read_predictions(pred_path)

    case_rows: list[dict[str, Any]] = []
    total_checks = 0
    passed_checks = 0
    checked_cases = 0
    missing_predictions = 0

    for sample in eval_rows:
        sample_id = str(sample.get("id", "")).strip()
        checks = sample.get("rule_checks") or []
        if not sample_id or not checks:
            continue
        checked_cases += 1
        pred = predictions.get(sample_id)
        if pred is None:
            missing_predictions += 1
            for check in checks:
                total_checks += 1
                case_rows.append(
                    {
                        "id": sample_id,
                        "scenario": sample.get("scenario", ""),
                        "rule_id": check.get("id", ""),
                        "pass": 0,
                        "reason": "missing_prediction",
                        "description": check.get("description", ""),
                        "answer": "",
                    }
                )
            continue
        answer = str(pred.get(args.answer_field, "") or "")
        for check in checks:
            total_checks += 1
            ok, reason = evaluate_check(answer, check)
            passed_checks += int(ok)
            case_rows.append(
                {
                    "id": sample_id,
                    "scenario": sample.get("scenario", ""),
                    "rule_id": check.get("id", ""),
                    "pass": int(ok),
                    "reason": reason,
                    "description": check.get("description", ""),
                    "answer": answer,
                }
            )

    failed_rows = [row for row in case_rows if not row["pass"]]
    summary = {
        "eval_jsonl": str(eval_path),
        "predictions": str(pred_path),
        "checked_cases": checked_cases,
        "missing_predictions": missing_predictions,
        "total_checks": total_checks,
        "passed_checks": passed_checks,
        "failed_checks": len(failed_rows),
        "pass_rate": round(passed_checks / total_checks, 4) if total_checks else 0.0,
    }

    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (out_dir / "rule_check_cases.csv").open("w", encoding="utf-8", newline="") as f:
        fieldnames = ["id", "scenario", "rule_id", "pass", "reason", "description", "answer"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(case_rows)
    with (out_dir / "rule_check_failures.csv").open("w", encoding="utf-8", newline="") as f:
        fieldnames = ["id", "scenario", "rule_id", "pass", "reason", "description", "answer"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(failed_rows)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[cases] {out_dir / 'rule_check_cases.csv'}")
    print(f"[failures] {out_dir / 'rule_check_failures.csv'}")


if __name__ == "__main__":
    main()
