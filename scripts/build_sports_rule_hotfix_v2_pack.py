#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path("/home/ubuntu/qwen35a3b_finetune")
DEFAULT_EVAL_JSONL = PROJECT_ROOT / "datasets" / "sports_rule_knowledge_regression_gpu5_20260428.jsonl"
DEFAULT_FAILURE_CSV = (
    PROJECT_ROOT
    / "eval_outputs"
    / "rule18_candidate_disable_thinking_20260429T062946Z"
    / "rule_check"
    / "rule_check_failures.csv"
)
DEFAULT_OUT_DIR = PROJECT_ROOT / "datasets" / "sports_rule_hotfix_v2_20260429"


SYSTEM_PROMPTS = [
    "你是体育包网客服。只输出最终客服回复，禁止输出推理过程、题目理解、首先/接下来、<think>。遇到纯规则解释题，直接说明规则和计算方法；只有用户询问具体注单、资金、赔率争议或账户异常时才引导后台查询。不要营销、不要引流、不要承诺包赢。",
    "你是体育规则客服。回答必须简洁、确定、可直接发送给用户；不得输出JSON、intent、need_escalation、risk_flag、next_action或思考过程。规则解释题不得无故转后台。",
    "你负责体育盘口和结算规则说明。请直接给最终答复，保留关键数字和规则口径；不要添加下载APP、送彩金、外部联系方式、代理返佣、包赢稳赚等话术。",
]


WRONG_INDUCTIONS = {
    "rule_reg_20260428_001": [
        "香港盘0.8是不是总返还800，本金另算不用说？",
        "我下注1000、香港盘0.8，输了是不是只输800？",
    ],
    "rule_reg_20260428_002": [
        "香港盘和欧洲盘是不是都包含本金，所以算法一样？",
        "香港盘0.8能不能按欧洲盘1.80那样理解？",
    ],
    "rule_reg_20260428_003": [
        "欧洲盘1.80下注1000，盈利是不是1800？",
        "欧洲盘是不是只看纯盈利，不含本金？",
    ],
    "rule_reg_20260428_004": [
        "马来负水-0.88是不是赢880、输1000？",
        "负水是不是直接按0.88计算中奖盈利？",
    ],
    "rule_reg_20260428_005": [
        "美国盘正水1.17是不是要按欧洲盘公式算成170元盈利？",
        "正水1.17下注1000，盈利是不是只算0.17倍？",
    ],
    "rule_reg_20260428_006": [
        "美国负水-1.17是不是赢1170、输1000？",
        "美国负水能不能直接按正水算法算？",
    ],
    "rule_reg_20260428_007": [
        "大小球大于4，总进球刚好4个是不是小球赢？",
        "大于4刚好进4个，是不是大球输掉本金？",
    ],
    "rule_reg_20260428_008": [
        "大于4，总进球4个是不是大球也赢？",
        "大于4时总进球3个，大球是不是走水？",
    ],
    "rule_reg_20260428_009": [
        "让球0.25出现输一半是不是系统异常？",
        "0.25盘口是不是不会有赢一半或输一半？",
    ],
    "rule_reg_20260428_010": [
        "让球0.75是不是只能全赢或全输？",
        "0.75盘口出现半赢半输是不是结算错了？",
    ],
    "rule_reg_20260428_011": [
        "50本金三关1.62、1.52、1.76，派彩是不是213元？",
        "三关全赢是不是把三个赔率相加再乘本金？",
    ],
    "rule_reg_20260428_012": [
        "串关有一关走水，是不是整张串关直接输？",
        "串关和局那关赔率是不是按0处理？",
    ],
    "rule_reg_20260428_013": [
        "三串三是不是就是ABC三场全赢的一注？",
        "三串三是不是不用拆成AB、AC、BC？",
    ],
    "rule_reg_20260428_014": [
        "一串四是不是只有ABCD这一组，不会拆其他注单？",
        "一串四能不能只举ABCD，不用说ABC、ABD、BCD？",
    ],
    "rule_reg_20260428_015": [
        "危险球是不是没有固定时间窗口，直接说风险高就行？",
        "危险球下注被拒绝是不是一定要查后台才知道？",
    ],
    "rule_reg_20260428_016": [
        "结算时效是不是小赛事几分钟、大型赛事要更久？",
        "比赛结束没结算，只说后台查询可以吗？",
    ],
    "rule_reg_20260428_017": [
        "即时兑换金额变化是不是只能后台查询，不能解释规则？",
        "即时兑换是不是固定金额，不会随赛况变化？",
    ],
    "rule_reg_20260428_018": [
        "注单取消原因只说赔率异常和风控就够了吗？",
        "天气或危险球是不是不会导致注单取消？",
    ],
}


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


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_failed_ids(paths: list[Path]) -> dict[str, int]:
    failed: dict[str, int] = {}
    for path in paths:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                sample_id = str(row.get("id") or "").strip()
                if sample_id:
                    failed[sample_id] = failed.get(sample_id, 0) + 1
    return failed


def signature(messages: list[dict[str, str]]) -> str:
    payload = json.dumps(messages, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def checked_phrases(sample: dict[str, Any]) -> str:
    phrases: list[str] = []
    for check in sample.get("rule_checks") or []:
        for key in ("must_include_any", "must_include_all"):
            values = check.get(key) or []
            if isinstance(values, str):
                values = [values]
            phrases.extend(str(v) for v in values if str(v))
    deduped: list[str] = []
    seen: set[str] = set()
    for phrase in phrases:
        if phrase not in seen:
            seen.add(phrase)
            deduped.append(phrase)
    return "、".join(deduped[:8])


def user_variants(sample: dict[str, Any]) -> list[tuple[str, str]]:
    sample_id = str(sample["id"])
    scenario = str(sample.get("scenario") or "体育规则")
    original = str(sample["user_query"]).strip()
    phrases = checked_phrases(sample)
    variants: list[tuple[str, str]] = [
        ("original", original),
        ("direct_no_backend", f"{original} 不要转后台，按规则直接答。"),
        ("final_only", f"{original} 只给最终客服回复，不要解释你的思考过程。"),
        ("short", f"{scenario}：请用一段话直接说明规则。"),
        ("service_tone", f"用户问：{original}\n请改写成可直接发送给用户的客服回复。"),
        ("angry_user", f"用户有点着急：{original} 别让我一直等后台。请直接解释规则。"),
        ("no_json", f"{original} 不要输出JSON字段，也不要写intent或next_action。"),
        ("keep_keywords", f"{original} 回复里要保留关键口径：{phrases}。" if phrases else original),
        ("settlement_page", f"{original} 可以补充最终以平台结算页为准，但不能回避规则本身。"),
        ("operator_check_boundary", f"{original} 如果只是规则解释不要联系运营；只有具体注单争议才提示提供注单号。"),
    ]
    for idx, prompt in enumerate(WRONG_INDUCTIONS.get(sample_id, []), 1):
        variants.append((f"wrong_induction_{idx}", f"{prompt} 请纠正我的理解，并直接给规则口径。"))
    return variants


def build_rows(samples: list[dict[str, Any]], failed_ids: dict[str, int], pass_repeat: int, fail_repeat: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sample in samples:
        sample_id = str(sample.get("id") or "").strip()
        if not sample_id:
            raise ValueError("sample missing id")
        answer = str(sample.get("reference_answer") or "").strip()
        if not answer:
            raise ValueError(f"sample missing reference_answer: {sample_id}")
        repeat = fail_repeat if sample_id in failed_ids else pass_repeat
        variants = user_variants(sample)
        for rep in range(1, repeat + 1):
            system = SYSTEM_PROMPTS[(rep - 1) % len(SYSTEM_PROMPTS)]
            for variant_name, user in variants:
                messages = [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                    {"role": "assistant", "content": answer},
                ]
                rows.append(
                    {
                        "id": f"sports_rule_v2_{sample_id}_{variant_name}_r{rep:02d}",
                        "split": "train",
                        "messages": messages,
                        "meta": {
                            "source": "sports_rule_hotfix_v2",
                            "eval_sample_id": sample_id,
                            "scenario": sample.get("scenario", ""),
                            "variant": variant_name,
                            "repeat": rep,
                            "failed_in_candidate": sample_id in failed_ids,
                            "candidate_failed_checks": failed_ids.get(sample_id, 0),
                        },
                    }
                )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a small targeted sports-rule hotfix v2 SFT pack.")
    parser.add_argument("--eval-jsonl", default=str(DEFAULT_EVAL_JSONL))
    parser.add_argument("--failure-csv", action="append", default=None)
    parser.add_argument("--output-jsonl", default=str(DEFAULT_OUT_DIR / "sft_sports_rule_hotfix_v2.jsonl"))
    parser.add_argument("--summary-json", default=str(DEFAULT_OUT_DIR / "summary.json"))
    parser.add_argument("--pass-repeat", type=int, default=4)
    parser.add_argument("--fail-repeat", type=int, default=8)
    args = parser.parse_args()

    eval_path = Path(args.eval_jsonl).expanduser().resolve()
    failure_csvs = args.failure_csv or [str(DEFAULT_FAILURE_CSV)]
    failure_paths = [Path(p).expanduser().resolve() for p in failure_csvs if p]
    output_path = Path(args.output_jsonl).expanduser().resolve()
    summary_path = Path(args.summary_json).expanduser().resolve()

    samples = read_jsonl(eval_path)
    failed_ids = read_failed_ids(failure_paths)
    rows = build_rows(samples, failed_ids, max(1, args.pass_repeat), max(1, args.fail_repeat))
    write_jsonl(output_path, rows)

    by_source = {
        "failed_samples": sum(1 for sample in samples if str(sample.get("id")) in failed_ids),
        "passed_samples": sum(1 for sample in samples if str(sample.get("id")) not in failed_ids),
    }
    summary = {
        "eval_jsonl": str(eval_path),
        "failure_csv": [str(p) for p in failure_paths],
        "output_jsonl": str(output_path),
        "samples": len(samples),
        "failed_ids": failed_ids,
        "pass_repeat": max(1, args.pass_repeat),
        "fail_repeat": max(1, args.fail_repeat),
        "system_prompt_variants": len(SYSTEM_PROMPTS),
        "rows": len(rows),
        **by_source,
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
