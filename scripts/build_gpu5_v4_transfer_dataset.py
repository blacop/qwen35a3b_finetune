#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from clean_sft_for_retrain import clean_dataset
from intent_escalation_guardrails import apply_guardrails, parse_bool


PROJECT_ROOT = Path("/home/ubuntu/qwen35a3b_finetune")
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "datasets" / "gpu5_v4_transfer_solution"
DEFAULT_INTENT_SOURCES = [
    PROJECT_ROOT / "datasets" / "eval_sports_customer_canary_120.jsonl",
    PROJECT_ROOT / "datasets" / "eval_sports_customer_targeted_parlay_v1_120.jsonl",
    PROJECT_ROOT / "datasets" / "eval_sports_customer_prod_500.jsonl",
]
DEFAULT_DOMAIN_GLOSSARY = PROJECT_ROOT / "rag" / "kb" / "v2" / "glossary" / "core_terms.jsonl"
DEFAULT_DOMAIN_TERMS = [
    "包网",
    "JT包网",
    "体育包网",
    "白标",
    "包网商",
    "代理",
    "总代",
    "流水",
    "风控",
    "限红",
    "RTP",
    "PNL",
    "负盈利",
    "返水",
    "洗码",
    "抽水",
    "对冲",
    "三方",
]

TERM_USAGE_HINTS = {
    "包网": "讨论整套平台能力、上线模式、技术托管和运营分工时会提到这个词。",
    "JT包网": "讨论 JT 这一类整套服务方案、后台能力和交付范围时会提到这个词。",
    "体育包网": "讨论体育盘口、赛果结算、滚球能力和体育运营方案时会提到这个词。",
    "白标": "讨论复用既有系统和牌照、快速挂品牌上线时会提到这个词。",
    "包网商": "讨论谁提供底层技术、支付、风控和后台时会提到这个词。",
    "代理": "讨论拉新、会员层级、下级管理和分成关系时会提到这个词。",
    "总代": "讨论最高层级代理、平台对接人和下级代理管理时会提到这个词。",
    "流水": "讨论投注量、活动门槛、打码要求或代理结算时会提到这个词。",
    "风控": "讨论账户审核、异常投注、套利识别和风险限制时会提到这个词。",
    "限红": "讨论单注上限、单场限额和玩家可投额度时会提到这个词。",
    "RTP": "讨论返奖率、游戏长期返还水平和产品模型时会提到这个词。",
    "PNL": "讨论玩家盈亏、平台盈亏和报表口径时会提到这个词。",
    "负盈利": "讨论代理结算口径、分润条件和下级会员亏损时会提到这个词。",
    "返水": "讨论按投注量返利、返点和会员返还时会提到这个词。",
    "洗码": "讨论按有效流水返佣、返利或返点时会提到这个词。",
    "抽水": "讨论平台从盘口或玩法中收取的利润空间时会提到这个词。",
    "对冲": "讨论相反方向下注、套利和风险对锁时会提到这个词。",
    "三方": "讨论三方支付、三方游戏或第三方服务接入时会提到这个词。",
}

TERM_MISCONCEPTIONS = {
    "包网": [
        ("包网是不是只给一个前台页面", "不是。包网通常是整套方案，包含后台、支付、风控、游戏接入和运营配置，不只是前台页面。"),
        ("包网是不是等于买一个域名", "不是。域名只是接入层的一部分，包网核心是底层系统和运营能力的整体交付。"),
    ],
    "JT包网": [
        ("JT包网是不是只卖体育盘口", "不是。JT包网通常提供后台、支付、风控、代理体系和多类游戏接入，不只是一条体育线路。"),
    ],
    "体育包网": [
        ("体育包网是不是只有足球玩法", "不是。体育包网通常覆盖足球、篮球、网球等赛事，以及让球、大小、独赢、串关、滚球等玩法。"),
    ],
    "白标": [
        ("白标是不是完全自研平台", "不是。白标更偏向复用现成系统和牌照资源，再挂自己的品牌运营。"),
    ],
    "包网商": [
        ("包网商是不是玩家或普通代理", "不是。包网商是提供底层平台能力的服务方，不是玩家账号，也不是普通推广代理。"),
    ],
    "代理": [
        ("代理是不是平台技术方", "不是。代理主要负责拓客和管理下级，不负责平台底层开发。"),
        ("代理是不是等于会员", "不是。会员是终端用户，代理是渠道或层级角色。"),
    ],
    "总代": [
        ("总代是不是平台", "不一定。总代通常是最高层级代理，直接对接平台方，但总代本身不等于平台技术或系统提供方。"),
        ("总代是不是普通客服", "不是。总代的职责是管理代理层级和渠道，不是前台客服岗位。"),
    ],
    "流水": [
        ("流水是不是余额", "不是。流水是投注或交易量口径，不等于账户余额。"),
    ],
    "风控": [
        ("风控是不是故意不让玩家赢钱", "不是。风控的核心是识别异常行为和控制风险敞口，不是简单针对正常玩家。"),
    ],
    "限红": [
        ("限红是不是封号", "不是。限红是额度限制，账户仍可能正常使用，只是投注上限被收紧。"),
    ],
    "RTP": [
        ("RTP 是不是实时赔率", "不是。RTP 是理论返奖率，描述长期返还水平，不是某一时刻的赔率。"),
    ],
    "PNL": [
        ("PNL 是不是利润表", "在这里不是。包网语境里的 PNL 更常指玩家或平台盈亏金额，不是财务报表里的损益表。"),
    ],
    "负盈利": [
        ("负盈利是玩家亏损还是平台亏损", "在代理结算语境里，负盈利通常指会员亏损、平台盈利的口径，代理据此分润，不是说平台亏了钱。"),
    ],
    "返水": [
        ("返水是不是中奖派彩", "不是。返水是按流水或规则返还的一部分金额，不等于注单中奖。"),
    ],
    "洗码": [
        ("洗码是不是洗钱", "不是。洗码在包网行业通常指按有效流水返利或返点的结算口径，不是违法资金处理。"),
    ],
    "抽水": [
        ("抽水是不是额外手续费", "不完全是。抽水更接近平台在玩法或盘口中的利润空间，不一定单独作为手续费展示。"),
    ],
    "对冲": [
        ("对冲是不是正常串关", "不是。对冲是通过相反方向下注来锁定风险或套利，和正常串关不是一个概念。"),
    ],
    "三方": [
        ("三方是不是平台自己的系统", "不一定。三方通常指第三方支付、第三方游戏或第三方数据服务，不等于平台自研模块。"),
    ],
}

TERM_RELATIONS = {
    "包网": ["体育包网", "白标", "包网商"],
    "JT包网": ["包网", "包网商"],
    "体育包网": ["包网", "风控", "限红"],
    "白标": ["包网"],
    "包网商": ["包网", "代理", "总代"],
    "代理": ["总代", "包网", "负盈利"],
    "总代": ["代理", "包网商"],
    "流水": ["返水", "洗码", "负盈利"],
    "风控": ["限红", "对冲"],
    "限红": ["风控", "对冲"],
    "RTP": ["PNL"],
    "PNL": ["RTP", "负盈利"],
    "负盈利": ["PNL", "代理"],
    "返水": ["流水", "洗码"],
    "洗码": ["流水", "返水"],
    "抽水": ["RTP", "PNL"],
    "对冲": ["风控", "限红"],
    "三方": ["包网", "风控"],
}

STRICT_MARKETING_PATTERNS = [
    re.compile(p, flags=re.I)
    for p in [
        r"POP(?:聊天软件)?",
        r"Mee Yoo",
        r"APP下载链接",
        r"下载链接",
        r"添加专属客服",
        r"APP store",
        r"苹果商场",
        r"复制到浏览器",
        r"APP链接",
        r"最新网址",
        r"升级通知",
        r"重新注册",
        r"注册激活",
        r"平台站点已被收购",
        r"签到福利",
        r"红包雨",
        r"(?:https?://)?[A-Za-z0-9.-]+\.(?:com|cc|cn|shop)\b",
        r"TT2(?:\.COM)?",
        r"99N(?:\.COM)?",
        r"太阳城",
        r"银河国际",
        r"(?:首存|次存|入款|充值).{0,10}(?:送|赠送|送彩金)",
        r"回归彩金",
        r"次存豪礼",
        r"春节福利",
        r"元宵次存",
        r"贵宾专属日福利",
        r"今日上分",
        r"送彩金",
    ]
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build a GPU5 transfer dataset from the GPU7 v4 corpus, "
            "GPU5 hotfix patches, parlay regression fixes, intent-routing SFT patches, "
            "then run marketing/POP cleanup."
        )
    )
    parser.add_argument(
        "--v4-dataset",
        default=str(PROJECT_ROOT / "datasets" / "sft_merged_v4_with_fix.jsonl"),
        help="Main v4 training dataset used by the stronger GPU7 model.",
    )
    parser.add_argument(
        "--hotfix-dataset",
        default=str(
            PROJECT_ROOT
            / "datasets"
            / "nohallucination_hotfix_pack_20260425"
            / "sft_openai_messages.cleaned.v3_merged.single75_exact.repair_v3.tydata_pdfcurated.nometa.marketingfix.opsclean.20260425.nohallucination_hotfix_merged.jsonl"
        ),
        help="GPU5 hotfix dataset. Unique samples are appended on top of v4.",
    )
    parser.add_argument(
        "--parlay-seeds",
        default=str(PROJECT_ROOT / "datasets" / "v5_fix_parlay_regression_seeds.jsonl"),
        help="Regression seeds used to patch v4's over-escalation on parlay rules.",
    )
    parser.add_argument(
        "--intent-source-jsonl",
        action="append",
        default=None,
        help=(
            "Evaluation JSONL used to synthesize intent/escalation transfer patches. "
            "Can be passed multiple times."
        ),
    )
    parser.add_argument(
        "--domain-glossary-jsonl",
        default=str(DEFAULT_DOMAIN_GLOSSARY),
        help="Glossary JSONL used to synthesize domain-knowledge SFT patches.",
    )
    parser.add_argument(
        "--domain-term",
        action="append",
        default=None,
        help=(
            "Glossary term to inject as domain knowledge. "
            "Can be passed multiple times. Defaults to a curated sports-pack list."
        ),
    )
    parser.add_argument(
        "--extra-sft-jsonl",
        action="append",
        default=None,
        help=(
            "Additional cleaned SFT jsonl files to append into the final GPU5 transfer dataset. "
            "Can be passed multiple times."
        ),
    )
    parser.add_argument(
        "--raw-output-jsonl",
        default=str(DEFAULT_OUTPUT_DIR / "sft_gpu5_v4_transfer_solution.raw.jsonl"),
        help="Raw merged dataset before cleaning.",
    )
    parser.add_argument(
        "--output-jsonl",
        default=str(DEFAULT_OUTPUT_DIR / "sft_gpu5_v4_transfer_solution.cleaned.jsonl"),
        help="Final cleaned SFT dataset path.",
    )
    parser.add_argument(
        "--clean-report-dir",
        default=str(DEFAULT_OUTPUT_DIR / "clean_reports"),
        help="Directory for cleanup reports.",
    )
    parser.add_argument(
        "--summary-json",
        default=str(DEFAULT_OUTPUT_DIR / "summary.json"),
        help="Output summary path.",
    )
    return parser


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if not isinstance(obj, dict):
                raise ValueError(f"{path}:{line_no} is not a JSON object")
            rows.append(obj)
    return rows


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def validate_messages(messages: Any, source: str) -> List[Dict[str, str]]:
    if not isinstance(messages, list) or not messages:
        raise ValueError(f"{source}: messages must be a non-empty list")
    normalized: List[Dict[str, str]] = []
    for idx, item in enumerate(messages):
        if not isinstance(item, dict):
            raise ValueError(f"{source}: messages[{idx}] must be an object")
        role = str(item.get("role", "")).strip()
        content = str(item.get("content", "")).strip()
        if role not in {"system", "user", "assistant"}:
            raise ValueError(f"{source}: invalid role {role!r}")
        if not content:
            raise ValueError(f"{source}: empty content at messages[{idx}]")
        normalized.append({"role": role, "content": content})
    return normalized


def message_signature(messages: Iterable[Dict[str, str]]) -> str:
    payload: List[Tuple[str, str]] = []
    for item in messages:
        payload.append((item["role"], item["content"].strip()))
    raw = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def normalize_record(record: Dict[str, Any], source_name: str, fallback_id: str) -> Dict[str, Any]:
    messages = validate_messages(record.get("messages"), source=f"{source_name}:{fallback_id}")
    out: Dict[str, Any] = {
        "id": str(record.get("id") or fallback_id),
        "split": str(record.get("split") or "train"),
        "messages": messages,
    }
    if "dialogue_hash" in record:
        out["dialogue_hash"] = str(record["dialogue_hash"])
    return out


def append_unique(
    output: List[Dict[str, Any]],
    seen: set[str],
    rows: Iterable[Dict[str, Any]],
) -> Tuple[int, int]:
    added = 0
    skipped = 0
    for row in rows:
        sig = message_signature(row["messages"])
        if sig in seen:
            skipped += 1
            continue
        seen.add(sig)
        output.append(row)
        added += 1
    return added, skipped


def parlay_answer(seed: Dict[str, Any]) -> str:
    query = str(seed.get("user_query", "")).strip()
    if "取消" in query or "作废" in query:
        detail = "如果串关中有一场取消或作废，剩余场次会按串关规则重新计算"
    else:
        detail = "如果串关中有一场延期或改期，整单会按串关规则处理"
    return (
        "您好，这类情况属于串关规则说明。"
        f"{detail}，是否按 1.0 计算、降关、作废或保留待结算，以平台规则为准。"
        "请以投注规则页、平台公告和最终结算结果为准。"
    )


def build_parlay_patch(seed_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    patch_rows: List[Dict[str, Any]] = []
    for idx, seed in enumerate(seed_rows, 1):
        record = {
            "id": f"gpu5_v4_parlay_patch_{seed.get('source_eval_id') or idx}",
            "split": "train",
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "请直接给最终回复，不要输出json，也不要出现intent、need_escalation这些字段："
                        f"{str(seed.get('user_query', '')).strip()}"
                    ),
                },
                {"role": "assistant", "content": parlay_answer(seed)},
            ],
        }
        patch_rows.append(normalize_record(record, source_name="parlay_patch", fallback_id=str(idx)))
    return patch_rows


def intent_template(intent: str, query: str, escalation: bool) -> str:
    if intent == "充值":
        return (
            "您好，关于这笔充值问题，需要先核对到账时间、订单号和资金状态。"
            "请提供账号、订单号和充值金额，这边为您提交后台查询。"
        )
    if intent == "提款":
        return (
            "您好，关于这笔提款问题，需要先核对审核进度和出款状态。"
            "请提供账号、订单号和提款金额，这边为您提交后台查询。"
        )
    if intent == "活动":
        return (
            "您好，当前活动内容请以活动规则为准。"
            "您可以重点查看参与门槛、流水要求和活动时效，避免影响领取与审核。"
        )
    if intent == "投诉":
        return (
            "抱歉给您带来不便，您的反馈这边已经记录。"
            "该问题会按投诉流程升级处理，核实后第一时间回复您。"
        )
    if intent == "注单异常":
        return (
            "您好，该问题涉及注单状态或结算异常。"
            "请提供账号和注单号，这边为您提交后台核实后回复。"
        )
    if intent == "串关规则":
        if "取消" in query or "作废" in query:
            return (
                "您好，串关里如果有场次取消或作废，剩余场次会按串关规则重新计算，"
                "具体以平台规则为准。"
            )
        return (
            "您好，串关里如果有场次延期或改期，整单会按串关规则处理，"
            "具体是否降关、保留待结算或按 1.0 计算，以平台规则为准。"
        )
    if intent == "滚球延迟":
        return (
            "您好，滚球延迟结算一般需要核对赛果同步和派彩状态。"
            "请提供账号和注单号，这边为您提交后台核实。"
        )
    if intent == "赔率异常":
        return (
            "您好，关于赔率变动问题，需要核对下单时间、锁定赔率和系统记录。"
            "请提供账号和注单号，这边为您提交后台复核。"
        )
    if intent == "限红风控":
        return (
            "您好，当前投注限额或风控调整需要结合账户风险记录核实。"
            "请提供账号，这边为您提交后台查询后回复。"
        )
    if intent == "赛事变更":
        return (
            "您好，赛事改期、延期或取消后的处理以平台公告和结算规则为准。"
            "您可以先按平台公告核对最终结算说明。"
        )
    if intent == "账户异常":
        return "您好，账户异常需要先核对账号状态和系统记录，请提供账号信息，这边为您进一步查询。"
    if intent == "技术故障":
        return "您好，该情况可能与系统状态有关，请提供账号、操作时间和报错现象，这边为您提交技术核查。"
    if escalation:
        return "您好，该问题需要进一步核实，请提供账号和相关单号，这边为您提交后台查询。"
    return "您好，这个问题可以按平台当前规则直接说明，具体以平台规则页和最终结算结果为准。"


def build_intent_transfer_answer(sample: Dict[str, Any]) -> str:
    query = str(sample.get("user_query", "")).strip()
    scenario = str(sample.get("scenario", "")).strip()
    gold_intent = str(sample.get("gold_intent", "")).strip() or "其他"
    gold_escalation = bool(parse_bool(sample.get("gold_need_escalation")))
    must_include = sample.get("must_include") or []
    if isinstance(must_include, str):
        must_include = [x.strip() for x in must_include.split(",") if x.strip()]

    draft = intent_template(intent=gold_intent, query=query, escalation=gold_escalation)
    patched = apply_guardrails(
        user_query=query,
        scenario=scenario,
        pred_intent=gold_intent,
        pred_need_escalation=gold_escalation,
        answer=draft,
        next_action="",
        must_include=must_include,
    )
    answer = str(patched["answer"]).strip()
    if answer.startswith("{") and answer.endswith("}"):
        raise ValueError(f"Generated structured answer unexpectedly for query: {query}")
    return answer


def build_intent_transfer_rows(source_rows: List[Dict[str, Any]], source_tag: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for idx, sample in enumerate(source_rows, 1):
        query = str(sample.get("user_query", "")).strip()
        if not query:
            continue
        answer = build_intent_transfer_answer(sample)
        row = {
            "id": f"intent_transfer_{source_tag}_{sample.get('id') or idx}",
            "split": "train",
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "请直接给最终回复，不要输出json，也不要出现intent、need_escalation这些字段："
                        f"{query}"
                    ),
                },
                {"role": "assistant", "content": answer},
            ],
        }
        rows.append(normalize_record(row, source_name=f"intent_transfer_{source_tag}", fallback_id=str(idx)))
    return rows


def build_domain_term_index(glossary_rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    index: Dict[str, Dict[str, Any]] = {}
    for row in glossary_rows:
        term = str(row.get("term", "")).strip()
        if not term:
            continue
        index[term] = row
    return index


def term_aliases(record: Dict[str, Any]) -> List[str]:
    term = str(record.get("term", "")).strip()
    aliases = [str(x).strip() for x in (record.get("aliases") or []) if str(x).strip()]
    out = [term]
    for alias in aliases:
        if alias not in out:
            out.append(alias)
    return out[:4]


def shorten_definition(term: str, definition: str) -> str:
    text = re.sub(r"\s+", " ", definition).strip()
    if term == "JT包网":
        return "JT包网是面向博彩/游戏平台的整套解决方案服务，通常提供后台管理、会员体系、资金审核、风控、支付通道和体育/真人/电子游戏接入等能力。"
    if term == "包网商":
        return "包网商是提供整套博彩/游戏平台能力的服务方，负责底层技术、支付、风控和游戏接入，客户主要负责品牌运营。"
    if term == "代理":
        return "代理是在包网平台里负责拓展会员和管理下级关系的中间角色，主要做渠道与用户拓展，不负责平台底层技术。"
    if term == "RTP":
        return "RTP 是返奖率/理论返还率，指游戏长期运行后理论上返还给玩家的投注比例，不是网络协议。"
    if term == "PNL":
        return "PNL 指盈亏金额。在包网/博彩语境里通常说玩家盈亏或平台盈亏，不是财务报表里的利润表。"
    if term == "白标":
        return "白标是复用现成系统和牌照资源、再挂自己品牌运营的模式，重点在于快速上线，不等于完全自研平台。"
    if term == "总代":
        return "总代是层级最高的代理，直接对接平台方，负责招募和管理下级代理，不是普通客服或玩家账号。"
    if term == "风控":
        return "风控是平台识别和限制异常投注、异常资金流和高风险账户的机制，常见动作包括限红、限玩法、审核提款等。"
    if term == "限红":
        return "限红是平台对单个玩家在单场或单注上的最高投注金额限制，目的是控制风险敞口和防止套利。"
    if term == "负盈利":
        return "负盈利一般指代理按下级会员亏损金额参与分成的结算口径，会员亏时代理分润，会员赢时通常不分润。"
    if len(text) <= 220:
        return text
    return text[:220].rstrip("，,；;。") + "。"


def build_domain_answer(record: Dict[str, Any]) -> str:
    term = str(record.get("term", "")).strip()
    definition = shorten_definition(term, str(record.get("definition", "")).strip())
    definition = f"{term}是：{definition}"
    if term in {"包网", "JT包网", "体育包网", "包网商"}:
        tail = "简单说，就是平台把技术、支付、风控、游戏接入和后台运营能力整体打包给客户使用。"
    elif term == "白标":
        tail = "简单说，就是复用现成系统或牌照资源后再挂自己的品牌运营，强调快速上线。"
    elif term in {"代理", "总代"}:
        tail = "这类角色主要负责拓展和管理下级会员或代理，不负责平台底层技术。"
    elif term in {"流水", "返水", "洗码", "抽水"}:
        tail = "这类词都和投注流水、返利或平台抽成口径有关，不能和普通电商/支付语境混用。"
    elif term in {"RTP", "PNL", "限红", "风控", "对冲"}:
        tail = "这是体育游戏包网场景里的业务术语，判断时要按博彩运营语境理解。"
    else:
        tail = ""
    return (definition + (" " + tail if tail else "")).strip()


def build_usage_answer(record: Dict[str, Any]) -> str:
    term = str(record.get("term", "")).strip()
    return (
        f"{build_domain_answer(record)} "
        f"{TERM_USAGE_HINTS.get(term, f'在体育游戏包网业务里，{term}通常作为一个业务术语使用。')}"
    ).strip()


def relation_statement(term: str, related: str) -> str:
    pair = {term, related}
    if pair == {"限红", "风控"}:
        return "风控是更上层的风险控制机制，限红是风控落地时最常见的一种执行方式。"
    if pair == {"RTP", "PNL"}:
        return "RTP 讲的是游戏或玩法长期返还水平，PNL 讲的是某个主体在某段时间里的实际盈亏，两者不是一个维度。"
    if pair == {"包网", "白标"}:
        return "两者都是合作模式，但包网更强调整套技术和运营能力打包，白标更强调复用现成系统和牌照再挂品牌。"
    if pair == {"包网", "体育包网"}:
        return "体育包网是包网在体育业务上的细分，重点在体育赛事数据、盘口、滚球和结算引擎。"
    if pair == {"包网", "包网商"}:
        return "包网描述的是合作模式，包网商描述的是提供这套模式和能力的服务方。"
    if pair == {"代理", "总代"}:
        return "总代是更高层级的代理，负责对接平台和管理下级代理，普通代理通常层级更低、管理范围更窄。"
    if pair == {"PNL", "负盈利"}:
        return "PNL 是泛化的盈亏指标，负盈利更常见于代理结算口径，强调会员亏损后代理可参与分润。"
    if pair == {"流水", "返水"}:
        return "流水是计算基础，返水是按规则从流水里返还的一部分金额，先有流水口径，后有返水计算。"
    if pair == {"流水", "洗码"}:
        return "洗码通常也是基于有效流水计算，和返水类似都依赖流水口径，但具体规则不一定相同。"
    if pair == {"对冲", "风控"}:
        return "对冲是风控重点关注的异常投注行为之一，因为它可能带来套利和风险外溢。"
    if pair == {"对冲", "限红"}:
        return "当系统识别到对冲风险时，常见处理动作就是限红、限玩法或进一步审核。"
    if pair == {"包网商", "总代"}:
        return "包网商是技术和系统提供方，总代是渠道层级角色，职责和位置完全不同。"
    if pair == {"包网商", "代理"}:
        return "包网商负责底层平台能力，代理负责拓展用户和管理层级，两者不是同一个角色。"
    if pair == {"三方", "风控"}:
        return "三方接入会影响资金链路和结算链路，所以通常也在风控重点核查范围内。"
    if pair == {"三方", "包网"}:
        return "三方通常是包网体系中的外部接入组件，比如支付、游戏或数据服务，并不是包网本身。"
    return f"{term}和{related}在业务上有关联，但不是同一个概念，判断时要先看它们分别处于哪一层业务口径。"


def build_relation_answer(record: Dict[str, Any], related_record: Dict[str, Any]) -> str:
    term = str(record.get("term", "")).strip()
    related = str(related_record.get("term", "")).strip()
    return (
        f"{term}和{related}的关系可以这样理解：{relation_statement(term, related)} "
        f"如果要区分，先把{term}按「{shorten_definition(term, str(record.get('definition', '')).strip())}」理解，"
        f"再把{related}按「{shorten_definition(related, str(related_record.get('definition', '')).strip())}」理解。"
    ).strip()


def build_comparison_answer(record: Dict[str, Any], related_record: Dict[str, Any]) -> str:
    term = str(record.get("term", "")).strip()
    related = str(related_record.get("term", "")).strip()
    return (
        f"{term}和{related}的区别在于：{relation_statement(term, related)} "
        f"简单记：{term}偏向{shorten_definition(term, str(record.get('definition', '')).strip())}"
        f"{related}偏向{shorten_definition(related, str(related_record.get('definition', '')).strip())}"
    ).strip()


def build_misconception_rows(record: Dict[str, Any]) -> List[Tuple[str, str]]:
    term = str(record.get("term", "")).strip()
    pairs = list(TERM_MISCONCEPTIONS.get(term, []))
    if term not in {"RTP", "PNL", "总代", "负盈利", "包网", "白标"}:
        pairs.append(
            (
                f"{term}是不是普通互联网术语",
                f"不建议按普通互联网语境理解。{build_domain_answer(record)} 在体育游戏包网场景里，{term}有明确业务口径。",
            )
        )
    return pairs


def build_domain_qa_pairs(record: Dict[str, Any], index: Dict[str, Dict[str, Any]]) -> List[Tuple[str, str]]:
    term = str(record.get("term", "")).strip()
    aliases = term_aliases(record)
    primary_alias = aliases[1] if len(aliases) > 1 else term
    queries: List[Tuple[str, str]] = []

    definition_answer = build_domain_answer(record)
    usage_answer = build_usage_answer(record)

    explain_queries = [
        f"{term}是什么",
        f"{term}是什么意思",
        f"{term}是做什么的",
        f"体育游戏包网里{term}指什么",
        f"请解释一下{term}",
        f"{term}在包网业务里怎么理解",
    ]
    if primary_alias != term:
        explain_queries.extend(
            [
                f"{primary_alias}是什么意思",
                f"{primary_alias}在体育包网里是什么意思",
            ]
        )
    if term in {"RTP", "PNL"}:
        explain_queries.extend(
            [
                f"{term}缩写代表什么",
                f"{term}在报表里通常表示什么",
            ]
        )
    for query in explain_queries:
        queries.append((query, definition_answer))

    usage_queries = [
        f"{term}一般会出现在什么业务场景",
        f"客服提到{term}通常在说什么",
        f"{term}在后台或报表里一般怎么用",
        f"{term}在体育包网运营里有什么实际作用",
        f"如果用户问到{term}，客服应该怎么解释",
        f"{term}在业务讨论里通常对应哪类问题",
    ]
    if term in {"包网", "白标", "包网商", "代理", "总代"}:
        usage_queries.append(f"{term}在合作模式里通常怎么提")
    if term in {"风控", "限红", "PNL", "RTP", "负盈利"}:
        usage_queries.append(f"{term}在运营报表或风控判断里怎么理解")
    for query in usage_queries:
        queries.append((query, usage_answer))

    for query, answer in build_misconception_rows(record):
        queries.append((query, answer))

    for related in TERM_RELATIONS.get(term, []):
        related_record = index.get(related)
        if not related_record:
            continue
        relation_answer = build_relation_answer(record, related_record)
        comparison_answer = build_comparison_answer(record, related_record)
        queries.extend(
            [
                (f"{term}和{related}是什么关系", relation_answer),
                (f"{term}跟{related}有啥关系", relation_answer),
                (f"{term}和{related}有什么区别", comparison_answer),
                (f"{term}跟{related}怎么区分", comparison_answer),
            ]
        )

    if term == "总代":
        queries.append(
            (
                "总代是不是平台",
                "总代不一定是平台。总代通常是最高层级代理，负责渠道和下级代理管理；平台方则负责系统、规则和底层运营能力。",
            )
        )
    if term == "负盈利":
        queries.append(
            (
                "负盈利是玩家亏损还是平台亏损",
                "在代理结算语境里，负盈利通常是指会员亏损、平台盈利的口径，代理按这部分亏损参与分润，不是说平台出现负利润。",
            )
        )
    if term == "RTP" and "PNL" in index:
        pnl_record = index["PNL"]
        queries.extend(
            [
                ("RTP 和 PNL 有什么区别", build_comparison_answer(record, pnl_record)),
                ("RTP 跟 PNL 不是一个东西吧", build_relation_answer(record, pnl_record)),
            ]
        )
    if term == "限红" and "风控" in index:
        risk_record = index["风控"]
        queries.extend(
            [
                ("限红和风控是什么关系", build_relation_answer(record, risk_record)),
                ("限红是不是风控的一部分", build_relation_answer(record, risk_record)),
            ]
        )
    if term == "包网" and "白标" in index:
        white_label_record = index["白标"]
        queries.extend(
            [
                ("包网和白标有什么区别", build_comparison_answer(record, white_label_record)),
                ("包网跟白标怎么选", build_relation_answer(record, white_label_record)),
            ]
        )

    deduped: List[Tuple[str, str]] = []
    seen_queries = set()
    for query, answer in queries:
        q = query.strip()
        a = answer.strip()
        if not q or not a or q in seen_queries:
            continue
        seen_queries.add(q)
        deduped.append((q, a))

    if len(deduped) < 20:
        filler_queries = [
            f"{term}这个词在体育客服里常怎么回复",
            f"新客服要怎么理解{term}",
            f"{term}和普通语境下的意思一样吗",
            f"{term}出现在用户问题里通常代表什么",
            f"{term}这个概念容易误解的点是什么",
        ]
        for query in filler_queries:
            if query not in seen_queries:
                deduped.append((query, usage_answer))
                seen_queries.add(query)
            if len(deduped) >= 20:
                break
    return deduped


def build_domain_transfer_rows(glossary_rows: List[Dict[str, Any]], terms: List[str]) -> List[Dict[str, Any]]:
    index = build_domain_term_index(glossary_rows)
    rows: List[Dict[str, Any]] = []
    for term in terms:
        if term not in index:
            continue
        record = index[term]
        for idx, (query, answer) in enumerate(build_domain_qa_pairs(record, index), 1):
            row = {
                "id": f"domain_glossary_{term}_{idx}",
                "split": "train",
                "messages": [
                    {
                        "role": "user",
                        "content": f"请直接解释这个体育游戏包网术语，不要带营销口吻：{query}",
                    },
                    {"role": "assistant", "content": answer},
                ],
            }
            rows.append(normalize_record(row, source_name=f"domain_glossary_{term}", fallback_id=str(idx)))
    return rows


def merge_sources(
    v4_rows: List[Dict[str, Any]],
    hotfix_rows: List[Dict[str, Any]],
    parlay_rows: List[Dict[str, Any]],
    intent_rows: List[Dict[str, Any]],
    domain_rows: List[Dict[str, Any]],
    extra_rows: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    merged_rows: List[Dict[str, Any]] = []
    seen: set[str] = set()

    v4_added, v4_skipped = append_unique(merged_rows, seen, v4_rows)
    hotfix_added, hotfix_skipped = append_unique(merged_rows, seen, hotfix_rows)
    parlay_added, parlay_skipped = append_unique(merged_rows, seen, parlay_rows)
    intent_added, intent_skipped = append_unique(merged_rows, seen, intent_rows)
    domain_added, domain_skipped = append_unique(merged_rows, seen, domain_rows)
    extra_added, extra_skipped = append_unique(merged_rows, seen, extra_rows)

    counts = {
        "v4_input": len(v4_rows),
        "hotfix_input": len(hotfix_rows),
        "parlay_seed_input": len(parlay_rows),
        "intent_transfer_input": len(intent_rows),
        "domain_glossary_input": len(domain_rows),
        "extra_sft_input": len(extra_rows),
        "v4_added": v4_added,
        "v4_skipped_duplicates": v4_skipped,
        "hotfix_added": hotfix_added,
        "hotfix_skipped_duplicates": hotfix_skipped,
        "parlay_added": parlay_added,
        "parlay_skipped_duplicates": parlay_skipped,
        "intent_transfer_added": intent_added,
        "intent_transfer_skipped_duplicates": intent_skipped,
        "domain_glossary_added": domain_added,
        "domain_glossary_skipped_duplicates": domain_skipped,
        "extra_sft_added": extra_added,
        "extra_sft_skipped_duplicates": extra_skipped,
        "raw_output_total": len(merged_rows),
    }
    return merged_rows, counts


def post_clean_strict_filter(cleaned_path: Path) -> Dict[str, int]:
    rows = read_jsonl(cleaned_path)
    kept: List[Dict[str, Any]] = []
    removed = 0
    for row in rows:
        assistant_text = "\n".join(
            str(m.get("content", ""))
            for m in row.get("messages", [])
            if isinstance(m, dict) and m.get("role") == "assistant"
        )
        if any(p.search(assistant_text) for p in STRICT_MARKETING_PATTERNS):
            removed += 1
            continue
        kept.append(row)
    write_jsonl(cleaned_path, kept)
    return {
        "post_clean_strict_removed": removed,
        "post_clean_final_rows": len(kept),
    }


def compute_term_coverage(dataset_path: Path, terms: List[str]) -> Dict[str, int]:
    counts = {term: 0 for term in terms}
    rows = read_jsonl(dataset_path)
    for row in rows:
        text = "\n".join(str(m.get("content", "")) for m in row.get("messages", []) if isinstance(m, dict))
        for term in terms:
            if term in text:
                counts[term] += 1
    return counts


def main() -> None:
    args = build_parser().parse_args()

    v4_path = Path(args.v4_dataset)
    hotfix_path = Path(args.hotfix_dataset)
    parlay_path = Path(args.parlay_seeds)
    raw_output_path = Path(args.raw_output_jsonl)
    output_path = Path(args.output_jsonl)
    report_dir = Path(args.clean_report_dir)
    summary_path = Path(args.summary_json)
    intent_source_paths = [Path(p) for p in (args.intent_source_jsonl or [str(p) for p in DEFAULT_INTENT_SOURCES])]
    domain_glossary_path = Path(args.domain_glossary_jsonl)
    domain_terms = args.domain_term or list(DEFAULT_DOMAIN_TERMS)
    extra_sft_paths = [Path(p) for p in (args.extra_sft_jsonl or [])]

    raw_output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    v4_rows = [
        normalize_record(row, source_name="v4", fallback_id=f"v4_{idx}")
        for idx, row in enumerate(read_jsonl(v4_path), 1)
    ]
    hotfix_rows = [
        normalize_record(row, source_name="hotfix", fallback_id=f"hotfix_{idx}")
        for idx, row in enumerate(read_jsonl(hotfix_path), 1)
    ]
    parlay_seed_rows = read_jsonl(parlay_path)
    parlay_rows = build_parlay_patch(parlay_seed_rows)
    glossary_rows = read_jsonl(domain_glossary_path)
    domain_rows = build_domain_transfer_rows(glossary_rows, terms=domain_terms)
    extra_rows: List[Dict[str, Any]] = []
    for source_path in extra_sft_paths:
        extra_rows.extend(
            normalize_record(row, source_name=f"extra_sft:{source_path.name}", fallback_id=f"{source_path.stem}_{idx}")
            for idx, row in enumerate(read_jsonl(source_path), 1)
        )

    intent_rows: List[Dict[str, Any]] = []
    for source_path in intent_source_paths:
        source_tag = source_path.stem.replace(".", "_")
        source_rows = read_jsonl(source_path)
        intent_rows.extend(build_intent_transfer_rows(source_rows, source_tag=source_tag))

    merged_rows, counts = merge_sources(
        v4_rows=v4_rows,
        hotfix_rows=hotfix_rows,
        parlay_rows=parlay_rows,
        intent_rows=intent_rows,
        domain_rows=domain_rows,
        extra_rows=extra_rows,
    )
    write_jsonl(raw_output_path, merged_rows)

    coverage_before = compute_term_coverage(raw_output_path, domain_terms)
    clean_summary = clean_dataset(raw_output_path, output_path, report_dir)
    strict_filter_summary = post_clean_strict_filter(output_path)
    coverage_after = compute_term_coverage(output_path, domain_terms)

    summary = {
        "v4_dataset": str(v4_path),
        "hotfix_dataset": str(hotfix_path),
        "parlay_seeds": str(parlay_path),
        "intent_source_jsonl": [str(p) for p in intent_source_paths],
        "domain_glossary_jsonl": str(domain_glossary_path),
        "domain_terms": domain_terms,
        "extra_sft_jsonl": [str(p) for p in extra_sft_paths],
        "raw_output_dataset": str(raw_output_path),
        "output_dataset": str(output_path),
        "clean_report_dir": str(report_dir),
        "counts": counts,
        "domain_term_coverage_raw": coverage_before,
        "domain_term_coverage_cleaned": coverage_after,
        "clean_summary": clean_summary,
        "strict_filter_summary": strict_filter_summary,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
