#!/usr/bin/env python3
import argparse
import csv
import hashlib
import json
import random
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List


def read_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


def dialog_hash(messages: List[dict]) -> str:
    packed = "|".join(f"{m.get('role', '')}:{normalize_text(m.get('content', ''))}" for m in messages)
    return hashlib.sha256(packed.encode("utf-8")).hexdigest()


def split_pick() -> str:
    p = random.random()
    if p < 0.94:
        return "train"
    if p < 0.97:
        return "val"
    return "test"


def infer_reply_bucket(text: str) -> str:
    if re.search(r"订单号|时间|金额|截图|支付凭证|交易尾号|提示|报错", text):
        return "collect_info"
    if re.search(r"刷新|重试|重新登录|更换网络|更换浏览器|清理缓存|重启|页面", text):
        return "troubleshoot"
    if re.search(r"恢复正常|状态|继续帮您处理|继续帮您查看|核实|跟进", text):
        return "status_followup"
    if re.search(r"欢迎来到在线客服中心|请问您当前遇到的具体情况|在线客服", text):
        return "opening"
    return "generic"


def load_reply_pool(path: Path) -> Dict[str, List[str]]:
    lines = [x.strip() for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    seen = set()
    uniq: List[str] = []
    for x in lines:
        nx = normalize_text(x)
        if nx in seen:
            continue
        seen.add(nx)
        uniq.append(x)

    pool: Dict[str, List[str]] = {
        "opening": [],
        "collect_info": [],
        "troubleshoot": [],
        "status_followup": [],
        "generic": [],
        "all": uniq,
    }
    for x in uniq:
        b = infer_reply_bucket(x)
        pool[b].append(x)

    for k in ["opening", "collect_info", "troubleshoot", "status_followup", "generic"]:
        if not pool[k]:
            pool[k] = uniq
    return pool


INTENT_CONFIG = {
    "充值/提现": {
        "scenario": "payment_support",
        "sub_intent": "充值未到账/提现处理中",
        "openers": [
            "我这边充值后还没到账，订单号<ORDER_ID>，麻烦查一下。",
            "提现申请提交后一直处理中，请帮我看下进度。",
            "支付成功了但余额没变化，能帮我核实吗？",
            "这笔充值显示成功但账户没到款，麻烦处理。",
        ],
        "detail_user": "账号<ACCOUNT>，时间<DATE_TIME>，金额<AMOUNT>，可提供截图。",
        "third_user": [
            "大概多久能有结果？",
            "我还需要补充什么资料吗？",
            "请问当前状态是什么？",
        ],
        "bucket_a2": "collect_info",
    },
    "技术故障": {
        "scenario": "tech_issue",
        "sub_intent": "页面异常/APP异常",
        "openers": [
            "APP一直加载不出来，页面转圈很久。",
            "游戏页面空白，进不去，能帮忙排查吗？",
            "网站打不开，提示异常，麻烦看下。",
            "按钮无法点击，操作不了，请协助处理。",
        ],
        "detail_user": "设备为<DEVICE>，网络<NETWORK>，已出现同样问题多次。",
        "third_user": [
            "我已经重试一次了，下一步怎么操作？",
            "如果还是不行，我应该怎么反馈？",
            "现在看起来还是异常，麻烦继续处理。",
        ],
        "bucket_a2": "troubleshoot",
    },
    "账户异常/风控": {
        "scenario": "account_review",
        "sub_intent": "审核中/限制提示",
        "openers": [
            "账号提示需要审核，暂时操作不了，麻烦帮我查下。",
            "系统提示账户状态异常，请协助核实原因。",
            "页面显示限制中，请问需要补什么资料？",
            "我这边被风控拦截了，怎么恢复？",
        ],
        "detail_user": "账号<ACCOUNT>，页面提示“审核中/限制中”，时间<DATE_TIME>。",
        "third_user": [
            "这个大概多久处理完？",
            "处理完成后会怎么通知我？",
            "是否需要我再提交一次申请？",
        ],
        "bucket_a2": "status_followup",
    },
    "活动优惠": {
        "scenario": "campaign_rules",
        "sub_intent": "资格核验/规则说明",
        "openers": [
            "我想确认一下活动资格，当前是否满足条件？",
            "活动页面规则看不太懂，麻烦帮我解释一下。",
            "我提交了活动申请，想确认审核状态。",
            "活动奖励还没显示，麻烦帮我核实。",
        ],
        "detail_user": "账号<ACCOUNT>，活动页显示状态“待核验”，时间<DATE_TIME>。",
        "third_user": [
            "如果不符合条件，能告诉我具体原因吗？",
            "后续状态变化会在哪里通知？",
            "请给我一个明确的处理路径。",
        ],
        "bucket_a2": "collect_info",
    },
    "投诉/催办": {
        "scenario": "complaint_followup",
        "sub_intent": "进度催办/服务投诉",
        "openers": [
            "这个问题反馈很久了还没结果，请帮我加急处理。",
            "我多次联系都在等待，麻烦给我明确进度。",
            "处理时间超过预期了，我要投诉并催办。",
            "请不要只让我等待，给我具体处理节点。",
        ],
        "detail_user": "账号<ACCOUNT>，问题类型<ISSUE_TYPE>，已等待<WAIT_MIN>分钟。",
        "third_user": [
            "请直接告诉我下一次反馈时间。",
            "如果超时未处理，升级路径是什么？",
            "我需要一个可追踪的处理编号。",
        ],
        "bucket_a2": "status_followup",
    },
}


def slot_value(name: str) -> str:
    if name == "<ORDER_ID>":
        return "ORD" + "".join(random.choice("0123456789") for _ in range(10))
    if name == "<ACCOUNT>":
        return "ACC" + "".join(random.choice("0123456789") for _ in range(6))
    if name == "<DATE_TIME>":
        return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    if name == "<AMOUNT>":
        return str(random.choice([100, 200, 300, 500, 800, 1000]))
    if name == "<DEVICE>":
        return random.choice(["Android", "iOS", "Web"])
    if name == "<NETWORK>":
        return random.choice(["4G/5G", "WiFi", "宽带"])
    if name == "<ISSUE_TYPE>":
        return random.choice(["充值未到账", "页面异常", "审核超时", "订单状态异常"])
    if name == "<WAIT_MIN>":
        return str(random.choice([20, 30, 45, 60, 90, 120]))
    return name


def fill_slots(text: str) -> str:
    for slot in ["<ORDER_ID>", "<ACCOUNT>", "<DATE_TIME>", "<AMOUNT>", "<DEVICE>", "<NETWORK>", "<ISSUE_TYPE>", "<WAIT_MIN>"]:
        if slot in text:
            text = text.replace(slot, slot_value(slot))
    return text


def pick(pool: Dict[str, List[str]], bucket: str) -> str:
    items = pool.get(bucket) or pool["all"]
    return random.choice(items)


def build_dialog(intent: str, pool: Dict[str, List[str]]) -> List[dict]:
    cfg = INTENT_CONFIG[intent]
    u1 = fill_slots(random.choice(cfg["openers"]))
    a1 = pick(pool, "opening")
    u2 = fill_slots(cfg["detail_user"])
    a2 = pick(pool, cfg["bucket_a2"])
    u3 = random.choice(cfg["third_user"])
    a3 = pick(pool, "status_followup")

    messages: List[dict] = [
        {"role": "user", "content": u1},
        {"role": "assistant", "content": a1},
        {"role": "user", "content": u2},
        {"role": "assistant", "content": a2},
        {"role": "user", "content": u3},
        {"role": "assistant", "content": a3},
    ]

    if random.random() < 0.35:
        messages += [
            {"role": "user", "content": "好的，我先按你说的操作。"},
            {"role": "assistant", "content": pick(pool, "generic")},
        ]
    return messages


def build_rows(per_intent: Dict[str, int], pool: Dict[str, List[str]], start_idx: int = 1) -> Dict[str, List[dict]]:
    sft_rows: List[dict] = []
    cls_rows: List[dict] = []
    faq_rows: List[dict] = []
    seen = set()
    seq = start_idx

    for intent, quota in per_intent.items():
        cfg = INTENT_CONFIG[intent]
        made = 0
        attempts = 0
        while made < quota and attempts < quota * 50:
            attempts += 1
            messages = build_dialog(intent, pool)
            h = dialog_hash(messages)
            if h in seen:
                continue
            seen.add(h)
            sid = f"KEFUCG_MULTI_{seq:06d}"
            seq += 1
            split = split_pick()
            turns = len(messages)
            row = {
                "id": sid,
                "scenario": cfg["scenario"],
                "intent": intent,
                "sub_intent": cfg["sub_intent"],
                "source": "kefucg_strict_reply_pool",
                "split": split,
                "turns": turns,
                "hash": h,
                "messages": messages,
            }
            sft_rows.append(row)
            cls_rows.append(
                {
                    "id": sid,
                    "text": messages[0]["content"],
                    "intent": intent,
                    "sub_intent": cfg["sub_intent"],
                    "scenario": cfg["scenario"],
                    "split": split,
                }
            )
            faq_rows.append(
                {
                    "id": sid,
                    "question": messages[0]["content"],
                    "answer": messages[-1]["content"],
                    "intent": intent,
                    "sub_intent": cfg["sub_intent"],
                    "scenario": cfg["scenario"],
                }
            )
            made += 1
    return {"sft": sft_rows, "cls": cls_rows, "faq": faq_rows}


def merge_sft(main_path: Path, new_rows: List[dict]) -> Dict[str, int]:
    old_rows = read_jsonl(main_path)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup = main_path.with_name(f"{main_path.stem}.backup.{ts}{main_path.suffix}")
    if main_path.exists():
        main_path.replace(backup)
        old_rows = read_jsonl(backup)

    old_hash = set()
    for r in old_rows:
        msgs = r.get("messages")
        if isinstance(msgs, list):
            old_hash.add(dialog_hash(msgs))

    merged = list(old_rows)
    added = 0
    for r in new_rows:
        h = r.get("hash") or dialog_hash(r["messages"])
        if h in old_hash:
            continue
        old_hash.add(h)
        merged.append(r)
        added += 1

    write_jsonl(main_path, merged)
    return {
        "backup_path": str(backup),
        "existing_before": len(old_rows),
        "generated": len(new_rows),
        "added": added,
        "merged_total": len(merged),
    }


def write_stats(out_dir: Path, sft_rows: List[dict], merge_info: Dict[str, int]) -> None:
    by_intent = Counter(x["intent"] for x in sft_rows)
    by_split = Counter(x["split"] for x in sft_rows)
    by_scenario = Counter(x["scenario"] for x in sft_rows)
    by_turns = Counter(str(x["turns"]) for x in sft_rows)

    csv_path = out_dir / "stats_kefucg_multiturn.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["group", "key", "value"])
        w.writerow(["total", "generated", len(sft_rows)])
        for gname, data in [
            ("by_intent", by_intent),
            ("by_scenario", by_scenario),
            ("by_split", by_split),
            ("by_turns", by_turns),
        ]:
            for k, v in sorted(data.items()):
                w.writerow([gname, k, v])
        for k, v in merge_info.items():
            w.writerow(["merge", k, v])

    summary = [
        "# KEFUCG Strict Pool Multiturn Summary",
        "",
        f"- generated: {len(sft_rows)}",
        f"- merged_total: {merge_info['merged_total']}",
        f"- added: {merge_info['added']}",
        "",
        "## by_intent",
    ]
    for k, v in sorted(by_intent.items()):
        summary.append(f"- {k}: {v}")
    summary.append("")
    summary.append("## by_split")
    for k, v in sorted(by_split.items()):
        summary.append(f"- {k}: {v}")
    summary.append("")
    summary.append("## merge")
    for k in ["backup_path", "existing_before", "generated", "added", "merged_total"]:
        summary.append(f"- {k}: {merge_info[k]}")
    (out_dir / "summary_kefucg_multiturn.md").write_text("\n".join(summary), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reply-pool",
        default="/home/ubuntu/qwen35a3b_finetune/datasets/kefucg_review_20260408/kefucg_strict_sft_candidates.txt",
    )
    parser.add_argument("--output-dir", default="/home/ubuntu/qwen35a3b_finetune/datasets/generated_kefucg_multiturn")
    parser.add_argument("--merged-sft", default="/home/ubuntu/qwen35a3b_finetune/datasets/sft_openai_messages.jsonl")
    parser.add_argument("--seed", type=int, default=20260408)
    parser.add_argument("--size", type=int, default=2000)
    args = parser.parse_args()

    random.seed(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pool = load_reply_pool(Path(args.reply_pool))

    intents = list(INTENT_CONFIG.keys())
    ratio = {
        "充值/提现": 0.26,
        "技术故障": 0.30,
        "账户异常/风控": 0.20,
        "活动优惠": 0.12,
        "投诉/催办": 0.12,
    }
    per_intent = {k: int(args.size * ratio[k]) for k in intents}
    diff = args.size - sum(per_intent.values())
    for i in range(diff):
        per_intent[intents[i % len(intents)]] += 1

    rows = build_rows(per_intent, pool)
    sft_rows = rows["sft"]
    cls_rows = rows["cls"]
    faq_rows = rows["faq"]

    write_jsonl(out_dir / "sft_openai_messages_kefucg_multiturn.jsonl", sft_rows)
    write_jsonl(out_dir / "intent_cls_kefucg_multiturn.jsonl", cls_rows)
    write_jsonl(out_dir / "faq_qa_kefucg_multiturn.jsonl", faq_rows)

    merge_info = merge_sft(Path(args.merged_sft), sft_rows)
    write_stats(out_dir, sft_rows, merge_info)

    print(
        json.dumps(
            {
                "generated": len(sft_rows),
                "per_intent": per_intent,
                "merge": merge_info,
                "outputs": {
                    "sft": str(out_dir / "sft_openai_messages_kefucg_multiturn.jsonl"),
                    "cls": str(out_dir / "intent_cls_kefucg_multiturn.jsonl"),
                    "faq": str(out_dir / "faq_qa_kefucg_multiturn.jsonl"),
                    "summary": str(out_dir / "summary_kefucg_multiturn.md"),
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
