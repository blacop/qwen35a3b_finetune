#!/usr/bin/env python3
import csv
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import yaml


def read_jsonl(path: Path) -> List[dict]:
    rows = []
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


def weighted_pick(items: List[dict], weight_key: str) -> dict:
    weights = [max(1, int(x.get(weight_key, 1))) for x in items]
    return random.choices(items, weights=weights, k=1)[0]


def parse_bucket(bucket: str) -> Tuple[int, int]:
    a, b = bucket.split("-")
    return int(a), int(b)


def weighted_pick_from_map(prob_map: Dict[str, float]) -> str:
    keys = list(prob_map.keys())
    probs = list(prob_map.values())
    return random.choices(keys, weights=probs, k=1)[0]


def sample_split(split_cfg: Dict[str, float]) -> str:
    return weighted_pick_from_map(split_cfg)


def random_dt_str() -> str:
    now = datetime.now(timezone.utc)
    delta = timedelta(minutes=random.randint(-720, 720))
    t = now + delta
    return t.strftime("%Y-%m-%d %H:%M:%S UTC")


def gen_from_regex(pattern: str) -> str:
    # Minimal deterministic regex renderer for known patterns in config.
    if pattern == "^[a-zA-Z0-9_]{6,14}$":
        ln = random.randint(6, 14)
        alphabet = "abcdefghijklmnopqrstuvwxyz0123456789_"
        return "".join(random.choice(alphabet) for _ in range(ln))
    if pattern == "^BET[0-9]{8}$":
        return "BET" + "".join(str(random.randint(0, 9)) for _ in range(8))
    if pattern == "^ORD[0-9]{10}$":
        return "ORD" + "".join(str(random.randint(0, 9)) for _ in range(10))
    if pattern == "^TK[0-9]{9}$":
        return "TK" + "".join(str(random.randint(0, 9)) for _ in range(9))
    return "X" + "".join(str(random.randint(0, 9)) for _ in range(8))


def gen_slot_value(slot: str, slot_cfg: dict) -> str:
    c = slot_cfg.get(slot, {})
    t = c.get("type", "enum")
    if t == "enum":
        vals = c.get("values", [slot])
        return str(random.choice(vals))
    if t == "int":
        return str(random.randint(int(c.get("min", 1)), int(c.get("max", 9999))))
    if t == "float":
        mn = float(c.get("min", 1.0))
        mx = float(c.get("max", 10.0))
        p = int(c.get("precision", 2))
        return str(round(random.uniform(mn, mx), p))
    if t == "datetime":
        return random_dt_str()
    if t == "regex":
        return gen_from_regex(str(c.get("pattern", "")))
    return f"<{slot.strip('<>')}_VAL>"


def fill_slots(template: dict, slot_cfg: dict) -> Dict[str, str]:
    required = template.get("required_slots", [])
    optional = template.get("optional_slots", [])
    chosen_optional = [s for s in optional if random.random() < 0.35]
    slots = {}
    for slot in list(dict.fromkeys(required + chosen_optional)):
        slots[slot] = gen_slot_value(slot, slot_cfg)
    return slots


def tone_text(base: str, tone: str) -> str:
    if tone == "empathetic":
        return "抱歉让您久等了，" + base
    if tone == "concise_actionable":
        return "已处理要点： " + base
    if tone == "firm_compliance":
        return "按平台规则说明： " + base
    if tone == "high_pressure_deescalation":
        return "我理解您着急，先给您明确处理路径： " + base
    return base


def get_sla(scenario: str, cfg: dict) -> str:
    by_scenario = cfg.get("by_scenario", {})
    if scenario in by_scenario:
        return random.choice(by_scenario[scenario])
    d = cfg.get("default", {})
    return random.choice([d.get("ack", "3-5分钟回执"), d.get("l1", "15分钟内复核")])


def get_escalation(cfg: dict) -> str:
    return weighted_pick_from_map(cfg.get("distribution", {"L1": 1.0}))


def render_dialog(template: dict, slots: dict, tone: str, turns: int, sla: str, escalation: str) -> List[dict]:
    scenario = template["scenario"]
    title = template["title"]
    account = slots.get("<ACCOUNT>", "<ACCOUNT>")
    bet_id = slots.get("<BET_ID>", "<BET_ID>")
    status = slots.get("<STATUS>", "待处理")
    reason = slots.get("<VOID_REASON>") or slots.get("<DELAY_REASON>") or slots.get("<RISK_REASON>", "规则核验中")
    ticket = slots.get("<TICKET_ID>", "TK000000001")
    next_action = slots.get("<NEXT_ACTION>", "提交工单")

    user_open = {
        "bet_abnormal": f"{title}，我的注单{bet_id}现在什么情况？",
        "parlay_settlement": f"{title}，串关注单{bet_id}还没结算。",
        "live_delay": f"{title}，滚球单{bet_id}为什么还在{status}？",
        "odds_anomaly": f"{title}，我下注时赔率和现在不一样。",
        "risk_limit": f"{title}，账号被限了，怎么处理？",
    }.get(scenario, f"{title}，请帮我处理。")

    a1 = tone_text("我先帮您核实信息，请提供账号与注单号。", tone)
    u2 = f"账号{account}，注单{bet_id}。"
    a2 = tone_text(
        f"已查询到当前状态为“{status}”。本单与“{reason}”相关，需按标准流程复核。",
        tone,
    )
    u3 = "那接下来怎么处理，多久有结果？"
    a3 = tone_text(
        f"我已为您{next_action}，工单号{ticket}，当前升级等级{escalation}。预计{sla}反馈，请您留意消息。",
        tone,
    )
    a4 = tone_text("若超时未更新，您直接回复本窗口，我会继续跟进到闭环。", tone)

    dialog = [
        {"role": "user", "content": user_open},
        {"role": "assistant", "content": a1},
        {"role": "user", "content": u2},
        {"role": "assistant", "content": a2},
        {"role": "user", "content": u3},
        {"role": "assistant", "content": a3},
    ]
    if turns >= 7:
        dialog += [{"role": "assistant", "content": a4}]
    if turns >= 8:
        dialog = dialog[:6] + [{"role": "user", "content": "收到，麻烦加急。"}, {"role": "assistant", "content": a4}]
    if turns >= 9:
        dialog += [{"role": "user", "content": "好的，我等通知。"}, {"role": "assistant", "content": "感谢配合，我会持续跟进。"}]
    if turns >= 10:
        dialog += [{"role": "assistant", "content": "本次处理路径已记录，后续同单可直接引用工单号。"}]

    return dialog


def normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


def dialog_hash(messages: List[dict]) -> str:
    packed = "|".join(f"{m.get('role','')}:{normalize_text(m.get('content',''))}" for m in messages)
    return hashlib.sha256(packed.encode("utf-8")).hexdigest()


def is_low_value_only(messages: List[dict], phrases: List[str]) -> bool:
    assistant_text = " ".join(m["content"] for m in messages if m.get("role") == "assistant").strip()
    return assistant_text in phrases


def quality_ok(messages: List[dict], qcfg: dict) -> bool:
    if qcfg.get("require_qa_pair", True):
        has_u = any(m.get("role") == "user" for m in messages)
        has_a = any(m.get("role") == "assistant" for m in messages)
        if not (has_u and has_a):
            return False
    min_chars = int(qcfg.get("min_chars_per_assistant_turn", 4))
    max_chars = int(qcfg.get("max_chars_per_assistant_turn", 320))
    for m in messages:
        if m.get("role") == "assistant":
            ln = len(m.get("content", ""))
            if ln < min_chars:
                return False
            if ln > max_chars:
                m["content"] = m["content"][:max_chars]
    lv = qcfg.get("drop_low_value_reply_only", {})
    if lv.get("enabled"):
        if is_low_value_only(messages, lv.get("phrases", [])):
            return False
    return True


def safety_tags(messages: List[dict], patterns: List[str]) -> List[str]:
    text = " ".join(m.get("content", "") for m in messages)
    tags = []
    if any(p in text for p in patterns):
        tags.append("policy_hit")
    return tags


def ensure_template_file(cfg: dict):
    tpath = Path(cfg["input"]["templates_jsonl"])
    if tpath.exists():
        return
    script = Path("/home/ubuntu/qwen35a3b_finetune/scripts/bootstrap_templates_100.py")
    raise SystemExit(
        f"templates missing: {tpath}. run: python3 {script}"
    )


def load_templates(cfg: dict) -> List[dict]:
    ensure_template_file(cfg)
    rows = read_jsonl(Path(cfg["input"]["templates_jsonl"]))
    req = set(cfg["input"].get("required_template_fields", []))
    if cfg["input"].get("strict_template_fields", True):
        for r in rows:
            miss = req - set(r.keys())
            if miss:
                raise ValueError(f"template missing fields {miss}: {r.get('template_id')}")
    return rows


def generate(cfg: dict) -> Tuple[List[dict], List[dict], List[dict], dict]:
    random.seed(int(cfg.get("seed", 42)))
    templates = load_templates(cfg)
    scenario_quota = cfg["generation"]["scenario_quota"]
    tone_dist = cfg["variation_rules"]["tone_distribution"]
    turn_dist = cfg["variation_rules"]["turn_bucket_distribution"]
    split_cfg = cfg["split"]
    slot_cfg = cfg["slot_filling"]["generators"]
    sla_cfg = cfg["sla_policy"]
    escalation_cfg = cfg["escalation_policy"]
    safety_patterns = cfg["safety"]["hard_block_patterns"]
    quality_cfg = cfg["quality"]

    by_scenario = defaultdict(list)
    for t in templates:
        by_scenario[t["scenario"]].append(t)

    sft_rows = []
    cls_rows = []
    faq_rows = []
    seen_hash = set()
    seq = 1

    for scenario, quota in scenario_quota.items():
        items = by_scenario.get(scenario, [])
        if not items:
            raise ValueError(f"no templates for scenario={scenario}")

        count = 0
        while count < int(quota):
            t = weighted_pick(items, cfg["generation"]["template_sampling"]["weight_field"])
            slots = fill_slots(t, slot_cfg)
            tone = weighted_pick_from_map(tone_dist)
            turn_bucket = weighted_pick_from_map(turn_dist)
            lo, hi = parse_bucket(turn_bucket)
            turns = random.randint(lo, hi)
            sla = get_sla(scenario, sla_cfg)
            escalation = get_escalation(escalation_cfg)
            messages = render_dialog(t, slots, tone, turns, sla, escalation)
            if not quality_ok(messages, quality_cfg):
                continue
            h = dialog_hash(messages)
            if h in seen_hash:
                continue
            seen_hash.add(h)
            tags = safety_tags(messages, safety_patterns)
            sample_id = f"GAP_{scenario.upper()}_{seq:06d}"
            seq += 1
            split = sample_split(split_cfg)

            sft_row = {
                "id": sample_id,
                "template_id": t["template_id"],
                "scenario": scenario,
                "intent": t["intent"],
                "sub_intent": t["sub_intent"],
                "tone": tone,
                "turns": turns,
                "sla_commit": sla,
                "escalation_level": escalation,
                "compliance_tags": tags,
                "quality_score": 1.0,
                "hash": h,
                "split": split,
                "messages": messages,
            }
            sft_rows.append(sft_row)
            cls_rows.append(
                {
                    "id": sample_id,
                    "text": messages[0]["content"],
                    "intent": t["intent"],
                    "sub_intent": t["sub_intent"],
                    "scenario": scenario,
                    "split": split,
                }
            )
            faq_rows.append(
                {
                    "id": sample_id,
                    "question": messages[0]["content"],
                    "answer": messages[-1]["content"],
                    "intent": t["intent"],
                    "sub_intent": t["sub_intent"],
                    "scenario": scenario,
                }
            )
            count += 1

    stats = {
        "total_generated": len(sft_rows),
        "by_scenario": dict(Counter(x["scenario"] for x in sft_rows)),
        "by_intent": dict(Counter(x["intent"] for x in sft_rows)),
        "by_split": dict(Counter(x["split"] for x in sft_rows)),
        "by_tone": dict(Counter(x["tone"] for x in sft_rows)),
    }
    return sft_rows, cls_rows, faq_rows, stats


def write_stats(root: Path, cfg: dict, stats: dict) -> None:
    out_csv = root / cfg["outputs"]["stats_csv"]
    out_md = root / cfg["outputs"]["summary_md"]

    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["group", "key", "value"])
        w.writerow(["total", "generated", stats["total_generated"]])
        for g in ["by_scenario", "by_intent", "by_split", "by_tone"]:
            for k, v in sorted(stats[g].items()):
                w.writerow([g, k, v])

    lines = [
        "# Generated Gap Pack Summary",
        "",
        f"- total_generated: {stats['total_generated']}",
        "",
        "## by_scenario",
    ]
    for k, v in sorted(stats["by_scenario"].items()):
        lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append("## by_intent")
    for k, v in sorted(stats["by_intent"].items()):
        lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append("## by_split")
    for k, v in sorted(stats["by_split"].items()):
        lines.append(f"- {k}: {v}")
    out_md.write_text("\n".join(lines), encoding="utf-8")


def merge_into_sft(cfg: dict, generated_rows: List[dict]) -> Dict[str, int]:
    target = Path(cfg["outputs"]["merged_sft_jsonl"])
    backup_prefix = Path(cfg["outputs"]["merged_backup_prefix"])
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_path = Path(f"{backup_prefix}.{ts}.jsonl")

    existing = read_jsonl(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        target.replace(backup_path)
        existing = read_jsonl(backup_path)

    existing_hashes = set()
    for row in existing:
        msgs = row.get("messages")
        if isinstance(msgs, list):
            existing_hashes.add(dialog_hash(msgs))

    merged = list(existing)
    added = 0
    for row in generated_rows:
        h = row.get("hash") or dialog_hash(row["messages"])
        if h in existing_hashes:
            continue
        existing_hashes.add(h)
        merged.append(row)
        added += 1

    write_jsonl(target, merged)
    return {
        "backup_path": str(backup_path),
        "existing_before": len(existing),
        "generated": len(generated_rows),
        "added": added,
        "merged_total": len(merged),
    }


def main():
    cfg_path = Path("/home/ubuntu/qwen35a3b_finetune/generator_config.yaml")
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    out_root = Path(cfg["outputs"]["root_dir"])
    out_root.mkdir(parents=True, exist_ok=True)

    sft_rows, cls_rows, faq_rows, stats = generate(cfg)
    write_jsonl(out_root / cfg["outputs"]["sft_jsonl"], sft_rows)
    write_jsonl(out_root / cfg["outputs"]["cls_jsonl"], cls_rows)
    write_jsonl(out_root / cfg["outputs"]["rag_faq_jsonl"], faq_rows)
    write_stats(out_root, cfg, stats)

    merge_info = merge_into_sft(cfg, sft_rows)
    print(json.dumps({"stats": stats, "merge": merge_info}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
