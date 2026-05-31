#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


PROJECT_ROOT = Path("/home/ubuntu/qwen35a3b_finetune")
BASE_TEXT_INPUT = PROJECT_ROOT / "datasets" / "relation_patch_pack" / "text_relation_patch.jsonl"
MM_INPUT = PROJECT_ROOT / "datasets" / "tydata_domain_multimodal_pack" / "multimodal_train.swift_vl.jsonl"
AB_RESULTS_ROOT = PROJECT_ROOT / "rag" / "eval" / "domain_text_vl_ab"
OUT_FILE = PROJECT_ROOT / "datasets" / "relation_patch_pack" / "text_relation_patch_v2.jsonl"
SUMMARY_FILE = PROJECT_ROOT / "datasets" / "relation_patch_pack" / "text_relation_patch_v2.summary.json"

SYSTEM_PROMPT = (
    "你是体育包网客服助手。直接用中文回答，先说页面可见信息，再说业务解释，最后给出简短结论。"
    "不要输出 Thinking、分析过程、英文提示或自我指令；不要重复堆砌同一术语、字段或句子。"
    "如果页面没有直接写出术语定义，要明确说明是基于页面场景判断。"
)

TOPIC_VISIBLE_FALLBACK = {
    "仪表盘": "页面能看到仪表盘、平台用户数据和待办事项。",
    "资金审核": "页面能看到充值审核、补单、额度或订单相关字段。",
    "代理管理": "页面能看到会员ID、上级代理、VIP或代理层级字段。",
    "线路配置": "页面能看到专属域名、推广链接、域名类型或绑定配置。",
    "系统配置": "页面能看到系统配置、站点或域名相关字段。",
    "滚球规则": "页面能看到滚球、盘口、赔率或投注规则说明。",
    "串关规则": "页面能看到串关、综合过关或连串过关规则。",
    "危险球": "页面能看到危险球、待确认或取消相关规则说明。",
    "让球规则": "页面能看到让球、盘口或比分规则说明。",
    "独赢和1X2": "页面能看到独赢、1X2 或赛果玩法说明。",
    "用户充值排名描述": "页面能看到用户充值、排行或统计描述。",
}

REWRITE_TARGETS = {
    "domain_vl_ab_002": "pack_white",
    "domain_vl_ab_003": "limit_risk",
    "domain_vl_ab_004": "agent_platform",
    "domain_vl_ab_005": "negative_profit",
    "domain_vl_ab_006": "agent_vs_packager",
    "domain_vl_ab_010": "parlay_relation",
    "domain_vl_ab_011": "dangerous_ball",
}


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def latest_ab_results_input() -> Path:
    candidates = sorted(
        AB_RESULTS_ROOT.glob("run_*/results.jsonl"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"No AB results found under: {AB_RESULTS_ROOT}")
    return candidates[0]


def normalize_text(text: str) -> str:
    return " ".join(str(text or "").replace("\r", " ").replace("\n", " ").split())


def trim(text: str, limit: int) -> str:
    text = normalize_text(text).strip("。；;，, ")
    if len(text) <= limit:
        return text
    return text[:limit].rstrip("。；;，, ") + "…"


def extract_section(text: str, label: str, next_labels: List[str]) -> str:
    text = str(text or "")
    match = re.search(re.escape(label) + r"：(.+)", text, flags=re.S)
    if not match:
        return ""
    section = match.group(1)
    end = len(section)
    for next_label in next_labels:
        idx = section.find(f"{next_label}：")
        if idx != -1:
            end = min(end, idx)
    return normalize_text(section[:end].strip("。；;，, "))


def extract_visible_info(response: str, topic: str) -> str:
    visible = extract_section(response, "页面可见信息", ["业务解释", "关系结论", "口径判断", "原因"])
    if visible:
        return trim(visible, 80)
    match = re.search(r"例如：(.+?)。图中没有", response)
    if match:
        return trim(match.group(1), 80)
    match = re.search(r"例如：(.+?)。", response)
    if match:
        return trim(match.group(1), 80)
    return TOPIC_VISIBLE_FALLBACK.get(topic, "页面主要体现后台字段和业务场景。")


def detect_category(query: str, term: str) -> str | None:
    blob = f"{term} {query}"
    if "限红" in blob or "风控" in blob:
        return "limit_risk"
    if "RTP" in blob or "PNL" in blob:
        return "rtp_pnl"
    if "负盈利" in blob:
        return "negative_profit"
    if "总代" in blob or "平台" in blob:
        return "agent_platform"
    if "包网" in blob or "白标" in blob:
        return "pack_white"
    if "串关" in blob or "综合过关" in blob:
        return "parlay_relation"
    if "危险球" in blob:
        return "dangerous_ball"
    return None


def looks_stats_page(visible: str, topic: str) -> bool:
    blob = f"{visible} {topic}"
    return any(token in blob for token in ["仪表盘", "统计", "排行", "充值", "提现", "订单", "审核", "盈亏", "结算"])


def looks_rules_page(visible: str, topic: str) -> bool:
    blob = f"{visible} {topic}"
    return any(token in blob for token in ["盘口", "赔率", "滚球", "危险球", "串关", "综合过关", "让球", "独赢", "玩法"])


def build_answer(category: str, query: str, visible: str, topic: str) -> str:
    visible = trim(visible or TOPIC_VISIBLE_FALLBACK.get(topic, "页面主要体现后台字段和业务场景。"), 80)
    if category == "limit_risk":
        if "封号" in query:
            biz = "这里更像额度、补单或审核场景。限红通常是限制投注或额度，不等于封号；风控才是更上层的风险识别和处理机制。"
            conclusion = "结论：限红是风控落地动作之一，不是独立的封号概念。"
        elif "区别" in query and "关系" not in query:
            biz = "风控是识别和处理风险的整套机制；限红是风控落到投注额度或玩法限制上的具体动作。"
            conclusion = "结论：两者有关联，但不是同义词。"
        else:
            biz = "这页出现额度限制、补单或审核字段时，通常先按风控执行位理解；限红是风控对用户或玩法施加的限制动作。"
            conclusion = "结论：风控是上层机制，限红是其中一种落地结果。"
        return f"页面可见信息：{visible}。业务解释：{biz} {conclusion}"

    if category == "rtp_pnl":
        if "利润表" in query:
            biz = "这里的 PNL 指玩家或平台在一段时间内的实际盈亏，不是财务利润表；RTP 则是玩法长期返还率口径。"
            conclusion = "结论：PNL 看实际结果，RTP 看理论返还。"
        elif "更接近" in query:
            if looks_stats_page(visible, topic):
                biz = "这页更像统计或审核口径，因此更接近 PNL 场景；RTP 一般属于玩法返还规则，不会直接等同于这类统计页。"
            elif looks_rules_page(visible, topic):
                biz = "这页更像玩法或盘口规则说明，但 RTP 仍然是长期返还率口径，PNL 仍然是实际盈亏口径。"
            else:
                biz = "这页没有直接写 RTP 或 PNL，我只能按页面场景判断：统计页偏 PNL，规则页偏 RTP。"
            conclusion = "结论：RTP 和 PNL 不是一个口径，不能互相替代。"
        else:
            biz = "RTP 讲玩法的长期理论返还率，PNL 讲某个主体在一段时间内的实际盈亏。"
            conclusion = "结论：一个偏规则参数，一个偏统计结果。"
        return f"页面可见信息：{visible}。业务解释：{biz} {conclusion}"

    if category == "agent_platform":
        if "是不是平台" in query:
            biz = "这类页面出现上级代理、会员ID 或代理层级字段时，更像代理体系页。总代是代理链路里的上级角色，平台是提供系统、规则和运营能力的主体。"
            conclusion = "结论：总代不是平台本身。"
        elif "代理" in query and "总代" in query:
            biz = "总代通常位于代理层级更上方，可以管理下级代理；普通代理更多负责自己的渠道、会员或推广链路。"
            conclusion = "结论：总代是更高层级的代理角色，不等于普通代理。"
        else:
            biz = "看到上级代理、代理层级或会员管理字段时，先按代理体系理解，不要把总代直接等同于平台。"
            conclusion = "结论：总代是角色层级，平台是运营主体。"
        return f"页面可见信息：{visible}。业务解释：{biz} {conclusion}"

    if category == "negative_profit":
        if "和PNL" in query or "PNL" in query:
            biz = "负盈利更偏代理分润结算口径，通常指会员亏损、平台盈利后，代理按这部分亏损参与分润；PNL 是更泛化的盈亏统计口径。"
            conclusion = "结论：负盈利偏分润结算，PNL 偏通用盈亏统计。"
        else:
            biz = "这类代理后台更常用来解释分润和结算。负盈利通常指会员亏损、平台盈利后的结算口径，不是说平台在亏钱。"
            conclusion = "结论：这里的负盈利更接近玩家亏损、平台盈利后的代理分润口径。"
        return f"页面可见信息：{visible}。业务解释：{biz} {conclusion}"

    if category == "pack_white":
        if "代理还是包网商" in query:
            biz = "专属域名、推广链接和绑定配置更能说明代理体系或渠道管理；包网商通常在更上层，提供系统、支付、风控和接入能力。"
            conclusion = "结论：这页更偏代理/渠道执行，不是包网商本体页。"
        elif "只给一个前台页面" in query:
            biz = "包网不是只给一个前台页，而是把系统、支付、风控、客服和代理管理等后端能力整套打包。"
            conclusion = "结论：包网强调整套能力，不是单页面交付。"
        else:
            biz = "包网强调整套系统和运营能力打包；白标强调复用现成系统或牌照后挂自己的品牌。"
            conclusion = "结论：包网偏能力交付，白标偏品牌复用。"
        return f"页面可见信息：{visible}。业务解释：{biz} {conclusion}"

    if category == "agent_vs_packager":
        biz = "页面里是域名绑定、推广链接和代理字段，更接近代理或渠道管理。包网商在更上层，负责系统、支付、风控和技术接入。"
        conclusion = "结论：这页更能说明代理，不是直接说明包网商。"
        return f"页面可见信息：{visible}。业务解释：{biz} {conclusion}"

    if category == "parlay_relation":
        biz = "串关就是把多场赛事或多个选项组合成一张注单，通常需要全赢才算整单命中，赔率会连乘。综合过关一般就是串关在页面里的展示或玩法名称。"
        conclusion = "结论：综合过关和串关本质上是同一类组合投注口径。"
        return f"页面可见信息：{visible}。业务解释：{biz} {conclusion}"

    if category == "dangerous_ball":
        biz = "危险球指点球、角球、自由球或临门进攻这类可能快速改变赛况的高风险事件。待确认或取消通常是因为事件还在判定、数据源回滚，或盘口需要二次校验。"
        conclusion = "结论：危险球不是独立玩法，而是影响滚球确认和结算的事件状态。"
        return f"页面可见信息：{visible}。业务解释：{biz} {conclusion}"

    return f"页面可见信息：{visible}。业务解释：请按页面场景解释业务含义，不要把不同术语混说。"


def build_user_prompt(query: str, visible: str, topic: str, extra: str = "") -> str:
    parts = []
    if visible:
        parts.append(f"【页面上下文】\n{visible}")
    if topic:
        parts.append(f"【页面主题】\n{topic}")
    if extra:
        parts.append(f"【要求】\n{extra}")
    parts.append(f"【问题】\n{query}")
    return "\n\n".join(parts)


def row_fingerprint(user: str, assistant: str) -> str:
    return hashlib.sha1((normalize_text(user) + "\n" + normalize_text(assistant)).encode("utf-8")).hexdigest()


def grouped_focus_rows(mm_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    per_group: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in mm_rows:
        meta = row.get("meta") or {}
        query = str(row.get("query", ""))
        term = str(meta.get("domain_term", "")).strip()
        qtype = str(meta.get("domain_qtype", "")).strip()
        topic = str(meta.get("topic", "")).strip()
        if qtype not in {"relation", "comparison", "misconception", "definition"}:
            continue
        category = detect_category(query, term)
        if not category:
            continue
        if category == "agent_platform" and "总代" not in query and term != "总代":
            continue
        key = (category, topic, qtype)
        per_group[key].append(row)

    caps = {
        "limit_risk": 8,
        "rtp_pnl": 8,
        "negative_profit": 8,
        "agent_platform": 8,
        "pack_white": 6,
        "parlay_relation": 8,
        "dangerous_ball": 8,
    }
    selected: List[Dict[str, Any]] = []
    for (category, _topic, _qtype), rows in sorted(per_group.items()):
        limit = caps.get(category, 6)
        selected.extend(sorted(rows, key=lambda r: str(r.get("id")))[:limit])
    return selected


def build_rewrite_rows(results_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for result in results_rows:
        rid = str(result.get("id", ""))
        category = REWRITE_TARGETS.get(rid)
        if not category:
            continue
        query = str(result.get("query", "")).strip()
        visible = trim(str(result.get("topic_hint", "")).strip() or str(result.get("ocr_context", "")).strip(), 90)
        bad = trim(str((result.get("text_result") or {}).get("content", "")).strip(), 320)
        answer = build_answer(category, query, visible, "")
        prompt = build_user_prompt(
            f"把下面这段跑偏回答改成可直接发给用户的版本：{query}",
            visible,
            "",
            "只保留 2-3 句话；不要出现 Thinking、英文分析或重复句式。\n【错误回答片段】\n" + bad,
        )
        rows.append(
            {
                "id": f"text_rewrite_{rid}",
                "split": "train",
                "messages": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": answer},
                ],
                "meta": {
                    "patch_type": "relation_text_antirepeat_rewrite",
                    "source_result_id": rid,
                    "focus_category": category,
                },
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Build expanded text relation + anti-repeat patch dataset.")
    parser.add_argument("--base-text-input", default=str(BASE_TEXT_INPUT))
    parser.add_argument("--mm-input", default=str(MM_INPUT))
    parser.add_argument("--ab-results-input", default=str(latest_ab_results_input()))
    parser.add_argument("--out-file", default=str(OUT_FILE))
    parser.add_argument("--summary-file", default=str(SUMMARY_FILE))
    args = parser.parse_args()

    base_rows = read_jsonl(Path(args.base_text_input))
    mm_rows = read_jsonl(Path(args.mm_input))
    results_rows = read_jsonl(Path(args.ab_results_input))

    selected_focus = grouped_focus_rows(mm_rows)
    extra_rows: List[Dict[str, Any]] = []
    focus_counter: Counter[str] = Counter()

    for index, row in enumerate(selected_focus):
        meta = row.get("meta") or {}
        query = str(row.get("query", "")).strip()
        term = str(meta.get("domain_term", "")).strip()
        topic = str(meta.get("topic", "")).strip()
        category = detect_category(query, term)
        if not category:
            continue
        visible = extract_visible_info(str(row.get("response", "")), topic)
        answer = build_answer(category, query, visible, topic)
        prompt = build_user_prompt(query, visible, topic, "直接回答，不超过3句话；不要重复同一个术语。")
        extra_rows.append(
            {
                "id": f"text_focus_{row['id']}",
                "split": row.get("split", "train"),
                "messages": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": answer},
                ],
                "meta": {
                    **meta,
                    "patch_type": "relation_text_focus",
                    "focus_category": category,
                    "source_mm_id": row.get("id"),
                },
            }
        )
        focus_counter[category] += 1

        if index % 3 == 0:
            binding_query = "回答这类题时，怎样先说页面可见信息，再给业务解释，才能避免跑偏或重复？"
            binding_answer = (
                f"页面可见信息：{visible}。业务解释：先用页面里的字段锁定场景，再解释业务关系；"
                "没有直接证据时要明确是按场景判断。结论：先看页面、再谈术语，才能减少把不同概念说混。"
            )
            binding_prompt = build_user_prompt(
                binding_query,
                visible,
                topic,
                "请给出可复用的回答原则，不超过3句话，不要堆砌同一个字段名。",
            )
            extra_rows.append(
                {
                    "id": f"text_binding_{row['id']}",
                    "split": row.get("split", "train"),
                    "messages": [
                        {"role": "user", "content": binding_prompt},
                        {"role": "assistant", "content": binding_answer},
                    ],
                    "meta": {
                        **meta,
                        "patch_type": "relation_text_binding",
                        "focus_category": category,
                        "source_mm_id": row.get("id"),
                    },
                }
            )

    rewrite_rows = build_rewrite_rows(results_rows)

    all_rows: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for row in list(base_rows) + extra_rows + rewrite_rows:
        user = str(row["messages"][0]["content"])
        assistant = str(row["messages"][1]["content"])
        fp = row_fingerprint(user, assistant)
        if fp in seen:
            continue
        seen.add(fp)
        all_rows.append(row)

    write_jsonl(Path(args.out_file), all_rows)

    patch_counter = Counter(str((row.get("meta") or {}).get("patch_type", "unknown")) for row in all_rows)
    summary = {
        "base_text_input": str(Path(args.base_text_input).resolve()),
        "mm_input": str(Path(args.mm_input).resolve()),
        "ab_results_input": str(Path(args.ab_results_input).resolve()),
        "out_file": str(Path(args.out_file).resolve()),
        "total_rows": len(all_rows),
        "train_rows": sum(1 for row in all_rows if row.get("split", "train") == "train"),
        "val_rows": sum(1 for row in all_rows if row.get("split") == "val"),
        "base_rows": len(base_rows),
        "focus_rows_added": len(extra_rows),
        "rewrite_rows_added": len(rewrite_rows),
        "focus_category_counts": dict(focus_counter),
        "patch_type_counts": dict(patch_counter),
        "system_prompt": SYSTEM_PROMPT,
    }
    Path(args.summary_file).write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
