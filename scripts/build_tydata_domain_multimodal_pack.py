#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple
from xml.etree import ElementTree as ET

from clean_sft_for_retrain import clean_dataset


PROJECT_ROOT = Path("/home/ubuntu/qwen35a3b_finetune")
DEFAULT_INPUT_DIR = Path("/tmp/tydata")
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "datasets" / "tydata_domain_multimodal_pack"
DEFAULT_DOMAIN_GLOSSARY = PROJECT_ROOT / "rag" / "kb" / "v2" / "glossary" / "core_terms.jsonl"
SYSTEM_PROMPT = (
    "你是体育包网客服与后台助手。请基于图片和上下文给出准确、简洁、合规的中文说明，"
    "不要编造不存在的按钮、字段、路径、规则或结算结果。不要输出思维链，不要重复堆砌同一字段、术语或句子；"
    "如果图里没有直接证据，要明确说是基于页面场景判断。"
)

DOCX_NS = {
    "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
    "pr": "http://schemas.openxmlformats.org/package/2006/relationships",
}
PPT_NS = {
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    "pr": "http://schemas.openxmlformats.org/package/2006/relationships",
}
DOCX_RID_KEY = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}embed"
PPT_RID_KEY = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}embed"

NOISE_TOKENS = {
    "",
    "|",
    "/",
    "……",
    "………………",
    "Contents",
    "How to play",
    "game rules",
    "physical education",
    "01",
    "02",
    "03",
    "04",
    "05",
}

MARKETING_PATTERNS = [
    re.compile(p, flags=re.I)
    for p in [
        r"APP下载",
        r"下载地址",
        r"下载页",
        r"APP更新",
        r"立即注册",
        r"优惠活动",
        r"VIP特权",
        r"弹窗广告",
        r"幸运大转盘",
        r"豪礼",
        r"中奖率",
        r"送彩金",
        r"首存",
        r"次存",
        r"落地页",
        r"开户链接",
        r"添加POP",
        r"联系POP",
        r"客服POP",
        r"最稳定",
        r"人气最高",
        r"备受玩家推崇",
        r"玩法多样化",
        r"各自的优势",
        r"开放下注盘口较多",
        r"登录验证类型",
        r"感谢您选择",
        r"共赢未来",
        r"重新定义博弈包网行业",
        r"多元化的娱乐服务",
        r"演示后台",
        r"演示前台",
        r"携手同行",
        r"后台密码",
        r"前台账号",
        r"前台密码",
        r"TG商务",
        r"广告图",
        r"热门推荐",
        r"注册页弹窗",
    ]
]

STRICT_TEXT_FILTER_PATTERNS = [
    re.compile(p, flags=re.I)
    for p in [
        r"POP(?:聊天软件)?",
        r"下载链接",
        r"添加专属客服",
        r"复制到浏览器",
        r"最新地址",
        r"平台注册",
        r"立即注册",
        r"(?:https?://)?[A-Za-z0-9.-]+\.(?:com|cc|cn|shop)\b",
    ]
]

OCR_NOISE_RE = re.compile(
    "|".join(
        [
            r"PLEASE ENTER YOUR TEXT HERE",
            r"PLEASE ENTER YOUR TEX",
            r"CONTENTS?",
            r"^\W+$",
        ]
    ),
    re.IGNORECASE,
)

DOMAIN_COMPARISONS = {
    "包网": "白标",
    "JT包网": "包网商",
    "体育包网": "包网",
    "白标": "包网",
    "包网商": "包网",
    "代理": "总代",
    "总代": "代理",
    "风控": "限红",
    "限红": "风控",
    "RTP": "PNL",
    "PNL": "RTP",
    "负盈利": "PNL",
    "返水": "洗码",
    "洗码": "返水",
    "抽水": "RTP",
    "对冲": "风控",
    "三方": "包网",
}

TERM_MISCONCEPTION_QUESTIONS = {
    "包网": "这张图对应的场景里，包网是不是只给一个前台页面？",
    "JT包网": "从这类后台图看，JT包网是不是只卖体育盘口？",
    "体育包网": "从这张体育玩法图看，体育包网是不是只有足球玩法？",
    "白标": "结合这类后台图，白标是不是完全自研平台？",
    "包网商": "从这类后台能力图看，包网商是不是普通代理？",
    "代理": "从这张代理相关页面看，代理是不是平台技术方？",
    "总代": "从这张代理相关页面看，总代是不是平台？",
    "风控": "结合这类页面，风控是不是故意不让玩家赢钱？",
    "限红": "结合这类体育盘口或额度页面，限红是不是封号？",
    "RTP": "结合这类体育玩法图，RTP 是不是实时赔率？",
    "PNL": "结合这类后台统计图，PNL 是不是财务利润表？",
    "负盈利": "结合这类代理后台图，负盈利是玩家亏损还是平台亏损？",
    "返水": "从这类后台记录图看，返水是不是中奖派彩？",
    "洗码": "从这类记录图看，洗码是不是洗钱？",
    "抽水": "从这类玩法图看，抽水是不是额外手续费？",
    "对冲": "结合这类体育规则图，对冲是不是正常串关？",
    "三方": "从这类补单页面看，三方是不是平台自研系统？",
}

PLATFORM_SHORT_DEFINITION = "平台方/系统方负责底层系统、规则、结算、支付和风控能力，不等于代理层级。"

RELATION_QUERY_VARIANTS = {
    ("风控", "限红"): [
        "限红和风控是什么关系？结合这张后台图说明。",
        "这张页面里能看到额度或审核信息时，应该把它理解成风控还是限红？",
        "请先说这张图里能看到什么，再说明风控和限红在业务上怎么对应。",
        "从这张图看，限红是独立功能，还是风控落地动作的一部分？为什么？",
    ],
    ("RTP", "PNL"): [
        "RTP 和 PNL 有什么区别？结合这张图说明。",
        "这张图更接近 RTP 场景还是 PNL 场景？请先说页面可见信息，再说业务解释。",
        "从这张页面能看出 RTP 和 PNL 分别属于什么口径吗？",
        "请结合图里可见信息说明：RTP 为什么不等于 PNL？",
    ],
    ("总代", "平台"): [
        "总代和平台是什么关系？结合这张代理相关页面说明。",
        "这张页面里的总代是不是平台方？请先说页面可见信息，再说业务解释。",
        "从这张代理页看，总代和平台分别扮演什么角色？",
        "图里出现代理、下级或域名信息时，为什么不能把总代直接理解成平台？",
    ],
}

RELATION_OFFTOPIC_KEYWORDS = [
    "短信提供商",
    "提醒①",
    "新版发布",
    "统计安装",
    "导航页配置",
    "充值任务",
    "充值分组",
    "域名解析",
    "添加CDN",
    "添加缓存",
    "站点管理",
    "域名盾",
    "证书",
    "跑量域名",
    "ITDOG",
    "上传APK",
    "安装完成",
    "我的网站",
    "回源地址",
    "DNS",
    "CNAME",
    "缓存配置",
]


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def run_cmd(cmd: List[str]) -> str:
    completed = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=True,
    )
    return completed.stdout


def normalize_text(text: str) -> str:
    text = str(text or "").replace("\r", "\n")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def read_glossary_index(path: Path) -> Dict[str, Dict[str, Any]]:
    rows: Dict[str, Dict[str, Any]] = {}
    if not path.exists():
        return rows
    for row in read_jsonl(path):
        term = normalize_text(row.get("term", ""))
        if term:
            rows[term] = row
    return rows


def has_cjk(text: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", text))


def looks_marketing(text: str) -> bool:
    content = normalize_text(text)
    if not content:
        return True
    return any(p.search(content) for p in MARKETING_PATTERNS)


def clean_lines(lines: Sequence[str], min_len: int = 6) -> List[str]:
    cleaned: List[str] = []
    for raw in lines:
        text = normalize_text(raw).strip(" ，,;；")
        if not text or len(text) < min_len:
            continue
        if text in NOISE_TOKENS:
            continue
        if re.fullmatch(r"[0-9A-Za-z]{1,3}", text):
            continue
        if re.fullmatch(r"[^\w\u4e00-\u9fff]+", text):
            continue
        if not has_cjk(text):
            continue
        if looks_marketing(text):
            continue
        if OCR_NOISE_RE.search(text):
            continue
        cleaned.append(text)

    deduped: List[str] = []
    seen = set()
    for text in cleaned:
        if text in seen:
            continue
        seen.add(text)
        deduped.append(text)
    return deduped


def should_keep_context(lines: Sequence[str], min_lines: int = 2, min_cjk: int = 12) -> bool:
    if len(lines) < min_lines:
        return False
    joined = " ".join(lines)
    if looks_marketing(joined):
        return False
    cjk_count = len(re.findall(r"[\u4e00-\u9fff]", joined))
    return cjk_count >= min_cjk


def valid_image(path: Path) -> bool:
    if not path.exists() or not path.is_file():
        return False
    if path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
        return False
    return path.stat().st_size >= 8_000


def pick_split(key: str, val_ratio: float) -> str:
    digest = hashlib.md5(key.encode("utf-8")).hexdigest()
    value = int(digest[:8], 16) / 0xFFFFFFFF
    return "val" if value < val_ratio else "train"


def infer_family(source_file: str) -> str:
    if "JT包网后台操作手册" in source_file:
        return "admin"
    return "sports"


def infer_topic(text: str, family: str) -> str:
    rules_admin = [
        ("仪表盘", r"仪表盘|今日新增注册|待办事项|今日盈利|排行榜"),
        ("权限管理", r"权限管理|角色分配|角色管理|功能权限"),
        ("管理员管理", r"管理员管理|后台人员账号"),
        ("系统配置", r"系统配置|默认密码|资金冻结比例"),
        ("货币配置", r"货币配置"),
        ("线路配置", r"线路|线路域名|DNS|CDN"),
        ("轮播图管理", r"轮播图管理|轮播图|首页|个人中心"),
        ("代理管理", r"代理列表|总代|代理"),
        ("资金审核", r"充值审核|提款|资金修正|补单额度"),
    ]
    rules_sports = [
        ("串关规则", r"串关|过关|综合过关"),
        ("危险球", r"危险球"),
        ("滚球规则", r"滚球|今日赛事"),
        ("独赢和1X2", r"独赢|1X2|标准盘"),
        ("让球规则", r"让球"),
        ("大小盘规则", r"大小盘|大小"),
        ("单双规则", r"单双"),
        ("波胆规则", r"波胆|AOS"),
        ("总入球规则", r"总入球"),
        ("半全场规则", r"半场/全场|半全场"),
        ("赔率盘型", r"香港盘|印尼盘|马来盘|美国盘|欧洲盘"),
        ("赛事结算", r"未结算|派彩|官方工作人员|确认注单"),
        ("注单取消", r"注单取消|拒绝|取消注单"),
        ("足球基础规则", r"五大联赛|比赛时间|球场|足球"),
    ]
    rules = rules_admin if family == "admin" else rules_sports
    for topic, pattern in rules:
        if re.search(pattern, text):
            return topic
    first_line = next((line for line in text.splitlines() if normalize_text(line)), "")
    first_line = normalize_text(first_line).strip("【】")
    return first_line[:18] if first_line else ("后台操作" if family == "admin" else "体育规则")


def focus_lines(lines: Sequence[str], max_lines: int = 4, max_chars: int = 220) -> List[str]:
    selected: List[str] = []
    total = 0
    for line in lines:
        text = normalize_text(line)
        if not text:
            continue
        if selected and text == selected[-1]:
            continue
        selected.append(text)
        total += len(text)
        if len(selected) >= max_lines or total >= max_chars:
            break
    return selected


def contains_any(text: str, keywords: Sequence[str]) -> bool:
    return any(keyword in text for keyword in keywords)


def compact_sentence(text: str, max_chars: int = 88) -> str:
    value = normalize_text(text).strip("。；;，, ")
    if not value:
        return ""
    if len(value) <= max_chars:
        return value
    return value[:max_chars].rstrip("，,；;。") + "…"


def relation_text_blob(record: Dict[str, Any], topic: str) -> str:
    parts = [topic] + list(record.get("lines") or [])
    return "\n".join(normalize_text(part) for part in parts if normalize_text(part))


def relation_offtopic(record: Dict[str, Any], topic: str) -> bool:
    blob = relation_text_blob(record, topic)
    return contains_any(blob, RELATION_OFFTOPIC_KEYWORDS)


def relation_anchor_ok(record: Dict[str, Any], topic: str, pair_name: str) -> bool:
    blob = relation_text_blob(record, topic)
    family = str(record.get("family", "sports"))
    if relation_offtopic(record, topic):
        return False

    if pair_name == "limit_risk":
        if family == "admin":
            return contains_any(blob, ["额度限制", "限额", "审核", "补单额度", "提款订单", "冻结", "资金审核"])
        return contains_any(blob, ["滚球", "串关", "危险球", "盘口", "赔率", "注单", "让球", "大小"])

    if pair_name == "rtp_pnl":
        if family == "admin":
            if contains_any(blob, ["支付", "通道", "支付宝", "JPAI", "编辑", "禁止充值用户ID"]):
                return False
            return (
                topic in {"仪表盘", "资金审核", "赛事结算"}
                or contains_any(blob, ["统计", "排行", "盈亏", "赛果", "已结算注单", "未结算注单", "用户充值排名"])
            )
        return contains_any(blob, ["赔率", "盘口", "返奖", "返还", "香港盘", "欧洲盘", "马来盘", "印尼盘", "玩法", "结算", "赛果"])

    if pair_name == "master_platform":
        return contains_any(blob, ["总代", "代理", "上级代理", "下级代理", "直属", "推广", "域名", "佣金", "分润"]) or topic == "代理管理"

    return False


def term_aliases(glossary_row: Dict[str, Any]) -> List[str]:
    aliases = [normalize_text(x) for x in (glossary_row.get("aliases") or []) if normalize_text(x)]
    term = normalize_text(glossary_row.get("term", ""))
    out: List[str] = []
    for item in [term] + aliases:
        if item and item not in out:
            out.append(item)
    return out[:5]


def shorten_definition(term: str, definition: str) -> str:
    text = normalize_text(definition)
    if term == "RTP":
        return "RTP 是返奖率/理论返还率，讲的是长期返还水平，不是某一时刻的赔率。"
    if term == "PNL":
        return "PNL 指盈亏金额，常用于玩家或平台盈亏统计，不是财务利润表。"
    if term == "总代":
        return "总代是最高层级代理，负责管理下级代理和对接平台，不等于平台方。"
    if term == "负盈利":
        return "负盈利通常指会员亏损、平台盈利时，代理按这部分亏损参与分润的结算口径。"
    if term == "白标":
        return "白标是复用现成系统或牌照再挂自己品牌的模式，不等于完全自研。"
    if len(text) <= 130:
        return text
    return text[:130].rstrip("，,；;。") + "。"


def direct_term_in_lines(term: str, glossary_row: Dict[str, Any], lines: Sequence[str]) -> bool:
    text = "\n".join(lines)
    return any(alias and alias in text for alias in term_aliases(glossary_row))


def image_domain_terms(record: Dict[str, Any], topic: str) -> List[str]:
    family = str(record.get("family", "sports"))
    text = "\n".join(record.get("lines") or [])
    candidates: List[str] = []

    def add(*terms: str) -> None:
        for term in terms:
            if term not in candidates:
                candidates.append(term)

    if family == "admin":
        if topic in {"仪表盘", "权限管理", "管理员管理", "系统配置", "线路配置", "资金审核"}:
            add("包网", "JT包网", "包网商")
        if topic in {"仪表盘", "系统配置", "线路配置"}:
            add("白标")
        if "代理" in text or topic == "代理管理":
            add("代理", "总代", "负盈利")
        if "返水" in text:
            add("返水")
        if "洗码" in text:
            add("洗码")
        if "三方" in text or "补单" in text or "提款订单" in text:
            add("三方")
        if any(key in text for key in ["额度限制", "审核", "冻结比例", "补单额度", "提款订单"]):
            add("风控", "限红")
        if topic == "仪表盘" or any(key in text for key in ["充值", "提现", "盈利", "统计", "排行"]):
            add("PNL")
    else:
        add("体育包网")
        if any(key in text for key in ["香港盘", "印尼盘", "马来盘", "欧洲盘", "美国盘", "赔率", "盘口"]):
            add("RTP", "抽水")
        if any(key in text for key in ["滚球", "危险球", "串关", "独赢", "让球", "大小", "波胆", "总入球"]):
            add("风控", "限红")
        if any(key in text for key in ["串关", "滚球", "危险球"]):
            add("对冲")
    return candidates


def domain_anchor_allowed(record: Dict[str, Any], topic: str) -> bool:
    source_file = str(record.get("source_file", ""))
    text = "\n".join(record.get("lines") or [])
    if "五大赛事电竞讲解" in source_file:
        return False
    bad_markers = [
        "下载：通过搜索条件筛选信息",
        "invaid character",
        "本 a | 录 ==",
    ]
    if any(marker in topic for marker in bad_markers):
        return False
    if any(marker in text for marker in ["invaid character", "本 a | 录 ==", "下载：通过搜索条件筛选信息"]):
        return False
    return True


def image_evidence_prefix(term: str, glossary_row: Dict[str, Any], lines: Sequence[str], topic: str, family: str) -> str:
    visible = "；".join(focus_lines(lines, max_lines=2, max_chars=80))
    if direct_term_in_lines(term, glossary_row, lines):
        return f"图里直接能看到和「{term}」相关的信息，例如：{visible}。"
    if family == "admin":
        return f"图里主要是「{topic}」相关后台页面，例如：{visible}。图中没有直接写出「{term}」，但这类后台页面能反映该术语的实际业务位置。"
    return f"图里主要是「{topic}」相关玩法/规则说明，例如：{visible}。图中没有直接写出「{term}」，但这类体育页面可以作为解释该术语的场景锚点。"


def select_visible_evidence(lines: Sequence[str], preferred_keywords: Sequence[str]) -> str:
    matched: List[str] = []
    for line in lines:
        text = normalize_text(line)
        if not text:
            continue
        if any(keyword in text for keyword in preferred_keywords):
            matched.append(text)
    focus = matched or list(lines)
    return "；".join(focus_lines(focus, max_lines=2, max_chars=96))


def relation_visible_prefix(record: Dict[str, Any], topic: str, preferred_keywords: Sequence[str]) -> str:
    lines = record.get("lines") or []
    family = str(record.get("family", "sports"))
    visible = select_visible_evidence(lines, preferred_keywords)
    if visible:
        return f"页面可见信息：{visible}。"
    if family == "admin":
        return f"页面可见信息：图里主要是「{topic}」相关后台页面，但没有直接出现目标术语字样。"
    return f"页面可见信息：图里主要是「{topic}」相关玩法/规则页面，但没有直接出现目标术语字样。"


def comparison_statement(term: str, related: str) -> str:
    pair = {term, related}
    if pair == {"包网", "白标"}:
        return "包网更强调整套技术和运营能力打包，白标更强调复用现成系统或牌照后挂自己的品牌。"
    if pair == {"JT包网", "包网商"}:
        return "JT包网更像具体服务方案或品牌名称，包网商是提供这类整套能力的服务方角色。"
    if pair == {"代理", "总代"}:
        return "总代层级更高，负责管理下级代理并直接对接平台；普通代理层级更低，职责更偏拓客和带下级。"
    if pair == {"风控", "限红"}:
        return "风控是更上层的风险控制机制，限红是风控最常见的一种落地动作。"
    if pair == {"RTP", "PNL"}:
        return "RTP 是长期返还率口径，PNL 是某个主体在某段时间里的实际盈亏口径。"
    if pair == {"负盈利", "PNL"}:
        return "PNL 是泛化的盈亏统计，负盈利更偏代理分润结算口径。"
    if pair == {"返水", "洗码"}:
        return "返水是按规则返还的一部分金额，洗码更强调按有效流水计算返利或记录的结算口径。"
    if pair == {"抽水", "RTP"}:
        return "抽水讲平台抽成或利润空间，RTP讲长期返还率，两者不是同一个指标。"
    if pair == {"对冲", "风控"}:
        return "对冲是风控重点关注的异常投注行为，风控会据此做限额、审核或清洗。"
    if pair == {"三方", "包网"}:
        return "三方通常是包网体系里的外部接入组件，不等于整个包网方案本身。"
    if pair == {"体育包网", "包网"}:
        return "体育包网是包网在体育业务上的细分，更聚焦体育盘口、滚球、串关和赛果结算能力。"
    return f"{term}和{related}相关，但不是同一个概念，判断时要先看它们处于产品能力、角色层级还是结算口径。"


def misconception_answer(term: str) -> str:
    mapping = {
        "包网": "不是。包网通常是整套后台、支付、风控、游戏接入和运营能力，不只是一个前台页面。",
        "JT包网": "不是。JT包网通常提供后台、支付、风控、代理体系和多类游戏接入，不只是一条盘口。",
        "体育包网": "不是。体育包网通常覆盖足球、篮球等赛事，以及滚球、串关、让球、大小等多类玩法。",
        "白标": "不是。白标强调复用现成系统或牌照资源，并不代表完全自研平台。",
        "包网商": "不是。包网商是底层能力提供方，不是普通代理或玩家账号。",
        "代理": "不是。代理主要负责渠道和下级管理，不负责平台底层技术。",
        "总代": "不一定。总代通常是最高层级代理，不等于平台技术方。",
        "风控": "不是。风控的核心是识别异常行为、控制风险敞口，不是单纯限制正常用户。",
        "限红": "不是。限红是额度限制，和封号不是一回事。",
        "RTP": "不是。RTP 是理论返还率，不是实时赔率。",
        "PNL": "不是。这里的 PNL 主要指盈亏金额，不是财务报表里的利润表。",
        "负盈利": "通常指会员亏损、平台盈利后代理按这部分亏损参与分润，不是说平台亏钱。",
        "返水": "不是。返水是按规则返还的一部分金额，不等于注单中奖派彩。",
        "洗码": "不是。洗码在这里是流水返利或记录口径，不是违法洗钱。",
        "抽水": "不完全是。抽水更偏平台抽成或利润空间，不一定以手续费形式单列。",
        "对冲": "不是。对冲是相反方向下注以锁定风险或套利，和正常串关不是一个概念。",
        "三方": "不是。三方通常指第三方支付、游戏或数据服务，不等于平台自研系统。",
    }
    return mapping.get(term, f"{term}不建议按普通互联网语境理解，需按体育包网业务口径判断。")


def domain_definition_answer(term: str, glossary_row: Dict[str, Any], record: Dict[str, Any], topic: str) -> str:
    lines = record.get("lines") or []
    family = str(record.get("family", "sports"))
    return (
        f"{image_evidence_prefix(term, glossary_row, lines, topic, family)} "
        f"{term}是：{shorten_definition(term, str(glossary_row.get('definition', '')))}"
    ).strip()


def domain_usage_answer(term: str, glossary_row: Dict[str, Any], record: Dict[str, Any], topic: str) -> str:
    lines = record.get("lines") or []
    family = str(record.get("family", "sports"))
    usage_tail = {
        "包网": "这类图能体现包网方案包含后台管理、配置和运营支撑能力。",
        "JT包网": "这类后台图能体现 JT 包网交付的是成套系统能力，而不是单一模块。",
        "体育包网": "这类玩法图体现的是体育包网里的赛事、赔率、盘口和结算能力。",
        "白标": "这类系统页常被拿来说明白标/包网方案能不能快速挂品牌上线。",
        "包网商": "这类后台图通常是包网商对外提供的运营和管理能力体现。",
        "代理": "这类页面一般用于代理层级管理、上级绑定、推广域名和代理数据维护。",
        "总代": "这类代理页常用于总代管理下级代理、域名和推广链路。",
        "风控": "这类页面常和额度、审核、异常处理联动，用来落地风控策略。",
        "限红": "这类页面常用来说明额度限制、审核限制和风险敞口控制。",
        "RTP": "这类玩法/赔率图可以作为解释 RTP 所属产品口径的场景，但图里看到的盘口本身不等于 RTP。",
        "PNL": "这类统计页常被用来观察平台盈利、充值提现差额和运营盈亏。",
        "负盈利": "这类代理后台常用于解释代理分润和负盈利结算逻辑。",
        "返水": "这类记录或比例配置图说明返水会落到具体的后台记录和比例设置。",
        "洗码": "这类记录页通常用于看洗码流水、洗码记录和返利结算。",
        "抽水": "这类盘口/赔率图可以帮助解释平台为什么会关注抽水和利润空间。",
        "对冲": "这类体育玩法图常用于说明平台为什么要监控相反方向下注和套利风险。",
        "三方": "这类补单/订单页面通常对应第三方支付或第三方回调处理链路。",
    }
    return (
        f"{image_evidence_prefix(term, glossary_row, lines, topic, family)} "
        f"{usage_tail.get(term, f'在这类页面里，{term}通常作为业务概念或后台能力来理解。')}"
    ).strip()


def domain_comparison_answer(
    term: str,
    related: str,
    glossary_index: Dict[str, Dict[str, Any]],
    record: Dict[str, Any],
    topic: str,
) -> str:
    glossary_row = glossary_index[term]
    related_row = glossary_index[related]
    lines = record.get("lines") or []
    family = str(record.get("family", "sports"))
    return (
        f"{image_evidence_prefix(term, glossary_row, lines, topic, family)} "
        f"{comparison_statement(term, related)} "
        f"其中 {term} 更接近「{shorten_definition(term, str(glossary_row.get('definition', '')))}」；"
        f"{related} 更接近「{shorten_definition(related, str(related_row.get('definition', '')))}」"
    ).strip()


def relation_limit_risk_answer(
    record: Dict[str, Any],
    topic: str,
    glossary_index: Dict[str, Dict[str, Any]],
) -> str:
    risk_row = glossary_index["风控"]
    limit_row = glossary_index["限红"]
    visible = relation_visible_prefix(record, topic, ["额度", "限制", "审核", "冻结", "补单", "盘口", "玩法", "注单"])
    return (
        f"{visible} "
        "业务解释：这类额度、审核、冻结、盘口限制页面通常是风控的执行位。"
        "风控负责识别异常投注、异常资金流动和风险敞口；限红则是把风险控制落实到用户、玩法或注单额度上的具体动作。 "
        f"关系结论：风控更接近「{shorten_definition('风控', str(risk_row.get('definition', '')))}」；"
        f"限红更接近「{shorten_definition('限红', str(limit_row.get('definition', '')))}」。"
    ).strip()


def relation_rtp_pnl_answer(
    record: Dict[str, Any],
    topic: str,
    glossary_index: Dict[str, Dict[str, Any]],
) -> str:
    rtp_row = glossary_index["RTP"]
    pnl_row = glossary_index["PNL"]
    family = str(record.get("family", "sports"))
    keywords = ["盈利", "盈亏", "充值", "提现", "统计", "排行", "赔率", "盘口", "玩法", "返奖"]
    visible = relation_visible_prefix(record, topic, keywords)
    page_binding = (
        "如果图里更偏统计、盈利、充值提现、排行看板，它更接近 PNL 场景；"
        "如果图里更偏玩法、赔率、盘口和返还逻辑，它更接近 RTP 所在的产品口径。"
    )
    if family == "sports":
        page_binding = (
            "这类体育玩法/赔率页更适合拿来解释 RTP 所在的产品规则口径；"
            "而 PNL 一般要落到后台统计、玩家或平台盈亏汇总页里看。"
        )
    return (
        f"{visible} "
        f"业务解释：{page_binding} "
        f"关系结论：RTP 更接近「{shorten_definition('RTP', str(rtp_row.get('definition', '')))}」；"
        f"PNL 更接近「{shorten_definition('PNL', str(pnl_row.get('definition', '')))}」。"
    ).strip()


def relation_master_platform_answer(
    record: Dict[str, Any],
    topic: str,
    glossary_index: Dict[str, Dict[str, Any]],
) -> str:
    master_row = glossary_index["总代"]
    visible = relation_visible_prefix(record, topic, ["总代", "代理", "上级", "下级", "域名", "推广", "佣金", "分润"])
    return (
        f"{visible} "
        "业务解释：这类页面如果出现代理列表、总代、下级代理、域名、推广链路或分润信息，说明它属于代理体系页面。"
        "总代是代理层级里最高的一层，负责管理下级代理并和平台对接；"
        "平台方/系统方则负责规则、结算、支付、风控和底层系统能力。 "
        f"关系结论：总代更接近「{shorten_definition('总代', str(master_row.get('definition', '')))}」；"
        f"平台更接近「{PLATFORM_SHORT_DEFINITION}」。"
    ).strip()


def build_limit_risk_relation_pairs(
    record: Dict[str, Any],
    topic: str,
    glossary_index: Dict[str, Dict[str, Any]],
) -> List[Tuple[str, str, str, str]]:
    if not relation_anchor_ok(record, topic, "limit_risk"):
        return []

    visible = relation_visible_prefix(record, topic, ["额度", "限制", "审核", "冻结", "补单", "盘口", "玩法", "注单"])
    risk_def = shorten_definition("风控", str(glossary_index["风控"].get("definition", "")))
    limit_def = shorten_definition("限红", str(glossary_index["限红"].get("definition", "")))
    return [
        (
            "限红",
            "relation",
            "限红和风控是什么关系？结合这张后台图说明。",
            f"{visible} 业务解释：这类额度限制、审核或补单字段更像风控执行位。风控负责识别和控制风险，限红负责把风险控制落到具体额度。 关系结论：限红通常是风控落地动作的一部分，不是独立于风控的另一套体系。",
        ),
        (
            "限红",
            "relation",
            "这张页面里能看到额度或审核信息时，应该把它理解成风控还是限红？",
            f"{visible} 判断：先按风控执行位理解，再把限红看成其中的额度控制动作。 原因：页面出现的是额度、审核或补单字段，说明它在执行风险策略，而不是单独介绍概念。",
        ),
        (
            "限红",
            "relation",
            "请先说这张图里能看到什么，再说明风控和限红在业务上怎么对应。",
            f"{visible} 业务对应：可见的额度限制、审核、补单字段，对应的是风控在后台的执行动作；其中限红负责限制下注或额度，风控负责整体识别和处理风险。",
        ),
        (
            "限红",
            "relation",
            "从这张图看，限红是独立功能，还是风控落地动作的一部分？为什么？",
            f"{visible} 结论：更像风控落地动作的一部分。 因为页面展示的是审核、限额、补单这类执行字段，说明它在落实风险策略，而不是单独介绍一套风险机制。",
        ),
        (
            "限红",
            "relation",
            "如果只根据页面可见信息，这页更像风控执行位还是普通运营页？限红放在什么位置理解更准确？",
            f"{visible} 只根据页面可见信息，这页更像风控执行位。 更准确的理解是：风控是上层机制，限红是其中控制下注额度的具体手段。",
        ),
        (
            "限红",
            "relation",
            "请用不超过3句话回答：这页里风控和限红怎么绑定？不要重复字段名。",
            f"{visible} 风控管整体风险识别和处置，限红管具体额度控制。 这类页面更像风控落地页，不要把限红单独说成另一套系统。 {compact_sentence(risk_def, 46)}；{compact_sentence(limit_def, 46)}。",
        ),
    ]


def build_rtp_pnl_relation_pairs(
    record: Dict[str, Any],
    topic: str,
    glossary_index: Dict[str, Dict[str, Any]],
) -> List[Tuple[str, str, str, str]]:
    if not relation_anchor_ok(record, topic, "rtp_pnl"):
        return []

    family = str(record.get("family", "sports"))
    visible = relation_visible_prefix(record, topic, ["盈利", "盈亏", "充值", "提现", "统计", "排行", "赔率", "盘口", "玩法", "返奖", "赛果", "结算"])
    if family == "sports":
        page_binding = "这类体育玩法、赔率或盘口页，更适合解释 RTP 所在的产品规则口径；PNL 一般要到后台统计或盈亏汇总页里看。"
        page_judgement = "如果图里主要是赔率、盘口、玩法或赛果信息，就先按 RTP 所在口径理解，不要直接把它说成 PNL 统计页。"
    else:
        page_binding = "这类统计、排行、充值提现看板，更接近 PNL 场景；RTP 是玩法长期返还率口径，不会在这种统计页直接展示。"
        page_judgement = "如果图里主要是统计、排行、充值提现或盈亏汇总，就先按 PNL 场景理解，不要把它硬说成 RTP 配置页。"
    rtp_def = shorten_definition("RTP", str(glossary_index["RTP"].get("definition", "")))
    pnl_def = shorten_definition("PNL", str(glossary_index["PNL"].get("definition", "")))
    return [
        (
            "RTP",
            "relation",
            "RTP 和 PNL 有什么区别？结合这张图说明。",
            f"{visible} 业务解释：{page_binding} 关系结论：RTP 讲长期返还率口径；PNL 讲某个主体在一段时间内的实际盈亏口径。",
        ),
        (
            "RTP",
            "relation",
            "这张图更接近 RTP 场景还是 PNL 场景？请先说页面可见信息，再说业务解释。",
            f"{visible} 业务解释：{page_judgement} 关系结论：RTP 不是实际盈亏统计，PNL 也不是玩法返还率，两者不能混说。",
        ),
        (
            "RTP",
            "relation",
            "从这张页面能看出 RTP 和 PNL 分别属于什么口径吗？",
            f"{visible} 口径判断：RTP 属于玩法和返还逻辑口径；PNL 属于统计和盈亏汇总口径。 看图时先判断页面偏规则还是偏统计，再决定往哪一边解释。",
        ),
        (
            "RTP",
            "relation",
            "请结合图里可见信息说明：RTP 为什么不等于 PNL？",
            f"{visible} 原因：RTP 讲的是长期理论返还水平，PNL 讲的是实际盈亏结果。 一个偏规则参数，一个偏统计结果，所以不能互相替代。",
        ),
        (
            "RTP",
            "relation",
            "回答这类图时，页面可见信息和业务解释应该怎么绑定，才能不把 RTP 和 PNL 说混？",
            f"{visible} 绑定方法：先说页面里是赔率/盘口还是统计/盈亏，再给业务解释。 看到玩法和返还逻辑时往 RTP 解释，看到统计和盈亏汇总时往 PNL 解释。",
        ),
        (
            "RTP",
            "relation",
            "请用不超过3句话回答：这页里 RTP 和 PNL 怎么区分？不要重复字段名。",
            f"{visible} RTP 看返还率和玩法口径，PNL 看盈亏统计口径。 简单说：RTP 不是盈亏表，PNL 也不是返奖率。 {compact_sentence(rtp_def, 44)}；{compact_sentence(pnl_def, 44)}。",
        ),
    ]


def build_master_platform_relation_pairs(
    record: Dict[str, Any],
    topic: str,
    glossary_index: Dict[str, Dict[str, Any]],
) -> List[Tuple[str, str, str, str]]:
    if not relation_anchor_ok(record, topic, "master_platform"):
        return []

    visible = relation_visible_prefix(record, topic, ["总代", "代理", "上级", "下级", "域名", "推广", "佣金", "分润", "直属"])
    master_def = shorten_definition("总代", str(glossary_index["总代"].get("definition", "")))
    return [
        (
            "总代",
            "relation",
            "总代和平台是什么关系？结合这张代理相关页面说明。",
            f"{visible} 业务解释：这类页面属于代理体系页面。 总代负责管理下级代理、推广链路或代理数据，对接平台；平台方负责规则、结算、支付、风控和底层系统能力。 关系结论：总代不等于平台。",
        ),
        (
            "总代",
            "relation",
            "这张页面里的总代是不是平台方？请先说页面可见信息，再说业务解释。",
            f"{visible} 业务解释：页面里出现代理、上级下级、域名或推广信息时，优先按代理层级理解，不要直接上升成平台方。 关系结论：总代是代理层级角色，平台是系统和规则提供方。",
        ),
        (
            "总代",
            "relation",
            "从这张代理页看，总代和平台分别扮演什么角色？",
            f"{visible} 角色划分：总代更偏代理管理、推广和下级维护；平台更偏底层系统、结算、支付和风控。 两者有对接关系，但不是同一个角色。",
        ),
        (
            "总代",
            "relation",
            "图里出现代理、下级或域名信息时，为什么不能把总代直接理解成平台？",
            f"{visible} 原因：这些字段说明页面在讲代理体系，不是在讲平台底层能力。 总代处在代理链路里，平台处在系统和规则供给侧，所以不能直接画等号。",
        ),
        (
            "总代",
            "relation",
            "如果只根据页面可见信息，这页更像代理体系页还是平台系统页？总代应该放在哪个角色层级理解？",
            f"{visible} 只根据页面可见信息，这页更像代理体系页。 更准确的理解是：总代属于最高层级代理，平台属于底层系统和规则提供方。",
        ),
        (
            "总代",
            "relation",
            "请用不超过3句话回答：总代和平台怎么区分？不要重复字段名。",
            f"{visible} 总代是代理链路里的最高层级角色，平台是底层系统和规则提供方。 看见代理、下级、推广或域名时，先按代理体系解释，不要直接说成平台。 {compact_sentence(master_def, 44)}；{compact_sentence(PLATFORM_SHORT_DEFINITION, 44)}。",
        ),
    ]


def build_special_relation_pairs(
    record: Dict[str, Any],
    topic: str,
    glossary_index: Dict[str, Dict[str, Any]],
) -> List[Tuple[str, str, str, str]]:
    terms = set(image_domain_terms(record, topic))
    pairs: List[Tuple[str, str, str, str]] = []
    family = str(record.get("family", "sports"))
    text = "\n".join(record.get("lines") or [])

    if {"风控", "限红"} & terms:
        pairs.extend(build_limit_risk_relation_pairs(record, topic, glossary_index))

    if ("RTP" in terms or "PNL" in terms or ("统计" in text and family == "admin")):
        pairs.extend(build_rtp_pnl_relation_pairs(record, topic, glossary_index))

    if ("总代" in terms or "代理" in terms or topic == "代理管理"):
        pairs.extend(build_master_platform_relation_pairs(record, topic, glossary_index))

    return pairs


def domain_misconception_answer(term: str, glossary_row: Dict[str, Any], record: Dict[str, Any], topic: str) -> str:
    lines = record.get("lines") or []
    family = str(record.get("family", "sports"))
    return (
        f"{image_evidence_prefix(term, glossary_row, lines, topic, family)} "
        f"{misconception_answer(term)}"
    ).strip()


def extract_steps(lines: Sequence[str], max_steps: int = 3) -> List[str]:
    verbs = ("点击", "选择", "输入", "保存", "进入", "配置", "绑定", "查看", "审核", "提交", "设置", "添加", "编辑")
    steps: List[str] = []
    for line in lines:
        text = normalize_text(line)
        if any(v in text for v in verbs):
            steps.append(text)
        if len(steps) >= max_steps:
            break
    if steps:
        return steps
    return focus_lines(lines, max_lines=max_steps, max_chars=120)


def make_text_answer(topic: str, family: str, lines: Sequence[str]) -> str:
    key_points = "；".join(focus_lines(lines))
    if family == "admin":
        steps = extract_steps(lines)
        answer_lines = [f"这段资料主要对应「{topic}」。"]
        if key_points:
            answer_lines.append(f"核心说明：{key_points}。")
        if steps:
            answer_lines.append("参考操作顺序：")
            for idx, step in enumerate(steps, 1):
                answer_lines.append(f"{idx}. {step}")
        answer_lines.append("如涉及权限、额度或资金变动，以当前后台角色权限和审核流程为准。")
        return "\n".join(answer_lines)

    tail = "如落到具体注单或结算争议，以平台规则页、赛果同步和最终结算结果为准。"
    return f"这段资料主要讲「{topic}」。关键点：{key_points}。{tail}"


def build_text_prompt(topic: str, family: str) -> str:
    if family == "admin":
        return f"{topic}这个后台模块是做什么的？请给出简明操作说明。"
    return f"{topic}这类体育玩法或规则应该怎么理解？请直接说明要点。"


def build_mm_pairs(topic: str, family: str, lines: Sequence[str]) -> List[Tuple[str, str]]:
    base_answer = make_text_answer(topic, family, lines)
    if family == "admin":
        scenario_answer = (
            f"这张图对应「{topic}」相关页面。建议先确认页面用途、当前账号权限和用户具体想做的动作，"
            f"再按图中字段或按钮逐步操作。{make_text_answer(topic, family, lines).splitlines()[-1]}"
        )
        return [
            ("这张后台截图对应什么功能？请简要说明。", base_answer),
            ("用户发来这张图时，客服应该怎么引导？", scenario_answer),
        ]
    scenario_answer = (
        f"这张图主要在解释「{topic}」。客服可以先讲清图里的核心规则，再提醒用户具体注单仍以平台规则页和最终结算为准。"
    )
    return [
        ("这张图主要讲的是什么体育规则或玩法？", base_answer),
        ("如果用户看不懂这张图，客服应该怎么解释？", scenario_answer),
    ]


def build_mm_domain_pairs(
    record: Dict[str, Any],
    topic: str,
    glossary_index: Dict[str, Dict[str, Any]],
) -> List[Tuple[str, str, str, str]]:
    if not domain_anchor_allowed(record, topic):
        return []

    family = str(record.get("family", "sports"))
    lines = record.get("lines") or []
    pairs: List[Tuple[str, str, str, str]] = []

    for term in image_domain_terms(record, topic):
        glossary_row = glossary_index.get(term)
        if not glossary_row:
            continue
        pairs.append(
            (
                term,
                "definition",
                f"结合这张图，{term}在这类页面或场景里通常指什么？",
                domain_definition_answer(term, glossary_row, record, topic),
            )
        )
        pairs.append(
            (
                term,
                "usage",
                f"看这张图，{term}在体育包网业务里一般怎么用？",
                domain_usage_answer(term, glossary_row, record, topic),
            )
        )

        related = DOMAIN_COMPARISONS.get(term)
        if related and related in glossary_index:
            pairs.append(
                (
                    term,
                    "comparison",
                    f"结合这张图，{term}和{related}有什么区别？",
                    domain_comparison_answer(term, related, glossary_index, record, topic),
                )
            )

        misconception_query = TERM_MISCONCEPTION_QUESTIONS.get(term)
        if misconception_query:
            pairs.append(
                (
                    term,
                    "misconception",
                    misconception_query,
                    domain_misconception_answer(term, glossary_row, record, topic),
                )
            )

        if term == "体育包网":
            pairs.append(
                (
                    term,
                    "comparison",
                    "这张体育玩法图能体现体育包网和普通包网有什么侧重点区别？",
                    domain_comparison_answer("体育包网", "包网", glossary_index, record, topic),
                )
            )
        if term == "PNL":
            pairs.append(
                (
                    term,
                    "misconception",
                    "这张后台统计图里的 PNL 是不是财务利润表？",
                    domain_misconception_answer(term, glossary_row, record, topic),
                )
            )
        if term == "RTP":
            pairs.append(
                (
                    term,
                    "comparison",
                    "RTP 和 PNL 有什么区别？",
                    domain_comparison_answer("RTP", "PNL", glossary_index, record, topic),
                )
            )
        if term == "限红":
            pairs.append(
                (
                    term,
                    "comparison",
                    "限红和风控是什么关系？",
                    domain_comparison_answer("限红", "风控", glossary_index, record, topic),
                )
            )
        if term == "总代":
            pairs.append(
                (
                    term,
                    "misconception",
                    "总代是不是平台？",
                    domain_misconception_answer(term, glossary_row, record, topic),
                )
            )
        if term == "负盈利":
            pairs.append(
                (
                    term,
                    "misconception",
                    "负盈利是玩家亏损还是平台亏损？",
                    domain_misconception_answer(term, glossary_row, record, topic),
                )
            )
        if term == "包网":
            pairs.append(
                (
                    term,
                    "comparison",
                    "包网和白标有什么区别？",
                    domain_comparison_answer("包网", "白标", glossary_index, record, topic),
                )
            )

    pairs.extend(build_special_relation_pairs(record, topic, glossary_index))

    deduped: List[Tuple[str, str, str, str]] = []
    seen = set()
    for term, qtype, query, answer in pairs:
        key = (term, qtype, query)
        if key in seen:
            continue
        seen.add(key)
        deduped.append((term, qtype, query, answer))
    return deduped


def parse_docx_relationships(zf: zipfile.ZipFile) -> Dict[str, str]:
    rels_path = "word/_rels/document.xml.rels"
    if rels_path not in zf.namelist():
        return {}
    root = ET.fromstring(zf.read(rels_path))
    mapping: Dict[str, str] = {}
    for rel in root.findall("pr:Relationship", DOCX_NS):
        rid = rel.attrib.get("Id", "").strip()
        target = rel.attrib.get("Target", "").strip()
        rel_type = rel.attrib.get("Type", "").strip()
        if rid and target and rel_type.endswith("/image"):
            mapping[rid] = Path(target).name
    return mapping


def extract_docx_assets(docx_path: Path, output_dir: Path) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    paragraphs: List[Dict[str, Any]] = []
    image_records: List[Dict[str, Any]] = []
    text_units: List[Dict[str, Any]] = []

    docx_output = output_dir / docx_path.stem
    images_dir = docx_output / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(docx_path, "r") as zf:
        rid_to_target = parse_docx_relationships(zf)
        for name in zf.namelist():
            if name.startswith("word/media/"):
                target_path = images_dir / Path(name).name
                with zf.open(name, "r") as src, target_path.open("wb") as dst:
                    shutil.copyfileobj(src, dst)

        root = ET.fromstring(zf.read("word/document.xml"))
        for idx, p in enumerate(root.findall(".//w:p", DOCX_NS), 1):
            texts = [(t.text or "") for t in p.findall(".//w:t", DOCX_NS)]
            joined = normalize_text("".join(texts))
            image_names: List[str] = []
            for blip in p.findall(".//a:blip", DOCX_NS):
                rid = blip.attrib.get(DOCX_RID_KEY, "").strip()
                target = rid_to_target.get(rid, "")
                if target:
                    image_names.append(target)
            paragraphs.append({"paragraph_id": idx, "text": joined, "image_names": image_names})

    non_empty = [p for p in paragraphs if p["text"] and not looks_marketing(p["text"])]
    chunk: List[str] = []
    for p in non_empty:
        text = str(p["text"])
        if sum(len(x) for x in chunk) + len(text) > 420 and chunk:
            lines = clean_lines(chunk, min_len=8)
            if should_keep_context(lines):
                text_units.append(
                    {
                        "id": f"docx_text_{len(text_units) + 1:04d}",
                        "source_file": docx_path.name,
                        "source_type": "docx_text",
                        "family": infer_family(docx_path.name),
                        "lines": lines,
                    }
                )
            chunk = []
        chunk.append(text)
    if chunk:
        lines = clean_lines(chunk, min_len=8)
        if should_keep_context(lines):
            text_units.append(
                {
                    "id": f"docx_text_{len(text_units) + 1:04d}",
                    "source_file": docx_path.name,
                    "source_type": "docx_text",
                    "family": infer_family(docx_path.name),
                    "lines": lines,
                }
            )

    for idx, p in enumerate(paragraphs):
        if not p["image_names"]:
            continue
        prev_text = ""
        next_text = ""
        for j in range(idx - 1, -1, -1):
            if paragraphs[j]["text"]:
                prev_text = str(paragraphs[j]["text"])
                break
        for j in range(idx + 1, len(paragraphs)):
            if paragraphs[j]["text"]:
                next_text = str(paragraphs[j]["text"])
                break
        lines = clean_lines([prev_text, str(p["text"]), next_text], min_len=6)
        if not should_keep_context(lines):
            continue
        for image_name in p["image_names"]:
            image_path = images_dir / image_name
            if not valid_image(image_path):
                continue
            image_records.append(
                {
                    "id": f"docx_img_{len(image_records) + 1:04d}",
                    "source_file": docx_path.name,
                    "source_type": "docx_image",
                    "family": infer_family(docx_path.name),
                    "image_path": str(image_path),
                    "lines": lines,
                }
            )
    return text_units, image_records


def parse_pptx_slide_relationships(zf: zipfile.ZipFile, slide_idx: int) -> List[str]:
    rels_path = f"ppt/slides/_rels/slide{slide_idx}.xml.rels"
    if rels_path not in zf.namelist():
        return []
    root = ET.fromstring(zf.read(rels_path))
    names: List[str] = []
    for rel in root.findall("pr:Relationship", PPT_NS):
        target = rel.attrib.get("Target", "").strip()
        rel_type = rel.attrib.get("Type", "").strip()
        if target and rel_type.endswith("/image"):
            names.append(Path(target).name)
    return names


def extract_pptx_assets(pptx_path: Path, output_dir: Path) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    text_units: List[Dict[str, Any]] = []
    image_records: List[Dict[str, Any]] = []

    pptx_output = output_dir / pptx_path.stem
    images_dir = pptx_output / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(pptx_path, "r") as zf:
        for name in zf.namelist():
            if name.startswith("ppt/media/") and Path(name).suffix.lower() in {".png", ".jpg", ".jpeg"}:
                target_path = images_dir / Path(name).name
                with zf.open(name, "r") as src, target_path.open("wb") as dst:
                    shutil.copyfileobj(src, dst)

        slide_files = sorted(
            [name for name in zf.namelist() if name.startswith("ppt/slides/slide") and name.endswith(".xml")],
            key=lambda value: int(re.search(r"slide(\d+)\.xml", value).group(1)),
        )
        for slide_file in slide_files:
            slide_idx = int(re.search(r"slide(\d+)\.xml", slide_file).group(1))
            root = ET.fromstring(zf.read(slide_file))
            raw_tokens = [t.text.strip() for t in root.findall(".//a:t", PPT_NS) if t.text and t.text.strip()]
            lines = clean_lines(raw_tokens, min_len=4)
            if not should_keep_context(lines):
                continue

            text_units.append(
                {
                    "id": f"pptx_slide_{len(text_units) + 1:04d}",
                    "source_file": pptx_path.name,
                    "source_type": "pptx_slide",
                    "family": infer_family(pptx_path.name),
                    "lines": lines,
                }
            )

            for image_name in parse_pptx_slide_relationships(zf, slide_idx):
                image_path = images_dir / image_name
                if not valid_image(image_path):
                    continue
                image_records.append(
                    {
                        "id": f"pptx_img_{len(image_records) + 1:04d}",
                        "source_file": pptx_path.name,
                        "source_type": "pptx_image",
                        "family": infer_family(pptx_path.name),
                        "image_path": str(image_path),
                        "lines": lines,
                    }
                )
    return text_units, image_records


def ocr_png(png_path: Path) -> List[str]:
    raw = run_cmd(["tesseract", str(png_path), "stdout", "-l", "chi_sim+eng", "--psm", "6"])
    lines: List[str] = []
    for raw_line in raw.splitlines():
        line = normalize_text(raw_line)
        if not line or len(line) < 6:
            continue
        if OCR_NOISE_RE.search(line):
            continue
        if not has_cjk(line):
            continue
        if looks_marketing(line):
            continue
        lines.append(line)
    return clean_lines(lines, min_len=6)


def extract_pdf_assets(pdf_path: Path, output_dir: Path) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    text_units: List[Dict[str, Any]] = []
    image_records: List[Dict[str, Any]] = []

    pdf_output = output_dir / pdf_path.stem
    pdf_output.mkdir(parents=True, exist_ok=True)
    prefix = pdf_output / "page"
    run_cmd(["pdftoppm", "-r", "180", "-png", str(pdf_path), str(prefix)])

    for page_png in sorted(pdf_output.glob("page-*.png")):
        lines = ocr_png(page_png)
        if not should_keep_context(lines, min_lines=2, min_cjk=24):
            continue
        page_no_match = re.search(r"page-(\d+)\.png$", page_png.name)
        page_no = int(page_no_match.group(1)) if page_no_match else 0
        text_units.append(
            {
                "id": f"pdf_text_{len(text_units) + 1:04d}",
                "source_file": pdf_path.name,
                "source_type": "pdf_page",
                "family": infer_family(pdf_path.name),
                "page": page_no,
                "lines": lines,
            }
        )
        image_records.append(
            {
                "id": f"pdf_img_{len(image_records) + 1:04d}",
                "source_file": pdf_path.name,
                "source_type": "pdf_page_image",
                "family": infer_family(pdf_path.name),
                "page": page_no,
                "image_path": str(page_png),
                "lines": lines,
            }
        )
    return text_units, image_records


def build_text_rows(text_units: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for unit in text_units:
        lines = unit.get("lines") or []
        family = str(unit.get("family", "sports"))
        source_file = str(unit.get("source_file", ""))
        topic = infer_topic("\n".join(lines), family)
        row = {
            "id": f"tytxt_{unit.get('id')}",
            "split": "train",
            "messages": [
                {"role": "user", "content": build_text_prompt(topic, family)},
                {"role": "assistant", "content": make_text_answer(topic, family, lines)},
            ],
            "meta": {
                "source_file": source_file,
                "source_type": unit.get("source_type"),
                "topic": topic,
                "family": family,
            },
        }
        rows.append(row)
    return rows


def build_mm_rows(
    image_records: Sequence[Dict[str, Any]],
    val_ratio: float,
    glossary_index: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for record in image_records:
        image_path = Path(str(record.get("image_path", "")))
        if not valid_image(image_path):
            continue
        lines = record.get("lines") or []
        family = str(record.get("family", "sports"))
        topic = infer_topic("\n".join(lines), family)
        for idx, (query, answer) in enumerate(build_mm_pairs(topic, family, lines), 1):
            sample_id = f"tymm_{record.get('id')}_{idx:02d}"
            rows.append(
                {
                    "id": sample_id,
                    "split": pick_split(sample_id, val_ratio),
                    "system": SYSTEM_PROMPT,
                    "query": query,
                    "response": answer,
                    "images": [str(image_path)],
                    "meta": {
                        "source_file": record.get("source_file"),
                        "source_type": record.get("source_type"),
                        "topic": topic,
                        "family": family,
                    },
                }
            )
        for idx, (term, qtype, query, answer) in enumerate(build_mm_domain_pairs(record, topic, glossary_index), 1):
            sample_id = f"tymm_domain_{record.get('id')}_{term}_{idx:02d}"
            rows.append(
                {
                    "id": sample_id,
                    "split": pick_split(sample_id, val_ratio),
                    "system": SYSTEM_PROMPT,
                    "query": query,
                    "response": answer,
                    "images": [str(image_path)],
                    "meta": {
                        "source_file": record.get("source_file"),
                        "source_type": record.get("source_type"),
                        "topic": topic,
                        "family": family,
                        "domain_term": term,
                        "domain_qtype": qtype,
                    },
                }
            )
    return rows


def strict_post_clean_text_filter(cleaned_path: Path) -> Dict[str, int]:
    rows = read_jsonl(cleaned_path)
    kept: List[Dict[str, Any]] = []
    removed = 0
    for row in rows:
        assistant_text = "\n".join(
            str(message.get("content", ""))
            for message in row.get("messages", [])
            if isinstance(message, dict) and message.get("role") == "assistant"
        )
        if any(pattern.search(assistant_text) for pattern in STRICT_TEXT_FILTER_PATTERNS):
            removed += 1
            continue
        kept.append(row)
    write_jsonl(cleaned_path, kept)
    return {"strict_removed": removed, "final_rows": len(kept)}


def compute_keyword_coverage_text(path: Path, keywords: Sequence[str]) -> Dict[str, int]:
    counts = {keyword: 0 for keyword in keywords}
    if not path.exists():
        return counts
    for row in read_jsonl(path):
        text = "\n".join(
            str(message.get("content", ""))
            for message in row.get("messages", [])
            if isinstance(message, dict)
        )
        for keyword in keywords:
            if keyword in text:
                counts[keyword] += 1
    return counts


def compute_keyword_coverage_mm(rows: Sequence[Dict[str, Any]], keywords: Sequence[str]) -> Dict[str, int]:
    counts = {keyword: 0 for keyword in keywords}
    for row in rows:
        text = "\n".join(
            [
                str(row.get("query", "")),
                str(row.get("response", "")),
                str((row.get("meta") or {}).get("topic", "")),
            ]
        )
        for keyword in keywords:
            if keyword in text:
                counts[keyword] += 1
    return counts


def compute_domain_qtype_counts(rows: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        qtype = str((row.get("meta") or {}).get("domain_qtype", "")).strip()
        if qtype:
            counts[qtype] += 1
    return dict(counts)


def compute_focus_relation_counts(rows: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    targets = {
        "限红_风控": ["限红和风控", "风控和限红"],
        "RTP_PNL": ["RTP 和 PNL", "RTP和PNL", "PNL 和 RTP", "PNL和RTP"],
        "总代_平台": ["总代和平台", "总代是不是平台", "总代 是不是 平台"],
        "页面可见绑定": ["页面可见信息", "先说页面可见信息", "只根据页面可见信息", "先说这张图里能看到什么"],
    }
    counts = {key: 0 for key in targets}
    for row in rows:
        if (row.get("meta") or {}).get("domain_qtype") != "relation":
            continue
        text = f"{row.get('query', '')}\n{row.get('response', '')}"
        for key, keywords in targets.items():
            if any(keyword in text for keyword in keywords):
                counts[key] += 1
    return counts


def build_pack(input_dir: Path, output_dir: Path, val_ratio: float) -> Dict[str, Any]:
    assets_dir = output_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    glossary_index = read_glossary_index(DEFAULT_DOMAIN_GLOSSARY)

    text_units: List[Dict[str, Any]] = []
    image_records: List[Dict[str, Any]] = []
    source_counter: Counter[str] = Counter()

    for path in sorted(input_dir.iterdir()):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix == ".docx":
            docx_text, docx_images = extract_docx_assets(path, assets_dir / "docx")
            text_units.extend(docx_text)
            image_records.extend(docx_images)
            source_counter["docx_text"] += len(docx_text)
            source_counter["docx_image"] += len(docx_images)
        elif suffix == ".pptx":
            pptx_text, pptx_images = extract_pptx_assets(path, assets_dir / "pptx")
            text_units.extend(pptx_text)
            image_records.extend(pptx_images)
            source_counter["pptx_text"] += len(pptx_text)
            source_counter["pptx_image"] += len(pptx_images)
        elif suffix == ".pdf":
            pdf_text, pdf_images = extract_pdf_assets(path, assets_dir / "pdf")
            text_units.extend(pdf_text)
            image_records.extend(pdf_images)
            source_counter["pdf_text"] += len(pdf_text)
            source_counter["pdf_image"] += len(pdf_images)

    text_rows = build_text_rows(text_units)
    mm_rows = build_mm_rows(image_records, val_ratio=val_ratio, glossary_index=glossary_index)

    debug_units_path = output_dir / "source_units.jsonl"
    debug_images_path = output_dir / "image_records.jsonl"
    text_raw_path = output_dir / "text_sft.raw.jsonl"
    text_clean_path = output_dir / "text_sft.cleaned.jsonl"
    clean_report_dir = output_dir / "clean_reports"
    mm_path = output_dir / "multimodal_train.swift_vl.jsonl"

    write_jsonl(debug_units_path, text_units)
    write_jsonl(debug_images_path, image_records)
    write_jsonl(text_raw_path, text_rows)
    write_jsonl(mm_path, mm_rows)

    clean_summary = clean_dataset(text_raw_path, text_clean_path, clean_report_dir)
    strict_summary = strict_post_clean_text_filter(text_clean_path)

    keywords = [
        "包网",
        "体育包网",
        "风控",
        "限红",
        "串关",
        "危险球",
        "滚球",
        "RTP",
        "PNL",
        "总代",
        "负盈利",
    ]
    summary = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "source_counts": dict(source_counter),
        "text_units": len(text_units),
        "image_records": len(image_records),
        "text_rows_raw": len(text_rows),
        "mm_rows": len(mm_rows),
        "mm_train": sum(1 for row in mm_rows if row["split"] == "train"),
        "mm_val": sum(1 for row in mm_rows if row["split"] == "val"),
        "mm_domain_rows": sum(1 for row in mm_rows if (row.get("meta") or {}).get("domain_term")),
        "mm_domain_qtype_counts": compute_domain_qtype_counts(mm_rows),
        "mm_focus_relation_counts": compute_focus_relation_counts(mm_rows),
        "text_clean_summary": clean_summary,
        "text_strict_summary": strict_summary,
        "text_keyword_coverage": compute_keyword_coverage_text(text_clean_path, keywords),
        "mm_keyword_coverage": compute_keyword_coverage_mm(mm_rows, keywords),
        "text_raw_output": str(text_raw_path),
        "text_clean_output": str(text_clean_path),
        "mm_output": str(mm_path),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build cleaned text + multimodal training pack from /tmp/tydata.")
    parser.add_argument("--input-dir", default=str(DEFAULT_INPUT_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--val-ratio", type=float, default=0.05)
    args = parser.parse_args()

    build_pack(
        input_dir=Path(args.input_dir).resolve(),
        output_dir=Path(args.output_dir).resolve(),
        val_ratio=args.val_ratio,
    )


if __name__ == "__main__":
    main()
