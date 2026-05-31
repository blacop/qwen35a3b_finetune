#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from mm_retrieval import MultiModalRetriever, clean_text


IMG_TOKEN_RE = re.compile(r"image\\s*\\d+\\.(png|jpg|jpeg|webp)", re.IGNORECASE)


DEFAULT_CASES: List[Dict[str, Any]] = [
    {"query": "如何配置后台角色权限", "expected_keywords": ["角色", "权限", "分配"]},
    {"query": "权限管理需要先做什么", "expected_keywords": ["先添加角色", "权限分配"]},
    {"query": "怎么新增管理员账号", "expected_keywords": ["管理员", "角色", "账号"]},
    {"query": "删除部门角色在哪里", "expected_keywords": ["删除", "角色管理", "部门角色"]},
    {"query": "后台登录需要做IP白名单吗", "expected_keywords": ["IP加白", "登录"]},
    {"query": "系统配置里验证码类型在哪", "expected_keywords": ["验证码类型", "系统配置"]},
    {"query": "如何配置短信提供商", "expected_keywords": ["短信", "提供商"]},
    {"query": "在线客服在后台哪里配置", "expected_keywords": ["在线客服", "配置"]},
    {"query": "APP下载地址在系统配置哪里", "expected_keywords": ["APP下载地址", "安卓", "IOS"]},
    {"query": "注册是否需要验证码开关在哪里", "expected_keywords": ["注册", "验证码", "开关"]},
    {"query": "登录验证码开关在什么位置", "expected_keywords": ["登录", "验证码", "开关"]},
    {"query": "重置默认密码相关配置在哪", "expected_keywords": ["重置", "默认密码", "系统配置"]},
    {"query": "如何做资金修正审核", "expected_keywords": ["资金修正", "审核"]},
    {"query": "待审核充值和待审核提款在什么页面", "expected_keywords": ["待审核充值", "待审核提款", "仪表盘"]},
    {"query": "如何处理出款", "expected_keywords": ["出款", "提款", "处理"]},
    {"query": "导入商户功能怎么用", "expected_keywords": ["导入商户", "老会员", "Excel"]},
    {"query": "充值广告图尺寸是多少", "expected_keywords": ["充值广告图", "尺寸"]},
    {"query": "首页轮播图尺寸要求", "expected_keywords": ["首页轮播图", "尺寸"]},
    {"query": "弹窗尺寸怎么配", "expected_keywords": ["弹窗尺寸", "注册页弹窗"]},
    {"query": "管理员GA开关是什么", "expected_keywords": ["GA开关", "管理员"]},
    {"query": "设置通道GA绑定在哪里", "expected_keywords": ["GA绑定", "通道"]},
    {"query": "风险用户验证方式在哪改", "expected_keywords": ["风险用户", "验证方式"]},
    {"query": "会员前端验证登录怎么配置", "expected_keywords": ["会员前端验证登录", "手机号", "验证码"]},
    {"query": "提款支付手动出款额度限制有哪些", "expected_keywords": ["提款额度", "手动出款", "额度"]},
    {"query": "活动额度和审核额度的区别", "expected_keywords": ["活动额度", "审核额度"]},
]


def load_cases(path: str | None) -> List[Dict[str, Any]]:
    if not path:
        return DEFAULT_CASES
    p = Path(path)
    rows: List[Dict[str, Any]] = []
    if p.suffix.lower() in {".json", ".jsonl"}:
        if p.suffix.lower() == ".json":
            data = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(data, list):
                rows = data
        else:
            for line in p.read_text(encoding="utf-8").splitlines():
                s = line.strip()
                if s:
                    rows.append(json.loads(s))
    if not rows:
        raise RuntimeError(f"No valid cases loaded from: {path}")
    return rows


def quantile(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    xs = sorted(values)
    idx = int(round((len(xs) - 1) * q))
    return float(xs[max(0, min(len(xs) - 1, idx))])


def duplicate_ratio(hits: List[Dict[str, Any]], sig_len: int = 100) -> float:
    if not hits:
        return 0.0
    sigs = []
    for h in hits:
        u = h.get("unit", {})
        raw = clean_text(f"{u.get('title', '')} {u.get('text', '')}").lower()
        raw = IMG_TOKEN_RE.sub(" ", raw)
        raw = clean_text(raw)
        sigs.append(raw[:sig_len])
    unique_cnt = len(set(sigs))
    return 1.0 - (float(unique_cnt) / float(len(sigs)))


def keyword_mrr(hits: List[Dict[str, Any]], expected_keywords: List[str], top_n: int = 3) -> float:
    kws = [k.strip().lower() for k in expected_keywords if k and str(k).strip()]
    if not kws:
        return 0.0
    for i, h in enumerate(hits[:top_n], start=1):
        u = h.get("unit", {})
        hay = clean_text(f"{u.get('title', '')} {u.get('text', '')}").lower()
        if any(k in hay for k in kws):
            return 1.0 / float(i)
    return 0.0


def has_keyword_hit(hits: List[Dict[str, Any]], expected_keywords: List[str], top_n: int = 3) -> bool:
    return keyword_mrr(hits, expected_keywords, top_n=top_n) > 0.0


def eval_variant(
    name: str,
    retriever: MultiModalRetriever,
    cases: List[Dict[str, Any]],
    search_kwargs: Dict[str, Any],
    top_k: int,
) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    latencies: List[float] = []

    for idx, c in enumerate(cases, start=1):
        query = str(c.get("query", "")).strip()
        expected_keywords = list(c.get("expected_keywords", []))

        t0 = time.perf_counter()
        res = retriever.search(query=query, top_k=top_k, **search_kwargs)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        latencies.append(latency_ms)

        hits = list(res.get("fused_hits", []))
        top1_modality = "none"
        top1_source_type = "none"
        top1_score = 0.0
        if hits:
            top1_modality = str(hits[0].get("unit", {}).get("modality", "none"))
            top1_source_type = str(hits[0].get("unit", {}).get("source_type", "none"))
            top1_score = float(hits[0].get("score", 0.0))

        num_images_topk = sum(1 for h in hits if str(h.get("unit", {}).get("modality", "")) == "image")
        kw_mrr = keyword_mrr(hits, expected_keywords, top_n=3)
        kw_hit = kw_mrr > 0.0

        rows.append(
            {
                "variant": name,
                "case_id": idx,
                "query": query,
                "expected_keywords": "|".join(expected_keywords),
                "top1_modality": top1_modality,
                "top1_source_type": top1_source_type,
                "top1_score": top1_score,
                "top1_is_text": int(top1_modality == "text"),
                "keyword_hit_top3": int(kw_hit),
                "keyword_mrr_top3": kw_mrr,
                "duplicate_ratio_topk": duplicate_ratio(hits),
                "num_hits": len(hits),
                "num_image_hits_topk": num_images_topk,
                "latency_ms": latency_ms,
            }
        )

    summary = {
        "variant": name,
        "index_meta": retriever.meta,
        "n_cases": len(rows),
        "top1_text_rate": float(sum(r["top1_is_text"] for r in rows) / max(1, len(rows))),
        "keyword_hit_rate_top3": float(sum(r["keyword_hit_top3"] for r in rows) / max(1, len(rows))),
        "keyword_mrr_top3_avg": float(sum(r["keyword_mrr_top3"] for r in rows) / max(1, len(rows))),
        "duplicate_ratio_topk_avg": float(sum(r["duplicate_ratio_topk"] for r in rows) / max(1, len(rows))),
        "image_hits_topk_avg": float(sum(r["num_image_hits_topk"] for r in rows) / max(1, len(rows))),
        "latency_ms_mean": float(statistics.mean(latencies) if latencies else 0.0),
        "latency_ms_p50": quantile(latencies, 0.50),
        "latency_ms_p95": quantile(latencies, 0.95),
        "search_kwargs": search_kwargs,
    }
    return {"rows": rows, "summary": summary}


def save_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare baseline vs improved multimodal retrieval")
    parser.add_argument("--baseline-index", required=True)
    parser.add_argument("--improved-index", required=True)
    parser.add_argument("--cases", default="")
    parser.add_argument("--out-dir", default="/home/ubuntu/generate/tydata_rag/mm_rag/eval_outputs")
    parser.add_argument("--top-k", type=int, default=8)
    args = parser.parse_args()

    cases = load_cases(args.cases or None)
    baseline = MultiModalRetriever(args.baseline_index)
    improved = MultiModalRetriever(args.improved_index)

    baseline_kwargs = {
        "top_k_text": 20,
        "top_k_image": 20,
        "text_weight": 0.65,
        "image_weight": 0.35,
        "min_score_text": 0.01,
        "min_score_image": 0.05,
        "prefer_text_for_text_query": False,
        "enable_rerank": False,
        "enable_dedup": False,
    }
    improved_kwargs = {
        "top_k_text": 20,
        "top_k_image": 20,
        "text_weight": 0.65,
        "image_weight": 0.35,
        "min_score_text": 0.01,
        "min_score_image": 0.05,
        "prefer_text_for_text_query": True,
        "enable_rerank": True,
        "enable_dedup": True,
        "dedup_text_signature_len": 100,
        "image_dedup_sim_threshold": 0.985,
    }

    b = eval_variant("baseline", baseline, cases, baseline_kwargs, top_k=args.top_k)
    n = eval_variant("improved", improved, cases, improved_kwargs, top_k=args.top_k)

    rows = b["rows"] + n["rows"]
    bs = b["summary"]
    ns = n["summary"]

    delta = {
        "top1_text_rate": float(ns["top1_text_rate"] - bs["top1_text_rate"]),
        "keyword_hit_rate_top3": float(ns["keyword_hit_rate_top3"] - bs["keyword_hit_rate_top3"]),
        "keyword_mrr_top3_avg": float(ns["keyword_mrr_top3_avg"] - bs["keyword_mrr_top3_avg"]),
        "duplicate_ratio_topk_avg": float(ns["duplicate_ratio_topk_avg"] - bs["duplicate_ratio_topk_avg"]),
        "image_hits_topk_avg": float(ns["image_hits_topk_avg"] - bs["image_hits_topk_avg"]),
        "latency_ms_mean": float(ns["latency_ms_mean"] - bs["latency_ms_mean"]),
        "latency_ms_p50": float(ns["latency_ms_p50"] - bs["latency_ms_p50"]),
        "latency_ms_p95": float(ns["latency_ms_p95"] - bs["latency_ms_p95"]),
    }

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(args.out_dir) / f"comparison_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    save_csv(out_dir / "comparison_cases.csv", rows)
    summary = {
        "generated_at_utc": ts,
        "n_cases": len(cases),
        "baseline": bs,
        "improved": ns,
        "delta_improved_minus_baseline": delta,
    }
    (out_dir / "comparison_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(str(out_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
