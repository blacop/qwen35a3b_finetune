#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


PROJECT_ROOT = Path("/home/ubuntu/qwen35a3b_finetune")
IMAGE_RECORDS = PROJECT_ROOT / "datasets" / "tydata_domain_multimodal_pack" / "image_records.jsonl"
DEFAULT_OUT = PROJECT_ROOT / "rag" / "eval" / "domain_text_vl_ab" / "eval_set.jsonl"


SPECS: List[Dict[str, Any]] = [
    {
        "record_id": "docx_img_0001",
        "dimension": "comparison",
        "query": "这张图更接近 PNL 统计页还是 RTP 配置页？请说明理由。",
        "rubric_groups": [["PNL", "盈亏"], ["RTP", "返奖率"], ["统计", "仪表盘", "运营数据"]],
    },
    {
        "record_id": "docx_img_0001",
        "dimension": "comparison",
        "query": "包网和白标有什么区别？结合这张后台图简要说明。",
        "rubric_groups": [["包网"], ["白标"], ["整套能力", "技术和运营能力", "系统能力"], ["牌照", "复用", "挂品牌"]],
    },
    {
        "record_id": "docx_img_0007",
        "dimension": "relation",
        "query": "限红和风控是什么关系？结合这张后台图说明。",
        "rubric_groups": [["限红"], ["风控"], ["额度限制", "限制下注", "限额"], ["风控的一部分", "落地动作", "执行方式"]],
    },
    {
        "record_id": "docx_img_0116",
        "dimension": "misconception",
        "query": "总代是不是平台？结合这张代理相关页面说明。",
        "rubric_groups": [["总代"], ["平台"], ["代理", "下级代理"], ["不是平台", "不一定是平台"]],
    },
    {
        "record_id": "docx_img_0116",
        "dimension": "misconception",
        "query": "负盈利是玩家亏损还是平台亏损？结合这张代理后台图说明。",
        "rubric_groups": [["负盈利"], ["会员亏损", "玩家亏损"], ["平台盈利"], ["代理分润", "分润"]],
    },
    {
        "record_id": "docx_img_0136",
        "dimension": "comparison",
        "query": "这张图里更能说明代理还是包网商？两者区别是什么？",
        "rubric_groups": [["代理"], ["包网商"], ["推广链接", "专属域名", "渠道"], ["技术方", "底层能力", "服务方"]],
    },
    {
        "record_id": "docx_img_0153",
        "dimension": "definition",
        "query": "三方是什么意思？这张补单页和三方是什么关系？",
        "rubric_groups": [["三方"], ["第三方", "外部"], ["补单", "回调"], ["支付", "通道", "订单"]],
    },
    {
        "record_id": "pptx_img_0001",
        "dimension": "comparison",
        "query": "RTP 和 PNL 有什么区别？结合这张体育玩法图说明。",
        "rubric_groups": [["RTP", "返奖率"], ["PNL", "盈亏"], ["长期", "理论"], ["实际盈亏", "统计"]],
    },
    {
        "record_id": "pptx_img_0001",
        "dimension": "definition",
        "query": "这张图能体现体育包网是什么吗？请简要解释。",
        "rubric_groups": [["体育包网"], ["盘口", "赔率"], ["玩法", "体育赛事"], ["滚球", "串关", "结算"]],
    },
    {
        "record_id": "pptx_img_0006",
        "dimension": "definition",
        "query": "串关是什么意思？这张图里的综合过关和串关是什么关系？",
        "rubric_groups": [["串关"], ["综合过关"], ["组合投注", "多场赛事"], ["全赢", "赔率相乘"]],
    },
    {
        "record_id": "pptx_img_0004",
        "dimension": "definition",
        "query": "危险球是什么意思？为什么会待确认或被取消？",
        "rubric_groups": [["危险球"], ["待确认"], ["取消"], ["点球", "角球", "自由球", "进攻"]],
    },
    {
        "record_id": "pptx_img_0001",
        "dimension": "comparison",
        "query": "香港盘和欧洲盘有什么区别？为什么这类图能说明体育包网能力？",
        "rubric_groups": [["香港盘"], ["欧洲盘"], ["赔率", "本金"], ["体育包网", "盘口", "玩法"]],
    },
]


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a small AB eval set for text-vs-VL domain comparison.")
    parser.add_argument("--image-records", default=str(IMAGE_RECORDS))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args()

    records = {row["id"]: row for row in read_jsonl(Path(args.image_records))}
    rows: List[Dict[str, Any]] = []

    for idx, spec in enumerate(SPECS, 1):
        record = records.get(spec["record_id"])
        if not record:
            raise KeyError(f"missing record_id: {spec['record_id']}")
        lines = record.get("lines") or []
        rows.append(
            {
                "id": f"domain_vl_ab_{idx:03d}",
                "record_id": spec["record_id"],
                "image_path": record["image_path"],
                "source_file": record["source_file"],
                "topic_hint": "；".join(lines[:3]),
                "ocr_context": "\n".join(lines[:6]),
                "query": spec["query"],
                "dimension": spec["dimension"],
                "rubric_groups": spec["rubric_groups"],
            }
        )

    write_jsonl(Path(args.out), rows)
    summary = {
        "total": len(rows),
        "output": str(Path(args.out)),
        "dimensions": sorted({row["dimension"] for row in rows}),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
