#!/usr/bin/env python3
"""AB test v31d (GPU5 :8014) vs GPU7 baseline (:8000)"""
import json
import time
import tempfile
import subprocess
from pathlib import Path
from collections import Counter

PROJECT_ROOT = Path("/home/ubuntu/qwen35a3b_finetune")
EVAL_SET = PROJECT_ROOT / "rag" / "eval" / "ab_eval_set.jsonl"
OUT_DIR = PROJECT_ROOT / "outputs" / f"ab_v31d_vs_gpu7_{int(time.time())}"

API_KEY = "sk-h2byzMZ66c53C8lZcz-9wcvN0Cwhny0AC7gf0qcOeksYCj-k"
SYSTEM_PROMPT = "你是体育包网客服，按标准流程处理注单异常，无法确认时提交后台并同步运营联系包网。输出必须是标准JSON格式，禁止输出任何推理过程。"

MODELS = {
    "v31d_gpu5": {"url": "http://127.0.0.1:8014", "model": "qwen35a3b-domain-text-vl-gpu5"},
    "gpu7_baseline": {"url": "http://127.0.0.1:8000", "model": "qwen35a3b-domain-text-vl-gpu7"},
}


def read_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def call_model(model_name, query):
    config = MODELS[model_name]
    body = {
        "model": config["model"],
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ],
        "max_tokens": 512,
        "temperature": 0,
    }
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".json", delete=False) as tmp:
        tmp.write(json.dumps(body, ensure_ascii=False))
        tmp_path = tmp.name

    safe_url = config["url"].rstrip("/") + "/v1/chat/completions"
    cmd = (
        f"curl -sS -m 60 "
        f"-H 'Content-Type: application/json' "
        f"-H 'Authorization: Bearer {API_KEY}' "
        f"'{safe_url}' --data-binary '@{tmp_path}'"
    )

    t0 = time.time()
    proc = subprocess.run(["bash", "-lc", cmd], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    Path(tmp_path).unlink(missing_ok=True)
    latency_s = round(time.time() - t0, 3)

    if proc.returncode != 0:
        return {"ok": False, "error": proc.stderr.strip(), "latency_s": latency_s}

    try:
        data = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return {"ok": False, "error": "invalid_json_response", "raw": proc.stdout[:500], "latency_s": latency_s}

    try:
        content = data["choices"][0]["message"]["content"]
    except Exception:
        return {"ok": False, "error": "missing_choices", "raw": data, "latency_s": latency_s}

    has_thinking = ("think" in content.lower()[:50] or "Thinking" in content or "<think>" in content)

    return {
        "ok": True,
        "content": content,
        "has_thinking": has_thinking,
        "latency_s": latency_s,
    }


def score_response(row, content):
    must_include_hits = sum(1 for term in row["must_include"] if term in content)
    must_not_hits = sum(1 for term in row["must_not_include"] if term in content)
    intent_hit = 1 if row["gold_intent"] in content else 0
    return {
        "must_include_hit": must_include_hits / len(row["must_include"]) if row["must_include"] else 1.0,
        "must_not_hit": must_not_hits == 0,
        "intent_hit": intent_hit,
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    eval_rows = read_jsonl(EVAL_SET)
    results = []

    for i, row in enumerate(eval_rows, 1):
        print(f"[{i}/{len(eval_rows)}] {row['id']}: {row['user_query'][:60]}...")

        v31d_result = call_model("v31d_gpu5", row["user_query"])
        gpu7_result = call_model("gpu7_baseline", row["user_query"])

        results.append({
            "id": row["id"],
            "scenario": row["scenario"],
            "user_query": row["user_query"],
            "gold_intent": row["gold_intent"],
            "v31d": {
                **v31d_result,
                "score": score_response(row, v31d_result.get("content", "")) if v31d_result.get("ok") else None,
            },
            "gpu7": {
                **gpu7_result,
                "score": score_response(row, gpu7_result.get("content", "")) if gpu7_result.get("ok") else None,
            },
        })

    # Save results
    with open(OUT_DIR / "results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # Calculate summary
    def safe_avg(items, key):
        vals = [item.get(key, 0) for item in items if item]
        return sum(vals) / max(len(vals), 1)

    v31d_scores = [r["v31d"]["score"] for r in results if r["v31d"].get("score")]
    gpu7_scores = [r["gpu7"]["score"] for r in results if r["gpu7"].get("score")]

    v31d_thinking = [r["v31d"].get("has_thinking", False) for r in results if r["v31d"].get("ok")]
    gpu7_thinking = [r["gpu7"].get("has_thinking", False) for r in results if r["gpu7"].get("ok")]

    summary = {
        "total_samples": len(results),
        "v31d": {
            "avg_intent_hit": safe_avg(v31d_scores, "intent_hit"),
            "avg_must_include": safe_avg(v31d_scores, "must_include_hit"),
            "must_not_compliance_rate": safe_avg(v31d_scores, "must_not_hit"),
            "thinking_rate": sum(v31d_thinking) / max(len(v31d_thinking), 1),
            "error_rate": 1 - len(v31d_scores) / max(len(results), 1),
        },
        "gpu7": {
            "avg_intent_hit": safe_avg(gpu7_scores, "intent_hit"),
            "avg_must_include": safe_avg(gpu7_scores, "must_include_hit"),
            "must_not_compliance_rate": safe_avg(gpu7_scores, "must_not_hit"),
            "thinking_rate": sum(gpu7_thinking) / max(len(gpu7_thinking), 1),
            "error_rate": 1 - len(gpu7_scores) / max(len(results), 1),
        },
    }

    with open(OUT_DIR / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n=== SUMMARY ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nResults saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
