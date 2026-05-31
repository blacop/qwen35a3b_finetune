#!/usr/bin/env python3
"""端到端工具调用评测：评估模型是否能正确识别意图、调用 tool、返回包含真实数据的答案"""
import json, sys, time, urllib.request, argparse
from concurrent.futures import ThreadPoolExecutor

def call_endpoint(url, model, query, api_key=None, use_agent=False, timeout=60):
    payload = {
        "model": model,
        "messages": [{"role":"user","content":query}],
        "max_tokens": 500,
        "temperature": 0.0,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    headers = {"Content-Type":"application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers=headers)
    t0 = time.time()
    try:
        resp = urllib.request.urlopen(req, timeout=timeout).read().decode()
        r = json.loads(resp)
        elapsed = (time.time() - t0) * 1000
        if "choices" not in r: return {"error": r, "latency_ms": elapsed}
        msg = r["choices"][0]["message"]
        return {
            "content": msg.get("content") or "",
            "tool_calls": msg.get("tool_calls") or [],
            "agent_trace": r.get("_agent", {}).get("trace", []),
            "latency_ms": elapsed,
        }
    except Exception as e:
        return {"error": str(e), "latency_ms": (time.time()-t0)*1000}

def evaluate(sample, result):
    expected_tool = sample["expected_tool"]
    expected_args = sample["expected_args"]
    must_inc = sample.get("must_include", [])
    
    tool_called = False
    tool_args_match = False
    if result.get("tool_calls"):
        for tc in result["tool_calls"]:
            if tc.get("function",{}).get("name") == expected_tool:
                tool_called = True
                try:
                    args = json.loads(tc["function"]["arguments"])
                    if all(args.get(k) == v for k, v in expected_args.items()):
                        tool_args_match = True
                except: pass
    if not tool_called and result.get("agent_trace"):
        for t in result["agent_trace"]:
            if t.get("tool") == expected_tool:
                tool_called = True
                if all(t.get("args",{}).get(k) == v for k, v in expected_args.items()):
                    tool_args_match = True
    
    content = result.get("content") or ""
    must_inc_hit = sum(1 for w in must_inc if w in content) / max(len(must_inc),1)
    
    return {
        "tool_called": int(tool_called),
        "tool_args_match": int(tool_args_match),
        "must_inc_score": must_inc_hit,
        "overall": (int(tool_called) * 0.4 + int(tool_args_match) * 0.3 + must_inc_hit * 0.3),
    }

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input-jsonl", required=True)
    p.add_argument("--url", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--api-key", default="")
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()
    
    with open(args.input_jsonl, encoding="utf-8") as f:
        samples = [json.loads(line) for line in f if line.strip()]
    
    def run_one(s):
        r = call_endpoint(args.url, args.model, s["user_query"], args.api_key)
        e = evaluate(s, r)
        return {**s, "result": r, "scores": e}
    
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        results = list(ex.map(run_one, samples))
    
    n = len(results)
    avg_tool = sum(r["scores"]["tool_called"] for r in results) / n
    avg_args = sum(r["scores"]["tool_args_match"] for r in results) / n
    avg_must = sum(r["scores"]["must_inc_score"] for r in results) / n
    avg_overall = sum(r["scores"]["overall"] for r in results) / n
    
    print(f"\n=== 工具调用评测：{args.url} ===")
    print(f"样本数: {n}")
    print(f"工具调用率 (tool_called):    {avg_tool:.4f}")
    print(f"参数正确率 (args_match):     {avg_args:.4f}")
    print(f"答案关键词命中 (must_inc):   {avg_must:.4f}")
    print(f"综合分 (overall):            {avg_overall:.4f}")
    return avg_overall, avg_tool

if __name__ == "__main__":
    main()
