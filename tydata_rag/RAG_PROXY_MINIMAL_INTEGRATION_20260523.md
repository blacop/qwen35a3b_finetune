# RAG Proxy Minimal Integration

## Goal

把 `tydata_rag/rag_api.py` 作为独立检索层挂到 GPU5/GPU7 模型前面，形成最小联调链路：

`Client -> RAG API -> retrieve -> 拼接 context -> GPU5/GPU7 /chat/completions -> 返回结果`

这样做的目的不是替代现有代理，而是先验证：

- 体育包网知识题是否因 RAG 明显提升
- 检索与模型路由是否能稳定工作
- 后续是否真的需要进入 Graph RAG / KG 阶段

## Current Entry

现有服务入口已经具备最小联调能力：

- `/retrieve`
- `/answer`
- `/v1/chat/completions`

其中：

- `/retrieve` 只做检索
- `/answer` 做检索 + 调模型
- `/v1/chat/completions` 走 OpenAI 兼容格式，最适合挂到现有代理后面

## GPU5 Example

环境文件当前已经支持 GPU5：

```env
RAG_INDEX_DIR=/video-storage/ai-customer/qwen35a3b_finetune/rag/typlay/rag_index_typlay_default_560_112
LLM_API_BASE=http://127.0.0.1:8014/v1
LLM_MODEL=qwen35a3b-domain-text-vl-gpu5
RAG_API_PORT=18080
RAG_LOCK_RETRIEVAL_PARAMS=1
```

启动：

```bash
set -a
source /home/ubuntu/tydata_rag/rag_api.env
set +a
/home/ubuntu/tydata_rag/start_rag_api.sh
```

## GPU7 Example

如果要切到 GPU7，只改模型上游：

```env
LLM_API_BASE=http://127.0.0.1:8015/v1
LLM_MODEL=qwen35a3b-domain-text-vl-gpu7
```

RAG 层不需要改代码。

## Minimal Proxy Routing

如果上层代理已经能按 path 或 model 转发，可以直接加两个稳定别名：

- `baowang-gpu5-rag`
- `baowang-gpu7-rag`

推荐映射：

- `baowang-gpu5-rag -> http://127.0.0.1:18080/v1/chat/completions`
- `baowang-gpu7-rag -> http://127.0.0.1:18081/v1/chat/completions`

其中：

- `18080` 绑定 GPU5 上游
- `18081` 绑定 GPU7 上游

## Request Shape

上层代理对 RAG 服务发 OpenAI 兼容请求即可：

```json
{
  "model": "baowang-gpu5-rag",
  "messages": [
    {"role": "user", "content": "香港盘和欧洲盘有什么区别？"}
  ],
  "temperature": 0,
  "max_tokens": 256
}
```

## Validation Path

联调阶段建议先只验证三类题：

1. 单文档知识抽取题
2. 多证据聚合题
3. 易被通用模型“自由发挥”的规则题

注意：

- 默认线上参数应先关闭 `enable_diversity`
- 文档族去重/多样性重排更适合用于“多证据聚合题”专项路由
- 否则会牺牲单文档题的 `Recall@1`

验证接口：

```bash
curl -sS http://127.0.0.1:18080/v1/chat/completions \
  -H 'Authorization: Bearer <RAG_API_KEY>' \
  -H 'Content-Type: application/json' \
  -d '{
    "model":"baowang-gpu5-rag",
    "messages":[
      {"role":"user","content":"PT、VG、bbin、FG 这四份百家乐规则里，是否都使用 8 副牌？"}
    ],
    "temperature":0,
    "max_tokens":256
  }'
```

## Decision Rule

只有在以下条件同时满足时，再考虑 Graph RAG / KG：

- 多证据专项集 `full_evidence_cover_rate` 仍明显不足
- 文档族去重 + 多样性重排后仍无法稳定提升
- 失败原因确实来自“跨节点关系推理”，而不是检索排序或题集口径

否则优先继续优化：

- query routing
- rerank
- source diversity
- evidence aggregation

## Current Recommendation

截至 2026-05-23 当前实验结论：

- 默认主链路：`enable_diversity=0`
- 多证据专项评测/候选路由：`enable_diversity=1`
- 不建议直接把多样性策略全量上线到所有知识题
