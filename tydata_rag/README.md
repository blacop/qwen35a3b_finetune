# tydata local RAG (Hybrid)

Build index:

```bash
/usr/bin/python3 /home/ubuntu/tydata_rag/build_rag_index.py \
  --input-dir /tmp/tydata \
  --output-dir /home/ubuntu/tydata_rag/index \
  --dense-backend auto \
  --dense-dim 256
```

Query top chunks:

```bash
/usr/bin/python3 /home/ubuntu/tydata_rag/query_rag.py \
  --index-dir /home/ubuntu/tydata_rag/index \
  --query "足球串关玩法是什么" \
  --top-k 5 \
  --mode hybrid \
  --fusion rrf
```

Generate a prompt with retrieved context (feed this prompt to your model):

```bash
/usr/bin/python3 /home/ubuntu/tydata_rag/query_rag.py \
  --index-dir /home/ubuntu/tydata_rag/index \
  --query "五大赛事有哪些核心规则" \
  --top-k 5 \
  --mode hybrid \
  --output prompt
```

Output artifacts in index dir:

- `chunks.jsonl`: chunked text + metadata
- `tfidf_matrix.npz`: sparse TF-IDF index
- `vectorizer.pkl`: vectorizer for query encoding
- `dense_vectors.npy`: dense vector index
- `dense_meta.json`: dense backend metadata
- `dense_word_vectorizer.pkl` + `dense_svd.pkl`: SVD dense artifacts
- `meta.json`: basic stats

## Local API

Start API service:

```bash
export RAG_INDEX_DIR=/home/ubuntu/tydata_rag/index
export LLM_API_BASE=http://127.0.0.1:8001/v1
export LLM_MODEL=your-model-name
# export LLM_API_KEY=optional

uvicorn rag_api:app --host 0.0.0.0 --port 18080 --app-dir /home/ubuntu/tydata_rag
```

Or one command:

```bash
LLM_API_BASE=http://127.0.0.1:8001/v1 \
LLM_MODEL=your-model-name \
/home/ubuntu/tydata_rag/start_rag_api.sh
```

Health check:

```bash
curl -sS http://127.0.0.1:18080/health | jq .
```

Retrieve endpoint:

```bash
curl -sS http://127.0.0.1:18080/retrieve \
  -H 'Content-Type: application/json' \
  -d '{
    "query":"世界杯和五大联赛",
    "top_k":5,
    "min_score":0.01,
    "retrieval_mode":"hybrid",
    "fusion":"rrf",
    "candidate_k":40,
    "enable_rerank":true
  }' | jq .
```

Answer endpoint (RAG + model):

```bash
curl -sS http://127.0.0.1:18080/answer \
  -H 'Content-Type: application/json' \
  -d '{
    "query":"串关怎么玩",
    "top_k":5,
    "include_context":true
  }' | jq .
```

Per-request LLM override:

```json
{
  "query": "串关怎么玩",
  "llm_base_url": "http://127.0.0.1:8001/v1",
  "model": "your-model-name",
  "api_key": "optional"
}
```

## A/B Compare (TF-IDF vs Hybrid)

```bash
set -a
source /home/ubuntu/tydata_rag/rag_api.env
set +a

/usr/bin/python3 /home/ubuntu/tydata_rag/ab_compare_retrieval.py \
  --api-base http://127.0.0.1:18080 \
  --api-key "$RAG_API_KEY" \
  --dataset /home/ubuntu/qwen35a3b_finetune/datasets/eval_sports_customer_prod_500.jsonl \
  --modes tfidf,hybrid \
  --top-k 5 \
  --fusion rrf
```

Outputs:

- `ab_results_long.csv`: per-query per-mode details
- `ab_results_pair.csv`: query-level pairwise delta
- `summary.json`: summary metrics and deltas

## Production Hardening

`rag_api.env` now supports fixed retrieval defaults (locked by default):

```env
RAG_LOCK_RETRIEVAL_PARAMS=1
RAG_DEFAULT_TOP_K=5
RAG_DEFAULT_MIN_SCORE=0.01
RAG_DEFAULT_MIN_SCORE_VEC=0.0
RAG_DEFAULT_MODE=hybrid
RAG_DEFAULT_FUSION=rrf
RAG_DEFAULT_CANDIDATE_K=40
RAG_DEFAULT_RRF_K=60
RAG_DEFAULT_WEIGHT_LEX=0.55
RAG_DEFAULT_WEIGHT_VEC=0.45
RAG_DEFAULT_ENABLE_RERANK=1
```

When locked, request payload retrieval knobs are ignored and forced to these defaults.

### Gray Switch

Use the helper script to toggle gray traffic without manual file edits:

```bash
# enable 20% canary traffic
/usr/bin/python3 /home/ubuntu/tydata_rag/set_rag_gray.py \
  --enable 1 \
  --ratio 0.2 \
  --canary-index-dir /home/ubuntu/tydata_rag/index_sbert_20260421T065404Z \
  --restart

# disable gray
/usr/bin/python3 /home/ubuntu/tydata_rag/set_rag_gray.py \
  --enable 0 \
  --ratio 0 \
  --restart
```

Optional request header:

- `X-Rag-Route: stable|canary` to force route for probing.

### Auto Rollback Guard

Install and start periodic guard (every 2 minutes):

```bash
/home/ubuntu/tydata_rag/install_rag_guard_timer.sh
```

Guard config file:

- `/home/ubuntu/tydata_rag/rag_guard.env`

Guard outputs:

- `/home/ubuntu/tydata_rag/runtime/rag_guard/events.jsonl`
- `/home/ubuntu/tydata_rag/runtime/rag_guard/guard.log`
