# Gateway Deployment And Architecture

Date: 2026-05-25

## 1. Summary

The current gateway is a two-layer design rather than a single proxy:

`Client -> external_api_gateway -> dpo_guardrails_proxy (gpu5/gpu7) -> upstream vLLM`

There is also a parallel direct RAG route for the `-rag` public aliases:

`Client -> external_api_gateway -> RAG API`

In practice, the responsibilities are split as follows:

- `external_api_gateway` is the public ingress.
- `dpo_guardrails_proxy` is the internal model proxy and policy layer.
- `vLLM` is the actual model serving backend.
- `RAG API` is exposed as a separate target for the public `-rag` aliases.

## 2. Main Components

### 2.1 External Public Gateway

Code:

- `scripts/external_api_gateway.py`

Config:

- `services/external-api-gateway.env`

Primary responsibilities:

- Validate external API keys.
- Enforce per-key route and model permissions.
- Apply per-key RPM rate limits.
- Expose stable public model aliases.
- Inject alias-level system prompt when the request does not already contain a real system prompt.
- Forward requests to internal proxies or direct RAG backends.
- Rewrite the response `model` field back to the public alias.
- Write metadata-only audit logs.

Public endpoints:

- `GET /health`
- `GET /v1/models`
- `POST /v1/chat/completions`
- `POST /v1/chat/completions/agent`

### 2.2 Internal Guardrails Proxy

Code:

- `scripts/dpo_guardrails_proxy.py`

Config:

- `services/dpo-guardrails-proxy.env`
- `services/dpo-guardrails-proxy-gpu5.env`

Primary responsibilities:

- Accept OpenAI-compatible chat requests from the external gateway.
- Remap public/internal model aliases to upstream model IDs.
- Force-inject model-specific system prompts.
- Inject glossary knowledge for definition-like queries.
- Inject RAG context for supported knowledge queries.
- Run multimodal VL RAG fallback when the first answer is weak.
- Apply security guardrails to input and output.
- Perform intent classification and front-route shortcutting for customer-service cases.
- Support agent mode with tool schema injection and tool-call loop.
- Expose a raw passthrough endpoint for evaluation.

Internal endpoints:

- `GET /health`
- `GET /v1/models`
- `POST /v1/chat/completions`
- `POST /v1/chat/completions/agent`
- `POST /v1/chat/completions/raw`
- `GET /v1/agent/health`

## 3. Current Deployment Topology

### 3.1 Public Alias Routing

From `services/external-api-gateway.env`:

| Public alias | Target base URL | Public behavior |
| --- | --- | --- |
| `baowang-gpu5` | `http://127.0.0.1:8001/v1` | Normal proxy path |
| `baowang-gpu7` | `http://127.0.0.1:8010/v1` | Normal proxy path |
| `baowang-gpu5-rag` | `http://127.0.0.1:18080/v1` | Direct RAG path |
| `baowang-gpu7-rag` | `http://127.0.0.1:18081/v1` | Direct RAG path |

Notes:

- `baowang-gpu5` and `baowang-gpu7` go through the internal `dpo_guardrails_proxy`.
- `baowang-gpu5-rag` and `baowang-gpu7-rag` do not use the normal proxy path; they are routed directly to the RAG service.
- The `-rag` aliases are configured with a backend API key and `allow_agent=false`.

### 3.2 Internal Proxy Upstreams

From `services/dpo-guardrails-proxy.env` and `services/dpo-guardrails-proxy-gpu5.env`:

| Internal proxy | Upstream base URL | Default public model | Upstream model override |
| --- | --- | --- | --- |
| GPU7 proxy | `http://127.0.0.1:8013/v1` | `gpu7-v5-combined` | `qwen35a3b-sft-v6-1-ops` |
| GPU5 proxy | `http://127.0.0.1:8014/v1` | `gpu5-v5.1f-merged` | `qwen35a3b-domain-text-vl-gpu5` |

Additional deployment notes:

- GPU7 agent requests also use `http://127.0.0.1:8013/v1`.
- GPU5 agent requests also use `http://127.0.0.1:8014/v1`.
- GPU5 has tool fallback enabled to `http://127.0.0.1:8013/v1`.
- Both proxies have RAG, multimodal RAG, glossary, and security guard features enabled.

## 4. End-To-End Request Flow

### 4.1 Standard Chat Path

1. Client sends a request to the external gateway with a public alias such as `baowang-gpu5`.
2. The external gateway authenticates the API key and checks whether that key can access the requested alias and route.
3. The external gateway injects an alias-level system prompt if the request does not already contain a real system prompt.
4. The external gateway rewrites the requested model alias to the configured upstream model name for that route and forwards the request.
5. The internal proxy receives the request and applies model alias mapping again, then force-injects the model-specific system prompt.
6. The internal proxy may inject glossary references or RAG references into the system message.
7. The internal proxy may short-circuit some customer-service intents through front-route templates instead of calling the model.
8. If the request reaches the model backend, the internal proxy forwards to vLLM with `thinking` disabled.
9. The internal proxy may apply VL RAG fallback for multimodal weak answers.
10. The internal proxy applies output guardrails and rewrites the response model field.
11. The external gateway rewrites the final response `model` field back to the public alias before returning it to the client.

### 4.2 Agent Path

1. Client calls `POST /v1/chat/completions/agent` on the external gateway.
2. The external gateway verifies that the selected alias allows agent mode.
3. The request is forwarded to the internal proxy agent endpoint.
4. The internal proxy injects tool schema, runs tool-call dispatch loops, and then returns the final model answer.

### 4.3 Direct RAG Path

1. Client selects `baowang-gpu5-rag` or `baowang-gpu7-rag`.
2. The external gateway authenticates the request and forwards directly to the configured RAG backend on `18080` or `18081`.
3. This path bypasses the normal `8001` and `8010` internal proxy entrypoints.

## 5. What Each Layer Owns

### External Gateway Owns

- External authentication.
- Route and model permission control.
- Public alias exposure.
- Rate limiting.
- Stable public API contract.
- Metadata audit logging.

### Internal Proxy Owns

- Prompt shaping and model override logic.
- Glossary injection.
- RAG injection and multimodal RAG fallback.
- Intent routing and front-route shortcuts.
- Security guardrails.
- Output post-processing.
- Agent and tools execution loop.

### Upstream Model Server Owns

- Actual text or multimodal generation.
- Streaming and standard OpenAI-compatible completion behavior.

## 6. Prompt Configuration

Both layers load alias/model prompt configuration from:

- `services/model-system-prompts.json`

Behavior split:

- The external gateway injects a prompt only when the request has no real system prompt.
- The internal proxy force-injects or merges the prompt on the model path.

This means prompt control is intentionally duplicated:

- the outer layer protects the public API contract,
- the inner layer enforces final model behavior.

## 7. Operational Notes

- External gateway audit log:
  - `/video-storage/ai-customer/qwen35a3b_finetune/runtime/external_api_gateway/audit.jsonl`
- GPU7 proxy audit log:
  - `/video-storage/ai-customer/qwen35a3b_finetune/runtime/gpu7_dpo_guardrails_proxy/audit.jsonl`
- GPU5 proxy audit log:
  - `/video-storage/ai-customer/qwen35a3b_finetune/runtime/gpu5_dpo_guardrails_proxy/audit.jsonl`

Key practical interpretation:

- The system exposed to callers is alias-driven and gateway-controlled.
- The real serving models are hidden behind the internal proxy.
- Most policy, RAG, glossary, and tool behavior happens in the internal proxy, not in the external gateway.
- The `-rag` aliases are a separate deployment path and should be treated as such in testing and debugging.

## 8. Source Files Used For This Document

- `scripts/external_api_gateway.py`
- `scripts/dpo_guardrails_proxy.py`
- `services/external-api-gateway.env`
- `services/dpo-guardrails-proxy.env`
- `services/dpo-guardrails-proxy-gpu5.env`
- `services/model-system-prompts.json`
