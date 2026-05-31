# GPU5 / GPU7 调用 API 工具原理（现网梳理）

- 生成时间（UTC）：2026-05-14
- 梳理范围：`external_api_gateway`、`dpo_guardrails_proxy`、skills/tool 执行链、systemd 与 nginx 入口
- 说明：本文仅描述当前配置与代码实现；密钥类信息已脱敏。

## 1. 总体架构（外网到工具执行）

```text
Client
  -> Nginx :80 (/v1/*)
    -> external-api-gateway :8025 (127.0.0.1)
      -> dpo-guardrails-proxy-gpu5 :8001 (0.0.0.0)
        -> vLLM GPU5 upstream :8014
          -> 模型 qwen35a3b-domain-text-vl-gpu5
      -> dpo-guardrails-proxy-gpu7 :8010 (0.0.0.0)
        -> vLLM GPU7 upstream :8013
          -> 模型 qwen35a3b-sft-v5-combined

在 dpo-guardrails-proxy 内：
  /v1/chat/completions/agent
    -> 自动注入 tools schema
    -> LLM 输出 tool_calls
    -> 本地执行 skills.tools.*
    -> 结果回填 role=tool
    -> 再次调用 LLM 汇总最终答案
```

## 2. 对外网关层（GPU5/GPU7 别名路由）

实现文件：`scripts/external_api_gateway.py`

### 2.1 核心职责

1. 鉴权：校验外部 API Key，按 key 限制可访问模型与路由。
2. 路由：把稳定别名映射到内部代理地址与内部模型名。
3. 限流：按 key 做 RPM 限流。
4. 审计：记录 metadata，不落用户原文。
5. 模型名改写：返回时把内部模型名重写成外部别名。

### 2.2 当前别名映射

来源：`services/external-api-gateway.env` 的 `GATEWAY_MODEL_ROUTES_JSON`

- `baowang-gpu5` -> `http://127.0.0.1:8001/v1`，上游模型 `gpu5-v5.1f-merged`
- `baowang-gpu7` -> `http://127.0.0.1:8010/v1`，上游模型 `gpu7-v5-combined`

对应代码位置：
- 默认路由定义：`external_api_gateway.py:27-40`
- 真正转发时替换 `payload.model`：`external_api_gateway.py:307-315`

### 2.3 对外接口

- `GET /v1/models`
- `POST /v1/chat/completions`
- `POST /v1/chat/completions/agent`

对应代码位置：`external_api_gateway.py:353-401`

## 3. 内部代理层（GPU5/GPU7 共用同一代理代码）

实现文件：`scripts/dpo_guardrails_proxy.py`

### 3.1 两个实例，仅环境不同

- GPU5 代理服务：`dpo-guardrails-proxy-gpu5.service`，监听 `:8001`
- GPU7 代理服务：`dpo-guardrails-proxy.service`，监听 `:8010`

二者都运行同一个 app：`scripts.dpo_guardrails_proxy:app`。

### 3.2 代理主要职责

1. OpenAI 兼容接口适配（`/v1/models`、`/v1/chat/completions`、`/agent`、`/raw`）。
2. 模型别名与对外模型隐藏。
3. 系统提示词注入。
4. 安全治理与输出清洗（例如去 reasoning 字段）。
5. RAG/术语词典/VL fallback 等增强。
6. Agent 工具调用循环与工具执行。

关键入口代码：
- `/v1/models`：`dpo_guardrails_proxy.py:3002-3014`
- `/v1/chat/completions`：`dpo_guardrails_proxy.py:3017-3251`
- `/v1/chat/completions/agent`：`dpo_guardrails_proxy.py:3268-3289`
- Agent 主循环：`dpo_guardrails_proxy.py:2719-2930`

## 4. GPU5 与 GPU7 的当前差异（配置层）

来源：
- `services/dpo-guardrails-proxy-gpu5.env`
- `services/dpo-guardrails-proxy.env`
- `services/vllm-gpu5-agent-upstream.env`
- `services/vllm-gpu7-baseline.env`

### 4.1 上游模型与端口

- GPU5 代理上游：`UPSTREAM_BASE_URL=http://127.0.0.1:8014/v1`
- GPU7 代理上游：`UPSTREAM_BASE_URL=http://127.0.0.1:8013/v1`

### 4.2 对外公共模型标识

- GPU5：`PUBLIC_MODEL_ID=gpu5-v5.1f-merged`
- GPU7：`PUBLIC_MODEL_ID=gpu7-v5-combined`

### 4.3 工具暴露策略

- GPU5：当前 `tools_total_exposed=198`（未强收敛 allowlist）
- GPU7：当前 `tools_total_exposed=43`（配置了 `AGENT_TOOLS_ALLOWLIST` + `AGENT_TOOLS_MAX=48`）

### 4.4 共同能力

二者都开启：
- `AGENT_INTENT_ROUTING_ENABLED=1`
- RAG + Glossary + VL fallback
- 安全注入与输出脱敏

## 5. API 工具调用闭环（/agent 原理）

以下是当前真实代码流程。

### 5.1 步骤 1：自动注入 tools schema

- 触发条件：请求没有显式 `tools`，且 `SKILLS_ENABLED`、`SKILLS_REGISTERED` 为真。
- 行为：调用 `_agent_select_tools_for_messages()`，按意图路由到指定工具组，设置 `tool_choice=auto`。

代码：`dpo_guardrails_proxy.py:2740-2747`、`963-987`

### 5.2 步骤 2：模型与请求规范化

- 应用模型 alias 与 system prompt。
- 强制关闭 `thinking`。
- Agent 路径强制 `stream=false`，避免 SSE 破坏 JSON 解析。
- 限制 `max_tokens <= AGENT_MAX_OUTPUT_TOKENS`。

代码：`dpo_guardrails_proxy.py:2750-2771`

### 5.3 步骤 3：Agent 多轮循环

- 循环最多 `SKILLS_MAX_ITER`。
- 调上游 `/chat/completions`。
- 提取 `tool_calls`。
- 支持 Swift 内联格式转标准 `tool_calls`。
- 校验工具名、参数、allowlist、required 字段。
- 执行本地 skill，追加 `role=tool` 消息。
- 回填后再次请求模型，直到无工具调用。

代码：`dpo_guardrails_proxy.py:2794-2921`

### 5.4 步骤 4：失败兜底（teacher fallback）

触发条件包括：
- 主模型无 choices
- 工具调用非法
- 需要工具却没产出 tool_calls

行为：切到 `TOOL_FALLBACK_BASE_URL` 再试，成功则继续循环。

代码：`dpo_guardrails_proxy.py:2269-2310`、`2817-2879`

### 5.5 步骤 5：返回

- 返回时保留 `_agent.trace`（可被配置为 redacted）。
- 返回模型名重写为公共模型名/别名。

代码：`dpo_guardrails_proxy.py:2923-2930`

## 6. tools 从哪里来（注册与执行）

### 6.1 注册机制

- `skills/registry.py` 通过 `autoload_tools()` 动态 import `skills/tools/*.py`
- 使用 `@register_skill` 自动注册
- `all_tools_schema()` 生成 OpenAI `tools` schema

### 6.2 执行机制

- 代理在 agent 循环中通过 `_skills_get(fn_name)` 取到 skill class
- 调 `await skill.execute(args)`
- 返回作为 `role=tool` 消息回灌模型

### 6.3 典型工具来源

- Mock 工具：`query_member`、`query_order`、`query_recharge`
- 业务 API 工具：`suncidi_tools.py`（HTTP 请求到 `SUNCIDI_API_BASE_URL`）
- 客服流程工具：`ai_customer_service_tools.py`（可切真实后端或 mock）

## 7. 对外请求路径与内部路径的区别

1. 外部调用建议走 Nginx `/v1/*` -> `external-api-gateway`。
2. 网关再按 alias 路由到 `:8001` 或 `:8010`。
3. 代理再路由到本机 vLLM（`:8014` / `:8013`）。
4. agent 模式在代理内部执行 tool，不是在 gateway 执行。

公网端口防护脚本：`scripts/apply_public_port_guard.sh`

## 8. 当前运行态快照（2026-05-14 UTC 实测）

### 8.1 进程与端口

- `dpo-guardrails-proxy-gpu5.service`：active，监听 `8001`
- `dpo-guardrails-proxy.service`：active，监听 `8010`
- `external-api-gateway.service`：active，监听 `127.0.0.1:8025`
- `vllm-qwen35-gpu5-agent-upstream.service`：active，监听 `8014`
- `vllm-qwen35-gpu7-baseline.service`：inactive（当前 `8013` 未监听）

### 8.2 健康检查

- `GET 8001/health`：`ok=true`，上游 `8014` 可达
- `GET 8010/health`：`ok=false`，上游 `8013` connection refused
- `GET 8025/health`：`ok=true`

### 8.3 功能实测

- `8025 /v1/models`：返回 `baowang-gpu5`、`baowang-gpu7`
- `8025 /v1/chat/completions`：
  - `baowang-gpu5` 可返回 200
  - `baowang-gpu7` 当前返回 502（根因是 8013 未启动）

## 9. 关键文件索引

- 网关代码：`scripts/external_api_gateway.py`
- 代理代码：`scripts/dpo_guardrails_proxy.py`
- GPU5 代理配置：`services/dpo-guardrails-proxy-gpu5.env`
- GPU7 代理配置：`services/dpo-guardrails-proxy.env`
- 外网网关配置：`services/external-api-gateway.env`
- GPU5 vLLM 配置：`services/vllm-gpu5-agent-upstream.env`
- GPU7 vLLM 配置：`services/vllm-gpu7-baseline.env`
- tools 注册：`skills/registry.py`
- tools 实现目录：`skills/tools/`
- Nginx 入口：`/etc/nginx/conf.d/ai.conf`

## 10. 一句话结论

当前 GPU5 / GPU7 的“API 工具调用”采用三层链路：`external-api-gateway（别名/鉴权） -> dpo-guardrails-proxy（治理+agent循环） -> vLLM（生成）`；工具执行实际发生在 proxy 的 `/agent` 循环中，GPU5 链路目前可用，GPU7 链路目前卡在上游 8013 未启动。
