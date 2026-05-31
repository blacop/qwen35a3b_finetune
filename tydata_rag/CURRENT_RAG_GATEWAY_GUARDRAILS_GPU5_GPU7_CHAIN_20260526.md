# 当前 RAG / Gateway / Guardrails / GPU5 / GPU7 推理链路梳理

更新时间：2026-05-26（基于当前进程、systemd、env、health 实测）

## 1. 当前在线服务总览

当前实际在线的关键服务和端口：

- `external-api-gateway`：`127.0.0.1:8025`
- `dpo-guardrails-proxy-gpu5`：`0.0.0.0:8001`
- `dpo-guardrails-proxy`（GPU7）：`0.0.0.0:8010`
- `vllm-qwen35-gpu5-agent-upstream`：`0.0.0.0:8014`
- GPU7 vLLM：`0.0.0.0:8013`
- `tydata-rag-api`：`0.0.0.0:18080`
- `tydata-rag-api-gpu7`：`0.0.0.0:18081`
- 旧版 `rag-server`：`0.0.0.0:8020`
- `mm-rag-api`：`0.0.0.0:18100`

实测 health 结果：

- `18080/health`：RAG 在线，知识索引 `953` chunks，上游指向 `http://127.0.0.1:8001/v1`，模型 `gpu5-v5.1f-merged`
- `18081/health`：RAG 在线，上游指向 `http://127.0.0.1:8010/v1`，模型 `gpu7-v5-combined`
- `8001/health`：guardrails 在线，上游指向 `http://127.0.0.1:8014/v1`
- `8010/health`：guardrails 在线，上游指向 `http://127.0.0.1:8013/v1`
- `8025/health`：gateway 在线，当前暴露 4 个模型别名：`baowang-gpu5`、`baowang-gpu7`、`baowang-gpu5-rag`、`baowang-gpu7-rag`

## 2. 当前主链路

### 2.1 普通推理链路（不走显式 RAG 别名）

GPU5：

`Client -> external-api-gateway:8025 -> baowang-gpu5 -> guardrails:8001 -> vLLM:8014`

GPU7：

`Client -> external-api-gateway:8025 -> baowang-gpu7 -> guardrails:8010 -> vLLM:8013`

这一层的作用分工：

- `gateway`：对外鉴权、模型别名、RPM 限流、审计
- `guardrails`：system prompt 注入、术语/安全/越权处理、前置模板路由、后处理
- `vLLM`：真正的模型推理

### 2.2 显式 RAG 推理链路（当前最清晰、最适合对外开放）

GPU5 RAG：

`Client -> external-api-gateway:8025 -> baowang-gpu5-rag -> tydata-rag-api:18080 -> guardrails:8001 -> vLLM:8014`

GPU7 RAG：

`Client -> external-api-gateway:8025 -> baowang-gpu7-rag -> tydata-rag-api-gpu7:18081 -> guardrails:8010 -> vLLM:8013`

这条链里，`rag_api.py` 做的事情是：

- 从 `messages` 里取最后一条 user query
- 调本地混合检索器 `HybridRetriever`
- 按规则拼接 `context_block`
- 把带参考资料的 system prompt 注入到 OpenAI 兼容请求
- 再转发给对应 guardrails / 上游模型

当前 `gateway` 中 `*-rag` 别名映射已经配置好：

- `baowang-gpu5-rag -> http://127.0.0.1:18080/v1`
- `baowang-gpu7-rag -> http://127.0.0.1:18081/v1`

并且 `gateway` 会自动把外部模型名改写成内部上游模型名，同时给 `RAG API` 补上后端 `Bearer RAG_API_KEY`。

## 3. Guardrails 里的“内嵌 RAG”现状

`dpo_guardrails_proxy.py` 自己也带了一套 RAG 注入逻辑，但当前要分开看：

- `MM RAG`：调用 `18100/mm-rag-api`，从日志上看确实在使用，命中的主要是 JT 包网后台操作手册类问题
- `Glossary`：术语字典增强也在实际生效
- `Sports RAG`：代码契约更像旧版 `rag-server:8020`，因为它期待 `/retrieve` 返回 `chunks`

这里有一个需要注意的点：

- `guardrails` 的 env 里把 `RAG_SERVER_URL` 指到了 `18080`
- 但 `18080` 的 `tydata-rag-api` `/retrieve` 返回的是 `hits`
- 旧版 `8020/rag-server` `/retrieve` 返回的才是 `chunks`

所以当前“guardrails 内嵌 sports RAG”这条链存在接口契约不一致风险；至少从代码结构看，它不是和 `tydata-rag-api.py` 原生对齐的。对外开放时，不建议把这条隐式链路当成主接口能力去承诺。

## 4. 当前最推荐的对外开放方式

### 方案 A：只对外开放 `external-api-gateway:8025`，这是首选

原因：

- 已有外部 API key 机制
- 已有模型别名和审计日志
- 已有对 `gpu5/gpu7/rag` 的稳定路由
- 不需要把 `8001/8010/18080/18081/8013/8014` 直接暴露到公网

建议只开放这 4 个别名：

- `baowang-gpu5`
- `baowang-gpu7`
- `baowang-gpu5-rag`
- `baowang-gpu7-rag`

这样对外统一就是一个 OpenAI 兼容入口：

- `POST /v1/chat/completions`
- `GET /v1/models`

### 方案 B：如果一定要单独开放 RAG 接口

分两类：

1. 如果只想给外部“RAG 问答能力”
- 仍建议只开放 `18080/18081` 的 `/v1/chat/completions`
- 不建议直接开放 `/retrieve` 给不受控调用方

2. 如果外部确实需要检索结果本身
- 再单独开放 `/retrieve`
- 但必须额外做鉴权、限流、IP 白名单、审计

原因是 `/retrieve` 属于“裸检索接口”，更容易被拿去扫知识库、打压测、试探召回策略。

## 5. 如果你要正式开放接口，应该怎么做

### 5.1 开放 GPU5 / GPU7 推理服务

推荐做法：

1. 保持 `8025` 作为唯一公网入口
2. 用 Nginx / Caddy / 云 LB 做 `443 -> 127.0.0.1:8025`
3. 继续让 `8025` 后面转 `8001/8010/18080/18081`
4. 不直接暴露 `8001/8010/8013/8014`

需要补的配置：

- 给公网域名加 TLS
- 维护 `GATEWAY_API_KEYS_FILE`
- 开启公网防火墙白名单或至少限源
- 增加请求日志轮转和报警

### 5.2 开放 RAG 接口

推荐分成两个公开产品面：

- `baowang-gpu5-rag`
- `baowang-gpu7-rag`

也就是继续复用 `8025` 的模型别名，不单独让外部知道 `18080/18081`。

如果确实要单独开放 `18080/18081`，至少要补：

- 反向代理层鉴权，不要只靠后端明文 token
- HTTPS
- IP 白名单
- QPS / RPM 限流
- 按路径分权：`/v1/chat/completions` 和 `/retrieve` 分开授权

### 5.3 哪些端口不要直接开放

不建议直接对公网开放：

- `8001`、`8010`：当前 health 显示 `proxy_api_key_required=false`，也就是 guardrails 自身没有开启外部鉴权
- `8013`、`8014`：这是底层 vLLM，上游模型口
- `18080`、`18081`：除非你就是要开放 RAG 原生接口，否则最好只让 gateway 访问
- `8020`：这是旧版 rag-server，建议保留内网用途

## 6. 落地建议

如果你的目标是“把 RAG 接口和 GPU5/GPU7 推理服务对外提供”，最稳的落地方式是：

`公网域名/负载均衡 -> 443 -> external-api-gateway:8025 -> {普通模型走 8001/8010，RAG 模型走 18080/18081}`

也就是说：

- 普通问答：用 `baowang-gpu5`、`baowang-gpu7`
- RAG 问答：用 `baowang-gpu5-rag`、`baowang-gpu7-rag`

这样最少改动、边界最清晰、风险最低。

## 7. 我认为当前还需要你重点确认的 3 件事

1. `GPU7` 对外是准备开放“普通推理”还是“RAG 推理”还是两者都要
2. 外部是否真的需要 `/retrieve` 裸检索接口
3. 是否要把 `guardrails` 内嵌的旧 sports RAG 逻辑统一到 `tydata-rag-api.py`，避免后续链路解释不一致
