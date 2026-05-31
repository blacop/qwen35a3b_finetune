# Gateway 部署情况、服务清单与整体架构

日期：2026-05-25

## 1. 文档范围

本文档只覆盖当前这条对外推理链路中直接相关的在线服务与配置：

- 公共入口 `nginx`
- 外层 `external_api_gateway`
- 内层 `dpo_guardrails_proxy`
- 上游 `vLLM`
- 文本 `RAG API`
- 多模态 `MM RAG API`
- 当前仍在线但不在主链路上的 `rag-server`
- 与上述端口保护相关的 `ai-public-port-guard`

不展开说明的同机其他服务：

- `open_webui`
- `mlflow`
- `tensorboard`
- `openclaw-gateway`
- `hermes-gateway`
- 其他训练/实验型 systemd 服务

## 2. 先说结论

当前对外调用链路不是单层 gateway，而是多层组合：

### 2.1 普通模型调用

`Client -> nginx :80 -> external_api_gateway :8025 -> dpo_guardrails_proxy :8001/:8010 -> vLLM :8014/:8013`

### 2.2 `-rag` 公共别名调用

`Client -> nginx :80 -> external_api_gateway :8025 -> RAG API :18080/:18081 -> dpo_guardrails_proxy :8001/:8010 -> vLLM :8014/:8013`

这点需要特别说明：

- `baowang-gpu5-rag` 和 `baowang-gpu7-rag` 虽然是先进入 `RAG API`
- 但 `RAG API` 本身的 `LLM_API_BASE` 仍然指向 `8001` 或 `8010`
- 所以它不是“RAG 直接打模型”，而是“RAG 先组装上下文，再回调 proxy，再由 proxy 去打 vLLM”

### 2.3 内层 proxy 的 RAG 注入

`dpo_guardrails_proxy :8001/:8010 -> /retrieve@18080 -> 原请求继续在 proxy 内部完成 -> vLLM`

这里的 `18080` 主要用于检索接口，不是完整回答链路回环。

## 3. 当前在线服务总表

下表按“当前 Gateway 主链路和直接相关依赖”整理，重点列出：

- 服务名
- 当前状态
- 监听端口
- systemd 单元文件位置
- 代码或工作目录
- 关键配置文件
- 日志位置
- 它和其他服务的关系

| 服务名 | 当前状态 | 监听 | systemd / 启动位置 | 代码或工作目录 | 关键配置 | 日志 | 关系说明 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `nginx.service` | `active` | `:80` | `/etc/nginx/conf.d/ai.conf` | 系统级 nginx | `/etc/nginx/conf.d/ai.conf` | 系统 nginx 日志 | 对外公共入口，`/v1/` 转发到 `127.0.0.1:8025` |
| `external-api-gateway.service` | `active` | `127.0.0.1:8025` | `/home/ubuntu/.config/systemd/user/external-api-gateway.service` | `/home/ubuntu/qwen35a3b_finetune` | `services/external-api-gateway.env` | `/video-storage/ai-customer/qwen35a3b_finetune/runtime/external_api_gateway/systemd.log` | 对外别名、鉴权、限流、路由分发 |
| `dpo-guardrails-proxy-gpu5.service` | `active` | `0.0.0.0:8001` | `/home/ubuntu/.config/systemd/user/dpo-guardrails-proxy-gpu5.service` | `/home/ubuntu/qwen35a3b_finetune` | `services/dpo-guardrails-proxy-gpu5.env` | `runtime/gpu5_dpo_guardrails_proxy/systemd.log` | GPU5 内层 proxy，承接普通请求和 RAG 回调 |
| `dpo-guardrails-proxy.service` | `active` | `0.0.0.0:8010` | `/home/ubuntu/.config/systemd/user/dpo-guardrails-proxy.service` | `/home/ubuntu/qwen35a3b_finetune` | `services/dpo-guardrails-proxy.env` | `runtime/gpu7_dpo_guardrails_proxy/systemd.log` | GPU7 内层 proxy，承接普通请求和 RAG 回调 |
| `vllm-qwen35-gpu5-agent-upstream.service` | `active` | `0.0.0.0:8014` | `/home/ubuntu/.config/systemd/user/vllm-qwen35-gpu5-agent-upstream.service` | `/home/ubuntu/qwen35a3b_finetune` | `services/vllm-gpu5-agent-upstream.env` | `tracking/logs/vllm_qwen35_gpu5_agent_upstream.log` | GPU5 实际模型服务，上游供 `8001` 使用 |
| `GPU7 vLLM 实际进程` | `进程在线，但非当前 systemd 托管` | `0.0.0.0:8013` | 当前进程 `PID 2259638`，`PPID 1` | `/home/ubuntu/qwen35a3b_finetune/outputs/swift_sft_v6_1_gpu7_20260515T024304Z/merged_vllm` | 运行参数内嵌；仓库里最接近的是 `services/vllm-gpu7-baseline.env`，但不一致 | 未通过当前活动 unit 统一管理 | GPU7 实际模型服务，上游供 `8010` 使用 |
| `tydata-rag-api.service` | `active` | `0.0.0.0:18080` | `/home/ubuntu/.config/systemd/user/tydata-rag-api.service` | `/home/ubuntu/tydata_rag` | `/home/ubuntu/tydata_rag/rag_api.env` | `/home/ubuntu/tydata_rag/rag_api.log` | GPU5 文本 RAG 服务，`LLM_API_BASE=http://127.0.0.1:8001/v1` |
| `tydata-rag-api-gpu7.service` | `active` | `0.0.0.0:18081` | `/home/ubuntu/.config/systemd/user/tydata-rag-api-gpu7.service` | `/home/ubuntu/tydata_rag` | `/home/ubuntu/tydata_rag/rag_api_gpu7.env` | `/home/ubuntu/tydata_rag/rag_api_gpu7.log` | GPU7 文本 RAG 服务，`LLM_API_BASE=http://127.0.0.1:8010/v1` |
| `mm-rag-api.service` | `active` | `0.0.0.0:18100` | `/home/ubuntu/.config/systemd/user/mm-rag-api.service` | `/home/ubuntu/tydata_rag/mm_rag` | `/home/ubuntu/tydata_rag/mm_rag/mm_rag.env` | `/home/ubuntu/tydata_rag/mm_rag/runtime_mm_rag.log` | 多模态 RAG / VL fallback 服务，当前指向 `8014` |
| `rag-server.service` | `active` | `0.0.0.0:8020` | `/home/ubuntu/.config/systemd/user/rag-server.service` | `/home/ubuntu/qwen35a3b_finetune/rag` | `rag/configs/current.yaml` | `rag/runtime/logs/rag_server.log` | 老的 / 独立检索服务，当前默认主链路未直接使用 |
| `ai-public-port-guard.service` | `active (exited)` | 不监听端口 | `/home/ubuntu/qwen35a3b_finetune/services/ai-public-port-guard.service` | `/home/ubuntu/qwen35a3b_finetune/scripts` | `scripts/apply_public_port_guard.sh` | 系统 journal | 用 iptables 限制部分内部端口只允许本机访问 |

## 4. 各服务所在文件夹

为便于别的同事协作，下面按文件夹说明“代码、配置、unit、日志”各自落在哪里。

### 4.1 `qwen35a3b_finetune`

主目录：

- `/home/ubuntu/qwen35a3b_finetune`

核心代码目录：

- `/home/ubuntu/qwen35a3b_finetune/scripts`
  - `external_api_gateway.py`
  - `dpo_guardrails_proxy.py`
  - `apply_public_port_guard.sh`

服务配置目录：

- `/home/ubuntu/qwen35a3b_finetune/services`
  - `external-api-gateway.env`
  - `dpo-guardrails-proxy.env`
  - `dpo-guardrails-proxy-gpu5.env`
  - `vllm-gpu5-agent-upstream.env`
  - `vllm-gpu7-baseline.env`
  - 各类 `install_*.sh`

RAG 老链路目录：

- `/home/ubuntu/qwen35a3b_finetune/rag`
  - `service/rag_server.py`
  - `configs/current.yaml`
  - `runtime/logs/`

项目内日志目录：

- `/home/ubuntu/qwen35a3b_finetune/runtime`
- `/home/ubuntu/qwen35a3b_finetune/tracking/logs`

### 4.2 `tydata_rag`

主目录：

- `/home/ubuntu/tydata_rag`

文本 RAG 代码与配置：

- `/home/ubuntu/tydata_rag/rag_api.py`
- `/home/ubuntu/tydata_rag/start_rag_api.sh`
- `/home/ubuntu/tydata_rag/rag_api.env`
- `/home/ubuntu/tydata_rag/rag_api_gpu7.env`

文本 RAG 日志：

- `/home/ubuntu/tydata_rag/rag_api.log`
- `/home/ubuntu/tydata_rag/rag_api_gpu7.log`

多模态 RAG 目录：

- `/home/ubuntu/tydata_rag/mm_rag`
  - `mm_rag_api.py`
  - `start_mm_rag_api.sh`
  - `mm_rag.env`
  - `runtime_mm_rag.log`

### 4.3 本机 systemd user 单元目录

当前大部分在线服务的 unit 文件不在仓库内，而在用户本地目录：

- `/home/ubuntu/.config/systemd/user`

当前确认存在的相关 unit 包括：

- `external-api-gateway.service`
- `dpo-guardrails-proxy.service`
- `dpo-guardrails-proxy-gpu5.service`
- `vllm-qwen35-gpu5-agent-upstream.service`
- `vllm-qwen35-gpu7-baseline.service`
- `tydata-rag-api.service`
- `tydata-rag-api-gpu7.service`
- `mm-rag-api.service`
- `rag-server.service`

### 4.4 系统级入口配置目录

- `/etc/nginx/conf.d/ai.conf`

这一层不在仓库里，属于机器本地配置。

## 5. 服务关系图

### 5.1 对外普通请求

```text
Internet
  -> nginx :80
    -> external_api_gateway :8025
      -> dpo_guardrails_proxy-gpu5 :8001
        -> vLLM GPU5 :8014

Internet
  -> nginx :80
    -> external_api_gateway :8025
      -> dpo_guardrails_proxy-gpu7 :8010
        -> vLLM GPU7 :8013
```

### 5.2 对外 `-rag` 请求

```text
Internet
  -> nginx :80
    -> external_api_gateway :8025
      -> tydata-rag-api :18080
        -> dpo_guardrails_proxy-gpu5 :8001
          -> vLLM GPU5 :8014

Internet
  -> nginx :80
    -> external_api_gateway :8025
      -> tydata-rag-api-gpu7 :18081
        -> dpo_guardrails_proxy-gpu7 :8010
          -> vLLM GPU7 :8013
```

### 5.3 proxy 内部增强链路

```text
dpo_guardrails_proxy :8001/:8010
  -> 文本检索：18080 (/retrieve)
  -> 多模态补强：18100
  -> 最终模型：8014 / 8013
```

说明：

- `18080` 既服务于对外 `-rag` 别名，也服务于 proxy 内部的文本检索。
- `18081` 当前主要是 GPU7 对外 `-rag` 入口。
- `18100` 当前是 MM RAG / VL fallback 入口。
- `8020` 虽然在线，但不在当前默认主链路上。

## 6. 目前已确认的关系与配置事实

### 6.1 nginx 关系

当前机器上的 nginx 配置明确写了：

- `/v1/` 转发到 `127.0.0.1:8025`
- upstream 名称为 `external_api_gateway_upstream`

也就是说，真正对外暴露给客户端的是 nginx，不是 `8025` 端口本身。

### 6.2 external gateway 关系

当前 `external-api-gateway.service`：

- 监听 `127.0.0.1:8025`
- 使用 `services/external-api-gateway.env`
- 模型别名包括：
  - `baowang-gpu5 -> 8001`
  - `baowang-gpu7 -> 8010`
  - `baowang-gpu5-rag -> 18080`
  - `baowang-gpu7-rag -> 18081`

### 6.3 GPU5 / GPU7 proxy 关系

当前两个 proxy 健康检查都正常：

- `8001 -> upstream_base_url=http://127.0.0.1:8014/v1`
- `8010 -> upstream_base_url=http://127.0.0.1:8013/v1`

但二者目前都显示：

- `proxy_api_key_required=false`

这意味着：

- 内层 proxy 本身没有再加一层访问鉴权
- 安全边界主要依赖端口暴露策略和外围入口控制

### 6.4 RAG API 关系

当前两个文本 RAG 服务的实际回调目标是：

- `18080 -> LLM_API_BASE=http://127.0.0.1:8001/v1`
- `18081 -> LLM_API_BASE=http://127.0.0.1:8010/v1`

所以：

- `-rag` 路径不会绕过 proxy
- 它只是把“检索+拼接上下文”前置到了一个独立服务里

### 6.5 MM RAG 关系

当前 `18100` 的配置是：

- `LLM_API_BASE=http://127.0.0.1:8014/v1`
- `LLM_MODEL=qwen35a3b-domain-text-vl-gpu5`

因此当前 MM RAG 逻辑实际更偏 GPU5 路径。

## 7. 当前问题清单

下面这部分是给团队协作用的。每个问题都尽量写清楚：

- 现象
- 影响
- 证据
- 建议谁来处理

### 7.1 高优先级问题

#### 问题 A：GPU7 的 `8013` 实际在线，但不在当前有效 systemd 托管下

现象：

- `8013` 端口当前有在线进程
- 但 `vllm-qwen35-gpu7-baseline.service` 状态是 `inactive (dead)`
- 当前在线进程是 `PID 2259638`，`PPID 1`

影响：

- 机器重启或进程异常退出后，`8013` 是否能自动恢复不确定
- 同事看到 systemd 状态会误以为 GPU7 上游未启动
- 当前运行模型与仓库里的 `services/vllm-gpu7-baseline.env` 也不一致

证据：

- `systemctl --user status vllm-qwen35-gpu7-baseline.service`
- `ps -fp 2259638`

建议处理：

- 推理同学：把当前 GPU7 vLLM 启动参数固化为正式 unit
- 运维同学：确认是否存在机器重启后的自恢复机制
- 项目负责人：决定以 `v6.1` 还是 `baseline env` 作为真实标准

#### 问题 B：内部关键端口大面积绑定 `0.0.0.0`

现象：

- `8001`
- `8010`
- `8013`
- `8014`
- `18080`
- `18081`
- `18100`
- `8020`

当前都在 `0.0.0.0` 上监听。

影响：

- 服务本身并没有通过监听地址限制外部访问
- 安全边界依赖机器 iptables 或外层网络策略
- 如果防火墙规则丢失，内部链路会直接暴露出去

证据：

- `ss -ltnp`

建议处理：

- 运维同学：统一确认这些端口的外网暴露面
- 服务同学：能改成 `127.0.0.1` 的尽量直接改为本地监听
- 安全同学：复核当前 iptables / 安全组 / Nginx 暴露策略

#### 问题 C：`ai-public-port-guard` 只保护了部分端口，未覆盖 RAG / MM RAG 端口

现象：

`apply_public_port_guard.sh` 当前只保护：

- `8001`
- `8010`
- `8013`
- `8014`
- `8025`

没有保护：

- `8020`
- `18080`
- `18081`
- `18100`

影响：

- 文本 RAG、GPU7 RAG、MM RAG、老 `rag-server` 可能直接暴露在公网或内网非预期范围

证据：

- `/home/ubuntu/qwen35a3b_finetune/scripts/apply_public_port_guard.sh`
- `ss -ltnp`

建议处理：

- 运维同学：先确认这些端口是否已被别处拦截
- 平台同学：如无特殊需求，应把 `18080/18081/18100/8020` 纳入同一套防护

### 7.2 中优先级问题

#### 问题 D：`mm-rag-api.service` 的 unit 依赖已经过时

现象：

- `mm-rag-api.service` 的 unit 写的是：
  - `After=vllm-qwen35-dpo-gpu5.service`
  - `Wants=vllm-qwen35-dpo-gpu5.service`
- 但 `mm_rag.env` 实际调用的是：
  - `LLM_API_BASE=http://127.0.0.1:8014/v1`

影响：

- unit 依赖和真实调用链不一致
- 同事排障时会误判 MM RAG 的上游依赖

建议处理：

- 把 `mm-rag-api.service` 的依赖改成当前真实上游
- 或明确声明它依赖的是 `8001` / `8014` 哪一层

#### 问题 E：`rag-server.service :8020` 在线，但不在当前默认主链路中

现象：

- `rag-server.service` 运行正常
- 但 `dpo-guardrails-proxy*.env` 当前的 `RAG_SERVER_URL` 指向的是 `http://127.0.0.1:18080`

影响：

- 同时存在两套 RAG 服务：
  - `8020` 老检索服务
  - `18080/18081` 新 RAG API
- 新同事会难以判断哪套是生产主链路

建议处理：

- RAG 同学：明确 `8020` 的定位
- 如果只是历史遗留，建议在文档和服务命名上做“主用 / 备用 / 旧链路”标注

#### 问题 F：关键 unit 和入口配置不在仓库内，存在机器本地漂移

现象：

以下关键内容都在机器本地，而不是直接纳入仓库版本控制：

- `/home/ubuntu/.config/systemd/user/*.service`
- `/etc/nginx/conf.d/ai.conf`

仓库里只有：

- env 文件
- install 脚本
- Python 代码

影响：

- 同事只看仓库，拿不到完整线上配置
- 线上配置可能与 install 脚本生成逻辑发生漂移

建议处理：

- 运维同学：导出当前生效 unit 与 nginx 配置做版本备份
- 开发同学：把关键 unit 模板也纳入仓库

### 7.3 低优先级但建议关注

#### 问题 G：proxy 内部未开启独立 API key 防护

现象：

- `8001/8010` 健康检查均显示 `proxy_api_key_required=false`

影响：

- 如果内网端口暴露控制失效，任何人都可以绕过外层 gateway 直接访问 proxy

建议处理：

- 如果这些端口未来仍然保持 `0.0.0.0` 监听，建议至少补一层 proxy API key

## 8. 推荐的协作拆分

为了方便别的同事一起接手，建议按角色拆分：

### 8.1 运维 / 平台

- 核对 nginx 对外入口与域名解析
- 复核 `iptables` 是否真的拦住了 `18080/18081/18100/8020`
- 把当前生效的 user unit 和 nginx 配置纳入变更记录

### 8.2 推理 / 模型服务

- 把 GPU7 `8013` 的真实 vLLM 启动方式收敛成正式 systemd unit
- 统一 GPU5 / GPU7 的 vLLM 服务管理方式
- 明确 `8013` 当前应该使用哪一版模型与 served-model-name

### 8.3 RAG

- 确认 `8020` 是否仍需保留
- 明确 `18080/18081` 与 `8020` 的职责边界
- 校正 MM RAG 的 unit 依赖与真实上游关系

### 8.4 应用 / 网关

- 决定内层 proxy 是否需要强制开启 API key
- 决定 `-rag` 路径是否继续保留“RAG API -> proxy -> vLLM”这条嵌套结构
- 统一文档中对“直连 RAG”的表述，避免误导

## 9. 建议后续补充的文件

如果要把这套链路交给更多同事一起维护，建议再补三份文档：

1. `SYSTEMD_UNITS_CURRENT_STATE.md`
2. `PORT_AND_FIREWALL_MATRIX.md`
3. `RAG_PATHS_AND_OWNERSHIP.md`

## 10. 本文档依据的主要文件

仓库内：

- `scripts/external_api_gateway.py`
- `scripts/dpo_guardrails_proxy.py`
- `scripts/apply_public_port_guard.sh`
- `services/external-api-gateway.env`
- `services/dpo-guardrails-proxy.env`
- `services/dpo-guardrails-proxy-gpu5.env`
- `services/vllm-gpu5-agent-upstream.env`
- `services/vllm-gpu7-baseline.env`
- `rag/service/rag_server.py`

仓库外但当前机器生效：

- `/home/ubuntu/.config/systemd/user/external-api-gateway.service`
- `/home/ubuntu/.config/systemd/user/dpo-guardrails-proxy.service`
- `/home/ubuntu/.config/systemd/user/dpo-guardrails-proxy-gpu5.service`
- `/home/ubuntu/.config/systemd/user/vllm-qwen35-gpu5-agent-upstream.service`
- `/home/ubuntu/.config/systemd/user/vllm-qwen35-gpu7-baseline.service`
- `/home/ubuntu/.config/systemd/user/tydata-rag-api.service`
- `/home/ubuntu/.config/systemd/user/tydata-rag-api-gpu7.service`
- `/home/ubuntu/.config/systemd/user/mm-rag-api.service`
- `/home/ubuntu/.config/systemd/user/rag-server.service`
- `/etc/nginx/conf.d/ai.conf`
- `/home/ubuntu/tydata_rag/rag_api.py`
- `/home/ubuntu/tydata_rag/rag_api.env`
- `/home/ubuntu/tydata_rag/rag_api_gpu7.env`
- `/home/ubuntu/tydata_rag/mm_rag/mm_rag_api.py`
- `/home/ubuntu/tydata_rag/mm_rag/mm_rag.env`
