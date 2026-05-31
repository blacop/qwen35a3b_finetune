# 对外开放实施清单（SOP）

更新时间：2026-05-26

适用目标：

- 对外开放普通推理：`baowang-gpu5`、`baowang-gpu7`
- 对外开放 RAG 推理：`baowang-gpu5-rag`、`baowang-gpu7-rag`
- 统一通过一个公网域名 + HTTPS + OpenAI 兼容接口访问

本文按当前机器真实状态编写，不是通用模板。

## 0. 当前机器状态和结论

已确认：

- `external-api-gateway` 在 `127.0.0.1:8025`
- `dpo-guardrails-proxy-gpu5` 在 `0.0.0.0:8001`
- `dpo-guardrails-proxy`（GPU7）在 `0.0.0.0:8010`
- `tydata-rag-api` 在 `0.0.0.0:18080`
- `tydata-rag-api-gpu7` 在 `0.0.0.0:18081`
- GPU5 vLLM 在 `0.0.0.0:8014`
- GPU7 vLLM 在 `0.0.0.0:8013`
- `nginx` 已安装并运行
- 当前 `80` 端口在监听
- 当前 `443` 端口未监听
- `certbot` 当前不可用，报 `OpenSSL` / `josepy` 兼容错误

重要结论：

1. 最推荐的公网入口还是 `nginx:443 -> 127.0.0.1:8025`
2. 不建议直接暴露 `8001/8010/8013/8014/18080/18081`
3. `GPU7 vLLM` 当前是手工拉起进程，不是可靠的 `systemd` 托管，这一项在对外开放前必须补

## 1. 对外开放目标架构

推荐架构：

`Internet -> Nginx:443 -> external-api-gateway:8025 -> {普通模型走 8001/8010，RAG 模型走 18080/18081}`

对外暴露的能力只有：

- `GET /v1/models`
- `POST /v1/chat/completions`

对外暴露的模型别名：

- `baowang-gpu5`
- `baowang-gpu7`
- `baowang-gpu5-rag`
- `baowang-gpu7-rag`

默认不开放：

- `POST /v1/chat/completions/agent`
- `GET /health`
- `POST /retrieve`

## 2. 变更前准备

### 2.1 准备域名

假设最终公网域名为：

- `api.example.com`

你需要先准备好 DNS A 记录：

- `api.example.com -> <这台机器公网 IP>`

验证：

```bash
dig +short api.example.com
```

### 2.2 建立变更备份目录

```bash
TS="$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p /home/ubuntu/change_backups/"$TS"
```

备份当前关键文件：

```bash
cp /home/ubuntu/qwen35a3b_finetune/services/external-api-gateway.env /home/ubuntu/change_backups/"$TS"/
cp /video-storage/ai-customer/qwen35a3b_finetune/runtime/external_api_gateway/api_keys.json /home/ubuntu/change_backups/"$TS"/
cp /home/ubuntu/qwen35a3b_finetune/services/dpo-guardrails-proxy.env /home/ubuntu/change_backups/"$TS"/
cp /home/ubuntu/qwen35a3b_finetune/services/dpo-guardrails-proxy-gpu5.env /home/ubuntu/change_backups/"$TS"/
cp /home/ubuntu/tydata_rag/rag_api.env /home/ubuntu/change_backups/"$TS"/
cp /home/ubuntu/tydata_rag/rag_api_gpu7.env /home/ubuntu/change_backups/"$TS"/
cp /home/ubuntu/.config/systemd/user/external-api-gateway.service /home/ubuntu/change_backups/"$TS"/
cp /home/ubuntu/.config/systemd/user/dpo-guardrails-proxy.service /home/ubuntu/change_backups/"$TS"/
cp /home/ubuntu/.config/systemd/user/dpo-guardrails-proxy-gpu5.service /home/ubuntu/change_backups/"$TS"/
cp /home/ubuntu/.config/systemd/user/tydata-rag-api.service /home/ubuntu/change_backups/"$TS"/
cp /home/ubuntu/.config/systemd/user/tydata-rag-api-gpu7.service /home/ubuntu/change_backups/"$TS"/
sudo cp -a /etc/nginx /home/ubuntu/change_backups/"$TS"/nginx
```

记录当前进程和端口：

```bash
systemctl --user list-units --type=service --all | rg 'gateway|guard|rag|vllm'
ps -ef | rg 'vllm|rag_api|external_api_gateway|dpo_guardrails_proxy'
ss -ltnp | rg ':80|:443|:8025|:8001|:8010|:8013|:8014|:18080|:18081'
```

## 3. systemd 变更清单

## 3.1 保证 user service 能随开机自启动

只做一次：

```bash
sudo loginctl enable-linger ubuntu
```

验证：

```bash
loginctl show-user ubuntu | rg Linger
```

预期看到：

- `Linger=yes`

## 3.2 保持 gateway 继续只监听本机回环

当前 `external-api-gateway` 监听 `127.0.0.1:8025`，这正是推荐状态。

不要改成 `0.0.0.0:8025`。公网流量应该先到 `nginx:443`，而不是直接打网关端口。

当前 unit 可直接复用：

- [external-api-gateway.service](/home/ubuntu/.config/systemd/user/external-api-gateway.service)

## 3.3 为 GPU7 上游补一个正式 systemd unit

这是当前最重要的补洞项。

现状：

- `GPU7 vLLM` 当前进程在跑
- 但不是由可靠的 `systemd` unit 托管
- 一旦机器重启或进程异常退出，对外服务会断

### 3.3.1 新建 GPU7 上游 env 文件

文件：

- `/home/ubuntu/qwen35a3b_finetune/services/vllm-gpu7-upstream.env`

建议内容：

```env
MODEL_DIR=/home/ubuntu/qwen35a3b_finetune/outputs/swift_sft_v6_1_gpu7_20260515T024304Z/merged_vllm
PORT=8013
TP_SIZE=1
MAX_LEN=8192
GPU_MEM_UTIL=0.85
SERVED_MODEL_NAME=qwen35a3b-sft-v6-1-ops
API_KEY=<GPU7_UPSTREAM_API_KEY>
SERVE_EXTRA_ARGS=--enforce-eager --trust-remote-code --enable-auto-tool-choice --tool-call-parser hermes --limit-mm-per-prompt {"image":0,"video":0}
```

说明：

- `MODEL_DIR` 和 `SERVED_MODEL_NAME` 按当前在线进程实值写
- `API_KEY` 要与 `dpo-guardrails-proxy.env` 中 `UPSTREAM_API_KEY` 保持一致

### 3.3.2 新建 GPU7 上游 unit

文件：

- `/home/ubuntu/.config/systemd/user/vllm-qwen35-gpu7-upstream.service`

建议内容：

```ini
[Unit]
Description=vLLM GPU7 upstream for guardrails proxy
After=network.target

[Service]
Type=simple
WorkingDirectory=/home/ubuntu/qwen35a3b_finetune
Environment=CUDA_VISIBLE_DEVICES=7
Environment=VLLM_HOST_IP=0.0.0.0
Environment=UV_CACHE_DIR=/tmp/uv-cache
EnvironmentFile=/home/ubuntu/qwen35a3b_finetune/services/vllm-gpu7-upstream.env
ExecStart=/bin/bash -lc "set -euo pipefail; UV_CACHE_DIR=/tmp/uv-cache /home/ubuntu/.local/bin/vllm serve ${MODEL_DIR} --host 0.0.0.0 --port ${PORT} --tensor-parallel-size ${TP_SIZE} --max-model-len ${MAX_LEN} --dtype bfloat16 --gpu-memory-utilization ${GPU_MEM_UTIL} --served-model-name ${SERVED_MODEL_NAME} --api-key ${API_KEY} ${SERVE_EXTRA_ARGS:-}"
Restart=always
RestartSec=10
StandardOutput=append:/home/ubuntu/qwen35a3b_finetune/tracking/logs/vllm_qwen35_gpu7_upstream.log
StandardError=append:/home/ubuntu/qwen35a3b_finetune/tracking/logs/vllm_qwen35_gpu7_upstream.log

[Install]
WantedBy=default.target
```

### 3.3.3 让 GPU7 guardrails 显式依赖这个上游 unit

修改文件：

- [dpo-guardrails-proxy.service](/home/ubuntu/.config/systemd/user/dpo-guardrails-proxy.service)

把：

```ini
After=network.target
```

改成：

```ini
After=network.target vllm-qwen35-gpu7-upstream.service
Requires=vllm-qwen35-gpu7-upstream.service
```

原因：

- 避免 `8010` 先起来，但 `8013` 还没准备好
- 让 GPU7 普通推理和 GPU7 RAG 链路的依赖关系明确

## 3.4 GPU5 相关 unit 原则上可直接复用

当前 GPU5 上游已有正式 unit：

- [vllm-qwen35-gpu5-agent-upstream.service](/home/ubuntu/.config/systemd/user/vllm-qwen35-gpu5-agent-upstream.service)

当前 GPU5 RAG / guardrails / gateway 依赖链已基本成型，不需要额外结构性调整。

## 3.5 应用 systemd 变更

```bash
systemctl --user daemon-reload
systemctl --user enable vllm-qwen35-gpu7-upstream.service
systemctl --user restart vllm-qwen35-gpu7-upstream.service
systemctl --user restart dpo-guardrails-proxy.service
systemctl --user restart tydata-rag-api-gpu7.service
systemctl --user restart external-api-gateway.service
```

验证：

```bash
systemctl --user status vllm-qwen35-gpu7-upstream.service --no-pager
systemctl --user status dpo-guardrails-proxy.service --no-pager
systemctl --user status tydata-rag-api-gpu7.service --no-pager
curl -sS http://127.0.0.1:8010/health
curl -sS http://127.0.0.1:18081/health
curl -sS http://127.0.0.1:8025/health
```

## 4. 鉴权和 API Key 管理

## 4.1 不要把内部 key 直接当外部 key 发给客户

内部 key：

- `RAG_API_KEY`
- `UPSTREAM_API_KEY`
- vLLM 直接 API key

这些都只应留在内网链路中。

外部只应该使用：

- `external-api-gateway` 的外部访问 key

## 4.2 调整外部 key 文件

当前文件：

- `/video-storage/ai-customer/qwen35a3b_finetune/runtime/external_api_gateway/api_keys.json`

建议改造原则：

1. 每个调用方单独一把 key
2. 默认不开放 `agent` route
3. 按客户粒度限制可访问模型
4. 按客户粒度限制 RPM

建议结构：

```json
{
  "keys": [
    {
      "id": "client-a-all",
      "key": "sk-baowang-REPLACE_ME",
      "models": [
        "baowang-gpu5",
        "baowang-gpu7",
        "baowang-gpu5-rag",
        "baowang-gpu7-rag"
      ],
      "routes": [
        "models",
        "chat"
      ],
      "rpm": 60
    },
    {
      "id": "client-b-gpu5-only",
      "key": "sk-baowang-REPLACE_ME_2",
      "models": [
        "baowang-gpu5",
        "baowang-gpu5-rag"
      ],
      "routes": [
        "models",
        "chat"
      ],
      "rpm": 30
    }
  ]
}
```

注意：

- 除非你明确要开放工具调用，否则把 `agent` 从 `routes` 里去掉
- 对外不要公开内部模型名，只暴露 `baowang-*` 别名

变更后重启：

```bash
systemctl --user restart external-api-gateway.service
```

## 5. Nginx 反向代理配置

## 5.1 设计原则

Nginx 只代理到：

- `127.0.0.1:8025`

Nginx 对外只放行：

- `GET /v1/models`
- `POST /v1/chat/completions`

其它路径默认拒绝或不暴露。

## 5.2 先准备 ACME challenge 目录

```bash
sudo mkdir -p /var/www/acme/.well-known/acme-challenge
sudo chown -R www-data:www-data /var/www/acme
```

## 5.3 新建 Nginx 站点配置

文件：

- `/etc/nginx/sites-available/baowang-api.conf`

建议内容：

```nginx
limit_req_zone $binary_remote_addr zone=baowang_per_ip:10m rate=10r/s;

upstream baowang_gateway {
    server 127.0.0.1:8025;
    keepalive 64;
}

server {
    listen 80;
    listen [::]:80;
    server_name api.example.com;

    location ^~ /.well-known/acme-challenge/ {
        root /var/www/acme;
        default_type "text/plain";
        try_files $uri =404;
    }

    location / {
        return 301 https://$host$request_uri;
    }
}

server {
    listen 443 ssl http2;
    listen [::]:443 ssl http2;
    server_name api.example.com;

    ssl_certificate /etc/nginx/ssl/api.example.com/fullchain.pem;
    ssl_certificate_key /etc/nginx/ssl/api.example.com/key.pem;
    ssl_session_timeout 1d;
    ssl_session_cache shared:SSL:10m;
    ssl_protocols TLSv1.2 TLSv1.3;

    client_max_body_size 10m;

    proxy_http_version 1.1;
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header X-Forwarded-Proto https;
    proxy_read_timeout 300;
    proxy_send_timeout 300;

    location = /v1/models {
        limit_req zone=baowang_per_ip burst=20 nodelay;
        proxy_pass http://baowang_gateway/v1/models;
    }

    location = /v1/chat/completions {
        limit_req zone=baowang_per_ip burst=30 nodelay;
        proxy_pass http://baowang_gateway/v1/chat/completions;
    }

    location = /health {
        allow 127.0.0.1;
        deny all;
        proxy_pass http://baowang_gateway/health;
    }

    location / {
        return 404;
    }
}
```

启用站点：

```bash
sudo ln -s /etc/nginx/sites-available/baowang-api.conf /etc/nginx/sites-enabled/baowang-api.conf
```

说明：

- 当前默认站点只有 `80`，不会抢 `443`
- `default.bak` 可以先不动
- 等你确认新站点完全稳定后，再决定是否清理默认站点

## 6. 域名和证书

## 6.1 当前建议：不用 certbot，直接用 acme.sh

原因：

- 这台机器上的 `certbot` 当前已损坏
- 继续用它会在发证阶段卡住

## 6.2 安装 acme.sh

```bash
curl https://get.acme.sh | sh
source ~/.bashrc
```

如果 `source ~/.bashrc` 后还找不到命令，直接使用：

```bash
~/.acme.sh/acme.sh --version
```

## 6.3 申请证书

先确保：

- `api.example.com` 已解析到本机公网 IP
- `80` 端口可从公网访问
- 上面的 ACME challenge location 已写入 Nginx 并 reload

先测试 Nginx 配置：

```bash
sudo nginx -t
sudo systemctl reload nginx
```

然后发证：

```bash
~/.acme.sh/acme.sh --issue -d api.example.com -w /var/www/acme
```

## 6.4 安装证书到 Nginx 使用路径

```bash
sudo mkdir -p /etc/nginx/ssl/api.example.com
~/.acme.sh/acme.sh --install-cert -d api.example.com \
  --key-file /etc/nginx/ssl/api.example.com/key.pem \
  --fullchain-file /etc/nginx/ssl/api.example.com/fullchain.pem \
  --reloadcmd "sudo nginx -t && sudo systemctl reload nginx"
```

然后再次验证：

```bash
sudo nginx -t
sudo systemctl reload nginx
ss -ltnp | rg ':443 '
```

## 7. 网关开放验证

## 7.1 内网验证

先验证内部 gateway 仍正常：

```bash
curl -sS http://127.0.0.1:8025/health
```

验证模型列表：

```bash
curl -sS http://127.0.0.1:8025/v1/models \
  -H 'Authorization: Bearer <EXTERNAL_GATEWAY_KEY>'
```

验证 GPU5 普通推理：

```bash
curl -sS http://127.0.0.1:8025/v1/chat/completions \
  -H 'Authorization: Bearer <EXTERNAL_GATEWAY_KEY>' \
  -H 'Content-Type: application/json' \
  -d '{
    "model":"baowang-gpu5",
    "messages":[{"role":"user","content":"香港盘和欧洲盘有什么区别？"}],
    "temperature":0,
    "max_tokens":256
  }'
```

验证 GPU7 普通推理：

```bash
curl -sS http://127.0.0.1:8025/v1/chat/completions \
  -H 'Authorization: Bearer <EXTERNAL_GATEWAY_KEY>' \
  -H 'Content-Type: application/json' \
  -d '{
    "model":"baowang-gpu7",
    "messages":[{"role":"user","content":"香港盘和欧洲盘有什么区别？"}],
    "temperature":0,
    "max_tokens":256
  }'
```

验证 GPU5 RAG：

```bash
curl -sS http://127.0.0.1:8025/v1/chat/completions \
  -H 'Authorization: Bearer <EXTERNAL_GATEWAY_KEY>' \
  -H 'Content-Type: application/json' \
  -d '{
    "model":"baowang-gpu5-rag",
    "messages":[{"role":"user","content":"PT、VG、bbin、FG 这四份百家乐规则里，是否都使用 8 副牌？"}],
    "temperature":0,
    "max_tokens":256
  }'
```

验证 GPU7 RAG：

```bash
curl -sS http://127.0.0.1:8025/v1/chat/completions \
  -H 'Authorization: Bearer <EXTERNAL_GATEWAY_KEY>' \
  -H 'Content-Type: application/json' \
  -d '{
    "model":"baowang-gpu7-rag",
    "messages":[{"role":"user","content":"PT、VG、bbin、FG 这四份百家乐规则里，是否都使用 8 副牌？"}],
    "temperature":0,
    "max_tokens":256
  }'
```

## 7.2 公网验证

证书生效后，验证公网入口：

```bash
curl -sS https://api.example.com/v1/models \
  -H 'Authorization: Bearer <EXTERNAL_GATEWAY_KEY>'
```

```bash
curl -sS https://api.example.com/v1/chat/completions \
  -H 'Authorization: Bearer <EXTERNAL_GATEWAY_KEY>' \
  -H 'Content-Type: application/json' \
  -d '{
    "model":"baowang-gpu5-rag",
    "messages":[{"role":"user","content":"香港盘和欧洲盘有什么区别？"}],
    "temperature":0,
    "max_tokens":256
  }'
```

## 8. 日志和巡检

上线后重点看这几类日志：

- gateway：
  - `/video-storage/ai-customer/qwen35a3b_finetune/runtime/external_api_gateway/systemd.log`
  - `/video-storage/ai-customer/qwen35a3b_finetune/runtime/external_api_gateway/audit.jsonl`
- GPU5 guardrails：
  - `/home/ubuntu/qwen35a3b_finetune/runtime/gpu5_dpo_guardrails_proxy/systemd.log`
- GPU7 guardrails：
  - `/home/ubuntu/qwen35a3b_finetune/runtime/gpu7_dpo_guardrails_proxy/systemd.log`
- GPU5 RAG：
  - `/home/ubuntu/tydata_rag/rag_api.log`
- GPU7 RAG：
  - `/home/ubuntu/tydata_rag/rag_api_gpu7.log`
- Nginx：
  - `/var/log/nginx/access.log`
  - `/var/log/nginx/error.log`

建议上线后观察项：

1. `401` 是否异常增多
2. `429` 是否打满
3. `502` 是否来自 gateway 到后端链路
4. GPU7 上游 `8013` 是否稳定
5. `baowang-gpu7-rag` 是否有明显首包超时

## 9. 回滚 SOP

## 9.1 只回滚 Nginx 公网暴露

适用于：

- 内部服务正常
- 只是公网入口配置有问题

步骤：

```bash
sudo rm -f /etc/nginx/sites-enabled/baowang-api.conf
sudo nginx -t
sudo systemctl reload nginx
```

结果：

- 公网入口关闭
- 内部 `8025`、`8001`、`8010`、`18080`、`18081` 不受影响

## 9.2 回滚 GPU7 新增上游 unit

适用于：

- 新的 GPU7 systemd 化改造有问题

步骤：

```bash
systemctl --user stop vllm-qwen35-gpu7-upstream.service
systemctl --user disable vllm-qwen35-gpu7-upstream.service
mv /home/ubuntu/.config/systemd/user/dpo-guardrails-proxy.service /home/ubuntu/.config/systemd/user/dpo-guardrails-proxy.service.bad
cp /home/ubuntu/change_backups/<TS>/dpo-guardrails-proxy.service /home/ubuntu/.config/systemd/user/dpo-guardrails-proxy.service
systemctl --user daemon-reload
systemctl --user restart dpo-guardrails-proxy.service
systemctl --user restart tydata-rag-api-gpu7.service
```

如果需要临时恢复到现在这种“手工 GPU7 vLLM”方式，再手工拉起原命令。

## 9.3 回滚 gateway key / route 配置

```bash
cp /home/ubuntu/change_backups/<TS>/external-api-gateway.env /home/ubuntu/qwen35a3b_finetune/services/external-api-gateway.env
cp /home/ubuntu/change_backups/<TS>/api_keys.json /video-storage/ai-customer/qwen35a3b_finetune/runtime/external_api_gateway/api_keys.json
systemctl --user restart external-api-gateway.service
```

## 9.4 全量回滚

按顺序执行：

```bash
sudo rm -f /etc/nginx/sites-enabled/baowang-api.conf
sudo cp -a /home/ubuntu/change_backups/<TS>/nginx/* /etc/nginx/
sudo nginx -t
sudo systemctl reload nginx

cp /home/ubuntu/change_backups/<TS>/external-api-gateway.env /home/ubuntu/qwen35a3b_finetune/services/external-api-gateway.env
cp /home/ubuntu/change_backups/<TS>/dpo-guardrails-proxy.env /home/ubuntu/qwen35a3b_finetune/services/dpo-guardrails-proxy.env
cp /home/ubuntu/change_backups/<TS>/dpo-guardrails-proxy-gpu5.env /home/ubuntu/qwen35a3b_finetune/services/dpo-guardrails-proxy-gpu5.env
cp /home/ubuntu/change_backups/<TS>/rag_api.env /home/ubuntu/tydata_rag/rag_api.env
cp /home/ubuntu/change_backups/<TS>/rag_api_gpu7.env /home/ubuntu/tydata_rag/rag_api_gpu7.env
cp /home/ubuntu/change_backups/<TS>/external-api-gateway.service /home/ubuntu/.config/systemd/user/
cp /home/ubuntu/change_backups/<TS>/dpo-guardrails-proxy.service /home/ubuntu/.config/systemd/user/
cp /home/ubuntu/change_backups/<TS>/dpo-guardrails-proxy-gpu5.service /home/ubuntu/.config/systemd/user/
cp /home/ubuntu/change_backups/<TS>/tydata-rag-api.service /home/ubuntu/.config/systemd/user/
cp /home/ubuntu/change_backups/<TS>/tydata-rag-api-gpu7.service /home/ubuntu/.config/systemd/user/

systemctl --user daemon-reload
systemctl --user restart vllm-qwen35-gpu5-agent-upstream.service
systemctl --user restart dpo-guardrails-proxy-gpu5.service
systemctl --user restart dpo-guardrails-proxy.service
systemctl --user restart tydata-rag-api.service
systemctl --user restart tydata-rag-api-gpu7.service
systemctl --user restart external-api-gateway.service
```

## 10. 最终上线标准

以下条件全部满足，才算具备对外开放条件：

1. `https://api.example.com/v1/models` 正常
2. 4 个公开模型别名都能返回结果
3. GPU7 上游已由 `systemd` 托管，不再依赖手工进程
4. Nginx 只暴露 `443` 公网入口
5. 内部上游端口没有直接对公网放开
6. 外部 key 已按客户分权，不再复用内部 key
7. 有明确回滚包和回滚命令
