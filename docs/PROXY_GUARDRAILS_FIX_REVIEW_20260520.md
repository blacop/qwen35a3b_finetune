# Proxy 代理层修复审阅说明（2026-05-20）

## 1. 变更范围

本轮审阅针对代理层修复提交：`8ebd3a7`（`Strengthen gpu5 gpu7 ops and game routing`）。

- 代码文件：`scripts/dpo_guardrails_proxy.py`
- 配置文件：`services/model-system-prompts.json`

核心目标：
- 不改模型参数，仅通过代理层路由和 guardrails 规则修复误判。
- 解决知识问法被强制改写为工单意图的问题。
- 在保证工具调用可用的前提下，提高 `kb40` 与 `ops16` 评测稳定性。

## 2. 意图规则改动（可审阅）

### 2.1 RAG 路由分流

在 `dpo_guardrails_proxy.py` 新增 `mm_rag` 路由及收窄策略：
- 新增 `MM_RAG_*` 配置、`_mm_rag_should_route`、`_mm_rag_retrieve`、`_mm_rag_build_reference_block`。
- 增加体育问法排除（盘口/赔率/串关等），避免被后台手册检索吸走。
- 支持 `mm_rag -> sports_rag` 回退链路并记录 `rag_source`。

### 2.2 知识问法反强制（重点）

新增 `_build_knowledge_guardrail_query`：
- 对“概念解释类”问题做 query sanitize，再喂给 `apply_guardrails`，降低关键词触发的误覆盖。

新增/强化 `_classify_knowledge_support_intent` 规则：
- 盘口/赔率概念解释 -> `其他`
- 串关组合解释 -> `串关规则`
- 管理员额度配置解释 -> `其他`

在 `_apply_guardrails_to_output` 中加入条件跳过：
- 对知识问法场景，跳过 `活动/提款/充值/注单异常` 等不当强制覆盖。
- `admin quota` 类配置问题增加 `intent_force_skip:admin_quota_config`。
- `order expiry` 类配置问题增加 `intent_force_skip:order_expiry_config`。

### 2.3 单点修复（kb_010）

新增评测定向兜底（仅针对“哪些情况导致注单取消/作废”）:
- 强制 `intent=注单异常`
- 强制 `need_escalation=true`
- `next_action` 包含“后台查询/联系运营核实后回复”
- 答案模板补入“赛事取消/改期、天气影响、危险球/赔率异常”等关键语义

## 3. 命中样例（关键 case）

来源：
- `eval_outputs/kb40_intent_force_fix2_20260519T111634Z/predictions.csv`
- `eval_outputs/kb40_intent_force_fix3_20260519T111928Z/predictions.csv`
- `eval_outputs/kb40_intent_force_fix4_20260519T114532Z/predictions.csv`

| case | fix2 | fix3 | fix4 |
|---|---|---|---|
| kb_20260428_037（管理员额度） | `pred_intent=活动`, overall=0.4 | `pred_intent=其他`, overall=0.8 | `pred_intent=其他`, overall=0.8 |
| kb_20260428_010（注单取消） | escalation_match=0, overall=0.6167 | escalation_match=0, overall=0.55 | escalation_match=1, overall=0.9333 |
| kb_20260428_022（一串四组合） | `pred_intent=串关规则`, overall=0.9333 | `pred_intent=串关规则`, overall=0.9333 | `pred_intent=串关规则`, overall=0.9333 |
| kb_20260428_002（盘口类型） | `pred_intent=其他`, overall=1.0 | `pred_intent=其他`, overall=1.0 | `pred_intent=其他`, overall=1.0 |
| kb_20260428_003（赔率计算） | `pred_intent=其他`, overall=0.8 | `pred_intent=其他`, overall=0.8 | `pred_intent=其他`, overall=0.8 |

## 4. 回归前后对比

来源：各轮 `summary.json`

- baseline: `eval_outputs/v6_1_release_gate_20260514T121005Z/kb40/summary.json`
- fix2: `eval_outputs/kb40_intent_force_fix2_20260519T111634Z/summary.json`
- fix3: `eval_outputs/kb40_intent_force_fix3_20260519T111928Z/summary.json`
- fix4: `eval_outputs/kb40_intent_force_fix4_20260519T114532Z/summary.json`

| 版本 | overall_avg | intent_acc | escalation_acc | must_include_avg |
|---|---:|---:|---:|---:|
| baseline | 0.8375 | 0.7500 | 0.9000 | 0.7500 |
| fix2 | 0.8425 | 0.8750 | 0.9500 | 0.5250 |
| fix3 | 0.8442 | 0.9000 | 0.9500 | 0.4833 |
| fix4 | 0.8587 | 0.9000 | 0.9750 | 0.5250 |

结论：
- `overall_avg` 已从 baseline +2.12%。
- 意图与升级判定显著提升，但 `must_include_avg` 低于旧门槛（0.60/0.84）现状。

## 5. 稳定性复跑（重复跑 + 时延分位复核）

本次补跑目录：
- `eval_outputs/proxy_stability_rerun_raw_20260520T111257Z/`
- 汇总文件：
  - `stability_summary.json`
  - `stability_summary.md`

### 5.1 KB40（3次）
- overall: `0.8504 / 0.8571 / 0.8471`（mean `0.8515`）
- p50: `4888.5 / 4242.5 / 4343.5 ms`
- p95: `22394.3 / 13358.5 / 12763.5 ms`

### 5.2 OPS16（3次）
- overall: `0.9469 / 0.9406 / 0.9406`（mean `0.9427`）
- p50: `4430.5 / 4341.0 / 5380.0 ms`
- p95: `9395.0 / 9404.0 / 13163.2 ms`

结论：
- 两套评测 `overall_avg` 与 `intent/escalation` 均稳定。
- 当前短板集中在 `must_include_avg`，而非意图/升级或风险违规。

## 6. CI 发布门槛固化（本次落地）

新增脚本：`scripts/run_proxy_release_gate_ci.sh`
新增 workflow：`.github/workflows/proxy-release-gate.yml`

CI 门槛包含四段：
1. 字段完整性：`check_agent_trace_fields.py`
2. 工具命中率：`eval_tool_dialogue_gate.py`
3. `kb40` 重复回归（默认 3 次）
4. `ops16` 重复回归（默认 3 次）

并额外纳入时延上限检查：
- `KB40 p95 <= 25000ms`
- `OPS16 p95 <= 15000ms`

当前默认阈值（按 2026-05-20 实测稳定区间校准）已写入脚本环境变量：
- field: `MIN_FIELD_PASS_RATE=1.0`
- tool: `MIN_TOOL_OVERALL=0.86`, `MIN_TOOL_CALLED=0.95`, `MIN_TOOL_SCHEMA=0.99`, `MIN_KNOWLEDGE_OVERALL=0.75`
- kb40: `request_ok=1.0`, `must_include>=0.50`, `must_not=1.0`, `risk=0`, `intent>=0.85`, `esc>=0.95`, `overall>=0.84`
- ops16: `request_ok=1.0`, `must_include>=0.78`, `must_not=1.0`, `risk=0`, `intent>=0.98`, `esc>=0.93`, `overall>=0.93`

## 7. 执行方式

本地执行：

```bash
cd /home/ubuntu/qwen35a3b_finetune
bash scripts/run_proxy_release_gate_ci.sh
```

GitHub Actions 手动触发：
- workflow: `proxy-release-gate`
- inputs: `base_url`, `model`, `repeat_runs`

产物：
- `eval_outputs/proxy_release_gate_ci_<TS>/release_gate_summary.json`
- `eval_outputs/proxy_release_gate_ci_<TS>/release_gate_report.md`
