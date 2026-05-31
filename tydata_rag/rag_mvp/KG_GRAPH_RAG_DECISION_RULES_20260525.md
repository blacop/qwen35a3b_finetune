# KG / Graph RAG Decision Rules

## Goal

在传统 RAG 基线已经相对稳定的前提下，明确是否需要进入 KG / Graph RAG 实验阶段，避免因为少量失败样本或主观判断过早加复杂度。

## Current Baseline Focus

当前先以以下目标作为传统 RAG 的发布前门槛：

- 主集 `recall@1 >= 0.90`
- 主集 `answer_supported_rate >= 0.95`
- 多证据专项 `avg_evidence_coverage_rate >= 0.70`
- 多证据专项 `full_evidence_cover_rate` 持续提升，且不能明显拖累主集 `recall@1`

## What To Optimize Before KG

在进入 KG / Graph RAG 前，优先把以下传统 RAG 手段做完并稳定：

- chunk size / overlap 调优
- `min_score` / `candidate_k` / `rrf_k` 调优
- rerank 权重调优
- source family 去重与 diversity 重排
- query pattern routing
- 多证据题单独 profile，而不是全量主链路直接开 diversity
- 评测题集补齐跨文档、跨玩法、跨规则差异场景

## Enter KG / Graph RAG Only If

只有当以下条件同时满足时，才建议进入 KG / Graph RAG 实验：

1. 传统 RAG sweep 后，主集指标已经稳定在发布门槛附近或以上。
2. 多证据专项仍存在明显短板，尤其是：
   - `full_evidence_cover_rate` 长期偏低
   - 同类文档已做去重 / diversity 后仍无法补齐关键证据
3. 失败样本复盘显示问题核心是“跨节点关系组合”而不是：
   - 命中错文档
   - 首条排序不稳
   - 证据块切分不合理
   - 提示词没有把已检索到的证据说透

## Do Not Enter KG Yet If

满足以下任一情况，都不建议进入 KG / Graph RAG：

- 主集 `recall@1` 或 `answer_supported_rate` 还不稳
- 多证据题失败主要来自 top-k 没捞全，而不是关系推理
- 单文档题因为多样性或复杂路由出现回归
- 评测题集规模还不足以区分“检索问题”和“推理问题”

## Practical Recommendation

当前推荐流程：

1. 先用 `sweep_typlay_retrieval.py` 固化主集和多证据集的最优参数。
2. 再按新增题集复核跨文档 / 跨玩法 / 跨规则差异三类场景。
3. 若多证据专项在传统 RAG 最优参数下仍长期卡住，再抽失败样本做 Graph RAG / KG 小样本 AB。

## Expected KG Benefit Scope

KG / Graph RAG 更可能对以下问题有帮助：

- 需要跨多个规则文档拼接关系链的问答
- 同一概念在不同平台、不同玩法下存在多跳映射
- 单条证据不完整，必须通过实体关系合并才能得出结论

而对以下问题，收益通常有限：

- 单文档规则抽取题
- 只是首条排序不稳的问题
- 只是 chunk 切分导致证据被截断的问题
