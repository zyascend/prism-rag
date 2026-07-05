# Run 20260705-ragas-eval — RAGAS 生成层评测（Faithfulness + Answer Relevancy）

> 日期: 2026-07-05
> 分支: feat/ragas-faithfulness
> 环境: AutoDL RTX 4090 24GB，全量检索（BM25 + Dense + Visual + Rerank）
> 模型: ColQwen2-v1.0 (visual), BGE (dense/rerank), qwen2:7b (Judge LLM via Ollama), nomic-embed-text (embedding)

## 目的

首次 RAGAS 生成层评测，验证生成质量（Faithfulness & Answer Relevancy）。

## 评测方式

- 50 条 ViDoRe v3 Industrial 英文 query
- 检索 → 生成 → Faithfulness → Answer Relevancy
- Faithfulness: 声明分解 + LLM 逐条验证（qwen2:7b）
- Answer Relevancy: 反向问题生成 + cosine 相似度

## 结果

| 指标 | 数值 | 说明 |
|------|:----:|------|
| Faithfulness | **0.8867** | 回答 88.7% 的声明被检索上下文支持 |
| Relevancy | **0.8147** | 回答与问题高度相关 |
| 生成回答 | 45/50 (90%) | |
| 拒答 | 5/50 (10%) | 全部合理（文档无对应内容） |
| 耗时 | 8 min 35 s | RTX 4090 |

## 关键发现

### 1. Faithfulness 质量良好
- 中位数 **1.0000** — 超过一半 query 回答完全忠于上下文
- 仅 4 条存在部分声明不被支持（其中 1 条为真正的 hallucination）

### 2. `--skip-index` bug 修复
`run_ragas_metrics.py::build_retriever()` 中 `--skip-index` 同时跳过了 FAISS 索引加载和 BM25 拟合，导致仅 Dense-only 检索可用。修复后始终保持 FAISS 加载 + BM25 拟合。

### 3. 标尺问题
- **拒答误计：** 拒答回答的声明被 LLM judge 判为"不支持"，拉低 Faithfulness ≈0.02
- **Relevancy 标尺偏差：** cosine similarity 对词面不同但语义等价的 pair 区分度不足

## 文件

- ragas_metrics_default.json — 50 条完整评测结果（含每条声明分解和验证）
- badcase_ragas_analysis.md — Bad Case 分析

## 下一步

1. 全量 283 条评测（约 45 min 云上跑完）
2. 云 API Judge（gpt-4o-mini）替换 Ollama 加速到分钟级
3. 修拒答误计标尺缺陷