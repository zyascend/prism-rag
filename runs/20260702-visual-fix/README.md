# Run 20260702-visual-fix — Visual 路 OOM 修复验证

- **日期**: 2026-07-02
- **分支**: feat/visual-path-fix-and-eval-alignment
- **GPU**: NVIDIA GeForce RTX 4090 24GB
- **数据集**: vidore/vidore_v3_industrial (English subset, 283 queries)
- **CUDA OOM**: 0 次

## 消融结果

| Config | NDCG@5 | NDCG@10 | Recall@5 | Recall@10 | MRR | Lat(ms) |
|--------|--------|---------|----------|-----------|-----|---------|
| BM25_only | 0.4358 | 0.4432 | 0.4206 | 0.5154 | 0.5443 | 23.3 |
| Dense_only | 0.3838 | 0.3938 | 0.3739 | 0.4694 | 0.5137 | 92.2 |
| **Visual_only** | **0.0965** | **0.0988** | **0.1104** | **0.1331** | **0.1177** | 267.1 |
| BM25_Dense | 0.4403 | 0.4528 | 0.4389 | 0.5525 | 0.5595 | 119.8 |
| BM25_Dense_Visual | 0.4255 | 0.4388 | 0.4390 | 0.5596 | 0.5406 | 399.0 |
| Full_no_rerank | 0.4255 | 0.4388 | 0.4390 | 0.5596 | 0.5406 | 404.2 |
| **Full_with_rerank** | **0.5326** | **0.5362** | **0.4912** | **0.5855** | **0.6482** | 612.7 |

## 与上次运行对比 (20260701_2118, 全 1698 query)

| Config | 上次 | 本次 | 变化 |
|--------|------|------|------|
| Visual_only | ⚠️ 0.0000 (OOM) | ✅ 0.0988 | 修复 |
| Full_with_rerank | 0.3136 | 0.5362 | +71% (含语言过滤) |

## 修复要点

1. **ColPali + FAISS 显存分离**: Phase A 预编码 query → unload ColPali → Phase B 加载 FAISS
2. **MaxSim 分块计算**: ColPali query ~1032 patches, 全量 matmul 中间矩阵 21GB → 按页 batch (200页/批) 降至 ~800MB
3. **语言过滤**: English 283 query 子集，对齐 ViDoRe V3 论文评估口径
