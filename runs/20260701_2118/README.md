# Run 20260701_2118

- **日期**: 2026-07-01
- **平台**: AutoDL RTX 4090 24GB
- **数据集**: ViDoRe v3 Industrial (5244 pages, 1698 queries)
- **Python**: conda 3.12.3, PyTorch 2.8.0+cu128
- **模型**: ColPali v1.3 + BGE-large-en-v1.5 + BGE-Reranker-large

## 已知问题
- Visual 路 CUDA OOM：ColPali 模型占 ~11.4GB，24GB 显存不足以同时跑 MaxSim
- 含 Visual 的配置实际分数 = BM25_Dense

## 最佳结果
- Full_with_rerank: NDCG@5=0.2468, Recall@5=0.2331, MRR=0.3136
