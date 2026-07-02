# Run 20260702-query-fix — Query 编码修复（process_queries）

- **日期**: 2026-07-02
- **分支**: fix/visual-query-encoding-no-dummy-image
- **GPU**: RTX 4090 24GB
- **数据集**: vidore/vidore_v3_industrial (English subset, 283 queries)
- **修复**: encode_query/encode_queries_batch 从传 dummy 白图改为 process_queries()

## 消融结果

| Config | NDCG@10 | vs 上次(dummy图) |
|--------|---------|:--:|
| BM25_only | 0.4432 | — |
| Dense_only | 0.3938 | — |
| **Visual_only** | **0.1313** | +33% |
| BM25_Dense | 0.4528 | — |
| BM25_Dense_Visual | 0.4357 | — |
| **Full_with_rerank** | **0.5507** | +2.7% |

## 关键发现
- Visual 路从残废(0.099)恢复到 0.131，+33%
- 但距 ColPali v1.3 官方 Industrial 分数(0.470)仍有 3.6x 差距
- MaxSim 与官方 score() 排名完全一致，根因不在检索算法
