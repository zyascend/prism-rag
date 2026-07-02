# Run 20260702-page-query-fix — Page 编码修复（process_images）

- **日期**: 2026-07-02
- **分支**: fix/visual-query-encoding-no-dummy-image
- **GPU**: RTX 4090 24GB
- **数据集**: vidore/vidore_v3_industrial (English subset, 283 queries)
- **修复**: encode_pages 从 text=[""] 改为 process_images()（含 Describe the image. prompt）

## 消融结果

| Config | NDCG@10 | vs Query fix only |
|--------|---------|:--:|
| BM25_only | 0.4432 | — |
| Dense_only | 0.3938 | — |
| Visual_only | 0.1302 | ≈0 |
| BM25_Dense | 0.4528 | — |
| BM25_Dense_Visual | 0.4402 | +0.004 |
| **Full_with_rerank** | **0.5506** | ≈0 |

## 关键发现
- Page 编码 fix 几乎没有效果（Visual NDCG 不变）
- "Describe the image." prompt 不是瓶颈
- 已排查 MaxSim、query 编码、page 编码，均与官方一致
- Visual 路剩余 gap (0.13 vs 官方 0.47) 来源不明，建议转向换组件策略
