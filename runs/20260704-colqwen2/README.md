# Run 20260704-colqwen2 - ColQwen2-v1.0 视觉编码实验

> 日期: 2026-07-04
> 分支: feature/colqwen2-integration
> 模型: vidore/colqwen2-v1.0 (Qwen2-VL-2B backbone)
> GPU: NVIDIA GeForce RTX 4090 24GB

## 目的

验证 ColQwen2 替换 ColPali-v1.3 后 Visual_only 路是否能提升 NDCG@10。

## 对比

| Config | ColPali-v1.3 | ColQwen2-v1.0 | 变化 |
|--------|-------------|---------------|------|
| Visual_only NDCG@10 | 0.1365 | **0.1564** | +14.6% |
| Visual_only Recall@10 | 0.2848 | **0.3254** | +14.3% |
| BM25_Dense_Visual NDCG@10 | 0.4452 | **0.4525** | +1.6% |

## 结论

ColQwen2 相比 ColPali 有约 15% 提升，但 Visual_only 绝对分数仍很低（0.1564），
离预期 SOTA（~0.3-0.5）差距大。两个不同视觉 backbone 结果相近，
暗示问题不在模型选择，而在管道更深层（如 query 编码、attention_mask 处理、评分逻辑）。

## 文件

- ablation_results.json - 消融运行结果
- env.txt - 运行环境依赖版本
