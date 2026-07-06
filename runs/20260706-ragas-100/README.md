# RAGAS 100-Query Eval — 2026-07-06

## 配置

| 参数 | 值 |
|------|-----|
| 查询数 | 100 (ViDoRe v3 Industrial, English-only) |
| 视觉模型 | ColQwen2-v1.0 |
| 上下文压缩 | 0.4 (BGE 句级 cosine 过滤) |
| 置信度阈值 | 0.0 (禁用，待校准) |
| GPU | RTX 4090 24GB |
| 耗时 | 31 min |

## 结果

| 指标 | 值 |
|------|:--:|
| Faithfulness (excl. rejected) | **0.8943** |
| Faithfulness (incl. rejected) | 0.7601 |
| Answer Relevancy | **0.8186** |
| Context Relevance | **0.0866** |
| 生成回答 | 85 |
| 拒答 | 15 (15%) |

## Faithfulness 分布

| Score | Count |
|------:|:-----:|
| 0.0 | 17 (rejected) |
| 0.3 | 1 |
| 0.5 | 2 |
| 0.6 | 3 |
| 0.7 | 8 |
| 0.8 | 12 |
| 0.9 | 4 |
| 1.0 | **53** |

## 上下文压缩

- 压缩前: 平均 88 句
- 压缩后: 平均 35 句
- 实际压缩比: 0.39

## Rerank Score 分布

- P10: 0.033, P25: 0.146, P50: 0.684, P75: 0.917, P90: 0.972
- 建议重启阈值: 0.15

## 对比基线

| 指标 | 100-query | 283-query (旧) | 50-query (旧) |
|------|:--:|:--:|:--:|
| Faithfulness | 0.8943 | 0.7721 | 0.8867 |
| Relevancy | 0.8186 | 0.8104 | 0.8147 |
| CtxRel | 0.0866 | 0.0759 | — |

## 文件

- `ragas_metrics_default.json` — 完整 per-query 结果 (2.8MB)
- `observability/` — traces, metrics, ragas_details, report
- `ragas_100_v3.log` — 运行日志
