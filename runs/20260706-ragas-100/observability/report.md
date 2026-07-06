# Observability Report — 2026-07-06 11:07 UTC

**Run:** `ragas_results`

## Summary

| Config | N | P50 | P95 | Avg Latency | B-Hits | D-Hits | V-Hits | Faith | Relev | CtxRel |
|--------|---|---|-----|-------------|--------|--------|--------|-------|-------|--------|
| default | 200 | 18771ms | 25926ms | 18637ms | 20.0 | 20.0 | 20.0 | 0.760 | 0.819 | 0.087 |

## Alerts (0)

No alerts.

## Per-Config Detail
### default
- **Queries:** 200
- **Latency:** P50=18771ms, P95=25926ms, P99=29532ms, Avg=18637ms (min=9038, max=30347)
- **Hits:** BM25=20.0, Dense=20.0, Visual=20.0
- **HyDE cache hit rate:** 0.0%
- **Faithfulness:** 0.760
- **Answer Relevancy:** 0.819
- **Context Relevance:** 0.087

## RAGAS Score Distribution

### default (100 queries)

| Metric | Min | P25 | P50 | P75 | Max | Mean |
|--------|-----|-----|-----|-----|-----|------|
| Faithfulness | 0.000 | 0.741 | 1.000 | 1.000 | 1.000 | 0.760 |
| Answer Relevancy | 0.592 | 0.780 | 0.818 | 0.867 | 0.946 | 0.819 |
| Context Relevance | 0.0103 | 0.0462 | 0.0784 | 0.1231 | 0.2812 | 0.0893 |
