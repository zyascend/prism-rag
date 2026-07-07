# Observability Report — 2026-07-07 04:50 UTC

**Run:** `ragas_results`

## Summary

| Config | N | P50 | P95 | Avg Latency | B-Hits | D-Hits | V-Hits | Faith | Relev | CtxRel |
|--------|---|---|-----|-------------|--------|--------|--------|-------|-------|--------|
| default | 200 | 16836ms | 24812ms | 17308ms | 20.0 | 20.0 | 20.0 | 0.746 | 0.801 | 0.116 |

## Alerts (0)

No alerts.

## Per-Config Detail
### default
- **Queries:** 200
- **Latency:** P50=16836ms, P95=24812ms, P99=26629ms, Avg=17308ms (min=3918, max=29306)
- **Hits:** BM25=20.0, Dense=20.0, Visual=20.0
- **HyDE cache hit rate:** 0.0%
- **Faithfulness:** 0.746
- **Answer Relevancy:** 0.801
- **Context Relevance:** 0.116

## RAGAS Score Distribution

### default (100 queries)

| Metric | Min | P25 | P50 | P75 | Max | Mean |
|--------|-----|-----|-----|-----|-----|------|
| Faithfulness | 0.000 | 0.702 | 1.000 | 1.000 | 1.000 | 0.746 |
| Answer Relevancy | 0.000 | 0.767 | 0.832 | 0.879 | 0.951 | 0.801 |
| Context Relevance | 0.0123 | 0.0596 | 0.1038 | 0.1519 | 1.0000 | 0.1180 |
