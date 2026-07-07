# Observability Report — 2026-07-07 02:51 UTC

**Run:** `ragas_results`

## Summary

| Config | N | P50 | P95 | Avg Latency | B-Hits | D-Hits | V-Hits | Faith | Relev | CtxRel |
|--------|---|---|-----|-------------|--------|--------|--------|-------|-------|--------|
| default | 200 | 13770ms | 20788ms | 13894ms | 20.0 | 20.0 | 20.0 | 0.744 | 0.798 | 0.117 |

## Alerts (0)

No alerts.

## Per-Config Detail
### default
- **Queries:** 200
- **Latency:** P50=13770ms, P95=20788ms, P99=22394ms, Avg=13894ms (min=3541, max=25053)
- **Hits:** BM25=20.0, Dense=20.0, Visual=20.0
- **HyDE cache hit rate:** 0.0%
- **Faithfulness:** 0.744
- **Answer Relevancy:** 0.798
- **Context Relevance:** 0.117

## RAGAS Score Distribution

### default (100 queries)

| Metric | Min | P25 | P50 | P75 | Max | Mean |
|--------|-----|-----|-----|-----|-----|------|
| Faithfulness | 0.000 | 0.702 | 1.000 | 1.000 | 1.000 | 0.744 |
| Answer Relevancy | 0.000 | 0.774 | 0.826 | 0.866 | 0.951 | 0.798 |
| Context Relevance | 0.0123 | 0.0596 | 0.0955 | 0.1565 | 1.0000 | 0.1199 |
