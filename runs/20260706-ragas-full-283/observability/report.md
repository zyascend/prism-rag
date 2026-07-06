# Observability Report — 2026-07-06 06:46 UTC

**Run:** `ragas_results`

## Summary

| Config | N | P50 | P95 | Avg Latency | B-Hits | D-Hits | V-Hits | Faith | Relev | CtxRel |
|--------|---|---|-----|-------------|--------|--------|--------|-------|-------|--------|
|  | 283 | 652ms | 699ms | 653ms | 20.0 | 20.0 | 0.0 | 0.000 | 0.000 | — |
| default | 283 | 0ms | 0ms | 0ms | 0.0 | 0.0 | 0.0 | 0.772 | 0.810 | 0.076 |

## Alerts (0)

No alerts.

## Per-Config Detail
### 
- **Queries:** 283
- **Latency:** P50=652ms, P95=699ms, P99=719ms, Avg=653ms (min=555, max=1182)
- **Hits:** BM25=20.0, Dense=20.0, Visual=0.0
- **HyDE cache hit rate:** 0.0%
- **Faithfulness:** 0.000
- **Answer Relevancy:** 0.000

### default
- **Queries:** 283
- **Latency:** P50=0ms, P95=0ms, P99=0ms, Avg=0ms (min=0, max=0)
- **Hits:** BM25=0.0, Dense=0.0, Visual=0.0
- **HyDE cache hit rate:** 0.0%
- **Faithfulness:** 0.772
- **Answer Relevancy:** 0.810
- **Context Relevance:** 0.076
