# Golden ablation NDCG (Boot-A Job1)

| config | NDCG@10 | Recall@5 | MRR | avg_latency_ms |
|--------|---------|----------|-----|----------------|
| BM25_only | 0.4063 | 0.3884 | 0.4993 | 16.5 |
| Dense_only | 0.3638 | 0.3231 | 0.4718 | 112.3 |
| Visual_only | 0.1590 | 0.1020 | 0.1727 | 218.4 |
| BM25_Dense | 0.4208 | 0.3837 | 0.5245 | 127.2 |
| BM25_Dense_Visual | 0.4201 | 0.4029 | 0.4973 | 370.8 |
| Full_no_rerank | 0.4201 | 0.4029 | 0.4973 | 0.2 |
| Full_with_rerank | 0.5161 | 0.4631 | 0.6307 | 717.7 |
| Full_zerank2 | 0.5318 | 0.4730 | 0.6601 | 1190.0 |

**Full_no_rerank → Full_zerank2 ΔNDCG@10 = +0.1117**
**BM25_only → Full_zerank2 Δ = +0.1255**

Protocol: standard NDCG log2 + page dedupe; colqwen2; --no-hyde; 283 en queries.
