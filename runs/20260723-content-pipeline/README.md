# Boot-CP summary (20260723)

- mode: skip-index · Full_zerank2 · max_queries=`100` (FULL=0)
- visual: `colqwen2`
- arms: A (off/off) · B1 (expand) · B2 (boost)
- note: section_path/neighbors 在旧索引中为空；B1 用 **page** 模式仍可扩同页 chunk

## NDCG@10 (from ablation_results.json)

| arm-A | 0.3575 | file=runs/20260723-content-pipeline/arm-A/ablation_results.json
| arm-B1 | 0.3575 | file=runs/20260723-content-pipeline/arm-B1/ablation_results.json
| arm-B2 | 0.3575 | file=runs/20260723-content-pipeline/arm-B2/ablation_results.json

## Next

1. Compare Arm-A vs Boot-A Full_zerank2 baseline
2. Compare B1/B2 vs Arm-A
3. Optional: text re-ingest for A1/A3 then re-run Arm-A
4. scp results, shutdown GPU
