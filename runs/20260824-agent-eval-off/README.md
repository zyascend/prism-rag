# Phase2 agent dual-arm eval

- items: **46**
- created: `2026-08-24T12:22:33.029400+00:00`
- git: `unknown` @ `unknown`
- llm: `qwen2:7b`
- visual: `colqwen2`
- judge: `llm`
- skip_index: `True`
- grade_off: `False`

## Verdict (draft)

**NO_GO_DRAFT**

Draft only — human must confirm before any default enabled change. Even GO_DRAFT keeps agent.enabled=false unless separate config PR.

- [FAIL] `atomic_not_hurt`: atomic Δ=-0.1111 (need ≥ -0.02)
- [FAIL] `multi_hop_ge_pipeline`: multi_hop agent=0.7222 pipeline=0.7778
- [PASS] `searches_within_budget`: avg_searches=2.239 budget=3.0
- [PASS] `no_mass_degrade`: degrade=0/46 (0.0%)

## Arms

### pipeline

| metric | value |
|--------|------:|
| n | 46 |
| correct_rate | 0.6739 |
| reject_accuracy | 0.8000 |
| false_reject_count | 4 |
| degrade_count | 0 |
| errors | 0 |
| latency_ms mean/p50/p95 | 3093 / 2813 / 6172 |

| tag | n | correct_rate |
|-----|--:|-------------:|
| atomic | 18 | 0.5000 |
| multi_hop | 18 | 0.7778 |
| reject | 10 | 0.8000 |

### agent

| metric | value |
|--------|------:|
| n | 46 |
| correct_rate | 0.6304 |
| reject_accuracy | 0.9000 |
| false_reject_count | 4 |
| degrade_count | 0 |
| errors | 0 |
| avg_searches | 2.239 |
| avg_llm_calls | 3.522 |
| latency_ms mean/p50/p95 | 6513 / 5891 / 9405 |

| tag | n | correct_rate |
|-----|--:|-------------:|
| atomic | 18 | 0.3889 |
| multi_hop | 18 | 0.7222 |
| reject | 10 | 0.9000 |

## Delta (agent − pipeline)

- correct_rate: -0.0435
- reject_accuracy: +0.1000
- false_reject_count: +0
- latency_ms_mean: +3420.2174
- by_tag.atomic.correct_rate: -0.1111
- by_tag.multi_hop.correct_rate: -0.0556
- by_tag.reject.correct_rate: +0.1000

## Decision discipline

- Do **not** flip `agent.enabled: true` without explicit human Go + config PR.
- NDCG / Boot ablation never use agent path.
- See `results.json` for per-item trajectories.
