# Boot-R summary (20260725)

| 项 | 值 |
|----|-----|
| 主机 | SeetaCloud 4090D · skip-index · colqwen2 · qwen2:7b |
| 臂 | `base` · `r1` · `pl` · `cr` |
| 冻结 | `eval_via_generator=true` · CRAG off · Gate2 off · expand/boost off |
| 方案 | [p0-p1-refiner-planning-impl](../../docs/superpowers/plans/2026-07-25-p0-p1-refiner-planning-impl.md) |
| 对照分析 | [mrag-survey-optimization-analysis](../../docs/assessments/2026-07-25-mrag-survey-optimization-analysis.md) |
| 日志 | `boot_r_20260725.log`（本目录） |
| 墙钟 | ~1h57m（21:01–22:58 CST） |

## 臂含义

| arm | 变更 |
|-----|------|
| **base** | refiner=`bge` · planning off · crossref off |
| **r1** | refiner=**soft_rank**（`prune_below=null`，`keep_ratio=0.5`） |
| **pl** | `search_planning.enabled=true` · heuristic · `default_visual=false` |
| **cr** | `crossref_expand.enabled=true` · post_rerank |

## 指标总表（机器可读 JSON → 汇总）

| arm | Faith | Rel | CtxRel | RAGAS拒答 | E2E Correct | E2E Reject | latency | 误拒(可答) |
|-----|------:|----:|-------:|----------:|------------:|-----------:|--------:|-----------:|
| base | 0.898 | 0.814 | 0.256 | 17 | **0.62** | 0.90 | 2.25s | 8 |
| r1 | 0.881 | 0.811 | 0.210 | 17 | **0.66** | **0.95** | 2.31s | **6** |
| pl | 0.887 | 0.812 | 0.245 | 15 | 0.62 | 0.90 | **1.99s** | 7 |
| cr | **0.904** | 0.815 | 0.257 | 18 | **0.64** | 0.90 | 2.32s | 8 |

### Δ vs base

| arm | Faith | Correct | RejectAcc | latency | 误拒 |
|-----|------:|--------:|----------:|--------:|-----:|
| r1 | −1.7pt | **+4pt** | +5pt | ×1.03 | −2 |
| pl | −1.1pt | 0 | 0 | **×0.88 (−12%)** | −1 |
| cr | +0.6pt | **+2pt** | 0 | ×1.03 | 0 |

## Go / No-Go 定稿

| 臂 | 判定 | 理由 |
|----|------|------|
| **r1 soft_rank** | **Go · 可试开** | Correct +4pt；误拒↓；latency ≤×1.15；Faith −1.7 在 −2pt 边界内。CtxRel↓ 不作否决。 |
| **pl planning** | **Go · 可试开** | Correct 持平；**latency −12%**（主收益）。 |
| **cr crossref** | **弱 Go · 可灰度** | Correct +2pt；Faith 略升；误拒未降；单开收益小于 r1。 |
| CtxRel | 观察 only | r1 掉 4.6pt **不作上线否决** |

## 默认配置建议（合入后）

| 开关 | 建议 | 备注 |
|------|------|------|
| `refiner.mode` | **soft_rank**（可先生产灰度） | 保持 `prune_below: null` |
| `retrieval.search_planning.enabled` | **true**（可试） | 黄金消融仍应 always_full / planning off 保证可比 |
| `retrieval.crossref_expand.enabled` | 灰度 / 与 soft_rank 组合再验 | 单开中等增益 |
| CRAG / Gate2 always | **保持 false** | 与本次无关 |

> 改默认前建议：本地确认 `models.yaml` + 可选 best 臂（soft_rank+planning+crossref）未做，组合效果未知。

## 产物布局

```text
runs/20260725-boot-r/
  env.txt · README.md · boot_r_20260725.log
  base|r1|pl|cr/
    models.boot.yaml
    ragas/ragas_metrics_default.json · run.log
    e2e/e2e_qa_results.json · badcase_e2e_qa_analysis.md · run.log
```

## 下一步

1. 更新 `handoff.md` 写入本表与默认建议  
2. （可选）改 `config/models.yaml` 默认 · PR  
3. **关机省钱**（云 GPU 已空闲）
