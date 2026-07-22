# CRAG ON-only 云上对照（2026-07-22）

> 完整叙事与决策见仓库根目录 `handoff.md` §0.1–0.3。  
> 本目录为机器可读产物 + 简表。

## 协议

| 项 | 值 |
|----|-----|
| 臂 | **仅 ON**（OFF 不重跑） |
| OFF 基线 | `runs/20260721-self-rag-on-only/comparison_post_p0.json` → off_recomputed |
| CRAG | `enabled=true`；Gate2 **关**；`eval_via_generator=true` |
| 模型 | colqwen2 + ollama `qwen2:7b`；`--skip-index` |
| 任务 | RAGAS 100 → E2E 70 |
| 墙钟 | RAGAS ~25.4 min · E2E ~10 min · DONE 15:59:50 UTC |
| 机器 | SeetaCloud 4090D |

## 对照表

| 指标 | OFF (post-P0) | CRAG ON | Δ |
|------|-------------:|--------:|--:|
| Faith @100 | 0.9188 | 0.8937 | −2.5pt |
| Rel | 0.8157 | 0.8223 | +0.7pt |
| CtxRel | 0.2549 | **0.3688** | **+11.4pt** |
| RAGAS 拒答 | 17 | 29 | +12 |
| E2E Correct | **0.60** | **0.48** | **−12pt** |
| E2E Reject | 0.90 | 0.90 | 0 |
| 可答题误拒 | ~6 (post-P0) | **10** | ↑ |
| E2E latency | 2.24 s | **7.04 s** | **×3.1** |

## 结论（定稿）

1. **机制有效**：CtxRel 明显上升 → grade/filter 在去噪。  
2. **业务无效**：E2E Correct 大跌 + 可答题误拒↑ → 过严过滤/改写伤证据。  
3. **成本不回本**：延迟 ×3.1，Faith 未涨。  
4. **决策**：`retrieval.crag.enabled` **默认 false**；禁止用本次配置写「涨分」简历。  
5. **主矛盾仍在检索/分块**，不是再叠一层 LLM 审证据就能解。

可选下一轮（非必须）：关 reformulate、软过滤不丢 chunk、仅 low_rerank 触发。

## 产物

| 路径 | 说明 |
|------|------|
| `on/models.boot.yaml` | 冻结配置 |
| `on/ragas/ragas_metrics_default.json` | RAGAS 明细 |
| `on/e2e/e2e_qa_results.json` | E2E 明细 |
| `on/e2e/badcase_e2e_qa_analysis.md` | E2E badcase |
| `crag_on_100.log` | 主机完整日志 |
| `env.txt` | 启动环境快照 |
