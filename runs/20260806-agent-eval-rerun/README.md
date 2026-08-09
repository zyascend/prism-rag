# Phase2 agent dual-arm re-run（隔离 fix 后）— **仍 NO_GO**

| 项 | 值 |
|----|-----|
| 日期 | 2026-08-06 |
| 相对 | 对比 [`../20260806-agent-eval/`](../20260806-agent-eval/)（串台 bug 污染） |
| 机器 | SeetaCloud 4090D · skip-index · colqwen2 · qwen2:7b · judge=llm |
| 条数 | **46**（atomic 18 / multi_hop 18 / reject 10） |
| 修复 | `new_thread_id()` + batch `checkpoint=false` |

## Verdict

**NO_GO_DRAFT**（保持 `agent.enabled: false`）

| 门禁 | 结果 |
|------|------|
| atomic Δ ≥ −2pt | **FAIL** Δ=−0.0556 |
| multi_hop ≥ pipeline | **FAIL** 0.556 vs 0.778 |
| avg searches ≤ 3 | PASS 2.20 |
| no mass degrade | PASS 0/46 |

## 主表（修后 · 可辩护）

| arm | Correct | RejectAcc | 误拒 | lat mean | avg searches |
|-----|--------:|----------:|-----:|---------:|-------------:|
| **pipeline** | **0.674** | 0.80 | **4** | ~3s 量级 | — |
| **agent** | **0.587** | **0.90** | **7** | ~7–10s | 2.20 |

### 分层 Correct

| tag | pipeline | agent | Δ |
|-----|--------:|------:|--:|
| atomic | 0.500 | 0.444 | −0.056 |
| multi_hop | **0.778** | **0.556** | **−0.222** |
| reject | 0.800 | **0.900** | +0.100 |

### vs 串台前（同日 NO_GO 污染跑）

| 指标 | 串台前 agent | **修后 agent** |
|------|-------------:|---------------:|
| Correct | 0.217 | **0.587** |
| 误拒 | 33 | **7** |
| multi_hop Correct | 0.056 | **0.556** |
| evidence_n max | 611 | **23** |
| status abstain | 41/46 | 16/46 |

→ **隔离修复有效**；修后 agent 仍整体弱于 pipeline，**尤其 multi_hop 未兑现设计目标**。

## 隔离自检

- `evidence_n` min/max/mean = **5 / 23 / 13.1**（无跨题单调累加）
- agent 偶有 win：`mh_010`, `mh_018`（pipeline 错、agent 对）
- multi_hop 主要损失：`mh_002/008/011/013/014/017`（pipeline 对、agent 错/拒）

## 决策

- **不**改 `agent.enabled: true`
- 不自动路由进 agent
- 可选后续：关 grade 对照、改写 decompose prompt、仅 Demo 显式 `mode=agent`
- 产物：`results.json` · `env.txt` · `run.log`
