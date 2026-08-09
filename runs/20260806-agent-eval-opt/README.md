# Phase2 agent dual-arm · **证据汇合 + decompose v2 后**

| 项 | 值 |
|----|-----|
| 日期 | 2026-08-06 |
| 相对 | 隔离修后 [`../20260806-agent-eval-rerun/`](../20260806-agent-eval-rerun/) |
| 机器 | SeetaCloud 4090D · skip-index · colqwen2 · qwen2:7b · judge=llm · 46q |
| 代码 | diversify evidence + synthesis k scale + `agent_decompose` v2 |

## Verdict

**NO_GO_DRAFT**（multi_hop 仍 < pipeline；**不**改 `agent.enabled`）

| 门禁 | 结果 |
|------|------|
| atomic Δ ≥ −2pt | **PASS** Δ=+0.00 |
| multi_hop ≥ pipeline | **FAIL** 0.667 vs 0.778 |
| avg searches ≤ 3 | PASS 2.28 |
| no mass degrade | PASS |

## 主表

| arm | Correct | RejectAcc | 误拒 | lat mean | avg searches |
|-----|--------:|----------:|-----:|---------:|-------------:|
| pipeline | **0.674** | 0.80 | 4 | 3.2s | — |
| **agent (opt)** | **0.630** | 0.80 | **4** | 6.4s | 2.28 |

### 分层 Correct

| tag | pipeline | agent | Δ |
|-----|--------:|------:|--:|
| atomic | 0.500 | **0.500** | 0 |
| multi_hop | **0.778** | **0.667** | −0.111 |
| reject | 0.800 | 0.800 | 0 |

## 三版对比（同日同机 46q）

| 版本 | agent Correct | multi_hop | 误拒 | evidence_n max | atomic Δ |
|------|-------------:|----------:|-----:|---------------:|---------:|
| 串台污染 | 0.217 | 0.056 | 33 | 611 | −0.39 |
| 隔离修后 | 0.587 | 0.556 | 7 | 23 | −0.056 |
| **汇合+dec v2** | **0.630** | **0.667** | **4** | **22** | **0** |

→ 优化方向正确：总 Correct +4.3pt、multi_hop +11pt、误拒对齐 pipeline、atomic 不伤。  
→ 仍差 multi_hop **−11pt** / 总 **−4pt**，**延迟约 ×2**，不足以 Go。

## 解读

- **有效：** 双侧证据进窗口 + 少误拒；decompose 后 atomic 更稳。  
- **未够：** multi_hop 仍输 pipeline（only_pipe 5 vs only_agent 3）。  
- **成本：** lat mean 6.4s vs 3.2s；avg searches 2.3。

## 决策

- 保持 **`agent.enabled: false`**
- 可选下一刀：合并后全局 rerank、`per_subquery=3`、更强 LLM、仅 Demo `mode=agent`
- 产物：`results.json` · `env.txt` · `run.log`
