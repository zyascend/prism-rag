# Phase2 agent dual-arm eval — **NO_GO**

| 项 | 值 |
|----|-----|
| 日期 | 2026-08-06 |
| 机器 | SeetaCloud 4090D · `connect.cqa1.seetacloud.com:28596` |
| 索引 | colqwen2 industrial · `--skip-index` · 8835 chunks |
| LLM / judge | qwen2:7b (Ollama) · `judge=llm` |
| 条数 | **46**（atomic 18 / multi_hop 18 / reject 10） |
| 代码 | 本机 `feat/agent-phase2-eval` 上传包（云上无 git） |

## Verdict

**NO_GO_DRAFT → 书面 No-Go（保持 `agent.enabled: false`）**

| 门禁 | 结果 |
|------|------|
| atomic Δ ≥ −2pt | **FAIL** Δ=−0.3889 |
| multi_hop ≥ pipeline | **FAIL** 0.0556 vs 0.7778 |
| avg searches ≤ budget | PASS 2.20 ≤ 3 |
| no mass degrade | PASS 0/46 |

## 主表

| arm | Correct | RejectAcc | 误拒 | latency mean | avg searches |
|-----|--------:|----------:|-----:|-------------:|-------------:|
| **pipeline** | **0.652** | 0.80 | **4** | 3.1s | — |
| **agent** | **0.217** | 0.80 | **33** | 10.1s | 2.20 |

### 分层 Correct

| tag | pipeline | agent | Δ |
|-----|--------:|------:|--:|
| atomic | 0.444 | 0.056 | −0.39 |
| multi_hop | **0.778** | **0.056** | **−0.72** |
| reject | 0.800 | 0.800 | 0 |

## 根因（跑后诊断，非门禁内）

1. **大面积误拒**：agent `status=abstain` **41/46**；可答题 `false_reject=33`（pipeline 仅 4）。  
2. **状态疑似串台**：`counts.evidence_n` 随题号单调涨（at_001=20 → rj_010=**611**）；`trajectory_summary` 出现多轮 `decompose` 叠在同一条上。  
   → 高度怀疑 **MemorySaver / 图状态 / evidence reducer 在 batch 连续 `run_agent` 间未隔离**。  
3. 在污染证据上，`synthesize` 大量输出 “context does not contain…”，而 **同题 pipeline 能答对**。  
4. 因此本数字 **主要证明当前 agent 旁路不可上线**；修好状态隔离后需 **重跑** 才能评判拆问/多跳是否真有增益。

## 决策

- **不**改 `agent.enabled: true`  
- **不**做启发式自动路由进 agent  
- **隔离 fix 已合分支**（`runner.new_thread_id` + eval 关 checkpoint + 单测）→ 需 **云上 46 重跑** 才能更新决议  

---

## Fix status（同日后续）

| 项 | 状态 |
|----|------|
| 根因 | `thread_id="local"` + MemorySaver 合并 `evidence`/`trajectory` reducer |
| 修复 | `src/agent/runner.py` 默认 UUID thread；`agent_answer_for_eval` 强制 checkpoint off |
| 单测 | `tests/test_agent_runner.py` 防串台 + 共享 id 泄漏对照 |
| 重跑 | **未做** — 见下一次 `runs/YYYYMMDD-agent-eval-rerun/` |

## 产物

- `results.json` · `env.txt` · `run.log`  
- 云路径：`/root/prism-rag/runs/20260806-agent-eval/`  
