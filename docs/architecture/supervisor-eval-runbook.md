# Supervisor 云上双臂评测 Runbook（Phase 2 · 46q）

> 目标：验证 supervisor 派单（`agent.supervise.enabled: true`）相对 off（现有 agent）
> 是否有增益，过门禁才谈默认开。
> 关联：[`agent.md`](./agent.md) · 8 月 6 日 off 基线 [`runs/20260806-agent-eval-opt/`](../../runs/20260806-agent-eval-opt/)
> 分支：`feat/agent-supervisor`（PR #45）

---

## 1. 为什么跑两个 arm

supervise 默认关 → off = 现有 agent 行为。但本次分支**顺带修了 Phase 1 遗留 bug**
（grade/synthesize 拿不到 evidence → agent 实际永远 abstain），所以**不能用 8 月 6 日
的 off 数据当对照**——会混入 bug 修复的影响，无法干净归因 supervisor。

必须同机同协议跑两个 arm：

| arm | 命令 | 目的 |
|-----|------|------|
| off（新基线） | `run_agent_eval.py --execute --skip-index`（不加 `--supervise-on`） | 修 bug 后的真 off 基线 |
| on | `run_agent_eval.py --execute --skip-index --supervise-on` | supervisor 派单增益 |

同 46q（atomic 18 / multi_hop 18 / reject 10），同机 SeetaCloud 4090D，同 judge=llm。

---

## 2. 云上命令（各一次，`--output-dir` 分开）

```bash
# 前置：分支 + 依赖 + 索引（skip-index 复用已建 FAISS/BM25）
cd /root/prism-rag && git fetch && git checkout feat/agent-supervisor

# arm 1: supervise OFF（新基线，修 bug 后）
PYTHONPATH=. python scripts/run_agent_eval.py \
  --execute --skip-index \
  --qa-file data/agent_eval_qa.json \
  --judge llm \
  --output-dir runs/20260824-agent-eval-off

# arm 2: supervise ON
PYTHONPATH=. python scripts/run_agent_eval.py \
  --execute --skip-index \
  --qa-file data/agent_eval_qa.json \
  --judge llm \
  --supervise-on \
  --output-dir runs/20260824-agent-eval-supervise
```

> 注意：评测的 `search_fn` 是**单臂注入**（`retriever.search` 三路融合），supervise 派单
> 在单臂下**只影响配额分配**（每子问 `searches`），不影响臂选择（臂选择需要逐臂
> `search_fns` 注入，评测环境未提供）。这是可接受的范围——supervise 的配额感知
> 价值仍可量化。

---

## 3. 门禁判定（照抄 8 月 6 日协议）

| 门禁 | 判定 | 参照 |
|------|------|------|
| atomic Δ ≥ −2pt | on 不拖累 atomic | off 新基线（不是 8/6 的 0.00） |
| **multi_hop ≥ pipeline 0.778** | 核心价值 | pipeline 0.778（同机同跑） |
| budget | avg searches ≤ 3 | off 2.28 / on 应 ≤3 |
| 误拒 | ≤ 4（对齐 pipeline） | off 4 |
| 延迟 | lat mean ≤ ~7s | off 6.4s / on +1 次 LLM 调用 |

结果读 `results.json` 的 `agent.correct` / 分层 `multi_hop` / `atomic` / 误拒 / `latency_ms`。

---

## 4. 结论规则

- **multi_hop on ≥ 0.778 且 atomic 不拖累** → supervisor 有价值，可谈 Go（但仍默认关，
  需 config PR + 人工确认）。
- **multi_hop 无增益或拖累 atomic** → supervisor 不值（一个 LLM 调用换不来派单精度），
  保留规则、删 supervise 节点。

### 2026-08-24 实测结果（已完成）

| arm | Correct | atomic | multi_hop | reject | avg searches | lat |
|-----|--------:|-------:|----------:|-------:|-------------:|----:|
| pipeline | 0.674 | 0.500 | **0.778** | 0.800 | — | 3.1s |
| agent off | 0.630 | 0.389 | 0.722 | 0.900 | 2.24 | 6.5s |
| agent on | 0.630 | 0.389 | 0.722 | 0.900 | 2.24 | 7.8s |

**判定：NO_GO（supervise 零增益 + 慢 1.3s）**。supervise 确实执行（35/46 派单、0 fallback），
但评测是**单臂注入**（`retriever.search` 三路融合），`_search_arms` 无 `search_fns` 退化为
全臂 → 选臂被架空；配额在每次全量三路检索下无效（off=on=2.24）。

**决策（2026-08-24）**：supervise **保留但默认关**（已实现 + 测试全绿 + 零行为影响），
待**多臂注入评测**（每臂独立 search_fn）再验证选臂价值。`agent.supervise.enabled` 维持 false。

附带：Phase1 evidence bug 修复真实增益 multi_hop 8/6 0.667 → 本次 off 0.722（+5.5pt）；
agent 整体仍 < pipeline（0.630 vs 0.674）→ agent 保持默认关。

---

## 5. 产物归档

- `runs/20260824-agent-eval-off/`（README + results.json + run.log）
- `runs/20260824-agent-eval-supervise/`
- 结论更新 `handoff.md`，标注 Go / NO_GO。
