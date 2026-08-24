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

### 2026-08-24 实测结果（两轮已完成）

**第一轮（arms 架空）**：on 与 off 逐项相同（0.630/0.389/0.722），根因是评测单臂注入
退化全臂 → 测不出 supervise。

**第二轮（arms 透传修复后重跑）**：

| arm | Correct | atomic | multi_hop | reject | 误拒 | avg searches | lat |
|-----|--------:|-------:|----------:|-------:|-----:|-------------:|----:|
| pipeline | 0.674 | 0.500 | **0.778** | 0.800 | 5 | — | 3.1s |
| agent off | 0.630 | 0.389 | 0.722 | 0.900 | 4 | 2.24 | 6.5s |
| agent on | **0.565** | 0.389 | **0.611** | 0.800 | **6** | 2.26 | 7.8s |

**判定：NO_GO（选臂真实生效后全面变差）**。supervise 选臂导致 multi_hop 0.722→0.611
（-11pt）、reject 0.9→0.8、误拒 4→6。3 条 multi_hop 回归（mh_001/009 被选臂后直接
abstain）——只开单臂漏掉三路交叉 evidence。根因：**qwen2:7b 对「该用哪个臂」判断
不可靠，选臂是负收益，不如规则全臂**。

**决策（2026-08-24）**：supervise 不值得保留（选臂真实生效时有害，非评测架空）。
维持规则派单 + `agent.supervise.enabled: false`。除非换更强模型或「先粗召回再选臂」
的稳健设计再验证。

附带：Phase1 evidence bug 修复真实增益 multi_hop 8/6 0.667 → off 0.722（+5.5pt）；
agent 整体仍 < pipeline（0.630 vs 0.674）→ agent 保持默认关。

---

## 5. 产物归档

- `runs/20260824-agent-eval-off/`（README + results.json + run.log）
- `runs/20260824-agent-eval-supervise/`
- 结论更新 `handoff.md`，标注 Go / NO_GO。
