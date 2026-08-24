# Supervisor 双臂评测 · NO_GO（单臂评测环境下无增益）

| 项 | 值 |
|----|-----|
| 日期 | 2026-08-24 |
| 机器 | SeetaCloud 4090D · skip-index · colqwen2 · qwen2:7b · judge=llm · 46q |
| 代码 | feat/agent-supervisor（PR #45）· 含 Phase1 evidence bug 修复 |
| 对比 | runs/20260824-agent-eval-off（off 新基线）vs runs/20260824-agent-eval-supervise（on）|

## 主表

| arm | Correct | atomic | multi_hop | reject | 误拒 | avg searches | lat |
|-----|--------:|-------:|----------:|-------:|-----:|-------------:|----:|
| pipeline | 0.674 | 0.500 | **0.778** | 0.800 | 4 | — | 3.1s |
| agent off | 0.630 | 0.389 | 0.722 | 0.900 | 4 | 2.24 | 6.5s |
| **agent on** | **0.630** | **0.389** | **0.722** | **0.900** | **4** | **2.24** | **7.8s** |

## 门禁判定

| 门禁 | 判定 |
|------|------|
| multi_hop >= 0.778 | **FAIL** 0.722（= off，supervise 零增益）|
| atomic 不拖累 | PASS（0.389 持平）|
| budget | PASS（avg searches 2.24，supervise 未超）|
| 延迟 | on 7.8s vs off 6.5s（+1.3s = supervise 1 次 LLM 调用）|

## 根因：单臂评测架空 supervise

- supervise **确实执行**：trajectory_summary 显示 35/46 条派单、0 fallback。
- 但评测是**单臂注入**（retriever.search 三路融合 search_fn），supervise 的 arms 在
   无  时退化为全臂 → 选臂无效。
- 配额（searches）在每次全量三路检索下也无效：avg searches off=on=2.24。
- 结论：supervise 价值需要**多臂注入评测**（每臂独立 search_fn）才能体现，当前协议测不出。

## 附带结论

- Phase1 evidence bug 修复带来的真实增益：agent multi_hop 8/6 0.667 → 本次 off 0.722（+5.5pt）。
- 但 agent 仍 < pipeline（0.630 vs 0.674），atomic 弱（0.389）→ agent 整体仍 NO_GO（默认关）。

## 决策

- **supervise 不值得保留**（单臂评测无增益 + 更慢）。建议：保留规则派单，删 supervise 节点；
  除非未来建多臂评测环境再验证。 维持 false。
