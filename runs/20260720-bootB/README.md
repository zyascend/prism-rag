# Boot-B — Visual 路由对照 + RAGAS 生成侧（2026-07-20）

> 协议：eval-protocol v1；检索 Full_zerank2 + ColQwen2；生成 context_filter=**bge**  
> 样本：**150** 条英文 query（非全量 283）  
> 主机：SeetaCloud 4090；Ollama qwen2:7b（RAGAS 前需 `ollama serve`）

## 1. Visual 路由（检索）

| mode | NDCG@5 | NDCG@10 | Recall@5 | MRR | avg_latency_ms |
|------|--------|---------|----------|-----|----------------|
| **always**（每 query 开 Visual） | 0.4312 | **0.4362** | 0.3851 | 0.5559 | 1244 |
| **heuristic**（表/图意图才开） | 0.4009 | **0.4012** | 0.3614 | 0.5292 | **1019** |

### 结论

- 延迟：heuristic 相对 always **约 −18%**（1244 → 1019 ms）
- 质量：NDCG@10 **−0.035**（0.436 → 0.401）
- 面试表述：按需跳过 Visual 可降延迟，全量平均 NDCG 有可测代价；适合延迟敏感场景或后续做 query 分类器。

> 注：150q 子集与 Boot-A 全量 283q（Full_zerank2=0.5318）**不可直接比绝对分**（query 子集 + 路由开关不同）。

## 2. 生成侧 RAGAS（默认 BGE 句级压缩）

| 指标 | 值 |
|------|-----|
| num_queries | 150 |
| generated / rejected | 126 / 24 |
| **Faithfulness** | **0.909** |
| **Answer Relevancy** | **0.797** |
| **Context Relevancy** | **0.258** |

### 结论

- 在 **BGE 压缩默认管线** 上拿到可引用生成侧数字（「有数」）。
- CtxRel ~0.26 与历史压缩后口径（~0.26–0.29）同量级。
- Faith 0.91 在 150 样本上偏高，全量 283 时可能回落（历史全量 Faith ~0.77）；写简历时标明 **150q**。

未跑：LLM 句过滤对照（`RUN_LLM_FILTER=0`）。

## 3. 运行插曲

- 首轮 RAGAS 因 **Ollama 未启动** 大量 `Connection refused`；停掉后 `ollama serve` 再跑成功。
- `cloud_boot_b.sh` 后续应在 RAGAS 前自动拉起 Ollama。

## 4. 文件

```text
bootB_summary.json              # 精简汇总
routing-always/ablation_results.json
routing-heuristic/ablation_results.json
ragas/bge/ragas_metrics_default.json   # 全量明细（较大）
```

## 5. 与 Boot-A 分工

| Boot | 回答什么 |
|------|----------|
| A（283q） | 精排瓶颈：no_rerank 0.42 → zerank 0.53；漂移 0 |
| B（150q） | Visual 按需路由延迟–质量；生成侧 BGE 压缩基线 Faith/Rel/CtxRel |
