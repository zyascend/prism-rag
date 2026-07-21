# Self-RAG Gate2 A/B (20260721)

| 项 | 值 |
|----|----|
| 主机 | SeetaCloud RTX 4090 D (`connect.cqa1.seetacloud.com:28596`) |
| 代码 | `feat/self-rag-gate2`（上传 tarball，含 Gate2 + `cloud_self_rag_ab.sh`） |
| visual | colqwen2 |
| RAGAS | 100 queries, English |
| E2E | 50 answerable + 20 rejection (`data/e2e_qa.json`) |
| 两臂共同 | `eval_via_generator=true`；`visual_routing.enabled=false`；`context_filter.mode=bge`；LLM=`qwen2:7b` via Ollama |
| 唯一变量 | `generation.self_rag.enabled` false vs true |

## 指标表

| arm | Faith | Rel | CtxRel | RAGAS 拒答 | E2E Correctness | E2E Rejection | E2E latency | E2E 可答被拒 |
|-----|------:|----:|-------:|-----------:|----------------:|--------------:|------------:|-------------:|
| **off** (Gate2 关) | **0.830** | **0.822** | 0.255 | 2/100 | **0.60** | 0.25 | 2.24 s | 0 |
| **on** (Gate2 开) | 0.786 | 0.763 | 0.255 | 0/100 | **0.60** | **0.95** | 3.96 s | 3 |
| **Δ** | **−0.044** | **−0.059** | +0.000 | −2 | 0 | **+0.70** | **+77%** | +3 |

## 结论

1. **Gate2 未提升 Faithfulness**（0.830 → 0.786），Answer Relevancy 同步下降。  
   → **默认保持 `self_rag.enabled=false`**；不写入简历主 bullet ③ 涨分句。
2. **E2E 可答正确率持平 0.60**；ON 侧多 3 条可答题被误拒（over-abstain 风险）。
3. **拒答准确率**：OFF 仅 0.25（`eval_via_generator` 路径下基线偏爱硬答），ON 回到 **0.95**——Gate2 对「不该答」更有效，但对「该答」Faith 无益。
4. **延迟**：E2E 平均 2.24s → 3.96s（约 **×1.8**），符合 +judge（及偶发 regen）预期。
5. CtxRel 两臂几乎相同（0.255）——符合「Gate2 不改检索/压缩」的设计。

## 简历 / 口述建议

| 用途 | 写法 |
|------|------|
| 简历正文 | **不写** Faith↑；可只在口述写「做了生成后忠实性门对照实验」 |
| 口述（阴性有价值） | 「整答 Gate2 在 100q 上 Faith −4pt、延迟 ×1.8；拒答集准确率从 0.25 拉到 0.95，但可答侧有误杀；默认关闭、高风险场景可开」 |
| 后续若要翻盘 | 调阈值 / claim 级 / 换 judge 模型；或只对低 rerank 置信 query 开 Gate2 |

## 产物路径

- `off/ragas/ragas_metrics_default.json` / `on/ragas/...`
- `off/e2e/e2e_qa_results.json` / `on/e2e/...`
- `off/models.boot.yaml` / `on/models.boot.yaml`
- `pipeline.log`

## 运维

- 结果已拉回本地 `runs/20260721-self-rag-gate2/`
- **请关机释放 GPU 计费**
