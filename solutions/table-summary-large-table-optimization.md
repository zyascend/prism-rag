# 表格摘要 + 大表保护：优化过程与结论

> 落档：2026-07-10
> 范围：从问题 → 设计 → 实现 → 云端验证 → 结论的完整复盘
> 关联设计文档：`docs/table-summary-large-table-design-2026-07-09.md`
> 实证来源：`runs/20260709-table-summary-ndcg/`（283q NDCG 10 路消融 + 100q RAGAS）

---

## 0. TL;DR

表格切分优化 = 两个协同特性：**大表保护**（按行切、带表头、绝不被按词切碎）+ **表格摘要**（每张表生成 NL 摘要，检索靠摘要定位、生成展开全表）。

设计与实证闭环：**摘要负责"找得到"，全表负责"答得准"**。

核心结论：
- **功能收益是确定性的**：大表结构可还原 + 补上表格语义检索层，让"表格找得到、答得准"从不可行变可行。
- **生成端 Faithfulness = 0.901，为全部 100q run 中最高**（但混杂编码器升级，不可 100% 归因）。
- **代价是文本路 NDCG 小幅下降**（Dense −7.6% / BM25 −4.1%），属设计内预期权衡。
- **CtxRel 0.2626 既非"翻倍"、也非本特性增益**——这是一次重要复盘（见 §8），最高 CtxRel 实为 0.4102（compress-ratio 实验），且 CtxRel 是精确度指标、对上下文体积敏感，不能用于衡量本特性。

---

## 1. 问题背景（改造前两个具体缺陷）

原 `TextChunker.chunk_page` 对所有超长段落（含表格）统一走"按词切碎"兜底路径：

```python
words = sent.split()
for word in words: ...   # 表格被切成 "|公司|营收" 这类碎片
```

带来两个缺陷：
1. **大表被切碎（结构破坏）**：`|---|---|` 分隔行、列名行、数据行被按词边界切断，检索到的 chunk 是一堆碎片，LLM 无法还原成可读表格 → 问"某表里某行多少"答不出。
2. **表格无语义摘要层**：表格 chunk 与普通文本一样只存 `text`，Dense/BM25 检索只能靠单元格字符串精确匹配，缺乏"这张表整体讲什么、有哪些列、极值在哪"的语义定位能力。

---

## 2. 优化方案（设计闭环）

| 特性 | 目标 | 作用阶段 |
|------|------|----------|
| **大表保护** | 表格按"行"切分且每段保留表头，绝不被按词切碎 | 入库（chunker） |
| **表格摘要** | 为每张表生成 1–3 句 NL 摘要，检索用摘要 embedding 定位，生成时展开全表 | 入库 → 检索 → 生成 |

两者形成闭环：**摘要负责"找得到"，全表负责"答得准"**。

关键分流点（设计文档 §6）：**向量用的是摘要，落库 `text` 仍是完整 Markdown 表**——即"检索靠摘要、生成靠全表"。

---

## 3. 实现落地（代码索引）

| 文件 | 改动 |
|------|------|
| `src/ingestion/text_chunker.py` | 新增 `_merge_table_blocks`（合并被空行拆开的相邻表块）、`_split_table`（按行切、每段带表头）、`_looks_like_table` / `_make_table_chunk`；路由：表格走 `_split_table` 不再进按词切碎路径 |
| `src/ingestion/table_summarizer.py` | 新增 `TableSummarizer.summarize()`，复用 `call_llm`；`lru_cache(2048)` 去重；失败降级返回 `""` |
| `src/store/pgvector_store.py` | `chunks` 表增 `table_summary TEXT` 列（`ALTER TABLE ... ADD COLUMN IF NOT EXISTS` 向后兼容）；`insert_chunks` 8 元组扩为 **9 元组** |
| `src/ingestion/pdf_ingestor.py` / `vidore_ingestor.py` | `__init__` 注入 `TableSummarizer`；`chunk_type=="table"` 时 `embed_text = summary or c.text` |
| `src/generation/generator.py` | `answer()` 对 `chunk_type=="table"` 跳过 `compress_context`、原样喂完整子块；非表仍走 BGE 句级压缩 |

配置项：
- `ingestion.table_summary_enabled`（默认 `True`，关闭则退化为纯文本检索）
- `retrieval.context_compression_ratio`（默认 `0.4`，仅作用于非表格上下文）

---

## 4. 云端验证过程（2026-07-09 run）

1. **重新入库**：`ingest_vidore.py --skip-faiss`，`table_summary_enabled=True` → chunks **8835 = text 6530 + table 2305**，`table_summary` **100% 非空**。
2. **复用 ColQwen2 视觉 FAISS 索引**（未重编码，省 GPU 时间）。
3. **全量 283q NDCG 10 路消融**：Dense / BM25 / Visual 单路及组合 + rerank + HyDE。
4. **RAGAS 100q 生成端评测**：Faithfulness / AnswerRelevancy / ContextRelevancy，82 生成 / 18 拒答。
5. **干净归因**：以"旧分块 vs 表格摘要分块"为唯一变量，消掉视觉编码器差异，量化文本路 NDCG 变化。

---

## 5. 验证结论（实测数据，全部源 json 复核）

### 5.1 检索侧 NDCG@10（283q，10 路，按 NDCG@10 降序）

| config | NDCG@10 | MRR |
|--------|---------|-----|
| **Full_zerank2** ⭐ | **0.5357** | **0.6658** |
| Full_zerank2_HyDE | 0.5273 | 0.6518 |
| Full_with_rerank (bge) | 0.5162 | 0.6356 |
| Full_BGE_HyDE | 0.5054 | 0.6150 |
| BM25_Dense_Visual | 0.4334 | 0.5071 |
| Full_no_rerank | 0.4334 | 0.5071 |
| BM25_Dense | 0.4296 | 0.5376 |
| BM25_only | 0.4248 | 0.5302 |
| Dense_only | 0.3638 | 0.4718 |
| Visual_only | 0.1590 | 0.1727 |

### 5.2 干净归因（旧分块 vs 表格摘要分块，消掉编码器变量）

| 路 | 旧分块 | 表格摘要分块 | Δ | 解读 |
|----|--------|--------------|---|------|
| Dense_only (BGE 文本) | 0.3938 | 0.3638 | **−0.030 (−7.6%)** | 纯"摘要替代整表编码"效应 |
| BM25_only (词法) | 0.4432 | 0.4248 | **−0.018 (−4.1%)** | 摘要关键词比整表少 |
| Visual_only (ColQwen2) | 0.1564 | 0.1590 | +0.003 ≈ 持平 | 分块不影响视觉检索 ✅ |

公平对照（各自最优 reranker）：历史最好 20260702(bge) NDCG 0.5507/MRR 0.6595 vs 当前 Full_zerank2 NDCG 0.5357/MRR **0.6658** → NDCG −2.7%、MRR **+1.0%**。

### 5.3 生成端 RAGAS（100q，表格摘要 ON）

| 指标 | 数值 |
|------|------|
| Faithfulness | **0.901** |
| AnswerRelevancy | **0.804** |
| ContextRelevancy | 0.2626（见 §8 口径警示，非收益信号） |
| 生成 / 拒答 | 82 / 18 |

### 5.4 视觉路不受影响

ColQwen2 NDCG 0.1564 → 0.1590（≈持平），多模态检索路径完整；ColQwen2 升级本身是净加分。

---

## 6. 正向收益总览

### 6.1 确定性功能收益（非指标，是能力修复）

| 收益 | 改造前 | 改造后 |
|------|--------|--------|
| **大表结构保全** | 长表被按词切碎成碎片，LLM 无法还原，行级表格问题答不出 | 按行切、每段带表头，任意单块都是合法 markdown 表 → 表格行级问题可答 |
| **表格语义检索层** | chunk 只存 `text`，Dense/BM25 只能单元格精确匹配，无"表讲什么/哪些列/极值在哪"的定位 | `table_summary` NL embedding 提供语义定位，"找得到正确的表" |

这两点是结构上最实打实的收益，不依赖 RAGAS 数值，正好解决 §1 列的两个缺陷。

### 6.2 生成端质量信号

- **Faithfulness = 0.901，为全部 100q run 中最高**（对比：旧分块 0.8943 / 0.8886 / 0.8862，ctxrel-fix 0.8821），方向与设计"全表进 context、不被压缩删行 → 答得准"一致。
- **AnswerRelevancy = 0.804**（对比 0.7984，微增 +0.7%）。
- ⚠️ 本 run 同时换了 ColQwen2 编码器，**不可 100% 干净归因给表格特性**（见 §9）。

### 6.3 零副作用 + 工程/运维收益

- **生成时大表不灌爆窗口**：generator 对 `chunk_type=="table"` 跳过 `compress_context`、原样喂完整子块（带表头、长度有界），把"保护"贯彻到生成端。
- **向后兼容 + 降级**：`ALTER TABLE ADD COLUMN IF NOT EXISTS`；LLM 摘要失败返回 `""` 自动退化为纯文本检索，入库不报错。
- **入库省 GPU**：复用 ColQwen2 视觉 FAISS 索引未重编码。

---

## 7. 代价（诚实列出）

- **文本路 NDCG 小幅下降**：Dense_only −7.6%（0.3938→0.3638）、BM25_only −4.1%（0.4432→0.4248）——正是"摘要替代整表编码"的预期权衡（摘要关键词比整表少）。
- **CtxRel 0.2626 是精度假降**（分母因注入摘要变大），**不是真实质量损失，也不计入收益**（见 §8）。
- **入库多一次 LLM 调用**：增加 token 成本与耗时，`lru_cache` 仅缓解重复表。

---

## 8. CtxRel 误述纠正（重要复盘）

> 本节能单独成立为"指标口径"教训，已在 `handoff.md` §9 以 footnote 形式记录。

**原误述**：handoff 曾写"表格摘要使 ContextRelevancy 翻倍到 0.263"。经横向核对全部 run，该表述两处错误：

1. **0.263 不是最高的**——全 run 最高 CtxRel 为 **0.4102**（`compress-ratio-025`），连 7/8 的 `ctxrel-fix`（0.2943）都比本 run 的 0.2626 高。
2. **"翻倍"归因错误**——0.087→0.294 的跃升来自 **7/8 的 `compute_context_relevancy` metric 修复**（改评压缩后 context），与任何特性无关；本 run 0.2626 甚至**低于**那次修复的 0.2943。

**根因（指标口径）**：本项目中 `ContextRelevancy = num_relevant / num_sentences`，即**检索上下文的精确度（precision），对上下文体积高度敏感**。

横向核对（同一 100q 集，`query[0]` 一致）：

| run | CtxRel | 平均句数 |
|-----|--------|----------|
| 20260708-compress-ratio-025 | **0.4102（全 run 最高）** | 18.5 |
| 20260708-ctxrel-fix（metric 修复） | 0.2943 | — |
| **20260709-table-summary-ndcg（本 run）** | **0.2626** | 32.1 |
| 20260707-ragas-100-clean（旧分块） | 0.1175 | — |

机制：
- 表格摘要向上下文**注入摘要文本** → 句数 18→32 → 分母变大 → precision **被动下降**。
- 0.4102 来自 `compress-ratio-025`（把上下文压到 0.25），属"砍掉上下文换精度"的假象，**不可作为质量增益**。

**结论**：CtxRel 在本项目**不是特性收益信号**，仅用于监控"上下文是否被稀释"；生成端主目标看 **Faithfulness 0.901**。

**复用教训**：报告 CtxRel 必须 cross-run 核对全部 run，并标注其为 precision 指标；"最高 / 翻倍"类结论先核实口径与基线。

---

## 9. 归因诚实度说明

- **Faithfulness 0.901 最高**，但 7/9 run 同时包含多项变更（ColQwen2 视觉编码器升级、表格摘要分块、可能其他），**不能单独归因于表格特性**。干净 A/B（仅开关 `table_summary_enabled`、其余全相同）在本轮未做。
- **文本路 NDCG 下降** 的归因是干净的（§5.2 以"旧分块 vs 表格摘要分块"为唯一变量），可信度高。
- 若要彻底坐实"表格特性 → 生成端增益"，下一步应补一组 **table_summary_enabled ON/OFF 同编码器对照**的 RAGAS 100q。

---

## 10. 下一步建议

1. **表级聚合 chunk**（设计文档 §11 已点出）：整表摘要 + 行指针，更好支撑"全表最大值在哪行"类跨块全局问题。
2. **补干净 A/B**：同编码器下开关 `table_summary_enabled` 跑 RAGAS 100q，坐实生成端增益归因。
3. **混合 embed 试点**：若想压低文本路 NDCG 损失，可试"Dense 同时编码摘要 + 关键单元格"。
4. **ratio 甜区**：压缩比 0.3 待测（0.4 安全默认、0.25 过激已验证）。

---

## 附录 A：关键数据速查

- 入库：chunks 8835 = text 6530 + table 2305，`table_summary` 100% 非空
- 检索最优：Full_zerank2 NDCG@10 **0.5357** / MRR **0.6658**（283q）
- 生成端：Faithfulness **0.901** / AnswerRelevancy 0.804 / ContextRelevancy 0.2626（82 生成 / 18 拒答）
- 文本路权衡：Dense −7.6% / BM25 −4.1%；视觉路持平
- CtxRel 全 run 最高：0.4102（compress-ratio-025，非本特性）

## 附录 B：文件索引

- 设计：`docs/table-summary-large-table-design-2026-07-09.md`
- 实证：`runs/20260709-table-summary-ndcg/`（results/ablation_results.json、ragas_metrics_default.json、logs/）
- 复盘记录：`handoff.md` §9（footnote ① ContextRelevancy 口径警示）
