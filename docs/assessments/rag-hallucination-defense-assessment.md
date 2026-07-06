# RAG 幻觉防御体系 — 项目现状评估

> **日期**: 2026-07-06
> **范围**: 对照业界完整 RAG 幻觉防御方案（数据→检索→生成→评估 四维度），逐项检查 PrismRAG 项目实现覆盖度
> **当前表现**: Faithfulness=0.8867, 真实幻觉率 ~2% (50条中1条严重)

---

## 一、检索阶段

### 1. 混合检索 + 重排序

| 完整方案 | 项目实现 | 状态 |
|:--------|:---------|:----:|
| 混合检索（向量+关键词）+ Reranker 精排，剔除不相关文档块，只喂 Top-3~5 | BM25 + Dense(BGE 1024d) + Visual(ColPali/ColQwen2) 三路 → RRF 融合 → **双 Cross-encoder Reranker**(BGE-large / zerank-2) → Top-5 | **✅ 超配** |

做法：
- `PrismRAGRetriever.search()` — 见 `src/retrieval/vidore_adapter.py:62`
- Reranker 实现 — `src/retrieval/reranker.py`
- 实测 NDCG@10=0.5715，超过论文 pipeline SOTA (0.532)

### 2. 上下文压缩与过滤

| 完整方案 | 项目实现 | 状态 |
|:--------|:---------|:----:|
| 生成前使用小模型/规则提取 chunk 中关键句子，剔除无关内容 | **未实现**。retrieved chunks 全量拼入 context 送 LLM | **❌** |

缺口：
- `evaluate_generation()` 中 `context = "\n\n---\n\n".join([r.get("text") for r in retrieved])` — 见 `src/evaluation/ragas_metrics.py:643`
- 无 `ContextualCompressionRetriever` 或类似机制
- 噪音段落（如表格描述、列表编号）直接进入 LLM 上下文，可能干扰注意力

### 3. 查询重写

| 完整方案 | 项目实现 | 状态 |
|:--------|:---------|:----:|
| 检索前用 LLM 重写用户问题，补全缺失上下文 | **已实验，结论无效**。HyDE 查询改写 NDCG Δ<0.005 | **⚠️ 放弃** |

做法 & 结论：
- `HyDEGenerator` — 见 `src/retrieval/hyde.py`
- 消融对比：`Full_BGE_HyDE` (0.5458) vs `Full_BGE` (0.5506)；`Full_zerank2_HyDE` (0.5733) vs `Full_zerank2` (0.5715)
- Δ<0.005，不具备实用价值
- 本项目为单轮 QA（无多轮对话），无需对话历史重写

---

## 二、生成阶段

### 1. 严格的系统提示词

| 完整方案 | 项目实现 | 状态 |
|:--------|:---------|:----:|
| 明确划定边界：仅基于上下文、不足则拒答、严禁编造 | Prompt 覆盖三要素 | **✅** |

```python
GENERATION_PROMPT = """\
Answer the question based ONLY on the provided context.
If the context does not contain enough information, say "I cannot answer..."
Do NOT make up information."""
```
见 `src/evaluation/ragas_metrics.py:187`

**缺项**: 未要求引用资料编号（如 `[1]`），上下文拼入时也没有编号标记。

### 2. 引入思维链

| 完整方案 | 项目实现 | 状态 |
|:--------|:---------|:----:|
| 先展示证据提取过程，再给出最终答案 | **未实现**。生成直接输出答案 | **❌** |

缺项：Prompt 未包含 `<thinking>` 标签要求，LLM 直接在 `<answer>` 中输出。

### 3. 控制生成参数

| 完整方案 | 项目实现 | 状态 |
|:--------|:---------|:----:|
| Temperature 调低，降低随机性 | `temperature=0.1` | **✅** |

见 `src/evaluation/ragas_metrics.py:116`，Ollama 调用时设置。

---

## 三、数据处理阶段

### 1. 优化文档切分策略

| 完整方案 | 项目实现 | 状态 |
|:--------|:---------|:----:|
| 语义切分/父子块检索（检索小块→喂大块） | 段落边界 + 句子回退 + 512 token 上限 | **⚠️ 半实现** |

做法：
- `TextChunker` — 见 `src/ingestion/text_chunker.py`
- 按双换行切段落 → 近语义边界
- 长段落后按句号/换行切到 ≤512 tokens
- 超长单句按词切分

缺项：
- 非真正语义切分（纯字符级边界）
- 无父子块：检索 chunk 即喂 LLM 的 chunk，失去全局页码上下文

### 2. 提升知识库质量

| 完整方案 | 项目实现 | 状态 |
|:--------|:---------|:----:|
| 清洗噪声 + 元数据（标题/章节/时间戳）附加到 LLM 输入 | Chunk 携带元数据但未注入 LLM | **⚠️ 部分** |

做法：
- `Chunk` dataclass 含 `chunk_id, page_id, doc_id, page_number, chunk_type`
- 元数据存储在 pgvector/FAISS 关联记录中

缺项：
- 检索结果的 context 拼接时，**未**将 `doc_id/page_number` 等信息附加到 text 中
- LLM 不知道当前文字来自哪页、哪个文档

---

## 四、评估与兜底

### 1. RAGAS 评估框架

| 完整方案 | 项目实现 | 状态 |
|:--------|:---------|:----:|
| Faithfulness + Answer Relevance 定期评估 | 自实现三层评估，含告警 | **✅ 深度覆盖** |

做法：
- **Layer 1**: 检索层消融（ViDoRe, 10 路配置，NDCG/Recall/MRR）
- **Layer 2**: RAGAS 生成层（Faithfulness 声明分解→LLM验证 + Answer Relevancy 反向问题→cosine + Context Relevance 逐句判断）
- **Layer 3**: 端到端 QA（LLM-as-judge 答案正确性 + 拒答准确率）
- **Auto Alert**: AlertChecker 监控 Faithfulness<0.6 触发告警
- **Bad Case 分析**: 已落地氮气罐颜色编造等典型案例
- 见 `src/evaluation/ragas_metrics.py`, `src/evaluation/e2e_qa.py`, `src/observability/alerting.py`

**已知标尺缺陷**：
- 拒答误计入 Faithfulness（应跳过），影响 -0.02
- Relevancy cosine 对词面不同不敏感，应改 LLM 评分，影响 -0.02
- 仅 50 条评测（统计意义有限），需扩展到 283 条

### 2. 置信度阈值兜底

| 完整方案 | 项目实现 | 状态 |
|:--------|:---------|:----:|
| 检索最高分 < 阈值则拦截生成，返回"知识库无相关内容" | **未实现**。rerank_score 只用于排序，不拦截 | **❌** |

缺口：
- `Reranker.rerank()` 返回的 `rerank_score` 仅用于排序
- `PrismRAGRetriever.search()` 无分数检查逻辑
- `evaluate_generation()` 的拒答检测依赖 LLM self-rejection（"cannot answer..."），而非检索置信度
- 氮气罐颜色代码编造（Bad Case B1）的直接根因：检索未召回但 LLM 硬编

---

## 总结

### 覆盖度

| 维度 | 项数 | ✅ | ⚠️ | ❌ |
|:----|:---:|:-:|:-:|:-:|
| 检索阶段 | 3 | 1 | 1 | 1 |
| 生成阶段 | 3 | 2 | 0 | 1 |
| 数据处理 | 2 | 0 | 2 | 0 |
| 评估与兜底 | 2 | 1 | 0 | 1 |
| **合计** | **10** | **4** | **3** | **3** |

### 修复优先级

| 优先级 | 项 | 预估收益 | 工作量 |
|:-----:|:---|:--------:|:------:|
| **P0** | 置信度阈值兜底 | 拦截氮气罐类幻觉，Faith 预估 +0.01~0.02 | 小 |
| **P1** | 上下文压缩 | 减少噪音干扰，尤其对长 chunk 有效 | 中 |
| **P2** | chunk 元数据注入 LLM | 让 LLM 知道信息来源页码/文档 | 小 |
| **P3** | 思维链生成 Prompt | 强制先提取证据再回答 | 小 |
| **P4** | 父子块检索 | 改善检索召回 + 上下文完整性 | 大 |

### 推荐立竿见影组合

**置信度阈值兜底（P0）+ 上下文压缩（P1）** 两项落地后，Faithfulness 有望提升 0.02~0.03，对应 Bad Case B1/B2/B3 可直接拦截或缓解。