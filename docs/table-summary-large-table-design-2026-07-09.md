# 表格摘要 + 大表保护 设计与实现

> 落档日期：2026-07-09
> 分支：`feat/production-spine`
> 范围：设计 **并** 实现（代码已落地，本文为设计说明 + 实现索引）

## 1. 背景与要解决的问题

原 `TextChunker.chunk_page` 对所有超长段落（含表格）统一走"按词切碎"的兜底路径：

```python
# 原逻辑：长段落超长 → 按句子边界切 → 单句仍超长 → 按空格切词
words = sent.split()
for word in words: ...   # 表格被切成 "|公司|营收" 这类碎片
```

这带来两个具体缺陷：

- **大表被切碎（结构破坏）**：`|---|---|` 分隔行、列名行、数据行被按词边界切断，
  检索到的 chunk 是一堆 `| ... |` 碎片，LLM 无法还原成可读表格，用户问"某表里某行
  多少"时答不出来。
- **表格无语义摘要层**：表格 chunk 与普通文本一样只存 `text`，Dense/BM25 检索只能
  靠单元格字符串精确匹配，缺乏"这张表整体讲了什么、有哪些列、极值在哪"的语义定位能力。

本方案用两个协同的特性解决：

| 特性 | 目标 | 作用阶段 |
|------|------|----------|
| **大表保护** | 表格按"行"切分且每段保留表头，绝不被按词切碎 | 入库（chunker） |
| **表格摘要** | 为每张表生成 1–3 句 NL 摘要，检索用摘要 embedding 定位，生成时展开全表 | 入库 → 检索 → 生成 |

两者形成闭环：**摘要负责"找得到"，全表负责"答得准"**。

## 2. 端到端数据流

```
PDF/MD
  │  MinerU / 文本清洗
  ▼
TextChunker.chunk_page
  ├─ 普通段落 → 原逻辑（句子/词边界，不变）
  └─ 表格段落
        ├─ _merge_table_blocks ：拼回被空行拆开的相邻表块
        └─ _split_table       ：按行切分，每段带表头  ──► Chunk(chunk_type="table", text=完整表切片)
                                                    │
                          TableSummarizer.summarize(text)  ──► table_summary
                                                    │
PDFIngestor / VidoreIngestor.ingest
  ├─ embed_text = table_summary  （Dense 编码摘要，而非整表）
  └─ insert_chunks：9-tuple (... text=完整表, table_summary=摘要)
                                                    │
                                            PostgreSQL + pgvector
   ┌─────────────────────────────────────────────────────────────┐
   │ chunks(text, bge_vector, table_summary, chunk_type, ...)     │
   └─────────────────────────────────────────────────────────────┘
        │  检索（Dense/Visual 用摘要向量定位 → 找到相关表）
        ▼
Generator.answer
  ├─ chunk_type=="table" → 跳过 compress_context，直接喂完整 text
  └─ 其他 chunk        → 照常 BGE 句级压缩
```

## 3. 大表保护（chunker 改动）

文件：`src/ingestion/text_chunker.py`

### 3.1 表块合并 `_merge_table_blocks`
清洗后按 `\n\s*\n` 切段落时，长表常被空行断开。合并相邻"看起来像表"的段落，
避免表头/分隔行掉队：

```python
@staticmethod
def _merge_table_blocks(paragraphs):
    merged = []
    for para in paragraphs:
        if (merged and TextChunker._looks_like_table(merged[-1])
                and TextChunker._looks_like_table(para)):
            merged[-1] = merged[-1] + "\n" + para
        else:
            merged.append(para)
    return merged
```

### 3.2 按行切表 `_split_table`
核心保护逻辑 —— 按"行"而非"词"切分，每段保留表头（列名行 + `|---|---|` 分隔行）：

```python
def _split_table(self, table_md, page_id, doc_id, page_number, doc_ref):
    lines = table_md.split("\n")
    sep_idx = next((i for i, ln in enumerate(lines)
                    if re.match(r"^\s*\|?[\s:|-]*-[\s:|-]*\|?\s*$", ln) and "|" in ln), None)
    header = lines[:sep_idx + 1] if sep_idx is not None else lines[:1]
    body   = lines[sep_idx + 1:] if sep_idx is not None else lines[1:]
    # 逐行累加，超过 max_chars 就切一块（每块都带 header）
    ...
    return [self._make_table_chunk("\n".join(header + buf), ...) for buf in blocks]
```

- 启发式定位分隔行（`|---|---|` 形态）；找不到则退化用首行当 header。
- 每个子块都重新拼上 `header`，因此**任意单块都能独立还原成合法 markdown 表**。
- 极端情况（无 body）兜底保留原表，绝不丢内容。

### 3.3 路由与辅助
`chunk_page` 内：`if self._looks_like_table(para):` 走 `_split_table`，不再进入
按词切碎的长段落路径；非表格段落完全不变。

辅助方法：
- `_looks_like_table(text)`：前 5 行管道符计数 ≥ 3 即判定为表。
- `_make_table_chunk(...)`：产出 `Chunk(chunk_type="table", doc_ref=doc_ref)`，
  `chunk_id` 留空由调用方按序填充。

> 关键边界：每张表的子块共享同一 `page_id/doc_id`，生成时按 `page_id` 取同页
> 其他 chunk 可自然聚拢；跨块的大表在检索阶段靠摘要向量被整体召回。

## 4. 表格摘要（TableSummarizer）

文件：`src/ingestion/table_summarizer.py`（新增）

```python
class TableSummarizer:
    def __init__(self, enabled=True):
        self.enabled = enabled
        # 复用 ragas_metrics.call_llm，避免新增 LLM 客户端

    @lru_cache(maxsize=2048)
    def summarize(self, table_md: str) -> str:
        if not self.enabled or not table_md.strip():
            return ""
        try:
            resp = call_llm(_TABLE_SUMMARY_PROMPT.format(table=table_md))
            return resp.strip().strip("```markdown").strip("```").strip()
        except Exception:
            return ""   # 降级：摘要失败不影响入库，仅失去语义检索增强
```

要点：
- **摘要内容**：1–3 句事实性描述 —— 这张表关于什么、有哪些列、显著行/极值。
- **降级策略**：LLM 调用失败或禁用时返回 `""`，入库/检索不报错，仅退化为纯文本检索。
- **缓存**：`lru_cache(maxsize=2048)` 去重相同表格（重复页/重复文档省 token）。

## 5. 存储层（pgvector）

文件：`src/store/pgvector_store.py`

- `CREATE TABLE` 增加列：`table_summary TEXT NOT NULL DEFAULT ''`
- **向后兼容**：旧库自动 `ALTER TABLE chunks ADD COLUMN IF NOT EXISTS table_summary TEXT NOT NULL DEFAULT ''`
- `insert_chunks` 元组由 8 列扩为 **9 列**：
  `(chunk_id, page_id, doc_id, page_number, chunk_type, text, bge_vector, doc_ref, table_summary)`
- `search_by_vector` / `get_chunks_by_page_ids` 的 SELECT 均回带 `table_summary` 与 `chunk_type`。

## 6. 入库接线（两个 ingestor）

文件：`src/ingestion/pdf_ingestor.py`、`src/ingestion/vidore_ingestor.py`

`__init__` 新增 `self.summarizer = TableSummarizer(enabled=cfg.get("ingestion.table_summary_enabled", True))`。

`ingest` / `_ingest_text` 中，当 `chunk.chunk_type == "table"`：
```python
summary = self.summarizer.summarize(c.text)      # 生成 NL 摘要
embed_text = summary or c.text                    # Dense 编码摘要（而非整表）
# 行元组 9 列：(chunk_id, page_id, doc_id, page_number, chunk_type, text=完整表, None, doc_ref, summary)
```
即：**向量用的是摘要，落库 `text` 仍是完整 Markdown 表**。这正是"检索靠摘要、生成靠全表"的分流点。

## 7. 生成展开全表（generator）

文件：`src/generation/generator.py`

`answer()` 原逻辑把全部 `text` 丢进 `compress_context` 做句级压缩。改为：

```python
for i, r in enumerate(top):
    if r.get("chunk_type") == "table":
        table_parts[i] = r["text"]          # 整表跳过压缩，原样保留
    else:
        text_idx.append(i); text_texts.append(r["text"])

if text_texts:
    compressed = compress_context(query, text_texts, self.bge, ratio=...)  # 非表仍压缩
    table_parts[min(text_idx)] = (table_parts.get(min(text_idx), "") + "\n\n" + compressed).strip()

context = "\n\n".join(table_parts[i] for i in sorted(table_parts))  # 保持原排序
```

- 表格 chunk 的**完整 Markdown** 一定进 context，不被句级压缩删行。
- 非表格 chunk 行为不变（压缩 + 拼接）。
- 因大表已在入库时被切成带表头的子块，这里"全表"指**完整子块切片**，上下文长度有界，
  不会把原始巨表一次性灌爆窗口。

## 8. 配置项

| key | 默认 | 说明 |
|-----|------|------|
| `ingestion.table_summary_enabled` | `True` | 关闭后摘要返回 `""`，退化为纯文本检索 |
| `retrieval.context_compression_ratio` | `0.4` | 非表格上下文压缩保留比例（表格不参与） |

## 9. 兼容性与迁移

- **旧数据库**：`create_schema` 的 `ALTER TABLE ... ADD COLUMN IF NOT EXISTS` 自动补列，
  存量 chunk 的 `table_summary` 为 `''`，检索/生成逻辑对 `''` 摘要安全降级。
- **旧代码调用方**：`insert_chunks` 调用点只有 `pdf_ingestor` 与 `vidore_ingestor` 两处，
  均已同步改为 9 元组，无遗留 8 元组调用。

## 10. 验证建议

1. `python -m py_compile` 全部改动文件（已完成，通过）。
2. 单元验证 `_split_table`：构造 100 行表格，断言每个子块都以表头开头、且能
   `markdown` 解析为合法表。
3. 端到端：喂一张含大表的 PDF，检索"表里有哪几列/某行数值"，确认召回的 context
   含完整表格而非碎片。
4. 评测对比：在 RAGAS `context_precision` / `faithfulness` 上对比开/关 `table_summary_enabled`。

## 11. 局限与后续

- 摘要由 LLM 生成，增加入库耗时与 token 成本；`lru_cache` 仅缓解重复。
- 跨块大表的"全局"问题（如"全表最大值是哪行"）仍需多次召回子块拼合，
  后续可加"表级聚合 chunk"（整表摘要 + 行指针）进一步优化。
- 图片 captioning / OSS 占位回声属于另一范式（本项目用 ColPali 视觉多向量，
  不走文本摘要路线），不在本次范围。
