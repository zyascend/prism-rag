# Content Pipeline — PDF / 图表 / 表格入库与分块

> 状态：与当前实现对齐（`parser` · `text_chunker` · `table_summarizer` · `pdf_ingestor._ingest_pages`）  
> 更新：2026-07-23  
> 配套：索引生命周期与三路一致见 [ingestion.md](./ingestion.md)；表摘要设计见 `docs/table-summary-large-table-design-2026-07-09.md`；表摘要 context 默认开关决策见 `docs/table-context-default-decision-protocol.md`  
> 本文回答面试题：**PDF/图表/表格如何入库？chunk 怎么切？整条链路怎么串？演进踩过哪些坑？**

---

## 1. 一句话职责

把一页 PDF 拆成 **两条并行资产**：

| 资产 | 来自 | 粒度 | 去向 |
|------|------|------|------|
| **文本/表格 chunk** | 页内 markdown | chunk | BGE → pgvector + BM25（原文） |
| **整页视觉向量** | 页截图 `image` | page | ColPali/ColQwen2 → FAISS |

图表主要靠 **整页 Visual**（不单独抠图建库）；表格靠 **markdown 表结构 chunk + 可选 NL 摘要 embed**。  
表摘要可选 **同页邻段上下文**（`ingestion.table_summary_context_enabled`，默认关；见 Phase A1 路线图）。

---

## 2. 边界

| 做 | 不做 |
|----|------|
| 解析出 `Page(markdown, image)` | OCR 独立流水线（依赖 MinerU/PyMuPDF） |
| 清洗 + 段落/表格分块 | 跨页合并 chunk、文档级大纲树 |
| 表：保留 markdown + 摘要向量 | 把表当纯图进 Col* 单独索引 |
| 图/版式：整页截图进 Visual | 图块级目标检测 + 单独 embedding 库 |
| 编码写入三路 | 在线检索融合（属 Retrieval） |

---

## 3. 总览：整条链路怎么串

```mermaid
flowchart TB
  PDF["PDF 文件"]

  subgraph Parse["① 解析 Parser"]
    direction TB
    SP["simple: PyMuPDF<br/>text + 150DPI 截图<br/>blocks=None"]
    MU["mineru: CLI<br/>优先 content_list → ContentBlock<br/>缺失则 md 切页降级"]
    Page["Page<br/>page_number · markdown · image · blocks?"]
    SP --> Page
    MU --> Page
  end

  subgraph Chunk["② 分块 TextChunker"]
    direction TB
    Path{"Page.blocks?"}
    CB["chunk_blocks<br/>type→table/text/image"]
    CP["chunk_page（启发式）"]
    Clean["clean_to_markdown<br/>噪音清洗 + doc_ref"]
    Split["双换行切段<br/>+ 相邻表块合并"]
    Branch{"_looks_like_table?"}
    TSplit["表：归一化分隔行<br/>按行切，保留表头"]
    XSplit["文：≤512 tok 一段<br/>长段按句/词切"]
    Chunks["Chunk 列表<br/>text | table | image"]
    Path -->|yes MinerU list| CB --> Chunks
    Path -->|no / simple| CP --> Clean --> Split --> Branch
    Branch -->|yes| TSplit --> Chunks
    Branch -->|no| XSplit --> Chunks
  end

  subgraph Enrich["③ 表格增强 TableSummarizer"]
    direction LR
    Ctx["可选同页邻段 context<br/>table_summary_context_enabled"]
    Sum["LLM 1–3 句摘要<br/>lru_cache 去重"]
    Dual["存: text=完整 md<br/>embed 用 summary 若有"]
    Ctx --> Sum --> Dual
  end

  subgraph Encode["④ 编码 + 写入"]
    direction TB
    BGE["BGE.encode(embed_text)"]
    PG["pgvector chunks"]
    BM["BM25 fit_incremental<br/>用原文 text"]
    Col["Col*.encode_pages(image)"]
    FA["FAISS add_pages"]
    BGE --> PG
    BGE --> BM
    Col --> FA
  end

  PDF --> Parse
  Page --> Chunk
  Chunks --> Enrich
  Chunks --> Encode
  Page -->|整页 image| Encode
```

### 串起来的「一页」心智模型

```text
                    Page
                   /    \
          markdown        image (整页截图)
             │                 │
             ▼                 ▼
      TextChunker          ColPali/ColQwen2
      text / table chunks     多向量 / 页
             │                 │
     table → 摘要(可选)        │
             │                 │
        BGE(embed_text)        │
             │                 │
     ┌───────┴───────┐         │
     ▼               ▼         ▼
  pgvector         BM25      FAISS
  (向量+原文)     (原文分词)  (页向量)
```

**关键接缝（面试常问）：**  
Visual 只知道「哪一页相关」；真正喂 LLM 的字，来自同 `page_id` 的 pg chunk。这是 Visual → 文本 grounding 的工程接缝（检索侧实现，入库时通过共享 `page_id` 埋下）。

---

## 4. ① PDF 如何解析

### 4.1 统一产物 `Page`

```text
Page
  ├── page_number: int     # 1-based
  ├── markdown: str        # 页文本（simple 为纯 text；mineru 为 md，可含表格）
  └── image: PIL.Image     # 整页渲染图（图表/版式进 Visual 的唯一输入）
```

**没有**单独的 `Chart` / `Figure` 类型：图在页里，靠截图 + 视觉模型「看见」。

### 4.2 两种 Parser

| | `simple`（默认/本地 dev） | `mineru`（生产向） |
|--|--------------------------|-------------------|
| 实现 | `SimplePDFParser` | `MinerUParser` |
| 文本 | PyMuPDF `get_text` | MinerU 产出 `.md` |
| 图 | `get_pixmap(dpi=150)` | 输出 `images/*.png` 按 md 图位切页 |
| 依赖 | 仅 PyMuPDF | 需 `mineru` CLI |
| 配置 | `ingestion.parser: simple` | `ingestion.parser: mineru` |

```mermaid
flowchart LR
  subgraph Simple
    A[PDF] --> B[逐页 text]
    A --> C[逐页 pixmap 150DPI]
    B --> D[Page]
    C --> D
  end

  subgraph MinerU
    E[PDF] --> F[mineru CLI]
    F --> G[stem.md]
    F --> H[images/]
    G --> I[按图片 markdown 切段]
    H --> I
    I --> J[Page 列表]
  end
```

入口：`build_parser()` ← `cfg["ingestion.parser"]`。

---

## 5. ② Chunk 怎么切

入口：`TextChunker.chunk_page(page_id, doc_id, page_number, markdown_text)`。

### 5.1 总流程（一页内）

```mermaid
flowchart TD
  MD[页 markdown] --> C0{空?}
  C0 -->|yes| Empty[返回空列表]
  C0 -->|no| Clean[clean_to_markdown]
  Clean --> Ref[抽出 doc_ref<br/>TO 编号]
  Clean --> Paras[按双换行切段落]
  Paras --> Merge[_merge_table_blocks<br/>相邻表段落拼回]
  Merge --> Loop{每个段落}
  Loop -->|像表格| Norm[_normalize_separator_row]
  Norm --> TSplit[_split_table 按行<br/>每块带表头]
  TSplit --> TC[Chunk type=table]
  Loop -->|普通文本| Len{len ≤ max_chars?}
  Len -->|yes| TX[一整段 = 1 chunk text]
  Len -->|no| Sent[按 .?! 切句攒 buffer]
  Sent --> Word[单句过长则按词切]
  Word --> TX2[Chunk type=text]
  TC --> Out[List Chunk]
  TX --> Out
  TX2 --> Out
```

### 5.2 尺寸规则

| 参数 | 值 | 说明 |
|------|-----|------|
| `MAX_TOKENS` | **512** | 目标上限 |
| `TOKEN_EST_RATIO` | 4 | 英文约 4 chars/token |
| `max_chars` | **2048** | `512 × 4`，实现里用字符长度近似 token |

**文本：**

1. 短段（`len ≤ max_chars`）→ 1 chunk  
2. 长段 → 按 `(?<=[.?!])\s+` 切句，buffer 累加至上限再落盘  
3. 单句仍超长 → 按空格分词再切  

**表格：** 不按词硬切（会破坏 `|---|` 对齐），见 §6。

### 5.3 预处理 `clean_to_markdown`（TO 手册噪音）

面向工业/ViDoRe 类手册，顺序清洗：

| 步 | 作用 |
|----|------|
| 抽 `doc_ref` | 首个 `TO …` 引用，供 grounding（**不进** CtxRel 评估口径） |
| 断词修复 | `word-\nword` → `wordword` |
| 去空单元格表碎片行 | 如 `\|  \| TO WP … \|`，**保留**正常 md 表 |
| 去 TO 引用行 / 纯文档编号行 | 减检索噪音 |
| 去全大写短行 | 章节标题类 |
| 压缩 ≥3 空行 | → 双换行 |

### 5.4 Chunk 对象（当前）

```text
Chunk
  ├── chunk_id       # pg{page_id:05d}_ch{idx:03d}
  ├── page_id / doc_id / page_number
  ├── text           # 原文（表=完整 markdown 片段）
  ├── chunk_type     # "text" | "table" | "image"（image 仅 content_list 锚点）
  ├── doc_ref        # 页级 TO 编号（可空）
  ├── table_summary  # 入库时 Summarizer 填；初建可空
  ├── caption        # 图/表题注（A2/A3；可空）
  ├── section_path   # 标题栈路径（A3；语料无 # 时可全空）
  ├── prev_chunk_id  # 同批邻居（A3）
  └── next_chunk_id
```

入口双路径：

| 入口 | 何时 | 类型从哪来 |
|------|------|------------|
| `chunk_page(markdown)` | simple / 无 content_list / ViDoRe | 启发式 `_looks_like_table` |
| `chunk_blocks(blocks)` | MinerU content_list | 解析器 `type` 字段 |

演进脉络见 **§15**。

---

## 6. ③ 表格如何入库

### 6.1 识别

`_looks_like_table`：前 5 行 `|` 计数 **≥ 3** → 当表格处理。

### 6.2 结构保护

| 步骤 | 目的 |
|------|------|
| `_merge_table_blocks` | 空行拆开的相邻表段拼回，避免表头/分隔行掉队 |
| `_normalize_separator_row` | 缺 `\|---\|` 时按表头列数注入 GFM 分隔行；支持 caption 贴表头 |
| `_split_table` | 超长表 **按数据行** 切；**每块复制 header+sep**，列语义不丢 |

```text
长表切分示意：
  | ColA | ColB |     ← 每块都带
  | ---- | ---- |
  | r1   | ...  |     ← 块1 部分行
  --- 下一 chunk ---
  | ColA | ColB |
  | ---- | ---- |
  | r50  | ...  |     ← 块2 部分行
```

### 6.3 双表示：检索摘要 vs 生成原文

```mermaid
flowchart LR
  T[table markdown Chunk.text]
  T --> S[TableSummarizer.summarize]
  S -->|成功| E[embed_text = summary]
  S -->|失败/关闭| E2[embed_text = 原文 md]
  T --> Store["pg: text=完整 md<br/>table_summary=摘要"]
  E --> BGE[BGE 向量]
  E2 --> BGE
  Store --> Gen["生成侧: 用 text<br/>表格保护逻辑"]
```

| 字段 | 用途 |
|------|------|
| `text` | 完整 markdown 表；**BM25 用它**；生成入模优先用它（结构） |
| `table_summary` | NL 摘要；**Dense embed 优先用它**（语义更好、向量更稳） |
| BGE 输入 | `summary if summary else text`（`pdf_ingestor`） |

`TableSummarizer`：

- Prompt：`src/prompts/prompts/table_summary.yaml`（v1 孤表；v2 带 Surrounding context，默认不 active）  
- `lru_cache` 键含 `(table, context)`，相同表不同语境可分缓存  
- 任意异常 → `""`，**不阻塞入库**  
- 开关：`ingestion.table_summary_enabled`（默认 True）  
- 上下文：`ingestion.table_summary_context_enabled`（默认 **false**；同页非表 chunk 截断注入）

### 6.4 生成侧呼应（入库时埋下的约定）

生成压缩时对 `chunk_type==table` **保护整表 markdown**（`generator.py`），避免句级过滤把表切烂。  
入库写 `chunk_type=table` + 完整 `text` 就是为这一步服务。

---

## 7. ④ 图表 / 版式如何入库

**没有独立「图 chunk 表」。**

| 内容 | 策略 |
|------|------|
| 示意图、曲线、截图、复杂版式 | 整页 `Page.image` → Col* → FAISS |
| 图注/旁白文字 | 若解析进 `markdown` → 普通 text chunk |
| `use_visual=false` | 跳过视觉路；图依赖文本侧能否抽到字 |

```text
图表密集页：
  Visual 召回 page_id
       → 反查该页全部 text/table chunk
       → 拼 grounding 上下文
```

页级 `page_hash` 优先用 **图像 tobytes**，版式/图变更会触发 page-diff 重编码（见 [ingestion.md](./ingestion.md)）。

---

## 8. ⑤ `_ingest_pages`：编码与写入如何串

对指定 `page_numbers` 子集（全量或 page-diff 的 changed+new）：

```mermaid
sequenceDiagram
  participant I as PDFIngestor
  participant K as TextChunker
  participant S as TableSummarizer
  participant B as BGE
  participant P as pgvector
  participant C as Col*
  participant F as FAISS
  participant M as BM25

  loop 每个目标页
    I->>I: 新 page_id + page_hash
    I->>K: chunk_page(markdown)
    loop 每个 Chunk
      alt chunk_type=table
        I->>S: summarize(text)
        Note over I: embed_text=summary or text
      else text
        Note over I: embed_text=text
      end
      I->>I: 攒 all_rows / all_texts
    end
    opt use_visual
      I->>I: 攒 page image
    end
  end
  I->>B: encode(all_texts)
  I->>P: insert_chunks(原文+向量+summary+hash)
  opt use_visual
    I->>C: encode_pages(images)
    I->>F: add_pages + save
  end
  I->>M: fit_incremental(原文 text 的 chunk 字典)
```

### 三路写入对照

| 路 | 输入内容 | 粒度 |
|----|----------|------|
| Dense (pg) | table→摘要优先，否则原文 | chunk |
| BM25 | **始终原文** `text`（表是 md 全文） | chunk |
| Visual | 整页 `image` | page |

---

## 9. 端到端数据流（一张图）

```mermaid
flowchart TB
  subgraph In["入库"]
    PDF --> Parser
    Parser --> Page
    Page --> Chunker
    Chunker --> TextC[text chunks]
    Chunker --> TableC[table chunks]
    TableC --> Summarizer
    TextC --> BGE
    Summarizer --> BGE
    TextC --> BM25
    TableC --> BM25
    Page --> Visual[Col* page emb]
    BGE --> PG[(pgvector)]
    Visual --> FAISS[(FAISS)]
    BM25 --> Mem[内存倒排]
  end

  subgraph Out["查询时消费（简述）"]
    Q[Query] --> R1[BM25]
    Q --> R2[Dense]
    Q --> R3[Visual MaxSim]
    R3 --> Join[page_id → pg chunks]
    R1 --> RRF
    R2 --> RRF
    Join --> RRF
    RRF --> Rerank --> Gen[Generator<br/>表用完整 md]
  end

  PG --> R2
  PG --> Join
  FAISS --> R3
  Mem --> R1
```

---

## 10. 关键代码

| 路径 | 职责 |
|------|------|
| `src/ingestion/parser.py` | `Page`、Simple/MinerU、`build_parser` |
| `src/ingestion/text_chunker.py` | 清洗、段落/句切、表识别与按行切分 |
| `src/ingestion/table_summarizer.py` | 表 NL 摘要 + lru |
| `src/ingestion/pdf_ingestor.py` | `_ingest_pages` 串解析后全链路 |
| `src/ingestion/vidore_ingestor.py` | 基准语料：HF 行已有 image+markdown，同 chunk 逻辑 |
| `src/ingestion/encoders.py` | BGE / Col* |
| `src/prompts/prompts/table_summary.yaml` | 摘要 prompt |
| `src/generation/generator.py` | 表 chunk 入模保护（消费侧） |
| `tests/test_text_chunker.py` / `test_parser.py` / `test_pdf_ingestor.py` | 单测 |

---

## 11. 配置

| 键 | 作用 |
|----|------|
| `ingestion.parser` | `simple` \| `mineru` |
| `ingestion.table_summary_enabled` | 是否打表摘要 LLM（默认 True） |
| `ingestion.table_summary_context_enabled` | 摘要是否带同页邻段（默认 **false**） |
| `ingestion.table_summary_context_max_chars` | 邻段截断（默认 1500） |
| `ingestion.image_caption_chunks` | content_list 图 caption 是否落 text 锚点（默认 false） |
| `retrieval.use_visual` | 是否编码/写入 FAISS |
| `embedding.colpali_batch_size` | 页图编码 batch |
| `TextChunker(max_tokens=512)` | 构造参数可改上限 |

本地 dev 示例：`config/models.local-dev.yaml` 常设 `parser: simple`、`use_visual: false` 降依赖。

---

## 12. 排障速查

| 现象 | 可能原因 |
|------|----------|
| 表被切成乱码碎片 | 旧逻辑按词切；确认走 `_split_table` + 分隔行归一化 |
| Dense 搜不到表、BM25 可以 | 摘要偏题或摘要失败空串；查 `table_summary` 与 embed 输入 |
| 有图但 Visual 全无 | `use_visual=false` 或未 load FAISS / 未 encode_pages |
| MinerU 失败 | CLI 未装；本地改 `parser=simple` |
| 空页 0 chunk | markdown 清洗后为空；Visual 仍可能有页向量 |
| 表进了 LLM 却缺列 | 生成侧未走 table 保护；或入库时 `chunk_type` 非 table |

---

## 13. 已知限制

| 项 | 说明 |
|----|------|
| Token 用字符近似 | `chars/4`，非真实 tokenizer |
| 表识别启发式 | 管道符计数（`chunk_page`）；`chunk_blocks` 依赖解析器 type |
| 图不单独建库 | 复杂多图页依赖整页 Visual + 页内文字 |
| simple 解析弱表结构 | 生产表质量依赖 MinerU md / content_list |
| 摘要成本 | 入库 LLM；靠 lru 与可关闭缓解；context 默认关 |
| section_path | 依赖 markdown `#` / text_level；ViDoRe 页上可全空 |
| 跨页表 | 不合并，各页各自切 |

---

## 14. 20～40 秒口述（面试 · 现状）

> PDF 先解析成 **页**：一边 markdown 文本，一边整页截图。  
> 文本侧：清洗手册噪音 → 按段切块，上限大约 512 token；**表格单独识别**，按行切并保留表头，再可选生成自然语言摘要——**向量用摘要、BM25 和生成用完整 markdown**。  
> 图表不做单独抠图库，整页进 ColPali/ColQwen2 走 Visual；命中页后用 **page_id 回查** 该页 text/table chunk 做 grounding。  
> 一页写完：BGE→pgvector，原文→BM25，页图→FAISS，三路共享 doc/page 标识。

**深读：** 增量/删除/三路一致 → [ingestion.md](./ingestion.md)。  
**演进叙事：** 见下节 §15（面试「你怎么一步步做厚 chunk 的」）。

---

## 15. Chunk 切分技术迭代历程（面试叙事）

> 目的：用 **问题驱动** 讲清「为什么改、改了什么、怎么验收」，避免背版本号。  
> 证据：`git log -- src/ingestion/text_chunker.py` + 设计/run 文档；数字以 `runs/` 与 handoff 为准。

### 15.1 总时间线（一图）

```text
2026-06-30  P0  通用段落切分（512 tok 近似）
     │
2026-07-06  P1  TO 手册正则清洗
     │
2026-07-07  P2  doc_ref 元数据（grounding ≠ CtxRel）
     │
2026-07-09  P3  大表保护 + 表 NL 摘要（双表示）
     │         同日：GFM 分隔行归一化
     │
2026-07-18  P4  表摘要 prompt 进版本库
     │
2026-07-23  P5  上下文表摘要（可选）
     │         MinerU content_list 类型化分块
     │         section / caption / 邻居链
     ▼
  云上 Text re-ingest + Goal-A 正式评测
```

### 15.2 分阶段：问题 → 改动 → 面试一句

#### P0 · 检索 MVP 基线（2026-06-30 · `3d9fff3`）

| | |
|--|--|
| **问题** | 需要可消融的文本 chunk 进 BM25/Dense，服务 ViDoRe 三路检索。 |
| **做法** | `TextChunker`：双换行切段 → ≤512 token（`chars/4`）整段落盘 → 过长按句/词切；`chunk_id` 绑定 `page_id`。 |
| **局限** | 表与正文同一套「按词切」；工业手册噪音未处理；无类型字段。 |
| **口述** | 「先保证有稳定 chunk 粒度，能跑通三路和 NDCG，再针对坏 case 特化。」 |

#### P1 · TO 军事手册清洗（2026-07-06 · `424534c`）

| | |
|--|--|
| **问题** | ViDoRe/TO 类语料：断词、空单元格表碎片、TO 引用行、全大写短行污染 embedding 与 BM25。 |
| **做法** | `clean_to_markdown`：断词修复 → 去碎片表行（**保留**正常 `\| col \|` 表）→ 去 TO/编号行 → 去全大写短标题行 → 压空行。 |
| **口述** | 「不是通用 NLP 清洗，是 **领域噪音表**；误伤正常表是红线，所以空单元格碎片和正常管道表分开。」 |

#### P2 · `doc_ref` 元数据（2026-07-07 · `d867ae8`）

| | |
|--|--|
| **问题** | 生成需要 TO 文档编号做 grounding；若把编号塞进参与 CtxRel 的正文，会 **抬高/污染** 上下文相关性口径。 |
| **做法** | 清洗时抽出首个 TO 引用 → `Chunk.doc_ref`；正文仍去掉引用行。评测 CtxRel 不看 `doc_ref`。 |
| **口述** | 「**元数据通道 vs 可评估正文** 拆开——产品要引用，尺子要干净。」 |

#### P3 · 大表保护 + 表摘要双表示（2026-07-09 · `fe9ceba` + `7db8595`）

| | |
|--|--|
| **问题 A** | 长表走「按词切」→ `\|---\|` 碎成 token，LLM 无法还原表结构，表题答不出。 |
| **问题 B** | 整表 md 直接 BGE：噪声大、语义定位弱（「这张表讲什么」）。 |
| **做法** | ① `_looks_like_table` 分流；② `_merge_table_blocks` 拼回被空行拆开的表；③ `_split_table` **按数据行切、每块复制表头+分隔行**；④ `TableSummarizer` 1–3 句 NL；⑤ **Dense embed=summary（失败则原文），BM25+生成=完整 md**；⑥ 生成侧 `chunk_type==table` 不做句级硬压碎。 |
| **加固** | `_normalize_separator_row`：ViDoRe 大量「有管道无 `\|---\|`」→ 注入 GFM 分隔行，并处理 caption 贴表头。 |
| **设计档** | `docs/table-summary-large-table-design-2026-07-09.md`；复盘 `#25`。 |
| **口述** | 「表的矛盾是 **结构保真 vs 语义可检索**：所以检索用摘要、生成用原表，切分永不按词砍列。」 |

#### P4 · Prompt 版本化（2026-07-18 · `aeb7d1c`）

| | |
|--|--|
| **问题** | 摘要模板散落代码，难审计、难回滚。 |
| **做法** | `table_summary.yaml` 进 `PromptRegistry`；运行时 `get_active`。 |
| **口述** | 「入库 LLM 也是 prompt 资产，和生成侧同一套版本纪律。」 |

#### P5 · Content Pipeline Phase A（2026-07-23 · `06d4510` → `4044ddb`）

灵感：RAG-Anything 的 ContextExtractor / typed content_list——**只借鉴入库语义，不迁知识图谱**。

| 子项 | 问题 | 做法 |
|------|------|------|
| **A1 上下文摘要** | 孤表摘要丢章节/工况 → Dense 错表/miss | `summarize(..., context=)`；同页非表 chunk 截断注入；prompt **v2**；**默认关** |
| **A2 类型化分块** | 仅靠管道符猜表，复杂/OCR 脏 md 易错 | MinerU `content_list` → `ContentBlock` → `chunk_blocks`；无 list **降级** `chunk_page` |
| **A3 轻量层级** | 缺「在哪一节 / 邻居是谁」 | `section_path` 标题栈；`prev/next_chunk_id`；pg 列可迁移 |
| **运维** | `ON CONFLICT DO NOTHING` 无法更新旧 chunk | `truncate_chunks` + `--replace-text` Text re-ingest（FAISS 不动） |

**云上验收（可讲数字）：**

| 实验 | 结果 |
|------|------|
| Boot-CP 三臂 expand/boost @100q | NDCG **无差**（page 指标 + 插件默认关）→ **插件不背锅** |
| Text re-ingest context ON | 8835 chunk、2305 表摘要齐全；`section_path` 在本语料近 0（无 `#` 标题） |
| Goal-A 正式 | Full_zerank2 **283q NDCG@10 = 0.5337**（Boot-A 黄金 **0.5318**，**+0.19pt**）；E2E Correct **0.66** / Reject **0.95**（可答误拒 9） |

**口述：**  
「主收益来自 **表语义重灌**，不是再叠检索门。默认 context 仍可关——100q 只是弱阳性；正式 283 显示不降略升；是否 yaml 默认 true 要用 OFF 双臂 decide 协议。」

### 15.3 设计原则（贯穿始终 · 面试收束）

| 原则 | 在 chunk 上的体现 |
|------|-------------------|
| **问题驱动迭代** | 先 MVP 通，再清洗 → 表结构 → 摘要 → 上下文/类型化 |
| **双表示** | 检索语义通道 vs 生成结构通道（表最典型） |
| **尺子干净** | `doc_ref` / 拒答口径 / 压缩前后 context 不混比 |
| **失败可降级** | 摘要失败空串；MinerU 无 list 回退启发式；默认关新开关 |
| **可辩护评测** | 改分块要能指到 run（Boot-A / Goal-A），禁止无对照吹涨点 |

### 15.4 1～2 分钟口述稿（演进版）

> 我们 chunk 不是一次设计定终身。最早是通用段落切，保证三路 RAG 能跑。上 ViDoRe 工业手册后，先加 **领域清洗**，再把 **文档编号做成元数据**，避免污染 CtxRel。  
> 真正分水岭是 **表**：发现按词切碎了 markdown 表，就做了 **按行切 + 表头复用**，并加 **NL 摘要双表示**——Dense 找表、生成读表结构。  
> 最近一轮对照多模态开源实践，补了 **可选上下文摘要**、MinerU **类型化块** 和轻量邻居元数据；云上 Text re-ingest 后 **283 NDCG@10 到 0.5337**，相对原黄金表略升，E2E Correct 0.66。  
> 原则始终是：**结构保真、语义可检索、开关默认可回滚、数字可指到 run。**

### 15.5 关键 commit / 文档索引

| 阶段 | Commit（短） | 文档 / run |
|------|--------------|------------|
| P0 MVP | `3d9fff3` | — |
| P1 清洗 | `424534c` | — |
| P2 doc_ref | `d867ae8` | — |
| P3 表保护+摘要 | `fe9ceba` · `7db8595` | `docs/table-summary-large-table-design-2026-07-09.md` · PR #25 |
| P4 prompt | `aeb7d1c` | `src/prompts/prompts/table_summary.yaml` |
| P5 Phase A | `06d4510` · `a2dfd17` · `4044ddb` | `docs/superpowers/plans/2026-07-23-content-pipeline-phase-ab-roadmap.md` |
| 云验收 | Goal-A | `runs/20260723-on-goalA/` · `runs/20260723-text-reingest-full/` |
| 默认开关决策（未改 yaml） | — | `docs/table-context-default-decision-protocol.md` |

```bash
# 本地复盘
git log --oneline -- src/ingestion/text_chunker.py src/ingestion/table_summarizer.py
```
