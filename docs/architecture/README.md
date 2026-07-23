# Architecture — 模块架构设计

> 本目录记录 PrismRAG **按模块拆分**的架构设计，供开发与面试口述对齐代码。  
> 仓库入口：[README.md](../../README.md) 的「模块架构文档」表。  
> 全系统单页图：[prismrag-architecture.html](../prismrag-architecture.html)；早期总览：[industrial-pdf-rag-architecture.md](../industrial-pdf-rag-architecture.md)。  
> 此处聚焦 **单模块边界、分层、数据流与排障入口**（当前实现快照，非长篇 Spec）。

---

## 文档约定

| 项 | 约定 |
|----|------|
| 一个模块 | 一篇 `*.md`（或子目录，见下） |
| 命名 | 小写短横线，与代码域对齐，如 `trace.md`、`cache.md`、`retrieval.md` |
| 图 | 优先 **Mermaid**（可渲染、可 diff）；避免依赖外部画图产物作为唯一来源 |
| 代码依据 | 写清主路径源文件；实现变更后同步改本文 |
| 与设计稿关系 | `docs/*-design-*.md` / `*-spec-*.md` 偏方案与决策；本目录偏 **当前实现架构快照** |

### 建议章节模板（新模块可复制）

```markdown
# <模块名>

## 1. 一句话职责
## 2. 边界（做什么 / 不做什么）
## 3. 分层架构（Mermaid）
## 4. 核心对象
## 5. 主路径时序（Mermaid）
## 6. 关键代码
## 7. 配置与开关
## 8. 排障 / 运维入口
## 9. 已知限制与演进
```

---

## 模块索引

| 模块 | 文档 | 代码主路径 | 状态 |
|------|------|------------|------|
| Trace / 可观测请求链路 | [trace.md](./trace.md) | `src/observability/`、`src/api/routes.py` | ✅ 初版 |
| Cache（L3/L4） | [cache.md](./cache.md) | `src/cache/`、`vidore_adapter`、`routes.py` | ✅ 初版 |
| Ingestion / 索引与增量更新 | [ingestion.md](./ingestion.md) | `src/ingestion/`、`src/store/`、`vidore_adapter.delete_document` | ✅ 初版 |
| Content Pipeline / 解析分块入库 | [content-pipeline.md](./content-pipeline.md) | `parser` · `text_chunker` · `table_summarizer` · `_ingest_pages` | ✅ 含 chunk 迭代历程 §15 |
| Retrieval（三路 + 融合精排） | — | `src/retrieval/`、`vidore_adapter` | ⏳ 待写 |
| Generation / Context filter | — | `src/generation/` | ⏳ 待写 |
| Self-RAG Gate2 | — | `src/generation/self_rag.py` | ⏳ 待写 |
| Evaluation / 三层评测 | [evaluation.md](./evaluation.md) | `src/evaluation/`、`scripts/run_eval|ragas|e2e_qa.py` | ✅ 初版 |

### 进行中路线图

| 路线 | 文档 | 主线 |
|------|------|------|
| Content Pipeline Phase A/B | [plans/2026-07-23-content-pipeline-phase-ab-roadmap.md](../superpowers/plans/2026-07-23-content-pipeline-phase-ab-roadmap.md) | 上下文表摘要 · content_list 类型化 · 元数据 · expand / modality boost（检索 badcase P2） |

> 新增模块：按上表补一行 + 新建对应 md；复杂模块可建子目录（如 `architecture/retrieval/README.md` + 分篇）。

---

## 维护

- 模块行为或默认配置变更时，更新对应篇与本索引状态。  
- 不在此目录堆评测数字长文；数字归 `runs/` 与 handoff。  
