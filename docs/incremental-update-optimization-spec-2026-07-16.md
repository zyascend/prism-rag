# PrismRAG 增量更新与删除优化 Spec

> 版本：v1.0 ｜ 日期：2026-07-16 ｜ 状态：待评审（Draft）
> 配套文档：`docs/incremental-update-delete-design-2026-07-09.md`（前期提案，本 Spec 在其基础上补齐现状审计、市面最优解与分阶段实施计划）
> 约束：本地 macOS 32GB 禁全量评测/编码；耗时操作（ColQwen2 重编码、全量评测）必须上云 GPU。

---

## 0. 现状审计（已代码核实，非推测）

| 能力 | 代码位置 | 状态 |
|------|----------|------|
| 新文档 ingest 全流水线（解析→切块→BGE 落 pgvector→ColPali 落 FAISS） | `pdf_ingestor.ingest` | ✅ 已实现 |
| pgvector 按 doc_id 删 chunk 行 | `pgvector_store.delete_by_doc_id` (L173，已 commit) | ✅ 已实现 |
| FAISS 增量追加页面（flat/hnsw 均可，不重建） | `faiss_store.add_pages` (L325) | ✅ 已实现 |
| BM25 增量 fit | — | ❌ 仅 `fit_from_pgvector` 全量重建 O(N) |
| FAISS 删除旧文档 | `faiss_store` | ❌ 无任何 delete/remove/compact/tombstone |
| BM25 删除已删 chunk | `bm25_retriever` | ❌ 无 `remove_chunks`，已删内容仍被召回 |
| 内容哈希 / 幂等重入库 | `pdf_ingestor` | ❌ `doc_id = _rand_doc_id()` 随机生成，无 hash |
| 统一删除编排 | — | ❌ 无编排层，三路独立 |
| page→doc 映射 | `faiss_store` | ❌ 无 `_page_doc_ids`，删除顺序脆弱 |

**已知正确性缺陷（P0）**：
- **D2**：删除文档时 BM25 未清理 → 已删内容仍经 `bm25.search` 返回 → 进 RRF → **出现在最终答案**。这是唯一会直接污染答案的 bug。
- **D1**：FAISS 无删除 → 旧视觉向量成孤儿，仍参与 MaxSim、占显存、挤占候选位。
- **U1**：BM25 每次 ingest 全量重建，语料增大后成本线性上升。
- **U2**：无内容哈希 → 重入库产生副本，新旧两版同时被召回。

---

## 1. 目标与非目标

### 1.1 目标
1. **删除一致性**：删除一份文档后，pgvector / FAISS / BM25 三路均不再返回其内容（消除 D1、D2）。
2. **更新幂等**：同一文档重入库（含部分修改）不产生副本，且尽量复用已有编码（消除 U2，节省 ColQwen2 GPU 成本）。
3. **增量高效**：新增/小改文档时，BM25 与 FAISS 不触发全量重建（消除 U1）。
4. **崩溃安全**：删除/更新过程中进程崩溃，重启后能恢复一致，不出现"删了一半"的脏状态。
5. **大批量零停机**：对全量/大批量刷新支持原子切换，检索服务不中断。

### 1.2 非目标
- 不做跨存储的分布式事务（pgvector/FAISS/BM25 性质不同，采用"pg 为真相源 + 删除账本 + 启动对账"的终态一致模型）。
- 不替换 FAISS（MaxSim late-interaction 算子 pgvector/Chroma/Milvus 均不原生支持）。
- 不做 chunk 级语义重切分（属切块策略范畴，本 Spec 不涉及）。

---

## 2. 设计原则（取自市面最优解）

| 原则 | 来源 | 本项目落地形式 |
|------|------|----------------|
| 内容寻址幂等（Content-addressed） | 生产级 RAG 通用实践 | doc/page 级 SHA256 作主键/索引，重入库即覆盖 |
| 逻辑删除 + 异步压缩 | HNSW/IVF 标准（Milvus/Weaviate 同） | FAISS 墓碑 + 占比>20% 触发 compact |
| 原子别名切换（blue-green reindex） | Qdrant alias / ES alias swap | FAISS 快照文件原子 rename + pg 暂存表 swap |
| page 级 hash diff | 部分修改最外科手段 | 仅重编码哈希变化的页，省 ColQwen2 算力 |
| 跨存储一致性编排 + 账本（outbox） | 无分布式事务时的标准补偿 | 删除账本记录待删项，按存储顺序应用，可重放 |
| pg 为真相源 | 既有 PG 有事务+ACL | 启动对账以 pg `chunks` 为准重建/裁剪另两路 |

---

## 3. 目标架构（三存储 + 编排层）

```
                    ┌─────────────────────────────────────────┐
   ingest/update ──▶│          Ingestion Orchestrator          │
                    │  (content_hash 幂等 / page diff / 编排)   │
                    └───────┬──────────────┬──────────┬────────┘
                            │              │          │
                  ┌─────────▼──┐   ┌───────▼─────┐ ┌──▼──────────┐
                  │  pgvector  │   │   FAISS     │ │   BM25      │
                  │ (真相源,   │   │ tombstone+  │ │ incremental │
                  │ 事务+ACL)  │   │ compact     │ │ fit/remove  │
                  └─────────┬──┘   └───────┬─────┘ └──┬──────────┘
                            │              │          │
                            └──── deletion ledger (outbox) ────┘
                                     (崩溃可重放)
   大批量刷新 ──▶ snapshot swap (FAISS 文件原子 rename + pg 暂存表 COMMIT swap)
```

---

## 4. 详细设计

### 4.1 内容寻址幂等（消除 U2）
- 新增 `documents` 表：`(doc_id PK, content_hash TEXT UNIQUE, source_path, page_count, ingested_at, status)`。
- `content_hash = sha256(normalized_text)`（归一化后全文，忽略空白差异），page 级 `page_hash = sha256(page_markdown_or_image)` 存于 `chunks.page_hash`。
- `ingest` 改为 `upsert_document`：
  - 计算 `content_hash`；若已存在且 `status='active'` → 直接返回旧 `doc_id`（**幂等 no-op，跳过重编码**）。
  - 若不存在 → 生成 `doc_id`（可仍随机，但 `content_hash` 唯一索引保证不重），走正常 ingest。
- 部分修改场景由 4.2 的 page diff 处理（同一 `content_hash` 变化即视为新版本，旧版走删除流程）。

### 4.2 page 级 hash diff（部分修改外科，省 GPU）
- 重入库时先按 `doc_id` 拉出旧 `page_hash` 列表，与新解析的 `page_hash` 对齐：
  - **哈希相同页** → 复用旧 chunk 行 + 旧 FAISS 向量（**跳过 ColQwen2 重编码**，这是最大省钱点）。
  - **哈希变化/新增页** → 删除旧 + 重新切块 + 重新编码（BGE + ColQwen2 仅对这些页）。
  - **删除页** → 从三路移除。
- 依赖 `_page_doc_ids`（4.4）与 `chunks.page_hash`。

### 4.3 BM25 增量 fit + remove_chunks（消除 U1、D2）
- 持久化索引状态到 `indexes/bm25_corpus.pkl`：`_tokenized_corpus`、`_doc_len`、`_doc_freq`、`_idf`、`_chunk_id_order`。
- 新增方法：
  - `fit_incremental(new_chunks)`：仅对新 chunk 分词追加，`O(vocab)` 重算 idf（把 O(N) 降为 O(Δ+vocab)）。
  - `remove_chunks(chunk_ids)`：从语料摘除，递减 doc_freq，`O(vocab)` 重算 idf。
- 启动对账：`pg.count()` 与缓存 size 比较 → 多则增量 fit，少则 `remove_chunks` 差额，一致则跳过（消除每次全量重建）。

### 4.4 FAISS 墓碑 + 异步 compact（消除 D1，解耦 D4）
- `faiss_store` 新增字段：`_deleted_page_ids: set[int]`、`_page_doc_ids: Dict[int,int]`（page_id→doc_id，1:1）。
- 新增方法：
  - `delete_by_page_ids(page_ids)`：加入墓碑集（**不物理删**，HNSW 不支持高效中间删）。
  - `delete_by_doc_id(doc_id)`：经 `_page_doc_ids` 取 page_ids → 墓碑（修复 D4 删除顺序脆弱，无需先查 pg）。
  - `compact()`：墓碑占比 > 20% 时异步重建索引，排除墓碑页，清空墓碑。
  - `_rank_pages` 排序前**过滤墓碑页**（保证检索不返回已删内容）。
- 元数据持久化到 `indexes/faiss_meta.json`（page_ids / page_doc_ids / deleted / boundaries），save/load 时同步。

### 4.5 统一删除编排 + 删除账本（消除 U4，崩溃安全）
- 新增 `orchestrator.delete_document(doc_id)`，**严格顺序**：
  1. 写删除账本：`ledger.append({doc_id, page_ids=None, status:'pending'})`。
  2. `page_ids = pg_store.get_page_ids_by_doc_id(doc_id)`（**先取 page_id 再删 pg 行**，防 D4）。
  3. `pg_store.delete_by_doc_id(doc_id)` + commit（真相源先删，标记账本 `pg_done`）。
  4. `faiss_store.delete_by_doc_id(doc_id)`（墓碑）+ save。
  5. `bm25.remove_chunks(chunk_ids_of_doc)` + save。
  6. 账本标记 `done` 并清除。
- **崩溃恢复**：重启时扫描账本 `status != 'done'` 的项，按未完成步重放（各步幂等，可安全重试）。
- 失败回滚：若 FAISS/BM25 save 失败，下一轮对账以 pg 为准重建该 doc 的三路状态。

### 4.6 大批量原子快照切换（零停机刷新）
- **FAISS**：构建新索引到 `indexes/faiss_v2.index` + `faiss_meta_v2.json`，完成且校验后 `os.replace` 原子 rename 为正式文件名（POSIX rename 原子），旧文件后台回收。
- **pgvector**：大批量刷新用暂存表 `chunks_staging` 写入新数据，`BEGIN; RENAME chunks→chunks_old; RENAME chunks_staging→chunks; DROP chunks_old; COMMIT;`（单事务内完成，检索无感知）。
- **BM25**：新语料 fit 到 `bm25_corpus_v2.pkl`，原子 rename。
- 适用场景：全量重索引、整库版本升级、大规模文档替换——优于逐条删除编排。

---

## 5. 数据模型变更

| 对象 | 变更 |
|------|------|
| `documents`（新表） | `doc_id PK, content_hash UNIQUE, source_path, page_count, ingested_at, status` |
| `chunks` | 新增 `page_hash TEXT`（page 级幂等/diff 用） |
| `faiss_store` 内存 | 新增 `_deleted_page_ids`、`_page_doc_ids`；持久化 `faiss_meta.json` |
| `bm25_retriever` 内存 | 新增 `_chunk_id_order`；持久化 `bm25_corpus.pkl` |
| `deletion_ledger`（新，jsonl 或 pg 小表） | `{doc_id, page_ids, status, ts}` |

---

## 6. 一致性模型（跨三存储）

- **真相源 = pgvector**：唯一具备事务 + ACL 的存储。
- **终态一致策略**：FAISS/BM25 为派生存储；任何不一致以 pg 为准：
  - 正常路径：编排层同步推三路。
  - 异常路径：删除账本 + 启动对账保证最终一致。
- **读取路径不变**：检索仍走三路 → RRF → reranker，本 Spec 不改召回逻辑。

---

## 7. 实施阶段与优先级

| 阶段 | 内容 | 解决 | 风险/成本 |
|------|------|------|-----------|
| **P0（正确性）** | 4.3 `remove_chunks` + 4.5 最小编排（仅 pg+BM25 清理） | D2 | 小，本地可验证 |
| **P1（幂等）** | 4.1 `documents` 表 + content_hash 幂等 upsert | U2 | 中，需迁移旧数据补 hash |
| **P1（FAISS）** | 4.4 墓碑 + compact + `_page_doc_ids` | D1, D4 | 中，本地 macOS flat 需验证 compact |
| **P2（效率）** | 4.3 `fit_incremental` + 4.2 page diff | U1, 省 GPU | 中，diff 对齐逻辑需测 |
| **P2（规模）** | 4.6 快照 swap | 大批量零停机 | 低-中，文件原子 rename 简单 |

**建议交付顺序**：P0 → P1(FAISS) → P1(幂等) → P2。

---

## 8. 验证方案

- **单元/本地（≤10 查询，符合本地限制）**：
  - `delete_document` 后三路均查不到该 doc 内容（验证 D1/D2 消除）。
  - 同内容重入库 → `chunks` 行数不变（验证 U2 幂等）。
  - page diff → 仅变化页触发 ColQwen2 编码（mock 编码计数验证）。
  - 崩溃模拟 → 重启后账本重放恢复一致。
- **云端全量评测（上 ColQwen2）**：删除/更新前后 NDCG@10 不退化；大批量 swap 后检索无中断。

---

## 9. 风险与缓解

| 风险 | 缓解 |
|------|------|
| macOS FAISS HNSW segfault（已知） | 本地默认 `flat`；compact 在 flat 上同样适用（重排数组即可） |
| 崩溃致账本残留 | 启动对账 + 账本重放，步骤幂等 |
| hash 碰撞 | SHA256，碰撞概率可忽略 |
| 旧数据无 content_hash | 迁移脚本对存量 doc 补算 hash（一次性上云跑） |
| compact 期间显存峰值 | 异步 + 占比阈值（>20%）触发，避免频繁重建 |

---

## 10. 与现有设计文档关系

- 本 Spec **不推翻** `incremental-update-delete-design-2026-07-09.md`，而是：
  1. 对其提案做**现状审计**（标注哪些已落地、哪些未做）；
  2. 引入市面最优解（内容寻址、别名切换、page diff、账本）作为设计依据；
  3. 给出**分阶段、带优先级、可验证**的实施计划，便于排期。
- 后续实现应在此 Spec 框架下更新 `handoff.md` 与对应代码模块。
