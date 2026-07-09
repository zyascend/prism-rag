# 增量更新与逻辑删除设计方案

> 日期:2026-07-09
> 范围:BM25 增量 fit(只 fit 新 chunk)+ FAISS 视觉索引逻辑删除(tombstone + compaction)
> 关联代码:`src/retrieval/bm25_retriever.py`、`src/store/faiss_store.py`、`src/store/pgvector_store.py`、`src/ingestion/pdf_ingestor.py`、`src/api/routes.py`

## 1. 背景与目标

PrismRAG 当前是"双存储"架构:文本/语义走 PostgreSQL + pgvector(Dense + BM25),视觉多向量走 FAISS(ColPali + MaxSim)。
当新知识 / 新文档到来时,现有入库路径基本可用,但**删除是半成品**,且 BM25 每次入库都全量重建,存在明显的性能与一致性短板。

本方案目标:

1. **增量 BM25**:新文档入库时只对新 chunk 做 fit,不再重读全表、重新分词整个语料。
2. **FAISS 逻辑删除**:按文档删除视觉向量时采用"墓碑 + 异步压缩",避免每次删除都重建全量索引,并正确释放 VRAM。
3. **修正当前删除的不一致**:让 BM25 / FAISS / pgvector 三路在删除时都被清理,使"删除"成为一致、可用的能力(目前已删文档仍会出现在检索结果中)。

## 2. 当前实现短板分析

### 2.1 更新(ingest)的短板

| # | 短板 | 代码定位 | 影响 |
|---|------|---------|------|
| U1 | BM25 每次 ingest 全量重建 | `bm25_retriever.fit_from_pgvector` 读全表 + 全程重新分词 + 重建 `BM25Okapi`;`scripts/ingest_pdf.py:32`、`src/api/routes.py:219` 每次入库都调 | 代价 O(N),随语料线性增长,后期是每个 ingest 的主开销 |
| U2 | 没有"更新已有文档"语义 | `pdf_ingestor._rand_doc_id`(66 行)每次随机,无内容哈希 | 重复入库=产生副本;要"更新"只能手动 delete + insert |
| U3 | FAISS 只增不减 | `faiss_store.add_pages`(325 行)只有 append,无 delete | VRAM/内存随 ingest 单调增长,脏数据只能整体重建 |
| U4 | 跨存储无事务边界 | `ingest` 先 commit pg(`insert_chunks`),再 FAISS add+save,再 BM25 重建 | 若 FAISS save 失败→pg 有、FAISS 无,不一致;`/ingest` 失败路径(`routes.py:209-218`)只回滚 pg,但 `add_pages` 已改内存 `_vectors` 且不 save,导致**进程内残留孤儿**,下次成功 ingest 触发 save 时把孤儿一起落盘 |
| U5 | add_pages 每次全量 vstack | `faiss_store.py:349` `np.vstack([self._vectors, nv])` | 每次 ingest 都 O(N) 拷贝一次数组,大索引时明显 |

### 2.2 删除的短板(更严重)

| # | 短板 | 代码定位 | 影响 |
|---|------|---------|------|
| D1 | FAISS 完全没有删除能力 | `faiss_store` 只有 `add_pages`,无 `delete_pages` | `delete_by_doc_id` 后视觉向量残留为 **orphan**,仍参与 MaxSim 打分、占用 VRAM、挤占候选位;HNSW 模式下还污染图 |
| D2 | BM25 不随删除更新 → 文档没真删掉 | `BM25Retriever` 无任何 delete 方法;`delete_by_doc_id` 不碰 BM25 | 已删 chunk 仍被 `bm25.search` 返回 → 喂给 RRF 融合 → **最终答案里仍出现已删除文档内容**(一致性 bug) |
| D3 | 没有 DELETE API | `routes.py` 只有 `/ingest`,无删除端点 | 删除只能直接调 `pg.delete_by_doc_id`(脚本/CLI),且即便调了也不清理 BM25/FAISS |
| D4 | page_id→doc_id 不在 FAISS | `page_id` 是 `random.getrandbits(31)`(`pdf_ingestor.py:70`),FAISS 只存 `page_id` | 想按 doc 删 FAISS 页面,必须先知道哪些 page_id 属于该 doc;一旦先删了 pg 行,映射就丢了 → 删除顺序很脆弱 |

**结论**:"增"勉强可用(但有 U1/U3/U4 隐患),"删"基本是半成品——pgvector 删了,BM25 和 FAISS 两条路都没清理,检索结果里已删文档依然可见。这正是需要增量 BM25 + FAISS 逻辑删除的原因。

## 3. 设计 A:增量 BM25(只 fit 新 chunk)

### 3.1 原理

`rank_bm25.BM25Okapi` 的关键内部属性都是公开、可直接改写的:

- `term_freqs`:每文档词频 dict 列表
- `doc_len`:每文档长度列表
- `idf`:词→idf dict
- `doc_freqs`、`corpus_size`、`avgdl`

`fit_from_pgvector` 慢在 **重读全表 + 重新分词全部 N 个 chunk**。`idf` 重算本身只依赖词表(≈O(vocab)),并不依赖逐文档重算。

因此增量思路:**只对新 chunk 分词、追加到语料,再 O(vocab) 重算 idf**——把 O(N) 降为 O(Δ + vocab)。

### 3.2 改动点(`src/retrieval/bm25_retriever.py`)

- `__init__` 增加 `self._tokenized: List[List[str]]`(与 `self._chunks` 等长)。
- `fit_from_pgvector` 在 `fit` 之后额外填充 `_tokenized` 并落盘缓存。
- 新增 `fit_incremental(new_chunks)`、`remove_chunks(chunk_ids)`、`_recompute_idf()`、`_persist_corpus()` / `_load_corpus()`。
- 语料(分好词的 chunk 列表)持久化到 `indexes/bm25_corpus.pkl`,用于重启后免全量重建。

### 3.3 代码骨架

```python
import math, pickle
from pathlib import Path
from typing import List, Dict, Optional, Set

CORPUS_CACHE = Path("indexes/bm25_corpus.pkl")


class BM25Retriever:
    def __init__(self):
        self._bm25: Optional[BM25Okapi] = None
        self._chunks: List[dict] = []
        self._tokenized: List[List[str]] = []

    def fit_from_pgvector(self, pg_store: PgVectorStore):
        chunks = []                     # 现有全表读取逻辑不变
        offset = 0
        while True:
            rows = pg_store._fetch_page(offset)   # 原 SELECT ... LIMIT/OFFSET
            if not rows: break
            for r in rows: chunks.append({...})
            offset += len(rows)
        self.fit(chunks)
        self._persist_corpus()

    def fit(self, chunks: List[dict]):
        self._chunks = chunks
        self._tokenized = [self._tokenize(c["text"]) for c in chunks]
        self._bm25 = BM25Okapi(self._tokenized)

    # ── 增量 fit ──
    def fit_incremental(self, new_chunks: List[dict]):
        if self._bm25 is None:                  # 首次:退化为全量
            return self.fit(new_chunks)
        new_tok = [self._tokenize(c["text"]) for c in new_chunks]
        for c, t in zip(new_chunks, new_tok):
            self._chunks.append(c)
            self._tokenized.append(t)
            self._bm25.term_freqs.append(self._term_freqs_of(t))
            self._bm25.doc_len.append(len(t))
        self._recompute_idf()                   # O(vocab),非 O(N)
        self._persist_corpus()

    # ── 增量删除(配合统一删除编排)──
    def remove_chunks(self, chunk_ids: Set[str]):
        keep = [i for i, c in enumerate(self._chunks) if c["chunk_id"] not in chunk_ids]
        self._chunks    = [self._chunks[i] for i in keep]
        self._tokenized = [self._tokenized[i] for i in keep]
        self._bm25.term_freqs = [self._bm25.term_freqs[i] for i in keep]
        self._bm25.doc_len    = [self._bm25.doc_len[i] for i in keep]
        self._recompute_idf()
        self._persist_corpus()

    def _recompute_idf(self):
        doc_freqs = {}
        for t in self._tokenized:
            for term in set(t):
                doc_freqs[term] = doc_freqs.get(term, 0) + 1
        N = len(self._tokenized)
        self._bm25.corpus_size = N
        self._bm25.doc_freqs = doc_freqs
        self._bm25.avgdl = sum(len(t) for t in self._tokenized) / N if N else 0.0
        self._bm25.idf = {
            term: math.log(1 + (N - df + 0.5) / (df + 0.5))
            for term, df in doc_freqs.items()
        }

    @staticmethod
    def _term_freqs_of(tokens):
        d = {}
        for t in tokens: d[t] = d.get(t, 0) + 1
        return d

    def _persist_corpus(self):
        with open(CORPUS_CACHE, "wb") as f:
            pickle.dump({"chunks": self._chunks, "tokenized": self._tokenized}, f)

    def _load_corpus(self) -> bool:
        if not CORPUS_CACHE.exists():
            return False
        with open(CORPUS_CACHE, "rb") as f:
            d = pickle.load(f)
        self._chunks, self._tokenized = d["chunks"], d["tokenized"]
        self._bm25 = BM25Okapi(self._tokenized)
        return True
```

### 3.4 重启安全与接入点

- **接入点**:`pdf_ingestor.ingest` 在 BGE 编码后拿到 `all_rows`(含 chunk 文本),入库后改调 `bm25.fit_incremental(all_rows)` 代替 `fit_from_pgvector`;`src/api/routes.py` 的 `/ingest` 同理。
- **重启安全**:startup 先试 `_load_corpus()`,若 `len(self._chunks) == pg.count()` 则免全量重建,否则回退 `fit_from_pgvector`。
- **删除一致性**:统一删除编排(见 §5)调用 `bm25.remove_chunks(...)`,确保已删文档不再出现在 BM25 结果中(修复 D2)。

## 4. 设计 B:FAISS 逻辑删除(tombstone + compaction)

核心思想:**不物理删,先打墓碑,搜索时过滤,占比超阈值再异步压缩**。FAISS 的 `IndexFlatIP` / `IndexHNSWFlat` 都不支持高效中间行删除,墓碑是正解。

### 4.1 新增状态(`src/store/faiss_store.py`)

```python
self._page_doc_ids: Optional[np.ndarray] = None   # 与 _page_ids 等长,落盘 indexes/<name>_docids.npy
self._deleted_page_ids: Set[int] = set()          # 墓碑,落盘 indexes/<name>_deleted.json
```

> **关键**:`_page_doc_ids` 让 FAISS 能**按 doc_id 独立删除**,不必先查 pgvector(修复 D4)。`page_id` 是全局随机且唯一,一个 page 只属于一个 doc,映射是 1:1。

### 4.2 核心流程

1. **入库**:`add_pages` 携带 `page_doc_map`,在追加 `_page_ids` 的同时追加 `_page_doc_ids`。
2. **删除 = 打墓碑**(不碰向量):把目标 page_id 加入 `_deleted_page_ids`;墓碑占比 > 20% 时自动触发 `compact()`。
3. **搜索过滤**:三路 maxsim(`_maxsim_torch` / `_maxsim_exact` / `_maxsim_hnsw`)统一在 `_rank_pages` 排序前剔除墓碑 page_id。
4. **compaction**:物理重建 `_vectors` / `_page_ids` / `_page_doc_ids` 与 FAISS index,丢弃墓碑,清空墓碑集并 save。

### 4.3 代码骨架

```python
# ── 增量 add:携带 doc_id ──
def add_pages(self, page_embeddings: Dict[int, torch.Tensor],
              page_doc_map: Dict[int, str]):
    # ... 现有 append 逻辑 ...
    new_docids = [page_doc_map[int(pid)] for pid in sorted(page_embeddings.keys())]
    if self._page_doc_ids is None:
        self._page_doc_ids = np.array([], dtype=object)
    self._page_doc_ids = np.concatenate([self._page_doc_ids, np.array(new_docids, dtype=object)])

# ── 逻辑删除 ──
def delete_by_page_ids(self, page_ids: List[int]):
    self._deleted_page_ids.update(int(p) for p in page_ids)
    if len(self._deleted_page_ids) / max(1, self.num_pages) > 0.20:
        self.compact()
    self._persist_tombstone()

def delete_by_doc_id(self, doc_id: str):          # FAISS 可独立删除
    if self._page_doc_ids is not None:
        mask = np.isin(self._page_doc_ids, [doc_id])
        pids = self._page_ids[mask].tolist()
        self.delete_by_page_ids(pids)

# ── 排序前过滤墓碑(三路统一)──
@staticmethod
def _rank_pages(page_scores: Dict[int, float], k: int,
                deleted: Set[int] = set()) -> List[dict]:
    page_scores = {pid: s for pid, s in page_scores.items() if pid not in deleted}
    sorted_pages = sorted(page_scores.items(), key=lambda x: x[1], reverse=True)
    return [{"page_id": pid, "score": score} for pid, score in sorted_pages[:k]]

# ── 物理压缩 ──
def compact(self):
    if not self._deleted_page_ids:
        return
    keep = ~np.isin(self._page_ids, list(self._deleted_page_ids))
    self._vectors     = self._vectors[keep]
    self._page_ids    = self._page_ids[keep]
    if self._page_doc_ids is not None:
        self._page_doc_ids = self._page_doc_ids[keep]
    self._page_boundaries = self._rebuild_boundaries(self._page_ids)
    dim = self._vectors.shape[1]
    self._index = faiss.IndexFlatIP(dim)
    self._index.add(self._vectors)
    if self._index_type == "hnsw":
        self._rebuild_hnsw()
    if self._device.type == "cuda":
        self._vectors_torch = torch.from_numpy(self._vectors).to(self._device)
    self._deleted_page_ids.clear()
    self._num_pages = len(self._page_boundaries)
    self._num_patches = len(self._page_ids)
    self.save()
```

> `_rebuild_boundaries` 复用现有 `load()` 里的边界重算逻辑(按 `_page_ids` 连续段切分)。`search` 三路在调用 `_rank_pages` 时传入 `self._deleted_page_ids`。

### 4.4 save / load 兼容

- `save()`:额外写 `indexes/<name>_docids.npy` 与 `indexes/<name>_deleted.json`。
- `load()`:读回 `_page_doc_ids` 与 `_deleted_page_ids`;`_index_type` 仍依据 `_hnsw.faiss` 是否存在决定(现有逻辑不变)。
- 旧索引无 `_docids.npy` 时,`_page_doc_ids = None`,此时按 doc 删除需依赖编排层先查 pg(见 §5),仍可正常工作。

## 5. 统一删除编排与 API

新增一个编排函数,把三路删干净,解决 D1 / D2 / D3:

```python
# src/store/pgvector_store.py 新增
def get_page_ids_by_doc_id(self, doc_id: str) -> List[int]:
    with self.conn.cursor() as cur:
        cur.execute("SELECT DISTINCT page_id FROM chunks WHERE doc_id=%s", (doc_id,))
        return [r[0] for r in cur.fetchall()]

# 编排层(可放 pdf_ingestor 或独立 store 模块)
def delete_document(pg, faiss, bm25, doc_id: str):
    page_ids = pg.get_page_ids_by_doc_id(doc_id)        # 先取 page_id 再删 pg
    deleted_rows = pg.delete_by_doc_id(doc_id)
    faiss.delete_by_page_ids(page_ids)                  # 墓碑(或 faiss.delete_by_doc_id)
    # BM25 删除该 doc 的所有 chunk_id
    bm25.remove_chunks({c["chunk_id"] for c in <该 doc 的 chunks>})
    return deleted_rows

# src/api/routes.py 新增
@app.delete("/documents/{doc_id}")
async def delete_document_endpoint(doc_id: str):
    retriever = get_retriever()
    delete_document(retriever.pg, retriever.faiss, retriever.bm25, doc_id)
    return {"deleted": doc_id}
```

**关键顺序**:先 `get_page_ids_by_doc_id` 拿到 page_id,再删 pg 行。即便 FAISS 没存 `_page_doc_ids`,也能正确打墓碑;存了则两条路互不依赖(更稳,修复 D4)。

## 6. 落地优先级与风险

| 优先级 | 项 | 原因 |
|--------|----|------|
| P0 | 修 D2(BM25 不更新) | 唯一会造成"已删文档仍在答案里"的**正确性 bug**,改动最小(加 `remove_chunks` + 编排调用) |
| P1 | 修 D1 / FAISS 逻辑删除 | 解决 VRAM 泄漏与 orphan 干扰召回,需加 `_page_doc_ids` + 墓碑 + 过滤 |
| P2 | 上增量 BM25(U1) | 性能优化,语料到几十万 chunk 后才痛,但设计已就绪可一并做 |
| P3 | 修 U4 不一致 | `ingest` 失败路径在 `faiss.save()` 前先 `faiss.load()` 旧索引(或失败即丢弃内存态),避免孤儿落盘 |

**风险 / 注意事项**

- `_page_doc_ids` 用 `dtype=object`(存字符串),会增加少量内存,但比"每次删除都查 pg"更稳。
- `compact()` 是 O(N) 重建,应在低峰或异步触发;自动阈值 20% 可按需调整。
- BM25 corpus pickle 与 pg 行数需保持一致;若检测到不一致(如直接改库),应回退全量重建。
- 增量 BM25 依赖 `rank_bm25` 内部属性(`term_freqs` / `doc_len` / `idf` 等)保持公开可写;升级 `rank_bm25` 大版本时需回归测试 `get_scores` 是否仍读取这些字段。

## 7. 附录:改动文件清单

| 文件 | 改动 |
|------|------|
| `src/retrieval/bm25_retriever.py` | 加 `_tokenized`、缓存持久化、`fit_incremental`、`remove_chunks`、`_recompute_idf` |
| `src/store/faiss_store.py` | 加 `_page_doc_ids` / `_deleted_page_ids`、`add_pages` 携带 doc_id、墓碑删除、`compact`、`_rank_pages` 过滤、save/load 兼容 |
| `src/store/pgvector_store.py` | 加 `get_page_ids_by_doc_id` |
| `src/ingestion/pdf_ingestor.py` | `ingest` 入库后改调 `bm25.fit_incremental`;`add_pages` 传入 `page_doc_map` |
| `src/api/routes.py` | `/ingest` 改用增量 BM25;新增 `DELETE /documents/{doc_id}` + 编排调用 |
| `scripts/ingest_pdf.py` | 改用 `bm25.fit_incremental`(或保留全量作为 fallback) |
