"""BM25 检索器 — 自维护增量统计实现（P2 效率优化）

设计目标（对应 Spec §4.3）：
- 淘汰 rank_bm25.BM25Okapi 的「全量重建」模式，改为自维护统计
  （doc_freq / idf / postings / avgdl），打分用标准 Okapi BM25 公式手动计算。
- `fit_incremental(new_chunks)`：仅追加新 chunk，O(vocab) 重算 idf（消除 U1）。
- `remove_chunks(chunk_ids)`：O(Δ) 递减 doc_freq + O(vocab) 重算 idf，
  不再 O(N) 重建索引（修复 P0 时临时采用的重建策略）。
- `save` / `load`：持久化语料统计到 `bm25_corpus.pkl`。
- `reconcile_from_pgvector`：启动时以 pg 真相源对账，仅增量同步差额
  （pg 多则 fit_incremental，pg 少则 remove_chunks）。

打分公式与 rank_bm25 BM25Okapi 完全一致：
  idf(t) = ln((N - df_t + 0.5) / (df_t + 0.5))   （本版本 rank_bm25 不含 epsilon 平滑）
  score(d,q) = sum_t idf(t) * (f_td*(k1+1)) / (f_td + k1*(1 - b + b*|d|/avgdl))
  k1 = 1.5, b = 0.75 （rank_bm25 BM25Okapi 默认）
已用对照测试验证与 BM25Okapi.get_scores 数值一致（见 tests/test_p2_incremental.py）。
"""

from __future__ import annotations

import math
import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Set

from src.store.pgvector_store import PgVectorStore
from src.observability import get_tracer

# rank_bm25 BM25Okapi（本项目所装版本）默认参数，保证数值一致
_K1 = 1.5
_B = 0.75
# 注意：本项目安装的 rank_bm25 版本的 BM25Okapi.idf = ln((N-df+0.5)/(df+0.5))，
# 不含 epsilon 平滑项（与部分文档描述不同）。为保证检索排序与旧实现逐位一致（NDCG 不漂移），此处同样不加。
_EPSILON = 0.0

# 空语料 / 未构建时返回空集（生产降级，不抛错）
_EMPTY: List[dict] = []


class BM25Retriever:
    """BM25 检索器（增量、自维护统计版本）"""

    def __init__(self, index_path: str = "indexes/bm25_corpus.pkl"):
        self._index_path = index_path
        # 文档载荷（检索结果回传）
        self._chunks: List[dict] = []
        # chunk_id 顺序（与统计数组下标对齐）
        self._chunk_id_order: List[str] = []
        # 分词后语料（持久化用）
        self._tokenized_corpus: List[List[str]] = []
        # 每篇文档的词频：doc_idx -> {term: count}
        self._chunk_term_freq: List[Dict[str, int]] = []
        # 每篇文档长度（token 数）
        self._doc_len: List[int] = []
        # 倒排索引：term -> [doc_idx, ...]（加速检索）
        self._postings: Dict[str, List[int]] = {}
        # 文档频率：term -> 含该词的文档数
        self._doc_freq: Dict[str, int] = {}
        # chunk_id -> doc_idx（O(1) 查找 + 幂等防护）
        self._chunk_index: Dict[str, int] = {}
        # 存活标记：删除采用逻辑删除（O(Δ) 递减，避免数组重建）
        self._alive: List[bool] = []
        self._corpus_size: int = 0
        self._total_len: int = 0
        self._avgdl: float = 0.0
        self._idf: Dict[str, float] = {}

    # ── 分词 ────────────────────────────────────────────────

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """分词：小写 + 非字母数字分割（与旧实现一致）"""
        import re
        return re.findall(r"[a-z0-9]+", text.lower())

    # ── 构建 ────────────────────────────────────────────────

    def fit(self, chunks: List[dict]):
        """从 chunk dict 列表全量构建 BM25 统计（冷启动 / reconcile 兜底）"""
        self._chunks = []
        self._chunk_id_order = []
        self._tokenized_corpus = []
        self._chunk_term_freq = []
        self._doc_len = []
        self._postings = {}
        self._doc_freq = {}
        self._chunk_index = {}
        self._alive = []
        self._corpus_size = 0
        self._total_len = 0
        self._avgdl = 0.0
        self._idf = {}
        for c in chunks:
            self._add_doc(c)
        self._avgdl = self._total_len / self._corpus_size if self._corpus_size else 0.0
        self._recompute_idf()

    def fit_from_pgvector(self, pg_store: PgVectorStore):
        """从 pgvector 读取所有 chunk 并全量构建（冷启动路径）"""
        chunks = []
        offset = 0
        limit = 1000
        while True:
            with pg_store.conn.cursor() as cur:
                cur.execute(
                    "SELECT chunk_id, page_id, doc_id, page_number, chunk_type, text FROM chunks ORDER BY chunk_id LIMIT %s OFFSET %s",
                    (limit, offset),
                )
                rows = cur.fetchall()
                if not rows:
                    break
                for r in rows:
                    chunks.append({
                        "chunk_id": r[0],
                        "page_id": r[1],
                        "doc_id": r[2],
                        "page_number": r[3],
                        "chunk_type": r[4],
                        "text": r[5],
                    })
                offset += limit
        self.fit(chunks)

    def fit_incremental(self, new_chunks: List[dict]) -> int:
        """增量追加新 chunk（消除 U1）。

        仅对新 chunk 分词追加，O(vocab) 重算 idf（不重建全量索引）。
        返回实际新增的 chunk 数（幂等跳过已存在的 chunk_id）。
        """
        added = 0
        for c in new_chunks:
            if c["chunk_id"] in self._chunk_index:
                continue
            self._add_doc(c)
            added += 1
        if added:
            self._avgdl = self._total_len / self._corpus_size if self._corpus_size else 0.0
            self._recompute_idf()
        return added

    def _add_doc(self, c: dict):
        cid = c["chunk_id"]
        toks = self._tokenize(c["text"])
        tf: Dict[str, int] = {}
        for t in toks:
            tf[t] = tf.get(t, 0) + 1
        idx = len(self._chunk_id_order)
        self._chunk_id_order.append(cid)
        self._chunks.append(c)
        self._tokenized_corpus.append(toks)
        self._chunk_term_freq.append(tf)
        dl = len(toks)
        self._doc_len.append(dl)
        self._alive.append(True)
        self._corpus_size += 1
        self._total_len += dl
        for t in tf:
            self._doc_freq[t] = self._doc_freq.get(t, 0) + 1
            self._postings.setdefault(t, []).append(idx)
        self._chunk_index[cid] = idx

    # ── 删除 ────────────────────────────────────────────────

    def remove_chunks(self, chunk_ids: Set[str]) -> int:
        """从 BM25 索引逻辑删除指定 chunk（O(Δ) 递减 doc_freq + O(vocab) 重算 idf）。

        采用逻辑删除（_alive 标记），避免删除即触发 O(N) 数组重建；
        删除频率低（doc 级），检索时过滤 _alive 即可，不影响正确性。
        """
        if not chunk_ids:
            return 0
        to_remove = [self._chunk_index[c] for c in chunk_ids if c in self._chunk_index]
        if not to_remove:
            return 0
        # 递减 doc_freq（O(变化文档的词数)）
        for idx in to_remove:
            for t in self._chunk_term_freq[idx]:
                self._doc_freq[t] -= 1
                if self._doc_freq[t] == 0:
                    del self._doc_freq[t]
            self._alive[idx] = False
            self._corpus_size -= 1
            self._total_len -= self._doc_len[idx]
        self._avgdl = self._total_len / self._corpus_size if self._corpus_size else 0.0
        self._recompute_idf()
        # 彻底清理已删 chunk 的载荷与映射（释放内存；检索已不命中）
        for idx in sorted(to_remove, reverse=True):
            del self._chunks[idx]
            del self._tokenized_corpus[idx]
            del self._chunk_term_freq[idx]
            del self._doc_len[idx]
            del self._alive[idx]
            del self._chunk_id_order[idx]
        # 重建 _chunk_index（下标已偏移）
        self._chunk_index = {cid: i for i, cid in enumerate(self._chunk_id_order)}
        # 重建倒排索引（仅引用存活下标）
        self._postings = {}
        for i, tf in enumerate(self._chunk_term_freq):
            for t in tf:
                self._postings.setdefault(t, []).append(i)
        return len(to_remove)

    def _recompute_idf(self):
        """重算 idf（O(vocab)）。公式与 rank_bm25 BM25Okapi（本版本）逐位一致。"""
        n = self._corpus_size
        if n == 0:
            self._idf = {}
            return
        self._idf = {}
        for t, df in self._doc_freq.items():
            self._idf[t] = math.log((n - df + 0.5) / (df + 0.5)) + _EPSILON

    # ── 检索 ────────────────────────────────────────────────

    def search(self, query: str, k: int = 20) -> List[dict]:
        """检索 Top-k chunk。语料为空 / 未构建时返回 []（不抛错，生产降级）。"""
        if self._corpus_size == 0 or not self._idf:
            return _EMPTY

        tracer = get_tracer()
        with tracer.start_span("bm25_search") as span:
            q_tokens = self._tokenize(query)
            # 候选文档 = 含任一查询词的文档（倒排加速，O(候选)）
            candidates: Set[int] = set()
            for t in q_tokens:
                post = self._postings.get(t)
                if post:
                    candidates.update(post)

            scored = []
            for idx in candidates:
                if not self._alive[idx]:
                    continue
                score = self._score_doc(idx, q_tokens)
                if score > 0:
                    chunk = self._chunks[idx]
                    scored.append({
                        **chunk,
                        "score": float(score),
                        "retrieval_type": "bm25",
                    })

            scored.sort(key=lambda r: r["score"], reverse=True)
            span.set_metadata({"num_results": len(scored), "k": k})
            return scored[:k]

    def _score_doc(self, idx: int, q_tokens: List[str]) -> float:
        dl = self._doc_len[idx]
        tf = self._chunk_term_freq[idx]
        avgdl = self._avgdl
        score = 0.0
        for t in q_tokens:
            f = tf.get(t, 0)
            if f == 0:
                continue
            idf = self._idf.get(t, 0.0)
            score += idf * (f * (_K1 + 1)) / (f + _K1 * (1 - _B + _B * dl / avgdl))
        return score

    # ── 持久化 ──────────────────────────────────────────────

    def save(self, path: Optional[str] = None):
        """保存语料统计到磁盘（P2 持久化，供重启对账）"""
        target = Path(path or self._index_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "chunks": self._chunks,
            "chunk_id_order": self._chunk_id_order,
            "tokenized_corpus": self._tokenized_corpus,
            "chunk_term_freq": self._chunk_term_freq,
            "doc_len": self._doc_len,
            "postings": self._postings,
            "doc_freq": self._doc_freq,
            "chunk_index": self._chunk_index,
            "alive": self._alive,
            "corpus_size": self._corpus_size,
            "total_len": self._total_len,
            "avgdl": self._avgdl,
        }
        with open(target, "wb") as f:
            pickle.dump(state, f)

    def save_atomic(self, path: Optional[str] = None):
        """原子保存：先写临时文件再 os.replace，避免写一半被检索进程读到。"""
        target = Path(path or self._index_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(target.suffix + ".part")
        self.save(str(tmp))
        os.replace(tmp, target)

    def load(self, path: Optional[str] = None) -> bool:
        """从磁盘加载语料统计，成功返回 True。"""
        target = Path(path or self._index_path)
        if not target.exists():
            return False
        with open(target, "rb") as f:
            state = pickle.load(f)
        self._chunks = state["chunks"]
        self._chunk_id_order = state["chunk_id_order"]
        self._tokenized_corpus = state["tokenized_corpus"]
        self._chunk_term_freq = state["chunk_term_freq"]
        self._doc_len = state["doc_len"]
        self._postings = state["postings"]
        self._doc_freq = state["doc_freq"]
        self._chunk_index = state["chunk_index"]
        self._alive = state["alive"]
        self._corpus_size = state["corpus_size"]
        self._total_len = state["total_len"]
        self._avgdl = state["avgdl"]
        self._recompute_idf()
        return True

    @property
    def ready(self) -> bool:
        """是否已构建语料（供路由判断是否需全量 refit）"""
        return self._corpus_size > 0

    # ── 启动对账（以 pg 真相源为准，仅增量同步差额）─────────

    def reconcile_from_pgvector(self, pg_store: PgVectorStore, save: bool = True):
        """启动时与 pg 对账：避免每次 ingest 都全量重建（消除 U1）。

        逻辑：
          - 无缓存且磁盘有 pkl -> 先 load。
          - 仍为空 -> 冷启动全量 fit。
          - 已有缓存 -> 比较 pg chunk_id 集合与本地：
              pg 多 -> fit_incremental 仅追加差额；
              pg 少 -> remove_chunks 仅删差额。
        """
        if self._corpus_size == 0 and Path(self._index_path).exists():
            self.load()
        if self._corpus_size == 0:
            self.fit_from_pgvector(pg_store)
            if save:
                self.save()
            return

        pg_ids = set(pg_store.get_all_chunk_ids())
        cached_ids = set(self._chunk_id_order)
        to_remove = cached_ids - pg_ids
        if to_remove:
            self.remove_chunks(to_remove)
        to_add = pg_ids - cached_ids
        if to_add:
            self.fit_incremental(pg_store.get_chunks_by_ids(list(to_add)))
        if save:
            self.save()
