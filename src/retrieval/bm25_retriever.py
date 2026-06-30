"""BM25 检索器 — 基于 rank_bm25"""

from __future__ import annotations

import math
from typing import List, Optional

from rank_bm25 import BM25Okapi

from src.ingestion.text_chunker import Chunk
from src.store.pgvector_store import PgVectorStore


class BM25Retriever:
    """BM25 检索器"""

    def __init__(self):
        self._bm25: Optional[BM25Okapi] = None
        self._chunks: List[dict] = []

    def fit_from_pgvector(self, pg_store: PgVectorStore):
        """从 pgvector 读取所有 chunk 并构建 BM25 索引"""
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

    def fit(self, chunks: List[dict]):
        """从 chunk dict 列表构建 BM25 索引"""
        self._chunks = chunks
        tokenized_corpus = [self._tokenize(c["text"]) for c in chunks]
        self._bm25 = BM25Okapi(tokenized_corpus)

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """分词：小写 + 非字母数字分割"""
        import re
        return re.findall(r"[a-z0-9]+", text.lower())

    def search(self, query: str, k: int = 20) -> List[dict]:
        """检索 Top-k chunk"""
        if self._bm25 is None:
            raise RuntimeError("BM25 索引未构建，请先调用 fit()")

        tokenized_query = self._tokenize(query)
        scores = self._bm25.get_scores(tokenized_query)

        top_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True,
        )[:k]

        results = []
        for idx in top_indices:
            score = scores[idx]
            if score > 0:
                chunk = self._chunks[idx]
                results.append({
                    **chunk,
                    "score": float(score),
                    "retrieval_type": "bm25",
                })

        return results