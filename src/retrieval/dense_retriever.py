"""Dense 检索器 — pgvector HNSW 余弦相似度搜索"""

from __future__ import annotations

from typing import List

import numpy as np

from src.ingestion.encoders import BGEEmbedder
from src.store.pgvector_store import PgVectorStore


class DenseRetriever:
    """Dense 检索器：BGE encode query → pgvector HNSW 搜索"""

    def __init__(self, pg_store: PgVectorStore, embedder: BGEEmbedder):
        self.pg = pg_store
        self.bge = embedder

    def search(self, query: str, k: int = 20) -> List[dict]:
        """检索 Top-k chunk"""
        # 1. BGE 编码查询
        query_emb = self.bge.encode([query])
        query_vec = query_emb.cpu().numpy().astype(np.float32)[0]

        # 2. pgvector HNSW 搜索
        results = self.pg.search_by_vector(query_vec, k=k)

        # 3. 添加 retrieval_type 标记
        for r in results:
            r["retrieval_type"] = "dense"

        return results