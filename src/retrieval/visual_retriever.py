"""Visual 检索器 — ColPali + FAISS MaxSim + pgvector grounding 反查"""

from __future__ import annotations

from typing import List

from src.ingestion.encoders import ColPaliEmbedder
from src.store.faiss_store import FaissColPaliStore
from src.store.pgvector_store import PgVectorStore


class VisualRetriever:
    """Visual 检索器：ColPali encode → FAISS MaxSim → pgvector grounding 反查"""

    def __init__(
        self,
        faiss_store: FaissColPaliStore,
        pg_store: PgVectorStore,
        colpali_embedder: ColPaliEmbedder,
    ):
        self.faiss = faiss_store
        self.pg = pg_store
        self.colpali = colpali_embedder

    def search(self, query: str, k: int = 20) -> List[dict]:
        """检索 Top-k 页 → 反查该页所有 chunk"""
        # 1. ColPali 编码查询
        q_emb = self.colpali.encode_query(query)

        # 2. FAISS MaxSim 搜索 → Top-k 页
        page_results = self.faiss.maxsim_search(q_emb, k=k)

        if not page_results:
            return []

        # 3. Grounding 反查：命中页的所有 BGE chunk
        page_ids = [pr["page_id"] for pr in page_results]
        page_score_map = {pr["page_id"]: pr["score"] for pr in page_results}

        chunks = self.pg.get_chunks_by_page_ids(page_ids)

        # 4. 合并分数
        results = []
        for chunk in chunks:
            results.append({
                **chunk,
                "score": page_score_map[chunk["page_id"]],
                "retrieval_type": "visual",
            })

        return results
