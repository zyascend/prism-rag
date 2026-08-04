"""Visual 检索器 — ColPali + FAISS MaxSim + pgvector grounding 反查"""

from __future__ import annotations

from typing import Dict, List

import torch

from src.ingestion.encoders import ColPaliEmbedder
from src.store.faiss_store import FaissColPaliStore
from src.store.pgvector_store import PgVectorStore
from src.observability import get_tracer


class VisualRetriever:
    """Visual 检索器：ColPali encode → FAISS MaxSim → pgvector grounding 反查

    Grounding 必须保留 MaxSim 的页序：SQL ``page_id = ANY(...)`` 不保证顺序，
    若按 DB 返回序进 RRF，Visual_only / 融合中的 visual 路会变成乱序噪声。
    """

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
        """检索 Top-k 页 → 反查该页所有 chunk（按 MaxSim 页序展开）"""
        tracer = get_tracer()

        with tracer.start_span("visual_encode") as span:
            q_emb = self.colpali.encode_query(query)
            span.set_metadata({"batch_size": 1})

        with tracer.start_span("visual_search") as span:
            page_results = self.faiss.maxsim_search(q_emb, k=k)
            span.set_metadata({
                "num_pages": len(page_results),
                "k": k,
                "num_results": len(page_results),
            })

        return self._ground_pages(page_results)

    def search_with_embedding(self, q_emb: torch.Tensor, k: int = 20) -> List[dict]:
        """使用预编码 query embedding 执行检索（跳过 encode_query()）"""
        tracer = get_tracer()

        with tracer.start_span("visual_search") as span:
            page_results = self.faiss.maxsim_search(q_emb, k=k)
            span.set_metadata({
                "num_pages": len(page_results),
                "k": k,
                "pre_encoded": True,
                "num_results": len(page_results),
            })

        return self._ground_pages(page_results)

    def search_pages(self, query: str, k: int = 20) -> List[dict]:
        """页级检索：返回 MaxSim Top-k 页（不 expand chunk）。

        与官方 vidore-benchmark 同构，供 Visual_only_pages 消融 / 诊断。
        每项: ``{"page_id": int, "score": float}``
        """
        tracer = get_tracer()
        with tracer.start_span("visual_encode") as span:
            q_emb = self.colpali.encode_query(query)
            span.set_metadata({"batch_size": 1, "page_level": True})
        with tracer.start_span("visual_search") as span:
            page_results = self.faiss.maxsim_search(q_emb, k=k)
            span.set_metadata({
                "num_pages": len(page_results),
                "k": k,
                "page_level": True,
            })
        return page_results

    def search_pages_with_embedding(self, q_emb: torch.Tensor, k: int = 20) -> List[dict]:
        """页级检索（预编码 query embedding）。"""
        tracer = get_tracer()
        with tracer.start_span("visual_search") as span:
            page_results = self.faiss.maxsim_search(q_emb, k=k)
            span.set_metadata({
                "num_pages": len(page_results),
                "k": k,
                "pre_encoded": True,
                "page_level": True,
            })
        return page_results

    def _ground_pages(self, page_results: List[dict]) -> List[dict]:
        """按 MaxSim 页序展开 chunk；同页内保持 SQL 相对序。"""
        if not page_results:
            return []

        page_ids = [pr["page_id"] for pr in page_results]
        page_score_map = {pr["page_id"]: pr["score"] for pr in page_results}
        chunks = self.pg.get_chunks_by_page_ids(page_ids)

        by_page: Dict[int, List[dict]] = {}
        for chunk in chunks:
            pid = chunk["page_id"]
            by_page.setdefault(pid, []).append(chunk)

        results: List[dict] = []
        for pid in page_ids:
            score = page_score_map[pid]
            for chunk in by_page.get(pid, []):
                results.append({
                    **chunk,
                    "score": score,
                    "retrieval_type": "visual",
                })
        return results
