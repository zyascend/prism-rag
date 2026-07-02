"""ViDoRe 评测适配器

实现 PrismRAG 统一检索器，将检索管道包装成可调用接口。
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import torch
from datasets import load_dataset
from tqdm import tqdm

from src.config import cfg
from src.ingestion.encoders import BGEEmbedder, ColPaliEmbedder
from src.ingestion.text_chunker import Chunk, TextChunker
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.fusion import RRFFusion
from src.retrieval.hyde import HyDEGenerator
from src.retrieval.reranker import Reranker
from src.retrieval.visual_retriever import VisualRetriever
from src.store.faiss_store import FaissColPaliStore
from src.store.pgvector_store import PgVectorStore

logger = logging.getLogger(__name__)


class PrismRAGRetriever:
    """PrismRAG 统一检索器（vidore-benchmark 适配用）"""

    def __init__(
        self,
        pg_store: PgVectorStore,
        faiss_store: FaissColPaliStore,
        bge: BGEEmbedder,
        colpali: ColPaliEmbedder,
        chunker: TextChunker,
        bm25: BM25Retriever,
        dense: DenseRetriever,
        visual: VisualRetriever,
        fusion: RRFFusion,
        reranker: Reranker,
        hyde: Optional[HyDEGenerator] = None,
        zerank_reranker: Optional[Reranker] = None,
    ):
        self.pg = pg_store
        self.faiss = faiss_store
        self.bge = bge
        self.colpali = colpali
        self.chunker = chunker
        self.bm25 = bm25
        self.dense = dense
        self.visual = visual
        self.fusion = fusion
        self.reranker = reranker
        self.hyde = hyde
        self.zerank_reranker = zerank_reranker

    def search(
        self,
        query: str,
        k: int = 10,
        use_bm25: bool = True,
        use_dense: bool = True,
        use_visual: bool = True,
        use_rerank: bool = True,
        visual_query_embedding: Optional[torch.Tensor] = None,
        use_hyde: bool = False,
        reranker_type: str = "bge",
    ) -> List[dict]:
        """统一检索接口

        Args:
            query: 查询文本
            k: 返回 Top-k chunk
            use_bm25/dense/visual: 控制各路的开关（消融用）
            use_rerank: 是否使用 cross-encoder 重排
            visual_query_embedding: 可选，预编码的 visual query embedding。
                                    传入时 visual route 走 search_with_embedding() 跳过编码。
            use_hyde: 是否启用 HyDE 查询改写
            reranker_type: 重排器选择 ("bge" | "zerank")

        Returns:
            结果列表，每个 dict 含 chunk_id, page_id, score, retrieval_type 等
        """
        result = self.search_with_trace(
            query, k, use_bm25, use_dense, use_visual, use_rerank,
            visual_query_embedding=visual_query_embedding,
            use_hyde=use_hyde, reranker_type=reranker_type,
        )
        return result["results"]

    def search_with_trace(
        self,
        query: str,
        k: int = 10,
        use_bm25: bool = True,
        use_dense: bool = True,
        use_visual: bool = True,
        use_rerank: bool = True,
        visual_query_embedding: Optional[torch.Tensor] = None,
        use_hyde: bool = False,
        reranker_type: str = "bge",
    ) -> dict:
        """带 retrieval_trace 的统一检索接口

        Args:
            visual_query_embedding: 可选，预编码的 visual query embedding。
                                    传入时 visual route 走 search_with_embedding() 而非现场编码。
            use_hyde: 是否启用 HyDE 查询改写。启用时会额外用 LLM 生成假设文档
                      并作为额外查询送入 dense/visual 路线。
            reranker_type: 重排器选择 ("bge" | "zerank")

        Returns:
            {"results": [...], "retrieval_trace": {"bm25_top5": [...], "dense_top5": [...],
             "visual_top5": [...], "hyde": "<generated text>"}}
        """
        routes = []
        trace = {"bm25_top5": [], "dense_top5": [], "visual_top5": [], "hyde": ""}

        # ── HyDE: 生成假设文档 ────────────────────────────────
        hyde_answer = ""
        if use_hyde and self.hyde is not None:
            hyde_answer = self.hyde.generate(query)
            trace["hyde"] = hyde_answer[:500] if hyde_answer else "(failed)"

        # ── BM25 route（始终用原始 query）───────────────────
        if use_bm25:
            try:
                bm25_results = self.bm25.search(query, k=20)
                routes.append(bm25_results)
                trace["bm25_top5"] = [
                    {"chunk_id": r["chunk_id"], "page_id": r["page_id"], "score": r["score"]}
                    for r in bm25_results[:5]
                ]
            except RuntimeError:
                logger.warning("BM25 未就绪，跳过")

        # ── Dense route（原始 query + 可选 HyDE answer）─────
        if use_dense:
            dense_queries = [query]
            if hyde_answer:
                dense_queries.append(hyde_answer)

            all_dense = []
            for q in dense_queries:
                results = self.dense.search(q, k=20)
                routes.append(results)
                all_dense.extend(results)

            # Trace: 仅记录原始 query 的 top-5
            if all_dense:
                trace["dense_top5"] = [
                    {"chunk_id": r["chunk_id"], "page_id": r["page_id"], "score": r["score"]}
                    for r in all_dense[:5]
                ]

        # ── Visual route（原始 query + 可选 HyDE answer）────
        if use_visual:
            try:
                # 原始 query — 优先使用预编码 embedding
                if visual_query_embedding is not None:
                    vis_results = self.visual.search_with_embedding(visual_query_embedding, k=20)
                else:
                    vis_results = self.visual.search(query, k=20)
                routes.append(vis_results)

                # HyDE answer — 现场编码（可能因 ColPali 已卸载而失败，不影响原始结果）
                if hyde_answer:
                    try:
                        hyde_vis = self.visual.search(hyde_answer, k=20)
                        routes.append(hyde_vis)
                        vis_results = vis_results + hyde_vis
                    except Exception as e:
                        logger.debug(f"HyDE visual 跳过（ColPali 可能未加载）: {e}")

                trace["visual_top5"] = [
                    {"chunk_id": r["chunk_id"], "page_id": r["page_id"], "score": r["score"]}
                    for r in vis_results[:5]
                ]
            except Exception as e:
                logger.warning(f"Visual 检索跳过: {e}")

        if not routes:
            return {"results": [], "retrieval_trace": trace}

        fused = self.fusion.fuse(routes, k=min(k * 2, 40))

        # ── Rerank（支持双 reranker）────────────────────────
        if use_rerank and fused:
            reranker = self.zerank_reranker if reranker_type == "zerank" else self.reranker
            reranked = reranker.rerank(query, fused, top_k=k)
            return {"results": reranked, "retrieval_trace": trace}

        return {"results": fused[:k], "retrieval_trace": trace}
