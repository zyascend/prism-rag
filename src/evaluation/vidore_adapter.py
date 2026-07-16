"""ViDoRe 评测适配器

实现 PrismRAG 统一检索器，将检索管道包装成可调用接口。
"""

from __future__ import annotations

import logging
from typing import List, Optional

import torch

from src.ingestion.encoders import BGEEmbedder, ColPaliEmbedder
from src.ingestion.text_chunker import TextChunker
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.fusion import RRFFusion
from src.retrieval.hyde import HyDEGenerator
from src.retrieval.reranker import Reranker
from src.retrieval.visual_retriever import VisualRetriever
from src.observability import get_tracer, get_collector
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

    def delete_document(self, doc_id: str) -> dict:
        """删除一份文档，保证三路不再返回其内容（修复 D2 正确性缺陷）。

        严格顺序（修复 D4 删除顺序脆弱）：
          1. 先取该 doc 的 chunk_id / page_id（pg 行删前取，避免丢引用）
          2. pg.delete_by_doc_id（真相源，已 commit）
          3. bm25.remove_chunks（修复 D2：已删内容不再进入 RRF/答案）
          4. faiss.delete_by_doc_id（墓碑，修复 D1：视觉向量不再参与 MaxSim）
          5. faiss.maybe_compact（墓碑占比高时物理回收）
        """
        chunk_ids = set(self.pg.get_chunk_ids_by_doc_id(doc_id))
        page_ids = self.pg.get_page_ids_by_doc_id(doc_id)

        deleted_rows = self.pg.delete_by_doc_id(doc_id)
        removed_bm25 = self.bm25.remove_chunks(chunk_ids)

        faiss_removed = self.faiss.delete_by_doc_id(doc_id)
        faiss_compacted = self.faiss.maybe_compact()

        return {
            "doc_id": doc_id,
            "pg_deleted_rows": deleted_rows,
            "bm25_removed": removed_bm25,
            "faiss_removed": faiss_removed,
            "faiss_compacted": faiss_compacted,
            "page_ids": page_ids,
        }

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
        config_label: str = "",
    ) -> dict:
        """带 retrieval_trace 的统一检索接口

        Args:
            visual_query_embedding: 可选，预编码的 visual query embedding。
                                    传入时 visual route 走 search_with_embedding() 而非现场编码。
            use_hyde: 是否启用 HyDE 查询改写。启用时会额外用 LLM 生成假设文档
                      并作为额外查询送入 dense/visual 路线。
            reranker_type: 重排器选择 ("bge" | "zerank")
            config_label: 检索配置标签，用于 observability 聚合。若已有父 Trace
                         则继承其 label。

        Returns:
            {"results": [...], "retrieval_trace": {"bm25_top5": [...], "dense_top5": [...],
             "visual_top5": [...], "hyde": "<generated text>"}}
        """
        tracer = get_tracer()
        collector = get_collector()

        # 检测是否已有父 Trace（由 evaluate_generation 等上层创建）
        existing = tracer.current_trace()
        if existing is not None:
            # 挂载到已有 Trace，继承其 config_label
            owns_trace = False
        else:
            tracer.start_trace(query=query, config_label=config_label)
            owns_trace = True

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
            if owns_trace:
                obs_trace = tracer.finish_trace()
                if obs_trace:
                    collector.ingest_trace(obs_trace)
            return {"results": [], "retrieval_trace": trace}

        fused = self.fusion.fuse(routes, k=min(k * 2, 40))

        # ── Rerank（支持双 reranker）────────────────────────
        if use_rerank and fused:
            reranker = self.zerank_reranker if reranker_type == "zerank" else self.reranker
            # Wrap fusion+rerank in a span
            with tracer.start_span("fusion_rerank") as fusion_span:
                reranked = reranker.rerank(query, fused, top_k=k)
                # Extract rerank score stats from results
                rerank_scores = [r.get("rerank_score", 0.0) for r in reranked]
                fusion_span.set_metadata({
                    "num_fused_input": len(fused),
                    "num_reranked_output": len(reranked),
                    "num_results": len(reranked),  # collector 据此追踪 fused hits
                    "max_rerank_score": round(max(rerank_scores), 4) if rerank_scores else 0.0,
                    "min_rerank_score": round(min(rerank_scores), 4) if rerank_scores else 0.0,
                    "mean_rerank_score": round(sum(rerank_scores) / len(rerank_scores), 4) if rerank_scores else 0.0,
                })
            if owns_trace:
                obs_trace = tracer.finish_trace()
                if obs_trace:
                    collector.ingest_trace(obs_trace)
            return {"results": reranked, "retrieval_trace": trace}

        if owns_trace:
            obs_trace = tracer.finish_trace()
            if obs_trace:
                collector.ingest_trace(obs_trace)
        return {"results": fused[:k], "retrieval_trace": trace}
