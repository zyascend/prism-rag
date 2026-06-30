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

    def search(
        self,
        query: str,
        k: int = 10,
        use_bm25: bool = True,
        use_dense: bool = True,
        use_visual: bool = True,
        use_rerank: bool = True,
    ) -> List[dict]:
        """统一检索接口

        Args:
            query: 查询文本
            k: 返回 Top-k chunk
            use_bm25/dense/visual: 控制各路的开关（消融用）
            use_rerank: 是否使用 cross-encoder 重排
        """
        routes = []

        if use_bm25:
            try:
                bm25_results = self.bm25.search(query, k=20)
                routes.append(bm25_results)
            except RuntimeError:
                logger.warning("BM25 未就绪，跳过")

        if use_dense:
            dense_results = self.dense.search(query, k=20)
            routes.append(dense_results)

        if use_visual:
            try:
                visual_results = self.visual.search(query, k=20)
                routes.append(visual_results)
            except Exception as e:
                logger.warning(f"Visual 检索跳过: {e}")

        if not routes:
            return []

        fused = self.fusion.fuse(routes, k=min(k * 2, 40))

        if use_rerank and fused:
            reranked = self.reranker.rerank(query, fused, top_k=k)
            return reranked

        return fused[:k]
