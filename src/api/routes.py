"""FastAPI 路由"""

from __future__ import annotations

import logging
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.config import cfg
from src.evaluation.vidore_adapter import PrismRAGRetriever
from src.ingestion.encoders import BGEEmbedder, ColPaliEmbedder
from src.ingestion.text_chunker import TextChunker
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.fusion import RRFFusion
from src.retrieval.reranker import Reranker
from src.retrieval.visual_retriever import VisualRetriever
from src.store.faiss_store import FaissColPaliStore
from src.store.pgvector_store import PgVectorStore

logger = logging.getLogger(__name__)

app = FastAPI(title="PrismRAG API", version="0.1.0")
_retriever: Optional[PrismRAGRetriever] = None


class SearchRequest(BaseModel):
    query: str
    k: int = 10
    use_rerank: bool = True


class SearchResult(BaseModel):
    chunk_id: str
    page_id: int
    doc_id: str
    text: str
    chunk_type: str
    score: float
    retrieval_type: str


class SearchResponse(BaseModel):
    query: str
    results: List[SearchResult]
    num_results: int


class HealthResponse(BaseModel):
    status: str
    index_pages: int = 0
    index_size_mb: float = 0.0


def get_retriever() -> PrismRAGRetriever:
    global _retriever
    if _retriever is None:
        cfg.load()
        pg_store = PgVectorStore()
        faiss_store = FaissColPaliStore()
        bge = BGEEmbedder()
        colpali = ColPaliEmbedder()
        chunker = TextChunker()
        bm25 = BM25Retriever()
        dense = DenseRetriever(pg_store, bge)
        visual = VisualRetriever(faiss_store, pg_store, colpali)
        fusion = RRFFusion(rrf_k=60)
        reranker = Reranker()

        _retriever = PrismRAGRetriever(
            pg_store=pg_store, faiss_store=faiss_store, bge=bge, colpali=colpali,
            chunker=chunker, bm25=bm25, dense=dense, visual=visual,
            fusion=fusion, reranker=reranker,
        )

        faiss_loaded = faiss_store.load()
        if faiss_loaded:
            bm25.fit_from_pgvector(pg_store)
            logger.info("API: 索引加载完成")
        else:
            logger.warning("API: FAISS 索引未找到，请先运行 ingest_vidore.py")
    return _retriever


@app.get("/health", response_model=HealthResponse)
async def health():
    retriever = get_retriever()
    return HealthResponse(
        status="ok",
        index_pages=retriever.faiss.num_pages,
        index_size_mb=round(retriever.faiss.index_size_mb, 1),
    )


@app.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    retriever = get_retriever()
    try:
        results = retriever.search(
            query=request.query,
            k=request.k,
            use_rerank=request.use_rerank,
        )
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    return SearchResponse(
        query=request.query,
        results=[SearchResult(**r) for r in results],
        num_results=len(results),
    )