"""FastAPI 路由"""

from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from src.config import cfg
from src.evaluation.vidore_adapter import PrismRAGRetriever
from src.generation.generator import Generator, GenerationError
from src.ingestion.encoders import BGEEmbedder, ColPaliEmbedder
from src.ingestion.pdf_ingestor import PDFIngestor
from src.ingestion.text_chunker import TextChunker
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.fusion import RRFFusion
from src.retrieval.reranker import Reranker
from src.retrieval.visual_retriever import VisualRetriever
from src.observability.middleware import ObservabilityMiddleware
from src.store.faiss_store import FaissColPaliStore
from src.store.pgvector_store import PgVectorStore

logger = logging.getLogger(__name__)

app = FastAPI(title="PrismRAG API", version="0.1.0")
app.add_middleware(ObservabilityMiddleware)
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


class RouteTraceItem(BaseModel):
    chunk_id: str
    page_id: int
    score: float


class RetrievalTrace(BaseModel):
    bm25_top5: List[RouteTraceItem] = []
    dense_top5: List[RouteTraceItem] = []
    visual_top5: List[RouteTraceItem] = []


class SearchResponse(BaseModel):
    query: str
    results: List[SearchResult]
    retrieval_trace: RetrievalTrace = RetrievalTrace()
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


_generator: Optional[Generator] = None


def get_generator(bge=None) -> Generator:
    global _generator
    if _generator is None:
        _generator = Generator(bge_embedder=bge)
    return _generator


def set_retriever(r):
    global _retriever
    _retriever = r


def set_generator(g):
    global _generator
    _generator = g


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
        result = retriever.search_with_trace(
            query=request.query,
            k=request.k,
            use_rerank=request.use_rerank,
        )
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    return SearchResponse(
        query=request.query,
        results=[SearchResult(**r) for r in result["results"]],
        retrieval_trace=RetrievalTrace(
            bm25_top5=[RouteTraceItem(**t) for t in result["retrieval_trace"]["bm25_top5"]],
            dense_top5=[RouteTraceItem(**t) for t in result["retrieval_trace"]["dense_top5"]],
            visual_top5=[RouteTraceItem(**t) for t in result["retrieval_trace"]["visual_top5"]],
        ),
        num_results=len(result["results"]),
    )


class IngestResponse(BaseModel):
    doc_id: str
    num_pages: int
    num_chunks: int


class Citation(BaseModel):
    chunk_id: str
    page_id: int
    doc_id: Optional[str] = None
    page_number: Optional[int] = None
    snippet: str


class AskRequest(BaseModel):
    query: str
    doc_id: Optional[str] = None
    k: int = 5
    use_rerank: bool = True


class AskResponse(BaseModel):
    query: str
    answer: str
    citations: List[Citation] = []
    retrieval_trace: RetrievalTrace = RetrievalTrace()


UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


@app.post("/ingest", response_model=IngestResponse)
async def ingest(file: UploadFile = File(...)):
    if not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(status_code=422, detail="仅支持 PDF 文件")
    doc_id = uuid.uuid4().hex[:12]
    pdf_path = UPLOAD_DIR / f"{doc_id}.pdf"
    pdf_path.write_bytes(await file.read())
    retriever = get_retriever()
    try:
        result = PDFIngestor(
            retriever.pg, retriever.faiss, retriever.bge,
            retriever.colpali, retriever.chunker,
        ).ingest(pdf_path, doc_id=doc_id)
    except Exception as e:
        retriever.pg.delete_by_doc_id(doc_id)
        logger.error(f"ingest failed: {e}")
        raise HTTPException(status_code=500, detail=f"入库失败: {e}")
    retriever.bm25.fit_from_pgvector(retriever.pg)
    if cfg.get("retrieval.use_visual", True):
        retriever.faiss.save()
    return IngestResponse(**result)


@app.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest):
    retriever = get_retriever()
    use_visual = cfg.get("retrieval.use_visual", True)
    try:
        results = retriever.search(
            request.query, k=request.k,
            use_visual=use_visual, use_rerank=request.use_rerank,
        )
    except Exception as e:
        logger.error(f"search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    if request.doc_id:
        results = [r for r in results if r.get("doc_id") == request.doc_id]
    if not results:
        return AskResponse(query=request.query,
                           answer="I don't have enough information to answer that question.",
                           citations=[])
    try:
        gen = get_generator(retriever.bge).answer(request.query, results, k_context=request.k)
    except GenerationError as e:
        raise HTTPException(status_code=502, detail=str(e))
    return AskResponse(
        query=request.query, answer=gen["answer"],
        citations=[Citation(**c) for c in gen["citations"]],
    )