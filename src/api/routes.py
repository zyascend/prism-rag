"""FastAPI 路由"""

from __future__ import annotations

import logging
import shutil
import uuid
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel
from src.cache.store import InMemoryLRUCache

from src.config import cfg
from src.evaluation.vidore_adapter import PrismRAGRetriever
from src.generation.generator import Generator, GenerationError
from src.generation.self_rag import SelfRAGOrchestrator, self_rag_config
from src.ingestion.encoders import BGEEmbedder, ColPaliEmbedder
from src.ingestion.pdf_ingestor import PDFIngestor
from src.ingestion.text_chunker import TextChunker
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.fusion import RRFFusion
from src.retrieval.reranker import Reranker
from src.retrieval.visual_retriever import VisualRetriever
from src.observability.middleware import ObservabilityMiddleware
from src.observability import get_collector
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
        use_visual = cfg.get("retrieval.use_visual", True)
        pg_store = PgVectorStore()
        faiss_store = FaissColPaliStore()
        bge = BGEEmbedder()
        # 本地 dev (use_visual=false) 免 ColPali 3.5B 下载：仅当启用 visual 路才构造
        colpali = ColPaliEmbedder() if use_visual else None
        chunker = TextChunker()
        bm25 = BM25Retriever()
        dense = DenseRetriever(pg_store, bge)
        visual = VisualRetriever(faiss_store, pg_store, colpali) if use_visual else None
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


@app.post("/cache/invalidate")
async def invalidate_cache():
    """失效检索缓存（重索引 / 语料变更后调用）。

    检索缓存以 index_version 为盐，本端点递增版本并清空内存缓存，
    确保后续请求重新检索而非命中旧结果。返回当前 index_version。
    """
    retriever = get_retriever()
    retriever.invalidate_cache()
    return {"status": "ok", "index_version": retriever.index_version}


@app.get("/prompts")
async def list_prompt_versions():
    """只读：列出所有 prompt 的当前生效版本与版本历史。

    用于排查线上当前生效的是哪一版 prompt。方案 A 不提供写入/切换端点，
    改 prompt 走"改 YAML → code review → 发版"流程。
    """
    from src.prompts import list_prompts

    return {"status": "ok", "prompts": list_prompts()}


@app.get("/health", response_model=HealthResponse)
async def health():
    retriever = get_retriever()
    return HealthResponse(
        status="ok",
        index_pages=retriever.faiss.num_pages,
        index_size_mb=round(retriever.faiss.index_size_mb, 1),
    )


@app.get("/trace/{trace_id}")
async def get_trace(trace_id: str):
    """按 X-Trace-Id 反查单条请求的完整 Trace（检索各路 + 生成层 context/citations）。

    用于生产态排查单条错误答案：拿到响应的 X-Trace-Id 后直接 GET 此端点，
    即可看 context 是否含正确答案、定位在检索层还是生成层。
    """
    trace = get_collector().get_trace(trace_id)
    if trace is None:
        raise HTTPException(status_code=404, detail="Trace not found (may have been evicted or never persisted)")
    return trace


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


class SelfRAGAttemptDetail(BaseModel):
    """单轮生成+Gate2 记录（fail→regen 回放）。"""
    attempt: int
    prompt_id: str
    answer: str = ""
    action: str = ""
    passed: Optional[bool] = None
    score: Optional[float] = None
    unsupported: List[str] = []
    latency_ms: Optional[float] = None
    gate_degraded: bool = False
    error: Optional[str] = None


class SelfRAGInfo(BaseModel):
    """Gate2 结果（向后兼容：默认关闭时仅 enabled=false）。"""
    enabled: bool = False
    passed: Optional[bool] = None
    score: Optional[float] = None
    attempts: Optional[int] = None
    final_action: Optional[str] = None
    gate_degraded: Optional[bool] = None
    attempts_detail: Optional[List[SelfRAGAttemptDetail]] = None


class AskResponse(BaseModel):
    query: str
    answer: str
    citations: List[Citation] = []
    retrieval_trace: RetrievalTrace = RetrievalTrace()
    self_rag: Optional[SelfRAGInfo] = None


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
            retriever.colpali, retriever.chunker, bm25=retriever.bm25,
        ).ingest(pdf_path, doc_id=doc_id)
    except Exception as e:
        retriever.pg.delete_by_doc_id(doc_id)
        pdf_path.unlink(missing_ok=True)
        if cfg.get("ingestion.parser") == "mineru":
            try:
                shutil.rmtree(Path("data/mineru_output") / doc_id, ignore_errors=True)
            except Exception:
                pass
        logger.error(f"ingest failed: {e}")
        raise HTTPException(status_code=500, detail="Ingestion failed")
    # P2-A：BM25 已通过 ingestor 增量维护；仅当为空（首篇/重启未加载）才全量 refit
    if not retriever.bm25.ready:
        retriever.bm25.fit_from_pgvector(retriever.pg)
    if cfg.get("retrieval.use_visual", True):
        retriever.faiss.save()
    return IngestResponse(**result)


@app.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest):
    retriever = get_retriever()
    use_visual = cfg.get("retrieval.use_visual", True)
    collector = get_collector()
    # L4 Answer 缓存的 config_label 与 L3 在线路径保持一致（空串：在线请求无 per-config 拆分）
    cache_label = ""

    # ── L4 Answer Cache ───────────────────────────────────────
    gen = get_generator(retriever.bge)
    answer_key = None
    cached_answer = None
    if cfg.cache.enabled and gen.cacheable:
        if retriever._answer_cache is None:
            retriever._answer_cache = InMemoryLRUCache(max_size=cfg.cache.max_size)
        answer_key = retriever.answer_cache_key(
            request.query, gen.model, request.k, request.doc_id,
        )
        cached_answer = retriever._answer_cache.get(answer_key)
    if cached_answer is not None:
        collector.record_cache_event("answer", hit=True, config_label=cache_label)
        rt = cached_answer["retrieval_trace"]
        trace = RetrievalTrace(
            bm25_top5=[RouteTraceItem(**t) for t in rt["bm25_top5"]],
            dense_top5=[RouteTraceItem(**t) for t in rt["dense_top5"]],
            visual_top5=[RouteTraceItem(**t) for t in rt["visual_top5"]],
        )
        sr_info = cached_answer.get("self_rag")
        return AskResponse(
            query=request.query, answer=cached_answer["answer"],
            citations=[Citation(**c) for c in cached_answer["citations"]],
            retrieval_trace=trace,
            self_rag=SelfRAGInfo(**sr_info) if sr_info else None,
        )

    try:
        search_result = retriever.search_with_trace(
            request.query, k=request.k,
            use_visual=use_visual, use_rerank=request.use_rerank,
        )
    except Exception as e:
        logger.error(f"search error: {e}")
        raise HTTPException(status_code=500, detail="Internal search error")
    results = search_result["results"]
    retrieval_trace = search_result["retrieval_trace"]
    if request.doc_id:
        results = [r for r in results if r.get("doc_id") == request.doc_id]
    trace = RetrievalTrace(
        bm25_top5=[RouteTraceItem(**t) for t in retrieval_trace["bm25_top5"]],
        dense_top5=[RouteTraceItem(**t) for t in retrieval_trace["dense_top5"]],
        visual_top5=[RouteTraceItem(**t) for t in retrieval_trace["visual_top5"]],
    )
    if not results:
        return AskResponse(
            query=request.query,
            answer="I don't have enough information to answer that question.",
            citations=[], retrieval_trace=trace,
        )
    try:
        sr_cfg = self_rag_config()
        if sr_cfg.get("enabled"):
            orchestrator = SelfRAGOrchestrator(gen)
            gen_out = orchestrator.answer(
                request.query, results, k_context=request.k
            )
        else:
            gen_out = gen.answer(request.query, results, k_context=request.k)
            gen_out.setdefault("self_rag", {"enabled": False})
    except GenerationError as e:
        raise HTTPException(status_code=502, detail=str(e))

    sr_payload = gen_out.get("self_rag") or {"enabled": False}
    # 响应暴露稳定字段 + attempts_detail（答案已在 orchestrator 侧截断）
    sr_public = {
        k: sr_payload.get(k)
        for k in (
            "enabled",
            "passed",
            "score",
            "attempts",
            "final_action",
            "gate_degraded",
            "attempts_detail",
        )
        if k in sr_payload or k == "enabled"
    }
    sr_public.setdefault("enabled", bool(sr_payload.get("enabled", False)))

    # 写入 L4 Answer 缓存（受全局开关 + 确定性守卫；命中率经 cache_label 聚合）
    if cfg.cache.enabled and gen.cacheable and answer_key is not None and retriever._answer_cache is not None:
        retriever._answer_cache.put(answer_key, {
            "answer": gen_out["answer"],
            "citations": gen_out["citations"],
            "retrieval_trace": retrieval_trace,
            "self_rag": sr_public,
        })
        collector.record_cache_event("answer", hit=False, config_label=cache_label)

    return AskResponse(
        query=request.query, answer=gen_out["answer"],
        citations=[Citation(**c) for c in gen_out["citations"]],
        retrieval_trace=trace,
        self_rag=SelfRAGInfo(**sr_public),
    )