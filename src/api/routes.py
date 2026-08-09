"""FastAPI 路由"""

from __future__ import annotations

import logging
import os
import shutil
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from src.cache.store import InMemoryLRUCache

from src.config import cfg
from src.evaluation.vidore_adapter import PrismRAGRetriever
from src.generation.generator import Generator, GenerationError
from src.generation.self_rag import SelfRAGOrchestrator, self_rag_config
from src.ingestion.encoders import BGEEmbedder, create_visual_encoder
from src.ingestion.pdf_ingestor import PDFIngestor
from src.ingestion.text_chunker import TextChunker
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.crag import CorrectiveRAG, crag_config
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.fusion import RRFFusion
from src.retrieval.reranker import Reranker
from src.retrieval.visual_retriever import VisualRetriever
from src.observability.middleware import ObservabilityMiddleware
from src.observability import get_collector
from src.rejection import abstain_message
from src.store.faiss_store import FaissColPaliStore
from src.store.pgvector_store import PgVectorStore

logger = logging.getLogger(__name__)

app = FastAPI(title="PrismRAG API", version="0.1.0")
app.add_middleware(ObservabilityMiddleware)

from fastapi.middleware.cors import CORSMiddleware

_cors = os.environ.get("PRISMRAG_CORS_ORIGINS", "").strip()
if _cors:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[o.strip() for o in _cors.split(",") if o.strip()],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-Trace-Id"],
    )
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
    # demo hover：命中页/chunk 文本预览（可选，旧缓存可无）
    text: Optional[str] = None
    doc_id: Optional[str] = None
    chunk_type: Optional[str] = None


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
        visual_backend = cfg.get("embedding.visual_backend", "colpali")
        pg_store = PgVectorStore()
        # colqwen2 与云上 283q 对齐；local-dev 可指向独立 demo FAISS 路径
        if str(visual_backend).startswith("colqwen2"):
            faiss_store = FaissColPaliStore(
                index_path=cfg.get("storage.faiss.colqwen2_index_path"),
                id_map_path=cfg.get("storage.faiss.colqwen2_id_map_path"),
            )
        else:
            faiss_store = FaissColPaliStore()
        bge = BGEEmbedder()
        # 仅当启用 visual 才加载 Col*（local-dev demo 数据量小可开）
        colpali = create_visual_encoder(visual_backend) if use_visual else None
        chunker = TextChunker(
            image_caption_chunks=cfg.get("ingestion.image_caption_chunks", False),
        )
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
            logger.info(
                "API: 索引加载完成 · visual=%s backend=%s",
                use_visual, visual_backend if use_visual else "off",
            )
        else:
            logger.warning(
                "API: FAISS 未找到（path=%s）；upload/ingest 后会创建。"
                "全量 ViDoRe 请用 ingest_vidore.py",
                faiss_store.index_path,
            )
            try:
                bm25.fit_from_pgvector(pg_store)
            except Exception as e:
                logger.warning("API: BM25 fit 跳过: %s", e)
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
    mode: str = "pipeline"  # pipeline | agent


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


class CRAGInfo(BaseModel):
    """Corrective RAG 检索纠错摘要（默认关闭）。"""
    enabled: bool = False
    applied: Optional[bool] = None
    final_action: Optional[str] = None
    query_original: Optional[str] = None
    query_used: Optional[str] = None
    num_relevant: Optional[int] = None
    sufficient: Optional[bool] = None
    skip_reason: Optional[str] = None
    grade_degraded: Optional[bool] = None


class AgentStepInfo(BaseModel):
    """Agent trajectory 单步（与 StepRecord 对齐）。"""
    step: int = 0
    node: str = ""
    tool: Optional[str] = None
    input_summary: str = ""
    output_summary: str = ""
    ok: bool = True
    error: Optional[str] = None
    latency_ms: Optional[float] = None
    counts: Dict[str, Any] = {}


class AgentInfo(BaseModel):
    """Agent 路径摘要（默认 null；mode=agent 且 enabled 时 used=true）。"""
    used: bool = False
    status: Optional[str] = None
    subqueries: List[str] = []
    trajectory: List[AgentStepInfo] = []
    counts: Dict[str, Any] = {}
    degraded_to_pipeline: bool = False
    thread_id: Optional[str] = None
    ignored_reason: Optional[str] = None


class AskResumeRequest(BaseModel):
    """HITL resume：继续被 interrupt 的 agent 线程。"""
    thread_id: str
    subqueries: Optional[List[str]] = None  # None = approve as-is（空列表→图内 fallback）
    k: int = 5
    use_rerank: bool = True
    doc_id: Optional[str] = None


class AskResponse(BaseModel):
    query: str
    answer: str
    citations: List[Citation] = []
    retrieval_trace: RetrievalTrace = RetrievalTrace()
    self_rag: Optional[SelfRAGInfo] = None
    crag: Optional[CRAGInfo] = None
    agent: Optional[AgentInfo] = None
    # 入模上下文（压缩/表保护后的最终 prompt context）；demo 链路透视用
    context: str = ""


UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# ── 异步入库任务（demo 嵌入页轮询）──
_ingest_jobs: Dict[str, Dict[str, Any]] = {}
_ingest_jobs_lock = threading.Lock()
_INGEST_JOB_TTL_SEC = 3600


class DocumentInfo(BaseModel):
    doc_id: str
    content_hash: str = ""
    source_path: str = ""
    created_at: Optional[str] = None
    num_chunks: int = 0
    num_pages: int = 0
    num_tables: int = 0
    num_text: int = 0
    page_from: Optional[int] = None
    page_to: Optional[int] = None


class CorpusStats(BaseModel):
    num_documents: int = 0
    num_document_rows: int = 0
    num_pages: int = 0
    num_chunks: int = 0
    num_table_chunks: int = 0
    index_pages: int = 0
    index_size_mb: float = 0.0
    use_visual: bool = True
    bm25_ready: bool = False


class DocumentsResponse(BaseModel):
    stats: CorpusStats
    documents: List[DocumentInfo] = []


class IngestJobCreateResponse(BaseModel):
    job_id: str
    doc_id: str
    filename: str
    status: str = "queued"


class IngestJobStatus(BaseModel):
    job_id: str
    doc_id: str
    filename: str = ""
    status: str  # queued | running | done | error
    phase: str = "queued"
    pct: float = 0.0
    message: str = ""
    events: List[dict] = []
    result: Optional[IngestResponse] = None
    error: Optional[str] = None
    created_at: float = 0.0
    updated_at: float = 0.0


def _job_update(job_id: str, **kwargs) -> None:
    with _ingest_jobs_lock:
        job = _ingest_jobs.get(job_id)
        if not job:
            return
        job.update(kwargs)
        job["updated_at"] = time.time()
        if "phase" in kwargs or "message" in kwargs:
            job.setdefault("events", []).append(
                {
                    "t": time.time(),
                    "phase": job.get("phase"),
                    "pct": job.get("pct"),
                    "message": job.get("message"),
                }
            )
            # 限制事件长度
            if len(job["events"]) > 80:
                job["events"] = job["events"][-80:]


def _run_ingest_job(job_id: str, pdf_path: Path, doc_id: str) -> None:
    """后台线程：解析 → 分块 → 编码 → 写库，并推进 job 进度。"""
    _job_update(job_id, status="running", phase="start", pct=1, message="启动入库…")
    retriever = get_retriever()

    def progress(phase: str, pct: float, message: str) -> None:
        _job_update(job_id, phase=phase, pct=pct, message=message, status="running")

    try:
        ingestor = PDFIngestor(
            retriever.pg, retriever.faiss, retriever.bge,
            retriever.colpali, retriever.chunker, bm25=retriever.bm25,
        )
        ingestor.set_progress_fn(progress)
        result = ingestor.ingest(pdf_path, doc_id=doc_id)
        if not retriever.bm25.ready:
            progress("bm25", 96, "BM25 全量 fit…")
            retriever.bm25.fit_from_pgvector(retriever.pg)
        if cfg.get("retrieval.use_visual", True):
            progress("faiss", 98, "保存 FAISS…")
            retriever.faiss.save()
        try:
            retriever.invalidate_cache()
        except Exception as e:
            logger.warning("post-ingest cache invalidate failed: %s", e)
        payload = IngestResponse(
            doc_id=result["doc_id"],
            num_pages=int(result.get("num_pages") or 0),
            num_chunks=int(result.get("num_chunks") or 0),
        )
        _job_update(
            job_id,
            status="done",
            phase="done",
            pct=100,
            message=f"完成 · status={result.get('status', 'ok')}",
            result=payload.model_dump(),
        )
    except Exception as e:
        logger.exception("ingest job %s failed: %s", job_id, e)
        try:
            retriever.pg.delete_by_doc_id(doc_id)
        except Exception:
            pass
        pdf_path.unlink(missing_ok=True)
        if cfg.get("ingestion.parser") == "mineru":
            try:
                shutil.rmtree(Path("data/mineru_output") / doc_id, ignore_errors=True)
            except Exception:
                pass
        _job_update(
            job_id,
            status="error",
            phase="error",
            pct=100,
            message=str(e),
            error=str(e),
        )


def _finalize_sync_ingest(retriever, result: dict) -> IngestResponse:
    if not retriever.bm25.ready:
        retriever.bm25.fit_from_pgvector(retriever.pg)
    if cfg.get("retrieval.use_visual", True):
        retriever.faiss.save()
    try:
        retriever.invalidate_cache()
    except Exception as e:
        logger.warning("post-ingest cache invalidate failed: %s", e)
    return IngestResponse(
        doc_id=result["doc_id"],
        num_pages=int(result.get("num_pages") or 0),
        num_chunks=int(result.get("num_chunks") or 0),
    )


@app.get("/documents", response_model=DocumentsResponse)
async def list_documents():
    """当前知识库文档列表 + 全库统计（demo 文档页）。"""
    retriever = get_retriever()
    try:
        docs = retriever.pg.list_documents()
        stats = retriever.pg.corpus_stats()
    except Exception as e:
        logger.exception("list_documents failed: %s", e)
        raise HTTPException(status_code=500, detail=f"list documents failed: {e}")
    stats_out = CorpusStats(
        **stats,
        index_pages=getattr(retriever.faiss, "num_pages", 0) or 0,
        index_size_mb=round(float(getattr(retriever.faiss, "index_size_mb", 0) or 0), 2),
        use_visual=bool(cfg.get("retrieval.use_visual", True)),
        bm25_ready=bool(getattr(retriever.bm25, "ready", False)),
    )
    return DocumentsResponse(
        stats=stats_out,
        documents=[DocumentInfo(**d) for d in docs],
    )


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
        logger.exception("ingest failed: %s", e)
        # 本地 demo 需要可读错误（PG 未起 / 模型未下 / 解析失败），勿只回笼统文案
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {e}")
    return _finalize_sync_ingest(retriever, result)


@app.post("/ingest/jobs", response_model=IngestJobCreateResponse)
async def create_ingest_job(file: UploadFile = File(...)):
    """异步入库：立刻返回 job_id，前端轮询 GET /ingest/jobs/{id} 看进度。

    编码在独立 daemon 线程执行，避免阻塞 uvicorn 事件循环，便于实时轮询。
    """
    if not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(status_code=422, detail="仅支持 PDF 文件")
    job_id = uuid.uuid4().hex[:16]
    doc_id = uuid.uuid4().hex[:12]
    filename = file.filename or f"{doc_id}.pdf"
    pdf_path = UPLOAD_DIR / f"{doc_id}.pdf"
    pdf_path.write_bytes(await file.read())
    now = time.time()
    with _ingest_jobs_lock:
        # 清理过期 job
        dead = [k for k, v in _ingest_jobs.items() if now - v.get("created_at", now) > _INGEST_JOB_TTL_SEC]
        for k in dead:
            _ingest_jobs.pop(k, None)
        _ingest_jobs[job_id] = {
            "job_id": job_id,
            "doc_id": doc_id,
            "filename": filename,
            "status": "queued",
            "phase": "queued",
            "pct": 0.0,
            "message": "已接收文件，排队中…",
            "events": [{"t": now, "phase": "queued", "pct": 0, "message": "queued"}],
            "result": None,
            "error": None,
            "created_at": now,
            "updated_at": now,
        }
    t = threading.Thread(
        target=_run_ingest_job,
        args=(job_id, pdf_path, doc_id),
        name=f"ingest-{job_id}",
        daemon=True,
    )
    t.start()
    return IngestJobCreateResponse(
        job_id=job_id, doc_id=doc_id, filename=filename, status="queued",
    )


@app.get("/ingest/jobs/{job_id}", response_model=IngestJobStatus)
async def get_ingest_job(job_id: str):
    with _ingest_jobs_lock:
        job = _ingest_jobs.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="job not found")
        # 浅拷贝，避免持锁序列化
        data = dict(job)
    result = data.get("result")
    return IngestJobStatus(
        job_id=data["job_id"],
        doc_id=data["doc_id"],
        filename=data.get("filename") or "",
        status=data["status"],
        phase=data.get("phase") or "",
        pct=float(data.get("pct") or 0),
        message=data.get("message") or "",
        events=list(data.get("events") or []),
        result=IngestResponse(**result) if result else None,
        error=data.get("error"),
        created_at=float(data.get("created_at") or 0),
        updated_at=float(data.get("updated_at") or 0),
    )


def _make_complete_fn(gen):
    """Build complete_fn from Generator (same chat client pattern as CRAG/self_rag)."""
    injected = getattr(gen, "_complete_fn", None)
    if callable(injected):
        return injected
    from src.generation.context_filter import openai_complete_fn

    client = getattr(gen, "client", None)
    model = getattr(gen, "model", None) or cfg.get("llm.model", "gpt-4o-mini")
    if client is None:
        raise RuntimeError("generator has no client for agent complete_fn")
    return openai_complete_fn(client, model)


def _agent_step_infos(trajectory: Optional[List[dict]], *, return_trajectory: bool) -> List[AgentStepInfo]:
    if not return_trajectory or not trajectory:
        return []
    out: List[AgentStepInfo] = []
    for t in trajectory:
        if not isinstance(t, dict):
            continue
        out.append(
            AgentStepInfo(
                step=int(t.get("step") or 0),
                node=str(t.get("node") or ""),
                tool=t.get("tool"),
                input_summary=str(t.get("input_summary") or ""),
                output_summary=str(t.get("output_summary") or ""),
                ok=bool(t.get("ok", True)),
                error=t.get("error"),
                latency_ms=t.get("latency_ms"),
                counts=dict(t.get("counts") or {}) if isinstance(t.get("counts"), dict) else {},
            )
        )
    return out


def _trace_from_evidence(evidence: Optional[List[dict]]) -> RetrievalTrace:
    """Synthesize a light retrieval_trace from agent evidence (dense_top5 only)."""
    items: List[RouteTraceItem] = []
    for e in (evidence or [])[:5]:
        if not isinstance(e, dict):
            continue
        page_raw = e.get("page_id")
        try:
            page_id = int(page_raw) if page_raw is not None else 0
        except (TypeError, ValueError):
            page_id = 0
        items.append(
            RouteTraceItem(
                chunk_id=str(e.get("chunk_id") or ""),
                page_id=page_id,
                score=float(e.get("score") or 0.0),
                text=(e.get("text") or "")[:200] if e.get("text") else None,
                doc_id=e.get("doc_id"),
            )
        )
    return RetrievalTrace(dense_top5=items)


def _agent_info_from_result(result, acfg: dict) -> AgentInfo:
    return AgentInfo(
        used=True,
        status=result.status,
        subqueries=list(result.subqueries or []),
        trajectory=_agent_step_infos(
            result.trajectory, return_trajectory=bool(acfg.get("return_trajectory", True))
        ),
        counts=dict(result.counts or {}),
        degraded_to_pipeline=(result.status == "degraded"),
        thread_id=result.thread_id,
    )


def _citations_safe(raw: Optional[List[dict]]) -> List[Citation]:
    out: List[Citation] = []
    for c in raw or []:
        if not isinstance(c, dict):
            continue
        try:
            out.append(
                Citation(
                    chunk_id=str(c.get("chunk_id") or ""),
                    page_id=int(c.get("page_id") or 0),
                    doc_id=c.get("doc_id"),
                    page_number=c.get("page_number"),
                    snippet=str(c.get("snippet") or (c.get("text") or "")[:200]),
                )
            )
        except Exception:
            continue
    return out


def _current_trace_id() -> str:
    from src.observability import get_tracer

    tr = get_tracer().current_trace()
    if tr is not None and getattr(tr, "trace_id", None):
        return str(tr.trace_id)
    return uuid.uuid4().hex[:16]


def _build_agent_search_fn(retriever, *, k: int, use_rerank: bool, doc_id: Optional[str], use_visual: bool):
    def search_fn(q, k_local=None):
        kk = int(k_local) if k_local is not None else k
        search_k = max(kk * 10, 50) if doc_id else kk
        hits = retriever.search(
            q,
            k=search_k,
            use_visual=use_visual,
            use_rerank=use_rerank,
        )
        if doc_id:
            hits = [r for r in hits if r.get("doc_id") == doc_id][:kk]
        return hits

    return search_fn


def _pipeline_fallback_fn(retriever, gen, *, query: str, k: int, use_rerank: bool, doc_id: Optional[str], use_visual: bool):
    def pipeline_fallback():
        search_k = max(k * 10, 50) if doc_id else k
        hits = retriever.search(
            query,
            k=search_k,
            use_visual=use_visual,
            use_rerank=use_rerank,
        )
        if doc_id:
            hits = [r for r in hits if r.get("doc_id") == doc_id][:k]
        out = gen.answer(query, hits, k_context=k)
        return {
            "answer": out.get("answer") or "",
            "citations": out.get("citations") or [],
            "context": out.get("context") or "",
        }

    return pipeline_fallback


@app.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest):
    retriever = get_retriever()
    use_visual = cfg.get("retrieval.use_visual", True)
    collector = get_collector()
    # L4 Answer 缓存的 config_label 与 L3 在线路径保持一致（空串：在线请求无 per-config 拆分）
    cache_label = ""
    req_mode = (request.mode or "pipeline").strip().lower()

    # ── L4 Answer Cache ───────────────────────────────────────
    gen = get_generator(retriever.bge)
    answer_key = None
    cached_answer = None
    if cfg.cache.enabled and gen.cacheable:
        if retriever._answer_cache is None:
            retriever._answer_cache = InMemoryLRUCache(max_size=cfg.cache.max_size)
        answer_key = retriever.answer_cache_key(
            request.query, gen.model, request.k, request.doc_id, mode=req_mode,
        )
        cached_answer = retriever._answer_cache.get(answer_key)
    if cached_answer is not None:
        collector.record_cache_event("answer", hit=True, config_label=cache_label)
        rt = cached_answer["retrieval_trace"]
        trace = RetrievalTrace(
            bm25_top5=[RouteTraceItem(**t) for t in rt.get("bm25_top5", [])],
            dense_top5=[RouteTraceItem(**t) for t in rt.get("dense_top5", [])],
            visual_top5=[RouteTraceItem(**t) for t in rt.get("visual_top5", [])],
        )
        sr_info = cached_answer.get("self_rag")
        crag_info = cached_answer.get("crag")
        agent_info = cached_answer.get("agent")
        return AskResponse(
            query=request.query, answer=cached_answer["answer"],
            citations=[Citation(**c) for c in cached_answer["citations"]],
            retrieval_trace=trace,
            self_rag=SelfRAGInfo(**sr_info) if sr_info else None,
            crag=CRAGInfo(**crag_info) if crag_info else None,
            agent=AgentInfo(**agent_info) if agent_info else None,
            context=cached_answer.get("context") or "",
        )

    # ── Agent branch（L4 miss 后、pipeline search 前）──
    from src.agent.config import agent_config

    acfg = agent_config()
    agent_ignored: Optional[AgentInfo] = None
    if req_mode == "agent":
        if not acfg.get("enabled"):
            # 继续 pipeline；响应标注 ignored
            agent_ignored = AgentInfo(
                used=False, ignored_reason="agent.enabled=false",
            )
        else:
            from src.agent.runner import run_agent

            search_fn = _build_agent_search_fn(
                retriever,
                k=request.k,
                use_rerank=request.use_rerank,
                doc_id=request.doc_id,
                use_visual=use_visual,
            )
            complete_fn = _make_complete_fn(gen)

            def generate_fn(q, hits, k_context=None):
                kk = int(k_context) if k_context is not None else int(request.k)
                return gen.answer(q, hits, k_context=kk)

            pipeline_fallback = _pipeline_fallback_fn(
                retriever,
                gen,
                query=request.query,
                k=request.k,
                use_rerank=request.use_rerank,
                doc_id=request.doc_id,
                use_visual=use_visual,
            )
            trace_id = _current_trace_id()
            try:
                result = run_agent(
                    request.query,
                    search_fn=search_fn,
                    complete_fn=complete_fn,
                    generate_fn=generate_fn,
                    pipeline_fallback_fn=pipeline_fallback,
                    trace_id=trace_id,
                )
            except Exception as e:
                logger.exception("agent path failed: %s", e)
                raise HTTPException(status_code=500, detail=f"Agent error: {e}")

            agent_public = _agent_info_from_result(result, acfg)
            # interrupted: empty answer ok (HITL pause)
            citations = _citations_safe(result.citations)
            rt_model = _trace_from_evidence(result.evidence)
            rt_dict = {
                "bm25_top5": [t.model_dump() for t in rt_model.bm25_top5],
                "dense_top5": [t.model_dump() for t in rt_model.dense_top5],
                "visual_top5": [t.model_dump() for t in rt_model.visual_top5],
            }
            context_out = result.context or ""
            # interrupted 不写 L4（答案未完成）；degraded/ok 可缓存
            if (
                result.status != "interrupted"
                and cfg.cache.enabled
                and gen.cacheable
                and answer_key is not None
                and retriever._answer_cache is not None
            ):
                retriever._answer_cache.put(answer_key, {
                    "answer": result.answer,
                    "citations": [c.model_dump() for c in citations],
                    "retrieval_trace": rt_dict,
                    "self_rag": None,
                    "crag": None,
                    "agent": agent_public.model_dump(),
                    "context": context_out,
                })
                collector.record_cache_event("answer", hit=False, config_label=cache_label)

            return AskResponse(
                query=request.query,
                answer=result.answer or "",
                citations=citations,
                retrieval_trace=rt_model,
                agent=agent_public,
                context=context_out,
            )

    # doc_id 过滤在检索后做：若只取 top-k，新上传小文档容易被大库挤掉。
    # 有 doc_id 时先多取再过滤，保证「上传后立刻问本篇」可用。
    search_k = request.k
    if request.doc_id:
        search_k = max(request.k * 10, 50)
    try:
        search_result = retriever.search_with_trace(
            request.query, k=search_k,
            use_visual=use_visual, use_rerank=request.use_rerank,
        )
    except Exception as e:
        logger.error(f"search error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal search error: {e}")
    results = search_result["results"]
    retrieval_trace = search_result["retrieval_trace"]
    if request.doc_id:
        results = [r for r in results if r.get("doc_id") == request.doc_id][: request.k]
        # RouteTraceItem 无 doc_id 字段，仅截断 top5 展示
        retrieval_trace = {
            "bm25_top5": retrieval_trace.get("bm25_top5", [])[:5],
            "dense_top5": retrieval_trace.get("dense_top5", [])[:5],
            "visual_top5": retrieval_trace.get("visual_top5", [])[:5],
        }

    # ── CRAG：检索后 grade / 可选改写再检索（默认关）──
    crag_public: dict = {"enabled": False, "applied": False}
    cr_cfg = crag_config()
    if cr_cfg.get("enabled") and results:
        def _research(q: str):
            rk = max(request.k * 10, 50) if request.doc_id else request.k
            second = retriever.search(
                q,
                k=rk,
                use_visual=use_visual,
                use_rerank=request.use_rerank,
            )
            if request.doc_id:
                second = [r for r in second if r.get("doc_id") == request.doc_id][: request.k]
            return second

        crag = CorrectiveRAG(
            search_fn=_research,
            client=gen.client,
            model=cr_cfg.get("judge_model") or gen.model,
            config=cr_cfg,
        )
        corrected = crag.correct(request.query, results, k=request.k)
        results = corrected.get("results") or results
        crag_payload = corrected.get("crag") or {}
        crag_public = {
            k: crag_payload.get(k)
            for k in (
                "enabled",
                "applied",
                "final_action",
                "query_original",
                "query_used",
                "num_relevant",
                "sufficient",
                "skip_reason",
                "grade_degraded",
            )
            if k in crag_payload or k == "enabled"
        }
        crag_public.setdefault("enabled", True)
        crag_public.setdefault("query_original", request.query)
        if "query_used" not in crag_public:
            crag_public["query_used"] = corrected.get("query_used", request.query)

    trace = RetrievalTrace(
        bm25_top5=[RouteTraceItem(**t) for t in retrieval_trace["bm25_top5"]],
        dense_top5=[RouteTraceItem(**t) for t in retrieval_trace["dense_top5"]],
        visual_top5=[RouteTraceItem(**t) for t in retrieval_trace["visual_top5"]],
    )
    if not results:
        return AskResponse(
            query=request.query,
            answer=abstain_message(),
            citations=[], retrieval_trace=trace,
            crag=CRAGInfo(**crag_public),
            agent=agent_ignored,
            context="",
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

    context_out = gen_out.get("context") or ""
    agent_dump = agent_ignored.model_dump() if agent_ignored is not None else None

    # 写入 L4 Answer 缓存（受全局开关 + 确定性守卫；命中率经 cache_label 聚合）
    if cfg.cache.enabled and gen.cacheable and answer_key is not None and retriever._answer_cache is not None:
        retriever._answer_cache.put(answer_key, {
            "answer": gen_out["answer"],
            "citations": gen_out["citations"],
            "retrieval_trace": retrieval_trace,
            "self_rag": sr_public,
            "crag": crag_public,
            "agent": agent_dump,
            "context": context_out,
        })
        collector.record_cache_event("answer", hit=False, config_label=cache_label)

    return AskResponse(
        query=request.query, answer=gen_out["answer"],
        citations=[Citation(**c) for c in gen_out["citations"]],
        retrieval_trace=trace,
        self_rag=SelfRAGInfo(**sr_public),
        crag=CRAGInfo(**crag_public),
        agent=agent_ignored,
        context=context_out,
    )


@app.post("/ask/resume", response_model=AskResponse)
async def ask_resume(request: AskResumeRequest):
    """Resume HITL-interrupted agent run (requires agent.enabled + checkpoint)."""
    from src.agent.config import agent_config
    from src.agent.runner import resume_agent

    acfg = agent_config()
    if not acfg.get("enabled"):
        raise HTTPException(status_code=400, detail="agent.enabled=false; cannot resume")
    ck = acfg.get("checkpoint") if isinstance(acfg.get("checkpoint"), dict) else {}
    if not ck.get("enabled", True):
        # default checkpoint.enabled is True in agent_config; explicit false → 400
        raise HTTPException(
            status_code=400,
            detail="agent.checkpoint.enabled=false; resume unavailable",
        )
    try:
        from src.agent.checkpoint import get_memory_saver

        get_memory_saver()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"checkpoint unavailable: {e}")

    retriever = get_retriever()
    gen = get_generator(retriever.bge)
    use_visual = cfg.get("retrieval.use_visual", True)

    search_fn = _build_agent_search_fn(
        retriever,
        k=request.k,
        use_rerank=request.use_rerank,
        doc_id=request.doc_id,
        use_visual=use_visual,
    )
    try:
        complete_fn = _make_complete_fn(gen)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"complete_fn unavailable: {e}")

    def generate_fn(q, hits, k_context=None):
        kk = int(k_context) if k_context is not None else int(request.k)
        return gen.answer(q, hits, k_context=kk)

    # None → [] so graph falls back to interrupted subqueries (approve as-is)
    approved = list(request.subqueries) if request.subqueries is not None else []

    pipeline_fallback = _pipeline_fallback_fn(
        retriever,
        gen,
        query="",  # resume path uses checkpointed query for generate; fallback is last resort
        k=request.k,
        use_rerank=request.use_rerank,
        doc_id=request.doc_id,
        use_visual=use_visual,
    )

    try:
        result = resume_agent(
            thread_id=request.thread_id,
            approved_subqueries=approved,
            search_fn=search_fn,
            complete_fn=complete_fn,
            generate_fn=generate_fn,
            pipeline_fallback_fn=pipeline_fallback,
        )
    except Exception as e:
        logger.exception("ask/resume failed: %s", e)
        raise HTTPException(status_code=400, detail=f"resume failed: {e}")

    agent_public = _agent_info_from_result(result, acfg)
    citations = _citations_safe(result.citations)
    rt_model = _trace_from_evidence(result.evidence)
    return AskResponse(
        query="",  # query lives in checkpoint; not re-sent on resume body
        answer=result.answer or "",
        citations=citations,
        retrieval_trace=rt_model,
        agent=agent_public,
        context=result.context or "",
    )


def _demo_static_dir() -> Path:
    """static/demo：优先仓库根（与 scripts/run_api.py cwd 一致），否则相对本文件回溯。"""
    candidates = [
        Path.cwd() / "static" / "demo",
        Path(__file__).resolve().parents[2] / "static" / "demo",
    ]
    for p in candidates:
        if p.is_dir():
            return p
    return candidates[0]


_DEMO_DIR = _demo_static_dir()
if _DEMO_DIR.is_dir():
    app.mount(
        "/demo",
        StaticFiles(directory=str(_DEMO_DIR), html=True),
        name="demo",
    )
else:
    logger.warning("Demo static dir missing: %s", _DEMO_DIR)