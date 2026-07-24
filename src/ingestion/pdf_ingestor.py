"""真实 PDF 导入管道：Parser → 分块 → BGE→pgvector + ColPali→FAISS 增量

P2 改动（对应 Spec §4.2 / §4.3）：
- 注入可选 `bm25`，ingest 时增量维护 BM25（消除 U1：不再每次全量重建）。
- 新增 page 级 `page_hash`：重入库时按 page_number 对齐哈希，
  未变页直接复用三路（跳过 ColQwen2 重编码，省 GPU），仅变化/新增页重编码，
  删除页三路清理。这是「部分修改」场景最大省钱点。
- 同内容重入库 → 幂等 no-op（不再整篇重编码）。
"""
from __future__ import annotations
import hashlib
import logging
import random
from pathlib import Path
from typing import Callable, Dict, List, Optional

from src.config import cfg
from src.ingestion.parser import Page, build_parser
from src.ingestion.table_summarizer import TableSummarizer

logger = logging.getLogger(__name__)


class PDFIngestor:
    def __init__(
        self,
        pg,
        faiss,
        bge,
        colpali,
        chunker,
        parser=None,
        delete_fn: Optional[Callable[[str], dict]] = None,
        bm25=None,
    ):
        self.pg = pg
        self.faiss = faiss
        self.bge = bge
        self.colpali = colpali
        self.chunker = chunker
        self.parser = parser or build_parser()
        # 幂等删除回调（可选）：保留以兼容旧调用方；P2 已在 ingestor 内联三路删除。
        self.delete_fn: Optional[Callable[[str], dict]] = delete_fn
        # P2-A：可选 BM25 引用，ingest 时增量维护（消除 U1 全量重建）
        self.bm25 = bm25
        self.summarizer = TableSummarizer(
            enabled=cfg.get("ingestion.table_summary_enabled", True),
            context_enabled=cfg.get("ingestion.table_summary_context_enabled", False),
            context_max_chars=cfg.get("ingestion.table_summary_context_max_chars", 1500),
        )
        # progress_fn(phase: str, pct: float 0-100, message: str) — 可选，供 demo 实时追踪
        self.progress_fn: Optional[Callable[[str, float, str], None]] = None

    def set_progress_fn(self, fn: Optional[Callable[[str, float, str], None]]) -> None:
        self.progress_fn = fn

    def _progress(self, phase: str, pct: float, message: str) -> None:
        if self.progress_fn is None:
            return
        try:
            self.progress_fn(phase, float(pct), message)
        except Exception as e:
            logger.debug("progress_fn error: %s", e)

    def ingest(self, pdf_path: Path, doc_id: Optional[str] = None) -> Dict:
        pdf_path = Path(pdf_path)
        self._progress("hash", 2, "计算内容哈希…")
        content_hash = hashlib.sha256(pdf_path.read_bytes()).hexdigest()

        # 空库/首次本地 demo：必须先建表，再查 content_hash（否则 documents 不存在 → 500）
        self._progress("schema", 5, "确保数据库 schema…")
        self.pg.create_schema()

        # P1：同内容已入库 → 幂等 no-op（不再整篇重编码，避免浪费 GPU）
        existing = self.pg.get_doc_id_by_content_hash(content_hash)
        if existing is not None:
            logger.info("内容哈希命中，幂等跳过 doc_id=%s", existing)
            self._progress("done", 100, f"内容已存在，跳过编码 doc_id={existing}")
            return {"doc_id": existing, "num_pages": 0, "num_chunks": 0, "status": "noop_identical"}

        doc_id = doc_id or _rand_doc_id()

        # P2-B：同 doc_id 修改版 → page diff UPDATE 路径（省 GPU）
        if doc_id and self.pg.document_exists(doc_id):
            return self._ingest_update(pdf_path, doc_id, content_hash)
        return self._ingest_fresh(pdf_path, doc_id, content_hash)

    # ── 全新增 ──────────────────────────────────────────────

    def _ingest_fresh(self, pdf_path: Path, doc_id: str, content_hash: str) -> Dict:
        self._progress("parse", 10, f"解析 PDF：{pdf_path.name}")
        pages = self.parser.parse(pdf_path)
        if not pages:
            raise RuntimeError("解析未产出任何页面")
        self._progress("parse", 20, f"解析完成 · {len(pages)} 页")
        hash_by_pn = {p.page_number: self._page_hash_of(p) for p in pages}
        num_chunks = self._ingest_pages(pages, doc_id, [p.page_number for p in pages], hash_by_pn)
        self._progress("index", 95, "写入 documents 元数据…")
        self.pg.upsert_document(doc_id, content_hash, str(pdf_path))
        self._progress("done", 100, f"入库完成 · {len(pages)} 页 · {num_chunks} chunks")
        return {"doc_id": doc_id, "num_pages": len(pages), "num_chunks": num_chunks, "status": "inserted"}

    # ── 修改版更新（page diff，省 GPU）──────────────────────

    def _ingest_update(self, pdf_path: Path, doc_id: str, content_hash: str) -> Dict:
        self._progress("parse", 10, f"解析 PDF（更新）· {pdf_path.name}")
        pages = self.parser.parse(pdf_path)
        if not pages:
            raise RuntimeError("解析未产出任何页面")
        self._progress("parse", 18, f"解析完成 · {len(pages)} 页 · page-diff")

        old_pages = {pn: pid for (pid, pn) in self.pg.get_pages_by_doc_id(doc_id)}
        old_hashes = self.pg.get_page_hashes_by_doc_id(doc_id)
        new_hash_by_pn = {p.page_number: self._page_hash_of(p) for p in pages}
        new_pn_set = set(new_hash_by_pn)

        unchanged, changed, new_pns, deleted_pns = [], [], [], []
        for p in pages:
            pn = p.page_number
            h = new_hash_by_pn[pn]
            if pn in old_hashes and old_hashes[pn] == h:
                unchanged.append(pn)          # 复用，三路都不动
            elif pn in old_pages:
                changed.append(pn)            # 重编码（新 page_id）
            else:
                new_pns.append(pn)            # 新增页
        for pn in old_pages:
            if pn not in new_pn_set:
                deleted_pns.append(pn)        # 旧页被删

        # 三路删除：changed 的旧页 + 删除页（用旧 page_id）
        remove_page_ids = [old_pages[pn] for pn in (changed + deleted_pns)]
        if remove_page_ids:
            old_chunk_ids = set(self.pg.get_chunk_ids_by_page_ids(remove_page_ids))
            self.pg.delete_chunks_by_page_ids(remove_page_ids)
            self.faiss.delete_by_page_ids(remove_page_ids)
            if self.bm25 is not None and old_chunk_ids:
                self.bm25.remove_chunks(old_chunk_ids)
            self.faiss.maybe_compact()

        # 仅重编码 changed + new 页（跳过 unchanged，省 ColQwen2 GPU）
        reencode_pns = changed + new_pns
        num_chunks = 0
        if reencode_pns:
            num_chunks = self._ingest_pages(pages, doc_id, reencode_pns, new_hash_by_pn)

        self.pg.update_document(doc_id, content_hash, str(pdf_path))
        logger.info(
            "page diff 更新 doc_id=%s：unchanged=%d changed=%d new=%d deleted=%d",
            doc_id, len(unchanged), len(changed), len(new_pns), len(deleted_pns),
        )
        self._progress(
            "done", 100,
            f"更新完成 · unchanged={len(unchanged)} changed={len(changed)} new={len(new_pns)}",
        )
        return {
            "doc_id": doc_id, "num_pages": len(pages), "num_chunks": num_chunks,
            "unchanged": len(unchanged), "changed": len(changed),
            "new": len(new_pns), "deleted": len(deleted_pns), "status": "updated",
        }

    # ── 通用：编码并写入指定页（fresh 全量 / update 的变更子集共用）──

    def _ingest_pages(
        self,
        pages: List[Page],
        doc_id: str,
        page_numbers: List[int],
        hash_by_pn: Dict[int, str],
    ) -> int:
        use_visual = cfg.get("retrieval.use_visual", True)
        pn_set = set(page_numbers)
        all_rows, all_texts = [], []
        page_images, page_id_for_image = [], []
        faiss_page_hashes: Dict[int, str] = {}
        if hasattr(self.chunker, "reset_headings"):
            self.chunker.reset_headings()

        work_pages = [p for p in pages if p.page_number in pn_set]
        n_work = max(len(work_pages), 1)
        self._progress("chunk", 25, f"分块 · 处理 {len(work_pages)} 页…")

        for i, p in enumerate(work_pages):
            page_id = _rand_page_id()
            phash = hash_by_pn[p.page_number]
            if getattr(p, "blocks", None):
                chunks = self.chunker.chunk_blocks(
                    page_id=page_id,
                    doc_id=doc_id,
                    page_number=p.page_number,
                    blocks=p.blocks,
                )
            else:
                chunks = self.chunker.chunk_page(
                    page_id=page_id, doc_id=doc_id,
                    page_number=p.page_number, markdown_text=p.markdown,
                )
            page_ctx = self.summarizer.build_page_context(chunks)
            for c in chunks:
                summary = ""
                embed_text = c.text
                if c.chunk_type == "table":
                    summary = self.summarizer.summarize(c.text, context=page_ctx)
                    if summary:
                        embed_text = summary
                all_rows.append((
                    c.chunk_id, page_id, doc_id, c.page_number,
                    c.chunk_type, c.text, None, c.doc_ref, summary, phash,
                    getattr(c, "section_path", "") or "",
                    getattr(c, "caption", "") or "",
                    getattr(c, "prev_chunk_id", "") or "",
                    getattr(c, "next_chunk_id", "") or "",
                ))
                all_texts.append(embed_text)
            if use_visual:
                page_images.append(p.image)
                page_id_for_image.append(page_id)
                faiss_page_hashes[page_id] = phash
            # 25% → 45% 分块进度
            pct = 25 + 20 * (i + 1) / n_work
            if (i + 1) % max(1, n_work // 10) == 0 or i + 1 == n_work:
                self._progress(
                    "chunk", pct,
                    f"分块中 · 页 {p.page_number} · chunks 累计 {len(all_rows)}",
                )

        if all_texts:
            self._progress("embed_text", 50, f"BGE 编码 {len(all_texts)} 个 chunk…")
            embs = self.bge.encode(all_texts, batch_size=32)
            self._progress("write_pg", 70, "写入 pgvector…")
            for i in range(0, len(all_rows), 100):
                batch = all_rows[i:i + 100]
                vecs = embs[i:i + 100].cpu().numpy().tolist()
                self.pg.insert_chunks([
                    (r[0], r[1], r[2], r[3], r[4], r[5], v, r[7], r[8], r[9],
                     r[10], r[11], r[12], r[13])
                    for r, v in zip(batch, vecs)
                ])
                done = min(i + 100, len(all_rows))
                self._progress(
                    "write_pg", 70 + 10 * done / max(len(all_rows), 1),
                    f"pgvector {done}/{len(all_rows)}",
                )

        if use_visual and page_images:
            self._progress("embed_visual", 82, f"Visual 编码 {len(page_images)} 页…")
            page_embs = self.colpali.encode_pages(page_images)
            page_doc_map = {pid: doc_id for pid in page_id_for_image}
            self.faiss.add_pages(
                {pid: e for pid, e in zip(page_id_for_image, page_embs)},
                page_doc_ids=page_doc_map,
                page_hashes=faiss_page_hashes,
            )
            self.faiss.save()
            self._progress("embed_visual", 90, "FAISS 已保存")
        elif not use_visual:
            self._progress("embed_visual", 90, "跳过 Visual（use_visual=false）")

        # P2-A：BM25 增量维护（仅本次新增/变更页），无需全量重建
        self._progress("bm25", 92, "更新 BM25…")
        if self.bm25 is not None:
            bm25_dicts = [
                {
                    "chunk_id": r[0], "page_id": r[1], "doc_id": r[2],
                    "page_number": r[3], "chunk_type": r[4], "text": r[5],
                }
                for r in all_rows
            ]
            self.bm25.fit_incremental(bm25_dicts)

        return len(all_rows)

    # ── 工具 ────────────────────────────────────────────────

    @staticmethod
    def _page_hash_of(page: Page) -> str:
        """页面内容哈希：优先用渲染图字节（视觉保真），退化用 markdown。"""
        try:
            if page.image is not None:
                return hashlib.sha256(page.image.tobytes()).hexdigest()
        except Exception:
            pass
        return hashlib.sha256((page.markdown or "").encode()).hexdigest()


def _rand_doc_id() -> str:
    return f"doc_{random.getrandbits(31):08x}"


def _rand_page_id() -> int:
    return random.getrandbits(31)
