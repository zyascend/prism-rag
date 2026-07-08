"""真实 PDF 导入管道：Parser → 分块 → BGE→pgvector + ColPali→FAISS 增量"""
from __future__ import annotations
import logging
import random
from pathlib import Path
from typing import Dict, Optional

import torch
from src.config import cfg
from src.ingestion.parser import build_parser

logger = logging.getLogger(__name__)


class PDFIngestor:
    def __init__(self, pg, faiss, bge, colpali, chunker, parser=None):
        self.pg = pg
        self.faiss = faiss
        self.bge = bge
        self.colpali = colpali
        self.chunker = chunker
        self.parser = parser or build_parser()

    def ingest(self, pdf_path: Path, doc_id: Optional[str] = None) -> Dict:
        doc_id = doc_id or _rand_doc_id()
        use_visual = cfg.get("retrieval.use_visual", True)
        pages = self.parser.parse(Path(pdf_path))
        if not pages:
            raise RuntimeError("解析未产出任何页面")

        self.pg.create_schema()
        all_rows, all_texts = [], []
        page_images = []
        page_id_for_image = []
        for p in pages:
            page_id = _rand_page_id()
            chunks = self.chunker.chunk_page(
                page_id=page_id, doc_id=doc_id,
                page_number=p.page_number, markdown_text=p.markdown,
            )
            for c in chunks:
                all_rows.append((c.chunk_id, c.page_id, c.doc_id, c.page_number,
                                 c.chunk_type, c.text, None, c.doc_ref))
                all_texts.append(c.text)
            if use_visual:
                page_images.append(p.image)
                page_id_for_image.append(page_id)

        if all_texts:
            embs = self.bge.encode(all_texts, batch_size=32)
            for i in range(0, len(all_rows), 100):
                batch = all_rows[i:i + 100]
                vecs = embs[i:i + 100].cpu().numpy().tolist()
                self.pg.insert_chunks([
                    (r[0], r[1], r[2], r[3], r[4], r[5], v, r[7]) for r, v in zip(batch, vecs)
                ])

        if use_visual and page_images:
            page_embs = self.colpali.encode_pages(page_images)
            self.faiss.add_pages({pid: e for pid, e in zip(page_id_for_image, page_embs)})
            self.faiss.save()

        return {"doc_id": doc_id, "num_pages": len(pages), "num_chunks": len(all_rows)}


def _rand_doc_id() -> str:
    return f"doc_{random.getrandbits(31):08x}"


def _rand_page_id() -> int:
    return random.getrandbits(31)
