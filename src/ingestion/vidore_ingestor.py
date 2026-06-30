"""ViDoRe 数据集导入管道

数据流：
  HF Dataset (image + markdown)
    ├─ TextChunker → BGE encode → pgvector
    └─ ColPali encode → FAISS index
"""

from __future__ import annotations

import logging
from typing import List, Optional

import torch
from datasets import load_dataset
from tqdm import tqdm

from src.config import cfg
from src.ingestion.encoders import BGEEmbedder, ColPaliEmbedder
from src.ingestion.text_chunker import Chunk, TextChunker
from src.store.faiss_store import FaissColPaliStore
from src.store.pgvector_store import PgVectorStore

logger = logging.getLogger(__name__)


class ViDoReIngestor:
    """ViDoRe 数据集导入器"""

    def __init__(
        self,
        pg_store: PgVectorStore,
        faiss_store: FaissColPaliStore,
        bge_embedder: BGEEmbedder,
        colpali_embedder: ColPaliEmbedder,
        chunker: TextChunker,
    ):
        self.pg = pg_store
        self.faiss = faiss_store
        self.bge = bge_embedder
        self.colpali = colpali_embedder
        self.chunker = chunker

    def ingest(
        self,
        dataset_path: str = "vidore/vidore_v3_industrial",
        max_pages: Optional[int] = None,
        skip_faiss: bool = False,
    ):
        """执行导入流程"""
        logger.info(f"加载数据集: {dataset_path}")
        ds = load_dataset(dataset_path, "corpus", split="test")

        if max_pages:
            ds = ds.select(range(min(max_pages, len(ds))))

        total_pages = len(ds)
        logger.info(f"共 {total_pages} 页")

        # 1. 文本路：分块 → BGE → pgvector
        self._ingest_text(ds)

        # 2. 视觉路：ColPali 编码 → FAISS
        if not skip_faiss:
            self._ingest_visual(ds)

        logger.info("导入完成")

    def _ingest_text(self, ds):
        """文本路导入"""
        logger.info("=== 文本路: 分块 + BGE 编码 + pgvector 入库 ===")
        self.pg.create_schema()

        all_chunk_rows: List[tuple] = []
        all_texts: List[str] = []

        for idx in tqdm(range(len(ds)), desc="分块"):
            row = ds[idx]
            page_id = int(row["corpus_id"])
            doc_id = str(row.get("doc_id", ""))
            page_number = int(row.get("page_number_in_doc", 0))
            markdown = row.get("markdown", None)

            chunks: List[Chunk] = self.chunker.chunk_page(
                page_id=page_id,
                doc_id=doc_id,
                page_number=page_number,
                markdown_text=markdown,
            )

            for chunk in chunks:
                all_chunk_rows.append((
                    chunk.chunk_id,
                    chunk.page_id,
                    chunk.doc_id,
                    chunk.page_number,
                    chunk.chunk_type,
                    chunk.text,
                    None,  # bge_vector placeholder
                ))
                all_texts.append(chunk.text)

        logger.info(f"共 {len(all_chunk_rows)} 个 chunk，开始 BGE 编码...")

        # BGE 批量编码
        bge_embs = self.bge.encode(all_texts, batch_size=32, show_progress=True)

        # 填充向量并写入 pgvector
        batch_size = 100
        for i in range(0, len(all_chunk_rows), batch_size):
            batch = []
            for j in range(i, min(i + batch_size, len(all_chunk_rows))):
                chunk_id, page_id, doc_id, pn, ctype, text, _ = all_chunk_rows[j]
                vec = bge_embs[j].cpu().numpy().tolist()
                batch.append((chunk_id, page_id, doc_id, pn, ctype, text, vec))
            self.pg.insert_chunks(batch)

        logger.info(f"pgvector 入库完成, 共 {self.pg.count()} 条 chunk")

    def _ingest_visual(self, ds):
        """视觉路导入"""
        logger.info("=== 视觉路: ColPali 编码 + FAISS 建索引 ===")

        page_embeddings: dict = {}

        # 批次编码
        batch_size = 4
        for i in tqdm(range(0, len(ds), batch_size), desc="ColPali 编码"):
            batch_rows = [ds[j] for j in range(i, min(i + batch_size, len(ds)))]
            images = [row["image"] for row in batch_rows]
            page_ids = [int(row["corpus_id"]) for row in batch_rows]

            embs = self.colpali.encode_pages(images, batch_size=len(images))
            for pid, emb in zip(page_ids, embs):
                page_embeddings[pid] = emb

        logger.info(f"ColPali 编码完成, 共 {len(page_embeddings)} 页")

        # FAISS 建索引
        self.faiss.build_index(page_embeddings)
        self.faiss.save()

        logger.info(f"FAISS 索引完成, 共 {self.faiss.num_pages} 页, {self.faiss.index_size_mb:.1f} MB")