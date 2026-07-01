"""ViDoRe 数据集导入管道 — 支持断点续传

数据流：
  HF Dataset (image + markdown)
    ├─ TextChunker → BGE encode → pgvector (幂等: ON CONFLICT DO NOTHING)
    └─ ColPali encode → FAISS index (可恢复: 跳过已完成的页面)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Set

import torch
from datasets import load_dataset
from tqdm import tqdm

from src.config import cfg
from src.ingestion.encoders import BGEEmbedder, ColPaliEmbedder
from src.ingestion.progress import (
    clear_progress,
    load_page_embeddings,
    load_state,
    save_page_embeddings,
    save_state,
)
from src.ingestion.text_chunker import Chunk, TextChunker
from src.store.faiss_store import FaissColPaliStore
from src.store.pgvector_store import PgVectorStore

logger = logging.getLogger(__name__)


class ViDoReIngestor:
    """ViDoRe 数据集导入器（支持断点续传）"""

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
        resume: bool = False,
    ):
        logger.info(f"加载数据集: {dataset_path}")
        ds = load_dataset(dataset_path, "corpus", split="test")
        if max_pages:
            ds = ds.select(range(min(max_pages, len(ds))))
        total_pages = len(ds)
        logger.info(f"共 {total_pages} 页")

        state = load_state() if resume else {"text_phase_done": False, "completed_count": 0}
        text_phase_done = state.get("text_phase_done", False)
        completed_count = state.get("completed_count", 0)

        if resume and text_phase_done:
            logger.info(f"🔄 恢复模式: 文本路已跳过, 视觉路 {completed_count}/{total_pages} 页已完成")

        # 1. 文本路
        if not (resume and text_phase_done):
            self._ingest_text(ds)
            save_state(text_phase_done=True, completed_count=0)
        else:
            logger.info("⏭️  文本路已就绪，跳过")

        # 2. 视觉路
        if not skip_faiss:
            if resume and completed_count > 0:
                logger.info(f"🔄 从 {completed_count} 页已有编码继续...")
            self._ingest_visual(ds, total_pages, resume=resume)

        logger.info("✅ 导入完成")

    def _ingest_text(self, ds):
        logger.info("=== [1/2] 文本路: 分块 + BGE 编码 + pgvector 入库 ===")
        self.pg.create_schema()

        all_chunk_rows: List[tuple] = []
        all_texts: List[str] = []

        for idx in tqdm(range(len(ds)), desc="分块"):
            row = ds[idx]
            chunks = self.chunker.chunk_page(
                page_id=int(row["corpus_id"]),
                doc_id=str(row.get("doc_id", "")),
                page_number=int(row.get("page_number_in_doc", 0)),
                markdown_text=row.get("markdown", None),
            )
            for c in chunks:
                all_chunk_rows.append((c.chunk_id, c.page_id, c.doc_id, c.page_number, c.chunk_type, c.text, None))
                all_texts.append(c.text)

        logger.info(f"共 {len(all_chunk_rows)} 个 chunk，开始 BGE 编码...")
        bge_embs = self.bge.encode(all_texts, batch_size=32, show_progress=True)

        for i in tqdm(range(0, len(all_chunk_rows), 100), desc="pgvector 入库"):
            batch = []
            for j in range(i, min(i + 100, len(all_chunk_rows))):
                vec = bge_embs[j].cpu().numpy().tolist()
                entry = all_chunk_rows[j]
                batch.append((entry[0], entry[1], entry[2], entry[3], entry[4], entry[5], vec))
            self.pg.insert_chunks(batch)

        logger.info(f"✅ pgvector 入库完成, 共 {self.pg.count()} 条 chunk")

    def _ingest_visual(self, ds, total_pages: int, resume: bool = False):
        logger.info("=== [2/2] 视觉路: ColPali 编码 + FAISS 建索引 ===")

        # 恢复模式：加载已有缓存
        cached_page_ids: Set[int] = set()
        if resume:
            cached = load_page_embeddings()
            cached_page_ids = set(cached.keys())
            logger.info(f"  📊 已有 {len(cached_page_ids)} 页编码缓存")
        else:
            clear_progress()

        # 找出需要编码的页
        pages_to_encode: List[int] = []
        for idx in range(total_pages):
            page_id = int(ds[idx]["corpus_id"])
            if page_id not in cached_page_ids:
                pages_to_encode.append(idx)

        logger.info(f"  📊 还需编码 {len(pages_to_encode)} 页")

        if not pages_to_encode:
            logger.info("✅ 所有页面已编码，构建 FAISS...")
            all_embeddings = load_page_embeddings()
            self._build_faiss(all_embeddings, total_pages)
            return

        # 分批编码
        batch_size = 4
        new_embeddings: Dict[int, torch.Tensor] = {}
        completed_count = len(cached_page_ids)

        for batch_i in tqdm(range(0, len(pages_to_encode), batch_size), desc="ColPali 编码"):
            batch_indices = pages_to_encode[batch_i : batch_i + batch_size]
            batch_rows = [ds[idx] for idx in batch_indices]
            batch_page_ids = [int(row["corpus_id"]) for row in batch_rows]

            embs = self.colpali.encode_pages([r["image"] for r in batch_rows], batch_size=len(batch_rows))

            batch_embeddings: Dict[int, torch.Tensor] = {}
            for pid, emb in zip(batch_page_ids, embs):
                batch_embeddings[pid] = emb
                new_embeddings[pid] = emb
                completed_count += 1

            # 每 50 批保存一次（避免频繁 I/O）
            save_page_embeddings(batch_embeddings)
            if (batch_i // batch_size) % 50 == 0:
                save_state(text_phase_done=True, completed_count=completed_count)

        save_state(text_phase_done=True, completed_count=completed_count)
        logger.info(f"✅ ColPali 编码完成: {completed_count} 页")

        # 构建 FAISS
        all_embeddings = load_page_embeddings()
        self._build_faiss(all_embeddings, total_pages)

    def _build_faiss(self, all_embeddings: Dict[int, torch.Tensor], total_pages: int):
        if len(all_embeddings) < total_pages:
            logger.warning(f"⚠️  仅 {len(all_embeddings)}/{total_pages} 页可用")
        logger.info(f"FAISS 建索引: {len(all_embeddings)} 页...")
        index_type = cfg.get("storage.faiss.index_type", "flat")
        hnsw_m = cfg.get("storage.faiss.hnsw_m", 32)
        self.faiss.build_index(all_embeddings, index_type=index_type, hnsw_m=hnsw_m)
        self.faiss.save()
        logger.info(f"✅ FAISS 索引完成: {self.faiss.num_pages} 页, {self.faiss.index_size_mb:.1f} MB")
