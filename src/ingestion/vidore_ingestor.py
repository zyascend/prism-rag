"""ViDoRe 数据集导入管道 — 支持断点续传

数据流：
  HF Dataset (image + markdown)
    ├─ TextChunker → BGE encode → pgvector (幂等: ON CONFLICT DO NOTHING)
    └─ ColPali encode → FAISS index (可恢复: 跳过已完成的页面)

用法:
  python -m scripts.ingest_vidore --dataset vidore/vidore_v3_industrial      # 首次
  python -m scripts.ingest_vidore --dataset vidore/vidore_v3_industrial      # 恢复（自动检测进度）
  python -m scripts.ingest_vidore --dataset vidore/vidore_v3_industrial --resume  # 显式恢复
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
        """执行导入流程

        Args:
            dataset_path: HF 数据集路径
            max_pages: 限制处理的页数（调试用）
            skip_faiss: 跳过视觉路（仅文本路）
            resume: 是否从上次中断处恢复
        """
        logger.info(f"加载数据集: {dataset_path}")
        ds = load_dataset(dataset_path, "corpus", split="test")

        if max_pages:
            ds = ds.select(range(min(max_pages, len(ds))))

        total_pages = len(ds)
        logger.info(f"共 {total_pages} 页")

        # 恢复模式：加载已有进度
        state = load_state() if resume else {"text_phase_done": False, "completed_page_ids": []}
        completed_set: Set[int] = set(state.get("completed_page_ids", []))
        text_phase_done = state.get("text_phase_done", False)

        if resume and text_phase_done:
            logger.info(f"🔄 恢复模式: 文本路已跳过, 视觉路 {len(completed_set)}/{total_pages} 页已完成")

        # 1. 文本路：分块 → BGE → pgvector
        if not (resume and text_phase_done):
            self._ingest_text(ds)
            save_state(text_phase_done=True, completed_page_ids=completed_set)
        else:
            logger.info("⏭️  文本路已就绪，跳过")

        # 2. 视觉路：ColPali 编码 → FAISS
        if not skip_faiss:
            if resume and len(completed_set) > 0:
                logger.info(f"🔄 从 {len(completed_set)} 页已有编码继续...")
            self._ingest_visual(ds, resume=resume, completed_set=completed_set)

        logger.info("导入完成")

    def _ingest_text(self, ds):
        """文本路导入（幂等：ON CONFLICT DO NOTHING）"""
        logger.info("=== [1/2] 文本路: 分块 + BGE 编码 + pgvector 入库 ===")
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
                    None,
                ))
                all_texts.append(chunk.text)

        logger.info(f"共 {len(all_chunk_rows)} 个 chunk，开始 BGE 编码...")

        bge_embs = self.bge.encode(all_texts, batch_size=32, show_progress=True)

        batch_size = 100
        for i in tqdm(range(0, len(all_chunk_rows), batch_size), desc="pgvector 入库"):
            batch = []
            for j in range(i, min(i + batch_size, len(all_chunk_rows))):
                chunk_id, page_id, doc_id, pn, ctype, text, _ = all_chunk_rows[j]
                vec = bge_embs[j].cpu().numpy().tolist()
                batch.append((chunk_id, page_id, doc_id, pn, ctype, text, vec))
            self.pg.insert_chunks(batch)

        logger.info(f"✅ pgvector 入库完成, 共 {self.pg.count()} 条 chunk")

    def _ingest_visual(
        self,
        ds,
        resume: bool = False,
        completed_set: Optional[Set[int]] = None,
    ):
        """视觉路导入（支持断点续传）

        策略：
          - 每次编码一批就增量保存到缓存文件
          - FAISS 索引从全部缓存构建（幂等）
          - 中断后恢复只需重跑此方法，跳过已完成页
        """
        logger.info("=== [2/2] 视觉路: ColPali 编码 + FAISS 建索引 ===")
        total_pages = len(ds)

        # 恢复模式：加载已缓存的嵌入
        cached_embeddings: Dict[int, torch.Tensor] = {}
        pages_to_encode: List[int] = []

        if resume:
            cached_embeddings = load_page_embeddings()
            already_done = set(cached_embeddings.keys())
            if completed_set:
                already_done |= completed_set

            for idx in range(total_pages):
                page_id = int(ds[idx]["corpus_id"])
                if page_id in already_done:
                    continue
                pages_to_encode.append(idx)

            logger.info(f"  📊 已有 {len(already_done)} 页编码缓存, 还需编码 {len(pages_to_encode)} 页")
        else:
            clear_progress()
            pages_to_encode = list(range(total_pages))
            logger.info(f"  📊 全新编码 {len(pages_to_encode)} 页")

        if not pages_to_encode:
            logger.info("✅ 所有页面已编码，跳过 ColPali")
            self._build_faiss_from_all(cached_embeddings, total_pages)
            return

        # 批量 ColPali 编码（跳过已完成的页）
        batch_size = 4
        new_embeddings: Dict[int, torch.Tensor] = {}
        new_count = 0

        # 总批次 = 按 pages_to_encode 划分的批次数量
        total_batches = (len(pages_to_encode) + batch_size - 1) // batch_size

        for batch_i in tqdm(range(0, len(pages_to_encode), batch_size), desc=f"ColPali 编码"):
            batch_indices = pages_to_encode[batch_i : batch_i + batch_size]
            batch_rows = [ds[idx] for idx in batch_indices]
            batch_images = [row["image"] for row in batch_rows]
            batch_page_ids = [int(row["corpus_id"]) for row in batch_rows]

            embs = self.colpali.encode_pages(batch_images, batch_size=len(batch_images))

            for pid, emb in zip(batch_page_ids, embs):
                new_embeddings[pid] = emb
                new_count += 1

            # 增量保存到缓存
            save_page_embeddings(new_embeddings)

            # 更新进度
            updated_completed = set(cached_embeddings.keys()) | set(new_embeddings.keys())
            save_state(text_phase_done=True, completed_page_ids=updated_completed)

        logger.info(f"✅ ColPali 编码完成: {new_count} 页新编码")

        # 从全部缓存构建 FAISS 索引
        all_embeddings = {**cached_embeddings, **new_embeddings}
        self._build_faiss_from_all(all_embeddings, total_pages)

    def _build_faiss_from_all(self, all_embeddings: Dict[int, torch.Tensor], expected_pages: int):
        """从全部页面嵌入构建 FAISS 索引"""
        if len(all_embeddings) < expected_pages:
            logger.warning(f"⚠️  仅 {len(all_embeddings)}/{expected_pages} 页可用, 其余可能因错误跳过")
        logger.info(f"FAISS 建索引: {len(all_embeddings)} 页...")
        self.faiss.build_index(all_embeddings)
        self.faiss.save()
        logger.info(f"✅ FAISS 索引完成: {self.faiss.num_pages} 页, {self.faiss.index_size_mb:.1f} MB")
