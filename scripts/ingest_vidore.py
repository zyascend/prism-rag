#!/usr/bin/env python
"""ViDoRe 数据导入入口脚本

Text re-ingest（只重灌 pg 文本侧，保留 FAISS）示例::

  python scripts/ingest_vidore.py --skip-faiss --replace-text --table-context
  python scripts/ingest_vidore.py --skip-faiss --replace-text --table-context --max-pages 20
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import cfg
from src.ingestion.encoders import BGEEmbedder, create_visual_encoder
from src.ingestion.text_chunker import TextChunker
from src.ingestion.vidore_ingestor import ViDoReIngestor
from src.store.faiss_store import FaissColPaliStore
from src.store.pgvector_store import PgVectorStore

logger = logging.getLogger(__name__)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description="Import ViDoRe dataset into stores")
    parser.add_argument("--dataset", default="vidore/vidore_v3_industrial", help="HF dataset path")
    parser.add_argument("--max-pages", type=int, default=None, help="Limit pages (for testing)")
    parser.add_argument("--skip-faiss", action="store_true", help="Skip ColPali + FAISS (BGE only)")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    parser.add_argument(
        "--replace-text",
        action="store_true",
        help="TRUNCATE chunks 后重写文本路（Text re-ingest；配合 --skip-faiss 保 FAISS）",
    )
    parser.add_argument(
        "--table-context",
        action="store_true",
        help="开启 table_summary 同页邻段上下文（ingestion.table_summary_context_enabled）",
    )
    parser.add_argument(
        "--no-table-summary",
        action="store_true",
        help="关闭表摘要 LLM（只切块+embed 原文）",
    )
    parser.add_argument(
        "--visual-model",
        default="colqwen2",
        choices=["colpali", "colqwen2"],
        help="Visual embedding model (default: colqwen2)",
    )
    args = parser.parse_args()

    cfg.load()
    if args.table_context:
        cfg._data.setdefault("ingestion", {})["table_summary_context_enabled"] = True
    if args.no_table_summary:
        cfg._data.setdefault("ingestion", {})["table_summary_enabled"] = False

    logger.info(
        "ingest flags: skip_faiss=%s replace_text=%s table_context=%s table_summary=%s max_pages=%s",
        args.skip_faiss,
        args.replace_text,
        cfg.get("ingestion.table_summary_context_enabled", False),
        cfg.get("ingestion.table_summary_enabled", True),
        args.max_pages,
    )
    if args.replace_text and not args.skip_faiss:
        logger.warning(
            "--replace-text 未加 --skip-faiss 将重跑视觉路（贵）。Text-only 请同时 --skip-faiss。"
        )

    pg_store = PgVectorStore()
    if args.visual_model == "colqwen2":
        faiss_store = FaissColPaliStore(
            index_path=cfg.get("storage.faiss.colqwen2_index_path"),
            id_map_path=cfg.get("storage.faiss.colqwen2_id_map_path"),
        )
    else:
        faiss_store = FaissColPaliStore()
    bge = BGEEmbedder()
    if args.skip_faiss:
        class _NoVisual:
            def encode_pages(self, *a, **k):
                raise RuntimeError("skip_faiss: visual encoder not loaded")

        colpali = _NoVisual()
    else:
        colpali = create_visual_encoder(model_name=args.visual_model)
    chunker = TextChunker(
        image_caption_chunks=cfg.get("ingestion.image_caption_chunks", False),
    )

    ingestor = ViDoReIngestor(pg_store, faiss_store, bge, colpali, chunker)
    ingestor.ingest(
        dataset_path=args.dataset,
        max_pages=args.max_pages,
        skip_faiss=args.skip_faiss,
        resume=args.resume,
        replace_text=args.replace_text,
    )


if __name__ == "__main__":
    main()
