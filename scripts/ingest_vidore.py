#!/usr/bin/env python
"""ViDoRe 数据导入入口脚本"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import cfg
from src.ingestion.encoders import BGEEmbedder, ColPaliEmbedder
from src.ingestion.text_chunker import TextChunker
from src.ingestion.vidore_ingestor import ViDoReIngestor
from src.store.pgvector_store import PgVectorStore
from src.store.faiss_store import FaissColPaliStore


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description="Import ViDoRe dataset into stores")
    parser.add_argument("--dataset", default="vidore/vidore_v3_industrial", help="HF dataset path")
    parser.add_argument("--max-pages", type=int, default=None, help="Limit pages (for testing)")
    parser.add_argument("--skip-faiss", action="store_true", help="Skip ColPali + FAISS (BGE only)")
    args = parser.parse_args()

    # 初始化
    cfg.load()
    pg_store = PgVectorStore()
    faiss_store = FaissColPaliStore()
    bge = BGEEmbedder()
    colpali = ColPaliEmbedder()
    chunker = TextChunker()

    ingestor = ViDoReIngestor(pg_store, faiss_store, bge, colpali, chunker)
    ingestor.ingest(
        dataset_path=args.dataset,
        max_pages=args.max_pages,
        skip_faiss=args.skip_faiss,
    )


if __name__ == "__main__":
    main()