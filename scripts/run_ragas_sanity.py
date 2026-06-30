#!/usr/bin/env python
"""RAGAS 拒答评测入口"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import cfg
from src.evaluation.ragas_sanity import run_ragas_sanity
from src.evaluation.vidore_adapter import PrismRAGRetriever
from src.ingestion.encoders import BGEEmbedder, ColPaliEmbedder
from src.ingestion.text_chunker import TextChunker
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.fusion import RRFFusion
from src.retrieval.reranker import Reranker
from src.retrieval.visual_retriever import VisualRetriever
from src.store.faiss_store import FaissColPaliStore
from src.store.pgvector_store import PgVectorStore


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description="Run RAGAS rejection sanity")
    parser.add_argument("--rejection-qa", default="data/rejection_qa.json")
    parser.add_argument("--output-dir", default="results")
    args = parser.parse_args()

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

    retriever = PrismRAGRetriever(
        pg_store=pg_store, faiss_store=faiss_store, bge=bge, colpali=colpali,
        chunker=chunker, bm25=bm25, dense=dense, visual=visual,
        fusion=fusion, reranker=reranker,
    )

    if not faiss_store.load():
        logging.error("FAISS 索引未找到。请先运行 `python scripts/ingest_vidore.py` 构建索引")
        sys.exit(1)

    bm25.fit_from_pgvector(pg_store)
    run_ragas_sanity(retriever, rejection_qa_path=args.rejection_qa, output_dir=args.output_dir)


if __name__ == "__main__":
    main()