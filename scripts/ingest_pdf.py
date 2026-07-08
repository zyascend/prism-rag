"""CLI: 入库本地 PDF（容器内或本地，需 PG 可达 + 模型已下载）"""
from __future__ import annotations

import argparse
from pathlib import Path

from src.config import cfg
from src.evaluation.vidore_adapter import PrismRAGRetriever
from src.ingestion.encoders import BGEEmbedder, ColPaliEmbedder
from src.ingestion.pdf_ingestor import PDFIngestor
from src.ingestion.text_chunker import TextChunker
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.fusion import RRFFusion
from src.retrieval.reranker import Reranker
from src.retrieval.visual_retriever import VisualRetriever
from src.store.faiss_store import FaissColPaliStore
from src.store.pgvector_store import PgVectorStore


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", required=True)
    args = ap.parse_args()
    cfg.load()
    pg = PgVectorStore()
    faiss = FaissColPaliStore()
    bge = BGEEmbedder()
    colpali = ColPaliEmbedder()
    chunker = TextChunker()
    bm25 = BM25Retriever()
    dense = DenseRetriever(pg, bge)
    visual = VisualRetriever(faiss, pg, colpali)
    fusion = RRFFusion(rrf_k=cfg.get("retrieval.rrf_k", 60))
    reranker = Reranker()
    retriever = PrismRAGRetriever(
        pg, faiss, bge, colpali, chunker, bm25, dense, visual, fusion, reranker
    )
    faiss.load()
    bm25.fit_from_pgvector(pg)
    res = PDFIngestor(pg, faiss, bge, colpali, chunker).ingest(Path(args.pdf))
    print(f"入库完成: {res}")


if __name__ == "__main__":
    main()
