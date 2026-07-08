"""CLI: 入库本地 PDF（容器内或本地，需 PG 可达 + 模型已下载）"""
from __future__ import annotations

import argparse
from pathlib import Path

from src.config import cfg
from src.ingestion.encoders import BGEEmbedder, ColPaliEmbedder
from src.ingestion.pdf_ingestor import PDFIngestor
from src.ingestion.text_chunker import TextChunker
from src.retrieval.bm25_retriever import BM25Retriever
from src.store.faiss_store import FaissColPaliStore
from src.store.pgvector_store import PgVectorStore


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", required=True)
    args = ap.parse_args()
    cfg.load()
    use_visual = cfg.get("retrieval.use_visual", True)
    pg = PgVectorStore()
    faiss = FaissColPaliStore()
    bge = BGEEmbedder()
    # 本地 dev (use_visual=false) 免 ColPali 3.5B 下载：仅当启用 visual 路才构造
    colpali = ColPaliEmbedder() if use_visual else None
    chunker = TextChunker()
    bm25 = BM25Retriever()
    # 先入库（会 create_schema + 写入 chunks），再拟合 BM25 内存索引
    faiss.load()
    res = PDFIngestor(pg, faiss, bge, colpali, chunker).ingest(Path(args.pdf))
    bm25.fit_from_pgvector(pg)
    print(f"入库完成: {res}")


if __name__ == "__main__":
    main()
