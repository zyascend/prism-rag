#!/usr/bin/env python
"""端到端 QA 评测入口 — Answer Correctness + Rejection Accuracy

使用方式：
  # 完整评测（50 QA + 20 拒答）
  python scripts/run_e2e_qa.py

  # 快速评测（仅 10 条）
  python scripts/run_e2e_qa.py --max-queries 10

  # 跳过索引（已有索引时）
  python scripts/run_e2e_qa.py --skip-index

流程：
  1. 加载端到端 QA 数据集
  2. 构建/加载检索索引
  3. 对每条 question: 检索 → 生成 → 判正确性/拒答
  4. 输出汇总结果到 results/e2e_qa_results.json
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import cfg
from src.evaluation.e2e_qa import evaluate_e2e_qa
from src.evaluation.vidore_adapter import PrismRAGRetriever
from src.ingestion.encoders import BGEEmbedder, create_visual_encoder
from src.ingestion.text_chunker import TextChunker
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.fusion import RRFFusion
from src.retrieval.reranker import Reranker
from src.retrieval.visual_retriever import VisualRetriever
from src.store.faiss_store import FaissColPaliStore
from src.store.pgvector_store import PgVectorStore

logger = logging.getLogger(__name__)


def build_retriever(skip_index: bool = False, visual_model: str = "colpali") -> PrismRAGRetriever:
    """构造 PrismRAGRetriever 实例"""
    pg_store = PgVectorStore()

    if visual_model == "colqwen2":
        faiss_store = FaissColPaliStore(
            index_path=cfg.get("storage.faiss.colqwen2_index_path"),
            id_map_path=cfg.get("storage.faiss.colqwen2_id_map_path"),
        )
    else:
        faiss_store = FaissColPaliStore()

    bge = BGEEmbedder()
    visual_encoder = create_visual_encoder(model_name=visual_model)
    chunker = TextChunker()
    bm25 = BM25Retriever()
    dense = DenseRetriever(pg_store, bge)
    visual = VisualRetriever(faiss_store, pg_store, visual_encoder)
    fusion = RRFFusion(rrf_k=60)
    reranker = Reranker()

    # 加载 FAISS 索引 + BM25
    faiss_loaded = faiss_store.load()
    if faiss_loaded:
        bm25.fit_from_pgvector(pg_store)
        logger.info("索引加载成功")
    elif not skip_index:
        logger.warning("FAISS 索引不存在，请先运行 `python scripts/ingest_vidore.py`")
        sys.exit(1)
    else:
        logger.warning("FAISS 索引不存在，跳过 visual 检索，仅使用 Dense 检索")

    return PrismRAGRetriever(
        pg_store=pg_store, faiss_store=faiss_store, bge=bge, colpali=visual_encoder,
        chunker=chunker, bm25=bm25, dense=dense, visual=visual,
        fusion=fusion, reranker=reranker,
    )


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="端到端 QA 评测 — Answer Correctness + Rejection Accuracy"
    )
    parser.add_argument("--qa-file", default="data/e2e_qa.json",
                        help="QA 数据集路径")
    parser.add_argument("--max-queries", type=int, default=None,
                        help="最大查询数（用于快速验证）")
    parser.add_argument("--skip-index", action="store_true", default=False,
                        help="跳过 FAISS 索引加载（适用于已有索引）")
    parser.add_argument("--output-dir", default="results",
                        help="结果输出目录")
    parser.add_argument("--visual-model", default="colpali",
                        choices=["colpali", "colqwen2"],
                        help="视觉检索模型")
    parser.add_argument("--no-rerank", action="store_true", default=False,
                        help="禁用重排序")
    parser.add_argument("--k", type=int, default=5,
                        help="检索 top-k")
    args = parser.parse_args()

    cfg.load()

    # ── 构建检索器 ─────────────────────────────────────────────
    retriever = build_retriever(
        skip_index=args.skip_index,
        visual_model=args.visual_model,
    )

    # ── 运行端到端评测 ─────────────────────────────────────────
    result = evaluate_e2e_qa(
        retriever=retriever,
        qa_path=args.qa_file,
        k=args.k,
        use_rerank=not args.no_rerank,
        max_queries=args.max_queries,
        output_dir=args.output_dir,
    )

    logger.info("端到端 QA 评测完成。")


if __name__ == "__main__":
    main()