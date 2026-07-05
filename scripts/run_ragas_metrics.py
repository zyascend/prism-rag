#!/usr/bin/env python
"""RAGAS 生成层评测入口 — Faithfulness + Answer Relevancy

使用方式：
  # 完整评测（283 条 ViDoRe 英文查询）
  python scripts/run_ragas_metrics.py

  # 快速评测（仅 10 条查询）
  python scripts/run_ragas_metrics.py --max-queries 10

  # 跳过检索，仅评测已有答案（debug 模式）
  python scripts/run_ragas_metrics.py --skip-retrieval --input-json results/sample_answers.json

流程：
  1. 加载 ViDoRe queries
  2. 构建/加载检索索引
  3. 对每条 query: 检索 → 生成 → Faithfulness → Answer Relevancy
  4. 输出汇总结果到 results/ragas_metrics.json
"""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import cfg
from src.evaluation.ablation import load_eval_data
from src.evaluation.ragas_metrics import (
    RagasGenerationEvalResult,
    compute_answer_relevancy,
    compute_faithfulness,
    evaluate_generation,
    evaluate_generation_configs,
    generate_answer,
)
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

    # 加载 FAISS 索引 + BM25（无论是否 skip_index）
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

    parser = argparse.ArgumentParser(description="Run RAGAS generation layer evaluation")
    parser.add_argument("--dataset", default="vidore/vidore_v3_industrial")
    parser.add_argument("--max-queries", type=int, default=None)
    parser.add_argument("--skip-index", action="store_true", default=False)
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--language", default="en", choices=["en", "all"])
    parser.add_argument("--visual-model", default="colqwen2", choices=["colpali", "colqwen2"])
    parser.add_argument("--skip-retrieval", action="store_true",
                        help="跳过检索，从已有 JSON 文件读取答案（debug 用）")
    parser.add_argument("--input-json", type=str, default=None,
                        help="输入 JSON 文件路径（与 --skip-retrieval 配合）")
    parser.add_argument("--ablation", action="store_true",
                        help="运行生成层消融（不同检索配置下都评测）")
    args = parser.parse_args()

    cfg.load()

    # ── 调试模式：直接从已有 JSON 评测 ──────────────────────────
    if args.skip_retrieval and args.input_json:
        logger.info(f"调试模式：从 {args.input_json} 读取已有问答对进行评测")
        with open(args.input_json) as f:
            data = json.load(f)

        faithfulness_results = []
        relevancy_results = []
        for item in data:
            query = item.get("query", item.get("question", ""))
            answer = item.get("answer", "")
            context = item.get("context", "")

            # Faithfulness
            f_res = compute_faithfulness(answer, context)
            f_res.query = query
            faithfulness_results.append(f_res)

            # Answer Relevancy
            r_res = compute_answer_relevancy(query, answer)
            relevancy_results.append(r_res)

        avg_f = sum(r.faithfulness_score for r in faithfulness_results) / len(faithfulness_results) if faithfulness_results else 0.0
        avg_r = sum(r.relevancy_score for r in relevancy_results) / len(relevancy_results) if relevancy_results else 0.0

        logger.info(f"\n调试模式结果 ({len(data)} 条):")
        logger.info(f"  Faithfulness: {avg_f:.4f}")
        logger.info(f"  Relevancy:    {avg_r:.4f}")

        output_path = Path(args.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        with open(output_path / "ragas_metrics_debug.json", "w") as f:
            json.dump({
                "summary": {
                    "num_queries": len(data),
                    "avg_faithfulness": round(avg_f, 4),
                    "avg_relevancy": round(avg_r, 4),
                },
                "faithfulness": [r.to_dict() for r in faithfulness_results],
                "relevancy": [r.to_dict() for r in relevancy_results],
            }, f, indent=2, ensure_ascii=False)
        return

    # ── 标准模式：加载评测数据 ──────────────────────────────────
    logger.info(f"加载评测数据 (language={args.language})...")
    queries_ds, _ = load_eval_data(
        dataset_path=args.dataset,
        max_queries=args.max_queries,
        language=args.language,
    )
    num_queries = len(queries_ds)
    logger.info(f"评测 query 数量: {num_queries}")

    # ── 构建检索器 ─────────────────────────────────────────────
    retriever = build_retriever(
        skip_index=args.skip_index,
        visual_model=args.visual_model,
    )

    # ── 生成层评测 ─────────────────────────────────────────────
    if args.ablation:
        # 消融模式：多种检索配置对比
        configs = [
            {"name": "bm25_only", "k": 5, "use_rerank": False},
            {"name": "dense_only", "k": 5, "use_rerank": False},
            {"name": "full_no_rerank", "k": 5, "use_rerank": False},
            {"name": "full_with_rerank", "k": 5, "use_rerank": True},
        ]

        # XXX: 消融模式下每条 query 都会修改 retriever 的内部状态
        # 需要用不同配置重新构造或通过 search() 参数控制
        evaluate_generation_configs(
            retriever=retriever,
            queries_ds=queries_ds,
            configs=configs,
            max_queries=args.max_queries,
            output_dir=args.output_dir,
        )
    else:
        # 标准模式：默认配置（Full + rerank）
        evaluate_generation(
            retriever=retriever,
            queries_ds=queries_ds,
            k=5,
            use_rerank=True,
            max_queries=args.max_queries,
            output_dir=args.output_dir,
            label="default",
        )

    logger.info("评测完成。")


if __name__ == "__main__":
    main()