#!/usr/bin/env python
"""评测入口脚本 — 修正 ColPali 预编码 + FAISS 加载分离顺序"""

import argparse
import logging
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import cfg
from src.ingestion.encoders import BGEEmbedder, create_visual_encoder
from src.ingestion.text_chunker import TextChunker
from src.evaluation.ablation import (
    ABLATION_CONFIGS,
    GOLDEN_NO_HYDE_NAMES,
    load_eval_data,
    run_ablation,
)
from src.evaluation.vidore_adapter import PrismRAGRetriever
from src.observability import dump_collector
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.fusion import RRFFusion
from src.retrieval.hyde import HyDEGenerator
from src.retrieval.reranker import Reranker
from src.retrieval.visual_retriever import VisualRetriever
from src.store.faiss_store import FaissColPaliStore
from src.store.pgvector_store import PgVectorStore

logger = logging.getLogger(__name__)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description="Run PrismRAG evaluation")
    parser.add_argument("--dataset", default="vidore/vidore_v3_industrial")
    parser.add_argument("--max-queries", type=int, default=None)
    parser.add_argument("--skip-index", action="store_true")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--language", default="en", choices=["en", "all"],
                        help="评测语言（'en'=论文对齐英文子集, 'all'=全量 1698 query）")
    parser.add_argument("--expected-query-count", type=int, default=None,
                        help="预期 query 数量校验（默认: en=283, all=不校验）")
    parser.add_argument("--quick", action="store_true",
                        help="仅跑新增配置（跳过 7 个基线消融）")
    parser.add_argument("--config-filter", type=str, default=None,
                        help="仅跑名称包含该子串的消融配置（如 Visual 匹配 Visual_only、BM25_Dense_Visual 等）")
    parser.add_argument("--visual-model", default="colqwen2",
                        choices=["colpali", "colqwen2"],
                        help="Visual embedding model (default: colqwen2)")
    parser.add_argument(
        "--no-hyde",
        action="store_true",
        help="Boot-A 默认：仅跑 GOLDEN_NO_HYDE（排除 Full_*_HyDE，省 GPU）",
    )
    args = parser.parse_args()

    cfg.load()

    # ── Phase 0: 加载 & 过滤评测数据 ──────────────────────────
    logger.info(f"加载评测数据 (language={args.language})...")
    queries_ds, qrel_map = load_eval_data(
        dataset_path=args.dataset,
        max_queries=args.max_queries,
        language=args.language,
    )
    num_queries = len(queries_ds)
    logger.info(f"评测 query 数量: {num_queries}")

    # 校验 query 数量（fail fast，仅在全量评测时校验）
    expected = args.expected_query_count
    if expected is None:
        expected = 283 if args.language == "en" else None
    if expected is not None and args.max_queries is None and num_queries != expected:
        raise RuntimeError(
            f"query 数量校验失败: 预期 {expected}, 实际 {num_queries}。"
            f"请检查 dataset 的 language 字段分布。"
        )

    # ── 基础设施初始化 ────────────────────────────────────────
    pg_store = PgVectorStore()
    if args.visual_model == "colqwen2":
        faiss_store = FaissColPaliStore(
            index_path=cfg.get("storage.faiss.colqwen2_index_path"),
            id_map_path=cfg.get("storage.faiss.colqwen2_id_map_path"),
        )
    else:
        faiss_store = FaissColPaliStore()
    bge = BGEEmbedder()
    chunker = TextChunker()
    bm25 = BM25Retriever()
    dense = DenseRetriever(pg_store, bge)
    fusion = RRFFusion(rrf_k=60)
    reranker = Reranker()
    # 与 run_ablation 一致：仅当选中配置含 use_hyde 时才预计算（Boot-A --no-hyde 可省 ~40min）
    selected = list(ABLATION_CONFIGS)
    if args.quick:
        selected = [c for c in ABLATION_CONFIGS if c.name in (
            "Full_BGE_HyDE", "Full_zerank2_HyDE",
        )]
    if args.config_filter:
        selected = [
            c for c in selected
            if args.config_filter.lower() in c.name.lower()
        ]
    # --no-hyde：先 filter 再剔除 HyDE（避免 Full_zerank2 子串命中 Full_zerank2_HyDE）
    if args.no_hyde:
        selected = [c for c in selected if not c.use_hyde]
    need_hyde = any(c.use_hyde for c in selected)
    hyde = HyDEGenerator() if need_hyde else None
    # zerank-2 延迟到 ColPali 卸载后加载，避免三模型同时占满 24GB 显存

    # ── Phase A: 预编码 visual query ──────────────────────────
    # 仅占用 ColPali 模型显存，FAISS 向量尚未加载到 GPU
    pre_encoded_visual = None
    logger.info(f"预编码 visual query ({args.visual_model})...")
    visual_encoder = create_visual_encoder(model_name=args.visual_model)
    query_texts = [str(queries_ds[i]["query"]) for i in range(num_queries)]
    pre_encoded_visual = visual_encoder.encode_queries_batch(query_texts, batch_size=8)
    logger.info(f"完成 {len(pre_encoded_visual)} 条 query 预编码")
    visual_encoder.unload()
    logger.info(f"Visual encoder ({args.visual_model})已卸载，显存已释放")
    torch.cuda.empty_cache()

    # ── Phase A2: HyDE 预计算（仅 need_hyde；Ollama 独占 GPU，完成后释放）──
    if need_hyde:
        logger.info("HyDE 预计算（Ollama GPU 加速）...")
        import subprocess, time
        result = subprocess.run(["pgrep", "-f", "ollama serve"], capture_output=True)
        if result.returncode != 0:
            subprocess.Popen(
                ["ollama", "serve"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            time.sleep(5)
        hyde.precompute(query_texts)
        logger.info(f"HyDE 预计算完成，缓存 {len(hyde._cache)} 条")
        subprocess.run(["pkill", "-f", "ollama serve"], capture_output=True)
        time.sleep(3)
        torch.cuda.empty_cache()
        logger.info("Ollama 已关闭，显存已释放")
    else:
        logger.info(
            "跳过 HyDE 预计算（当前消融配置不含 HyDE；"
            f"selected={[c.name for c in selected]}）"
        )

    # ── 加载 zerank-2（ColPali 已卸载，Ollama 已释放，显存充足）──
    logger.info("加载 zerank-2 reranker (bf16)...")
    try:
        torch.cuda.empty_cache()
        zerank_reranker = Reranker(model_id=cfg.zerank_reranker_model_id,
                                   automodel_args={"torch_dtype": torch.bfloat16})
    except Exception as e:
        # 禁止置 None：Full_zerank2 配置会调用 .rerank 直接炸
        logger.warning(f"zerank-2 加载失败，降级到 BGE reranker: {e}")
        zerank_reranker = reranker

    # ── Phase B: Ingest / Load FAISS ──────────────────────────
    # 此时 Visual 模型已卸载，FAISS GPU 向量可以安全加载
    if not args.skip_index:
        encoder_for_ingest = create_visual_encoder(model_name=args.visual_model)
        from src.ingestion.vidore_ingestor import ViDoReIngestor
        ingestor = ViDoReIngestor(pg_store, faiss_store, bge, encoder_for_ingest, chunker)
        ingestor.ingest(dataset_path=args.dataset)
        bm25.fit_from_pgvector(pg_store)
        logger.info("BM25 索引构建完成")
        encoder_for_ingest.unload()
    else:
        faiss_loaded = faiss_store.load()
        if faiss_loaded:
            bm25.fit_from_pgvector(pg_store)
            logger.info("索引加载成功，跳过构建")
        else:
            logger.warning("FAISS 索引不存在，重新构建")
            encoder_for_ingest = create_visual_encoder(model_name=args.visual_model)
            from src.ingestion.vidore_ingestor import ViDoReIngestor
            ingestor = ViDoReIngestor(pg_store, faiss_store, bge, encoder_for_ingest, chunker)
            ingestor.ingest(dataset_path=args.dataset)
            bm25.fit_from_pgvector(pg_store)
            logger.info("BM25 索引构建完成")
            encoder_for_ingest.unload()

    # ── 构造检索器 ────────────────────────────────────────────
    visual = VisualRetriever(faiss_store, pg_store, visual_encoder)
    retriever = PrismRAGRetriever(
        pg_store=pg_store, faiss_store=faiss_store, bge=bge, colpali=visual_encoder,
        chunker=chunker, bm25=bm25, dense=dense, visual=visual,
        fusion=fusion, reranker=reranker, hyde=hyde, zerank_reranker=zerank_reranker,
    )

    # ── 执行消融实验 ──────────────────────────────────────────
    run_ablation(
        retriever,
        queries_ds=queries_ds,
        qrel_map=qrel_map,
        output_dir=args.output_dir,
        pre_encoded_visual=pre_encoded_visual,
        language=args.language,
        quick=args.quick,
        config_filter=args.config_filter,
        no_hyde=args.no_hyde,
    )

    dump_collector(f"ablation_{Path(args.output_dir).name}")


if __name__ == "__main__":
    main()