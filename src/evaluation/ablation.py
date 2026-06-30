"""消融实验运行器"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from datasets import load_dataset
from tqdm import tqdm

from src.evaluation.vidore_adapter import PrismRAGRetriever

logger = logging.getLogger(__name__)


@dataclass
class AblationConfig:
    name: str
    use_bm25: bool = True
    use_dense: bool = True
    use_visual: bool = True
    use_rerank: bool = True


ABLATION_CONFIGS = [
    AblationConfig(name="BM25_only", use_bm25=True, use_dense=False, use_visual=False, use_rerank=False),
    AblationConfig(name="Dense_only", use_bm25=False, use_dense=True, use_visual=False, use_rerank=False),
    AblationConfig(name="Visual_only", use_bm25=False, use_dense=False, use_visual=True, use_rerank=False),
    AblationConfig(name="BM25_Dense", use_bm25=True, use_dense=True, use_visual=False, use_rerank=False),
    AblationConfig(name="BM25_Dense_Visual", use_bm25=True, use_dense=True, use_visual=True, use_rerank=False),
    AblationConfig(name="Full_no_rerank", use_bm25=True, use_dense=True, use_visual=True, use_rerank=False),
    AblationConfig(name="Full_with_rerank", use_bm25=True, use_dense=True, use_visual=True, use_rerank=True),
]


def compute_ndcg(relevant: set, ranked: List[str], k: int) -> float:
    dcg, idcg = 0.0, 0.0
    for i in range(min(k, len(ranked))):
        if ranked[i] in relevant:
            dcg += 1.0 / (i + 1)
    for i in range(min(k, len(relevant))):
        idcg += 1.0 / (i + 1)
    return dcg / idcg if idcg > 0 else 0.0


def compute_recall(relevant: set, ranked: List[str], k: int) -> float:
    if not relevant:
        return 0.0
    return sum(1 for r in ranked[:k] if r in relevant) / len(relevant)


def compute_mrr(relevant: set, ranked: List[str]) -> float:
    for i, r in enumerate(ranked, start=1):
        if r in relevant:
            return 1.0 / i
    return 0.0


def run_ablation(
    retriever: PrismRAGRetriever,
    dataset_path: str = "vidore/vidore_v3_industrial",
    max_queries: Optional[int] = None,
    output_dir: str = "results",
) -> List[dict]:
    """运行全量消融实验"""
    logger.info("加载查询和 qrels...")
    queries_ds = load_dataset(dataset_path, "queries", split="test")
    qrels_ds = load_dataset(dataset_path, "qrels", split="test")

    if max_queries:
        queries_ds = queries_ds.select(range(min(max_queries, len(queries_ds))))

    qrel_map: Dict[int, set] = {}
    for qrel in qrels_ds:
        qid = int(qrel["query_id"])
        cid = int(qrel["corpus_id"])
        if qid not in qrel_map:
            qrel_map[qid] = set()
        qrel_map[qid].add(cid)

    results = []

    for config in ABLATION_CONFIGS:
        logger.info(f"=== 消融配置: {config.name} ===")
        latencies = []
        all_ranked_page_ids: List[List[str]] = []
        all_relevant: List[set] = []

        for q_idx in tqdm(range(len(queries_ds)), desc=f"  {config.name}"):
            q = queries_ds[q_idx]
            qid = int(q["query_id"])
            query_text = str(q["query"])

            start = time.time()
            retrieved = retriever.search(
                query=query_text, k=10,
                use_bm25=config.use_bm25, use_dense=config.use_dense,
                use_visual=config.use_visual, use_rerank=config.use_rerank,
            )
            latencies.append((time.time() - start) * 1000)

            ranked_page_ids = [str(r["page_id"]) for r in retrieved]
            all_ranked_page_ids.append(ranked_page_ids)
            relevant = {str(cid) for cid in qrel_map.get(qid, set())}
            all_relevant.append(relevant)

        n = len(all_ranked_page_ids)
        ndcg5 = sum(compute_ndcg(rel, ranked, 5) for rel, ranked in zip(all_relevant, all_ranked_page_ids)) / n
        ndcg10 = sum(compute_ndcg(rel, ranked, 10) for rel, ranked in zip(all_relevant, all_ranked_page_ids)) / n
        rec5 = sum(compute_recall(rel, ranked, 5) for rel, ranked in zip(all_relevant, all_ranked_page_ids)) / n
        rec10 = sum(compute_recall(rel, ranked, 10) for rel, ranked in zip(all_relevant, all_ranked_page_ids)) / n
        mrr = sum(compute_mrr(rel, ranked) for rel, ranked in zip(all_relevant, all_ranked_page_ids)) / n
        avg_lat = sum(latencies) / len(latencies) if latencies else 0

        result = {
            "config": config.name,
            "ndcg@5": round(ndcg5, 4), "ndcg@10": round(ndcg10, 4),
            "recall@5": round(rec5, 4), "recall@10": round(rec10, 4),
            "mrr": round(mrr, 4), "avg_latency_ms": round(avg_lat, 1),
            "num_queries": n,
        }
        results.append(result)
        logger.info(f"  NDCG@10={ndcg10:.4f}, Recall@5={rec5:.4f}, MRR={mrr:.4f}")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    with open(output_path / "ablation_results.json", "w") as f:
        json.dump(results, f, indent=2)

    logger.info("\n" + "=" * 80)
    logger.info("消融实验结果")
    header = f"{'Config':<25} {'NDCG@5':<10} {'NDCG@10':<10} {'Recall@5':<10} {'Recall@10':<10} {'MRR':<10} {'Lat(ms)':<10}"
    logger.info(header)
    logger.info("-" * 80)
    for r in results:
        logger.info(f"{r['config']:<25} {r['ndcg@5']:<10.4f} {r['ndcg@10']:<10.4f} {r['recall@5']:<10.4f} {r['recall@10']:<10.4f} {r['mrr']:<10.4f} {r['avg_latency_ms']:<10.0f}")

    return results
