"""消融实验运行器"""

from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional

from datasets import load_dataset as hf_load_dataset
from tqdm import tqdm

from src.evaluation.vidore_adapter import PrismRAGRetriever

if TYPE_CHECKING:
    import torch

logger = logging.getLogger(__name__)


@dataclass
class AblationConfig:
    name: str
    use_bm25: bool = True
    use_dense: bool = True
    use_visual: bool = True
    use_rerank: bool = True
    reranker_type: str = "bge"      # "bge" | "zerank"
    use_hyde: bool = False


ABLATION_CONFIGS = [
    # ── 基础消融：各路检索器独立/组合 ──
    AblationConfig(name="BM25_only", use_bm25=True, use_dense=False, use_visual=False, use_rerank=False),
    AblationConfig(name="Dense_only", use_bm25=False, use_dense=True, use_visual=False, use_rerank=False),
    AblationConfig(name="Visual_only", use_bm25=False, use_dense=False, use_visual=True, use_rerank=False),
    AblationConfig(name="BM25_Dense", use_bm25=True, use_dense=True, use_visual=False, use_rerank=False),
    AblationConfig(name="BM25_Dense_Visual", use_bm25=True, use_dense=True, use_visual=True, use_rerank=False),
    AblationConfig(name="Full_no_rerank", use_bm25=True, use_dense=True, use_visual=True, use_rerank=False),
    AblationConfig(name="Full_with_rerank", use_bm25=True, use_dense=True, use_visual=True, use_rerank=True),
    # ── 全管道 + HyDE / zerank-2 变体（消融对比用）──
    AblationConfig(name="Full_BGE_HyDE",
        use_bm25=True, use_dense=True, use_visual=True, use_rerank=True,
        reranker_type="bge", use_hyde=True),
    AblationConfig(name="Full_zerank2",
        use_bm25=True, use_dense=True, use_visual=True, use_rerank=True,
        reranker_type="zerank", use_hyde=False),
    AblationConfig(name="Full_zerank2_HyDE",
        use_bm25=True, use_dense=True, use_visual=True, use_rerank=True,
        reranker_type="zerank", use_hyde=True),
]


def compute_ndcg(relevant: set, ranked: List[str], k: int) -> float:
    """计算 NDCG@k，使用标准 log2(i+1) 折扣（与 pytrec_eval 一致）。
    对重复页面去重，仅保留首次出现（后续重复不计增益）"""
    dcg, idcg = 0.0, 0.0
    seen = set()
    pos = 0
    for rid in ranked:
        if pos >= k:
            break
        if rid in seen:
            continue
        seen.add(rid)
        if rid in relevant:
            dcg += 1.0 / math.log2(pos + 2)
        pos += 1
    for i in range(min(k, len(relevant))):
        idcg += 1.0 / math.log2(i + 2)
    return dcg / idcg if idcg > 0 else 0.0


def compute_recall(relevant: set, ranked: List[str], k: int) -> float:
    if not relevant:
        return 0.0
    seen = set()
    hits = 0
    for r in ranked[:k]:
        if r in seen:
            continue
        seen.add(r)
        if r in relevant:
            hits += 1
    return hits / len(relevant)


def compute_mrr(relevant: set, ranked: List[str]) -> float:
    for i, r in enumerate(ranked, start=1):
        if r in relevant:
            return 1.0 / i
    return 0.0


# ViDoRe dataset language field values (dataset uses full names, CLI uses short codes)
_LANGUAGE_MAP = {
    "en": "english",
    "fr": "french",
    "de": "german",
    "it": "italian",
    "es": "spanish",
    "pt": "portuguese",
}


def load_eval_data(
    dataset_path: str,
    max_queries: Optional[int] = None,
    language: str = "en",
) -> tuple:
    """加载评测数据，按语言过滤，支持 query 数量限制

    Args:
        dataset_path: HuggingFace dataset 路径
        max_queries: 可选，限制 query 数量
        language: "en"/"fr"/... 过滤对应语言，"all" 全部保留

    Returns:
        (queries_ds, qrel_map) 元组
        - queries_ds: HuggingFace Dataset（已过滤和截断）
        - qrel_map: Dict[int, set] — query_id -> set of corpus_ids
    """
    queries_ds = hf_load_dataset(dataset_path, "queries", split="test")
    qrels_ds = hf_load_dataset(dataset_path, "qrels", split="test")

    if language != "all":
        dataset_lang = _LANGUAGE_MAP.get(language, language)
        queries_ds = queries_ds.filter(lambda x: x["language"] == dataset_lang)

    if max_queries:
        queries_ds = queries_ds.select(range(min(max_queries, len(queries_ds))))

    qrel_map: Dict[int, set] = {}
    for qrel in qrels_ds:
        qid = int(qrel["query_id"])
        cid = int(qrel["corpus_id"])
        if qid not in qrel_map:
            qrel_map[qid] = set()
        qrel_map[qid].add(cid)

    return queries_ds, qrel_map


def run_ablation(
    retriever: PrismRAGRetriever,
    queries_ds,
    qrel_map: Dict[int, set],
    output_dir: str = "results",
    pre_encoded_visual: Optional[Dict[int, "torch.Tensor"]] = None,
    language: str = "en",
    quick: bool = False,
    config_filter: Optional[str] = None,
) -> List[dict]:
    """运行全量消融实验

    Args:
        retriever: PrismRAGRetriever 实例
        queries_ds: 已加载和过滤的 queries dataset
        qrel_map: query_id -> relevant corpus_id set
        output_dir: 结果输出目录
        pre_encoded_visual: {q_idx: tensor[1, n_q, 128]} 预编码的 visual query embedding
        language: 当前评测语言，会写入结果元数据
        quick: 仅跑新增配置（跳过基线消融）
        config_filter: 可选，按名称子串过滤消融配置（如 "Visual" 匹配 Visual_only、BM25_Dense_Visual）
    """

    configs = ABLATION_CONFIGS
    if quick:
        # 仅跑 HyDE + zerank-2 相关的新配置
        configs = [c for c in ABLATION_CONFIGS if c.name in (
            "Full_BGE_HyDE", "Full_zerank2_HyDE"
        )]
        logger.info(f"Quick 模式：仅跑 {len(configs)} 组新配置")
    if config_filter:
        configs = [c for c in configs if config_filter.lower() in c.name.lower()]
        logger.info(f"Config filter '{config_filter}': 仅跑 {len(configs)} 组配置 ({[c.name for c in configs]})")

    results = []

    for config in configs:
        logger.info(f"=== 消融配置: {config.name} ===")
        latencies = []
        all_ranked_page_ids: List[List[str]] = []
        all_relevant: List[set] = []

        for q_idx in tqdm(range(len(queries_ds)), desc=f"  {config.name}"):
            q = queries_ds[q_idx]
            qid = int(q["query_id"])
            query_text = str(q["query"])

            visual_q_emb = None
            if config.use_visual and pre_encoded_visual is not None and q_idx in pre_encoded_visual:
                visual_q_emb = pre_encoded_visual[q_idx]

            start = time.time()
            retrieved = retriever.search(
                query=query_text, k=10,
                use_bm25=config.use_bm25, use_dense=config.use_dense,
                use_visual=config.use_visual, use_rerank=config.use_rerank,
                visual_query_embedding=visual_q_emb,
                use_hyde=config.use_hyde, reranker_type=config.reranker_type,
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
            "language": language,
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
