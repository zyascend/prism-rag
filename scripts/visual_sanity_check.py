#!/usr/bin/env python3
"""Visual 路最小时间验证 — 5-query MaxSim sanity check

在云端（4090, 模型/依赖/索引均就绪）约 3-5 分钟完成。
用途：判定 ColQwen2 query 和 FAISS 索引是否在同一嵌入空间。

判读：
  - 大部分 query 的相关页面进入 top-10 → 索引/查询匹配 → 0.1564 的根因是
    eval 错配了索引（路径/symlink），修复路径后重跑 eval 即可，无需重建索引。
  - 大部分 query 的相关页面不在 top-10 → 索引/查询不匹配或索引本身有问题 →
    需要完整重建索引（Option B, ~15-18 min）。

用法：
  python scripts/visual_sanity_check.py --n 5 --k 10 --visual-model colqwen2
"""

import argparse
import sys
import time
from pathlib import Path

import torch
from datasets import load_dataset as hf_load_dataset

from src.config import cfg
from src.ingestion.encoders import create_visual_encoder
from src.store.faiss_store import FaissColPaliStore


def check_index_path(visual_model: str):
    """Step 0: 检查索引路径是否正确（~10 sec, 不需要 GPU）"""
    if visual_model == "colqwen2":
        idx_path = cfg.get("storage.faiss.colqwen2_index_path")
        id_map_path = cfg.get("storage.faiss.colqwen2_id_map_path")
    else:
        idx_path = cfg.get("storage.faiss.index_path")
        id_map_path = cfg.get("storage.faiss.id_map_path")

    print(f"[0] 索引路径检查")
    print(f"    index_path  = {idx_path}")
    print(f"    id_map_path = {id_map_path}")

    # 解析可能的 symlink / autodl-tmp 映射
    idx = Path(idx_path)
    if idx.exists():
        resolved = idx.resolve() if idx.is_symlink() else idx
        size_mb = idx.stat().st_size / (1024 * 1024)
        print(f"    ✅ 索引文件存在: {resolved} ({size_mb:.0f} MB)")
    else:
        print(f"    ❌ 索引文件不存在: {idx_path}")
        # 尝试 autodl-tmp 常见位置
        alt = Path("/root/autodl-tmp/indexes") / idx.name
        if alt.exists():
            print(f"    💡 找到了替代位置: {alt} ({alt.stat().st_size / 1e6:.0f} MB)")
        return False
    return True


def main():
    parser = argparse.ArgumentParser(description="Visual 路 MaxSim sanity check")
    parser.add_argument("--n", type=int, default=5, help="验证 query 数 (默认 5)")
    parser.add_argument("--k", type=int, default=10, help="Top-K (默认 10)")
    parser.add_argument("--visual-model", default="colqwen2",
                        choices=["colpali", "colqwen2"])
    args = parser.parse_args()

    t_total = time.time()

    # ── Step 0: 路径检查 ──
    if not check_index_path(args.visual_model):
        sys.exit(1)

    # ── Step 1: 加载 FAISS 索引 ──
    print(f"\n[1] 加载 FAISS 索引 ({args.visual_model})...")
    t1 = time.time()
    if args.visual_model == "colqwen2":
        store = FaissColPaliStore(
            index_path=cfg.get("storage.faiss.colqwen2_index_path"),
            id_map_path=cfg.get("storage.faiss.colqwen2_id_map_path"),
        )
    else:
        store = FaissColPaliStore()
    ok = store.load()
    if not ok:
        print("    ❌ FAISS 索引加载失败，检查索引文件是否完整（.faiss + _vectors.npy + -ids.npy）")
        sys.exit(1)
    print(f"    ✅ {store.num_pages:,} pages, {store._num_patches:,} patches, type={store.index_type}")
    print(f"    加载耗时: {time.time() - t1:.1f}s")

    # ── Step 2: 加载 ViDoRe 数据集 ──
    print("\n[2] 加载 ViDoRe V3 Industrial 数据集...")
    t2 = time.time()
    ds_path = "vidore/vidore_v3_industrial"
    queries_ds = hf_load_dataset(ds_path, "queries", split="test")
    qrels_ds = hf_load_dataset(ds_path, "qrels", split="test")

    # 过滤 English
    en_queries = queries_ds.filter(lambda x: x["language"] == "english")
    # 构建 qrel_map: query_id → {corpus_id, ...}
    qrel_map = {}
    for qrel in qrels_ds:
        qid = int(qrel["query_id"])
        cid = int(qrel["corpus_id"])
        qrel_map.setdefault(qid, set()).add(cid)

    print(f"    English queries: {len(en_queries)}, with qrels: {len(qrel_map)}")

    # ── Step 3: 加载 ColQwen2 / ColPali ──
    print(f"\n[3] 加载 {args.visual_model} 视觉编码器...")
    t3 = time.time()
    visual_encoder = create_visual_encoder(model_name=args.visual_model)
    print(f"    加载耗时: {time.time() - t3:.1f}s")

    # ── Step 4: 取 N 条 query 做 MaxSim sanity ──
    print(f"\n[4] MaxSim sanity check ({args.n} queries, top-{args.k})...")
    sample_queries = []
    for i, q in enumerate(en_queries):
        qid = int(q["query_id"])
        if qid in qrel_map and len(qrel_map[qid]) > 0:
            sample_queries.append((qid, str(q["query"]), qrel_map[qid]))
        if len(sample_queries) >= args.n:
            break

    hits = 0
    scores = []
    for idx, (qid, query_text, relevant) in enumerate(sample_queries):
        t_q = time.time()
        q_emb = visual_encoder.encode_query(query_text)  # [1, n_q, 128]
        results = store.maxsim_search(q_emb, k=args.k)
        top_ids = {r["page_id"] for r in results}
        hit = bool(top_ids & relevant)
        hits += int(hit)
        top_score = results[0]["score"] if results else 0.0
        scores.append(top_score)

        elapsed = (time.time() - t_q) * 1000
        marker = "✅" if hit else "❌"
        print(f"    {marker} Q{idx} (id={qid}): \"{query_text[:80]}...\"")
        print(f"       relevant={relevant} top10={sorted(top_ids)[:10]} score={top_score:.4f} ({elapsed:.0f}ms)")

    # ── 结论 ──
    print(f"\n{'='*60}")
    print(f"[5] 结论")
    actual_n = len(sample_queries)
    print(f"    命中: {hits}/{actual_n} (hit@{args.k})")
    import statistics
    if scores:
        print(f"    Top-1 score: {statistics.mean(scores):.4f} (avg), range [{min(scores):.4f}, {max(scores):.4f}]")
    print(f"    总耗时: {time.time() - t_total:.0f}s")

    if actual_n == 0:
        print(f"\n    ⚠️  没有找到可验证的 query（English 且有 ground-truth）。")
        print(f"       请检查数据集是否正确加载。")
    elif hits / max(actual_n, 1) >= 0.6:
        print(f"\n    ✅ 结论: 索引/查询匹配。0.1564 的原因是 eval 错配了索引（路径/symlink）。")
        print(f"       修复方向: 确保 run_eval.py 加载正确的 colqwen2 索引路径，重新跑 eval 即可。")
    else:
        print(f"\n    ❌ 结论: 索引/查询不匹配或索引本身有问题。")
        print(f"       需要完整重建索引: rm -f page_embeddings_cache.pkl && python scripts/run_eval.py \\")
        print(f"           --visual-model {args.visual_model} --config-filter Visual_only --max-queries 50")


if __name__ == "__main__":
    main()
