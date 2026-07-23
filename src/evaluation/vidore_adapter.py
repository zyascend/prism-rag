"""ViDoRe 评测适配器

实现 PrismRAG 统一检索器，将检索管道包装成可调用接口。
"""

from __future__ import annotations

import logging
import hashlib
import unicodedata
from typing import List, Optional

import torch

from src.ingestion.encoders import BGEEmbedder, ColPaliEmbedder
from src.ingestion.text_chunker import TextChunker
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.expand import expand_neighbors
from src.retrieval.fusion import RRFFusion
from src.retrieval.hyde import HyDEGenerator
from src.retrieval.query_intent import apply_modality_boost, detect_query_intent
from src.retrieval.reranker import Reranker
from src.retrieval.visual_retriever import VisualRetriever
from src.retrieval.visual_router import VisualRouter, build_visual_router_from_config
from src.observability import get_tracer, get_collector
from src.store.faiss_store import FaissColPaliStore
from src.store.pgvector_store import PgVectorStore
from src.config import cfg
from src.cache.store import InMemoryLRUCache

logger = logging.getLogger(__name__)


class PrismRAGRetriever:
    """PrismRAG 统一检索器（vidore-benchmark 适配用）"""

    def __init__(
        self,
        pg_store: PgVectorStore,
        faiss_store: FaissColPaliStore,
        bge: BGEEmbedder,
        colpali: ColPaliEmbedder,
        chunker: TextChunker,
        bm25: BM25Retriever,
        dense: DenseRetriever,
        visual: VisualRetriever,
        fusion: RRFFusion,
        reranker: Reranker,
        hyde: Optional[HyDEGenerator] = None,
        zerank_reranker: Optional[Reranker] = None,
    ):
        self.pg = pg_store
        self.faiss = faiss_store
        self.bge = bge
        self.colpali = colpali
        self.chunker = chunker
        self.bm25 = bm25
        self.dense = dense
        self.visual = visual
        self.fusion = fusion
        self.reranker = reranker
        self.hyde = hyde
        self.zerank_reranker = zerank_reranker
        # Visual 按需路由（None = 关闭，调用方 use_visual 原样生效）
        # 测试里可能把 cfg 换成无 .get 的 stub，失败时关闭路由
        try:
            self.visual_router: Optional[VisualRouter] = build_visual_router_from_config(
                cfg.get
            )
        except Exception:
            self.visual_router = None
        # ── 检索缓存（L3 结果缓存）──
        self.index_version = 0  # 语料版本盐；语料变更时 +1，旧 key 天然失效
        self._cache: "InMemoryLRUCache | None" = None  # 惰性创建（受 cache.enabled 控制）
        # ── 答案缓存（L4 整次生成结果缓存）── 同样受 cache.enabled + index_version 盐控制
        self._answer_cache: "InMemoryLRUCache | None" = None  # 惰性创建

    def delete_document(self, doc_id: str) -> dict:
        """删除一份文档，保证三路不再返回其内容（修复 D2 正确性缺陷）。

        严格顺序（修复 D4 删除顺序脆弱）：
          1. 先取该 doc 的 chunk_id / page_id（pg 行删前取，避免丢引用）
          2. pg.delete_by_doc_id（真相源，已 commit）
          3. bm25.remove_chunks（修复 D2：已删内容不再进入 RRF/答案）
          4. faiss.delete_by_doc_id（墓碑，修复 D1：视觉向量不再参与 MaxSim）
          5. faiss.maybe_compact（墓碑占比高时物理回收）
        """
        chunk_ids = set(self.pg.get_chunk_ids_by_doc_id(doc_id))
        page_ids = self.pg.get_page_ids_by_doc_id(doc_id)

        deleted_rows = self.pg.delete_by_doc_id(doc_id)
        removed_bm25 = self.bm25.remove_chunks(chunk_ids)

        faiss_removed = self.faiss.delete_by_doc_id(doc_id)
        faiss_compacted = self.faiss.maybe_compact()

        # 文档删除 = 语料变更 → 失效检索缓存（index_version 盐变化，旧 key 天然查不到）
        self.invalidate_cache()

        return {
            "doc_id": doc_id,
            "pg_deleted_rows": deleted_rows,
            "bm25_removed": removed_bm25,
            "faiss_removed": faiss_removed,
            "faiss_compacted": faiss_compacted,
            "page_ids": page_ids,
        }

    def invalidate_cache(self) -> None:
        """失效检索缓存：递增 index_version 并清空内存缓存。

        语料任何变更（删除/重索引）后调用，确保旧缓存 key 因版本盐不匹配而失效，
        杜绝脏读。调用方：delete_document、POST /cache/invalidate（重索引后）。
        """
        self.index_version += 1
        if self._cache is not None:
            self._cache.clear()
        if self._answer_cache is not None:
            self._answer_cache.clear()

    def _safe_cfg(self, path: str) -> dict:
        """读取嵌套 dict 配置；测试 stub 无 .get 时返回 {}。"""
        try:
            val = cfg.get(path)
        except Exception:
            return {}
        return val if isinstance(val, dict) else {}

    def _expand_cfg(self) -> dict:
        """读取 neighbor_expand 配置；测试 stub 无 .get 时视为关闭。"""
        return self._safe_cfg("retrieval.neighbor_expand")

    def _boost_cfg(self) -> dict:
        return self._safe_cfg("retrieval.modality_boost")

    def _expand_cache_salt(self) -> str:
        """neighbor_expand + modality_boost 配置盐。"""
        parts = []
        ex = self._expand_cfg()
        if not ex.get("enabled"):
            parts.append("expand=off")
        else:
            parts.append(
                f"expand=on:{ex.get('mode', 'page')}:"
                f"{ex.get('max_extra', 2)}:{ex.get('stage', 'post_rerank')}"
            )
        mb = self._boost_cfg()
        if not mb.get("enabled"):
            parts.append("boost=off")
        else:
            parts.append(
                f"boost=on:{mb.get('table_score_bonus', 0.02)}:"
                f"{mb.get('image_score_bonus', 0.02)}:"
                f"{mb.get('force_visual_on_visual_intent', False)}"
            )
        return "|".join(parts)

    def _cache_key(
        self, query: str, k: int,
        use_bm25: bool, use_dense: bool, use_visual: bool, use_rerank: bool,
        visual_query_embedding: Optional[torch.Tensor], use_hyde: bool, reranker_type: str,
        expand_salt: str = "expand=off",
    ) -> str:
        """构造检索缓存 key：归一化 query + 全部检索开关 + reranker + index_version 盐。

        key 必须包含影响结果的所有维度，否则不同配置会串结果。
        index_version 保证语料变更后旧 key 自动失效（不依赖 TTL）。
        """
        norm = unicodedata.normalize("NFKC", query).lower().strip()
        norm = " ".join(norm.split())
        vr_mode = "off"
        if self.visual_router is not None:
            vr_mode = self.visual_router.mode
        parts = [
            f"q={norm}",
            f"k={k}",
            f"bm25={use_bm25}", f"dense={use_dense}", f"visual={use_visual}",
            f"rerank={use_rerank}", f"hyde={use_hyde}", f"rt={reranker_type}",
            f"vr={vr_mode}",
            expand_salt,
            f"v={self.index_version}",
        ]
        if visual_query_embedding is not None:
            try:
                tb = visual_query_embedding.detach().cpu().numpy().tobytes()
                parts.append("ve=" + hashlib.sha256(tb).hexdigest()[:16])
            except Exception:
                parts.append("ve=unknown")
        else:
            parts.append("ve=none")
        return "|".join(parts)

    def _maybe_expand(
        self, hits: list, *, stage: str, k: int, trace: dict
    ) -> list:
        """按配置在 pre/post_rerank 阶段做邻居 expand。"""
        ex = self._expand_cfg()
        if not ex.get("enabled"):
            return hits
        if (ex.get("stage") or "post_rerank") != stage:
            return hits
        mode = ex.get("mode") or "page"
        max_extra = int(ex.get("max_extra") or 2)
        # post：可略高于 k；pre：给 rerank 更多候选
        cap = max(k + max_extra * max(len(hits), 1), k) if stage == "post_rerank" else None
        expanded, info = expand_neighbors(
            hits, self.pg, mode=mode, max_extra=max_extra, cap=cap,
        )
        trace["neighbor_expand"] = info
        if stage == "post_rerank" and k:
            # 主 hits 优先，再附扩块；cap 必须 >k 否则 expand 永远被裁掉（Boot-CP 教训）
            primary_ids = {h.get("chunk_id") for h in hits}
            primary = [r for r in expanded if r.get("chunk_id") in primary_ids]
            extra = [r for r in expanded if r.get("chunk_id") not in primary_ids]
            total_cap = max(k + max_extra * min(len(primary), 5), k)
            expanded = (primary + extra)[:total_cap]
        return expanded

    def answer_cache_key(
        self, query: str, model: str, k_context: int, doc_id: Optional[str]
    ) -> str:
        """构造 L4 Answer 缓存 key：归一化 query + model + k_context + doc_id + index_version 盐。

        与 L3 不同，doc_id 影响最终答案（路由层在检索后做确定性后置过滤），必须纳入 key；
        index_version 保证语料变更后旧 key 自动失效（不依赖 TTL）。
        Self-RAG 开关/阈值变化也必须入 key，避免开/关串答案。
        """
        from src.generation.self_rag import self_rag_cache_salt
        from src.retrieval.crag import crag_cache_salt

        norm = unicodedata.normalize("NFKC", query).lower().strip()
        norm = " ".join(norm.split())
        parts = [
            f"q={norm}",
            f"model={model}",
            f"kctx={k_context}",
            f"doc={doc_id or '*'}",
            f"v={self.index_version}",
            self_rag_cache_salt(),
            crag_cache_salt(),
        ]
        return "|".join(parts)

    def search(
        self,
        query: str,
        k: int = 10,
        use_bm25: bool = True,
        use_dense: bool = True,
        use_visual: bool = True,
        use_rerank: bool = True,
        visual_query_embedding: Optional[torch.Tensor] = None,
        use_hyde: bool = False,
        reranker_type: str = "bge",
    ) -> List[dict]:
        """统一检索接口

        Args:
            query: 查询文本
            k: 返回 Top-k chunk
            use_bm25/dense/visual: 控制各路的开关（消融用）
            use_rerank: 是否使用 cross-encoder 重排
            visual_query_embedding: 可选，预编码的 visual query embedding。
                                    传入时 visual route 走 search_with_embedding() 跳过编码。
            use_hyde: 是否启用 HyDE 查询改写
            reranker_type: 重排器选择 ("bge" | "zerank")

        Returns:
            结果列表，每个 dict 含 chunk_id, page_id, score, retrieval_type 等
        """
        result = self.search_with_trace(
            query, k, use_bm25, use_dense, use_visual, use_rerank,
            visual_query_embedding=visual_query_embedding,
            use_hyde=use_hyde, reranker_type=reranker_type,
        )
        return result["results"]

    def search_with_trace(
        self,
        query: str,
        k: int = 10,
        use_bm25: bool = True,
        use_dense: bool = True,
        use_visual: bool = True,
        use_rerank: bool = True,
        visual_query_embedding: Optional[torch.Tensor] = None,
        use_hyde: bool = False,
        reranker_type: str = "bge",
        config_label: str = "",
    ) -> dict:
        """带 retrieval_trace 的统一检索接口

        Args:
            visual_query_embedding: 可选，预编码的 visual query embedding。
                                    传入时 visual route 走 search_with_embedding() 而非现场编码。
            use_hyde: 是否启用 HyDE 查询改写。启用时会额外用 LLM 生成假设文档
                      并作为额外查询送入 dense/visual 路线。
            reranker_type: 重排器选择 ("bge" | "zerank")
            config_label: 检索配置标签，用于 observability 聚合。若已有父 Trace
                         则继承其 label。

        Returns:
            {"results": [...], "retrieval_trace": {"bm25_top5": [...], "dense_top5": [...],
             "visual_top5": [...], "hyde": "<generated text>"}}
        """
        tracer = get_tracer()
        collector = get_collector()

        # 检测是否已有父 Trace（由 evaluate_generation 等上层创建）
        existing = tracer.current_trace()
        if existing is not None:
            # 挂载到已有 Trace，继承其 config_label
            owns_trace = False
        else:
            tracer.start_trace(query=query, config_label=config_label)
            owns_trace = True

        # ── 查询意图（B2，纯规则；默认不改变路径）──
        intent = detect_query_intent(query)
        boost_cfg = self._boost_cfg()
        force_visual = bool(
            boost_cfg.get("enabled")
            and boost_cfg.get("force_visual_on_visual_intent")
            and intent.visual
        )

        # ── Visual 按需路由（在缓存 key 之前解析 effective_visual）──
        effective_visual = use_visual
        visual_routed: bool | None = None
        if use_visual and self.visual_router is not None:
            effective_visual = self.visual_router.should_use_visual(query)
            visual_routed = effective_visual
        if force_visual:
            effective_visual = True
            visual_routed = True

        # ── L3 Retrieval Cache ───────────────────────────────
        cache_on = cfg.cache.enabled
        cache_key: str | None = None
        cached = None
        if cache_on:
            if self._cache is None:
                self._cache = InMemoryLRUCache(max_size=cfg.cache.max_size)
            cache_key = self._cache_key(
                query, k, use_bm25, use_dense, effective_visual, use_rerank,
                visual_query_embedding, use_hyde, reranker_type,
                expand_salt=self._expand_cache_salt(),
            )
            cached = self._cache.get(cache_key)
        if cached is not None:
            # 命中：记录 cache 事件；仅当我们拥有该 trace 时发出轻量 trace（供 GET /trace/{id} 可见）
            collector.record_cache_event("retrieval", hit=True, config_label=config_label)
            if owns_trace:
                with tracer.start_span("retrieval") as _span:
                    _span.set_metadata({
                        "cache_hit": True,
                        "cache_layer": "retrieval",
                        "num_results": len(cached.get("results", [])),
                        "visual_routed": visual_routed,
                    })
                _t = tracer.finish_trace()
                if _t:
                    collector.ingest_trace(_t)
            return cached

        routes = []
        trace = {
            "bm25_top5": [],
            "dense_top5": [],
            "visual_top5": [],
            "hyde": "",
            "visual_routed": visual_routed,
            "visual_routing_mode": (
                self.visual_router.mode if self.visual_router is not None else None
            ),
        }

        # ── HyDE: 生成假设文档 ────────────────────────────────
        hyde_answer = ""
        if use_hyde and self.hyde is not None:
            hyde_answer = self.hyde.generate(query)
            trace["hyde"] = hyde_answer[:500] if hyde_answer else "(failed)"

        # ── BM25 route（始终用原始 query）───────────────────
        if use_bm25:
            try:
                bm25_results = self.bm25.search(query, k=20)
                routes.append(bm25_results)
                trace["bm25_top5"] = [
                    {"chunk_id": r["chunk_id"], "page_id": r["page_id"], "score": r["score"]}
                    for r in bm25_results[:5]
                ]
            except RuntimeError:
                logger.warning("BM25 未就绪，跳过")

        # ── Dense route（原始 query + 可选 HyDE answer）─────
        if use_dense:
            dense_queries = [query]
            if hyde_answer:
                dense_queries.append(hyde_answer)

            all_dense = []
            for q in dense_queries:
                results = self.dense.search(q, k=20)
                routes.append(results)
                all_dense.extend(results)

            # Trace: 仅记录原始 query 的 top-5
            if all_dense:
                trace["dense_top5"] = [
                    {"chunk_id": r["chunk_id"], "page_id": r["page_id"], "score": r["score"]}
                    for r in all_dense[:5]
                ]

        # ── Visual route（原始 query + 可选 HyDE answer；可被 router 跳过）────
        if effective_visual:
            try:
                # 原始 query — 优先使用预编码 embedding
                if visual_query_embedding is not None:
                    vis_results = self.visual.search_with_embedding(visual_query_embedding, k=20)
                else:
                    vis_results = self.visual.search(query, k=20)
                routes.append(vis_results)

                # HyDE answer — 现场编码（可能因 ColPali 已卸载而失败，不影响原始结果）
                if hyde_answer:
                    try:
                        hyde_vis = self.visual.search(hyde_answer, k=20)
                        routes.append(hyde_vis)
                        vis_results = vis_results + hyde_vis
                    except Exception as e:
                        logger.debug(f"HyDE visual 跳过（ColPali 可能未加载）: {e}")

                trace["visual_top5"] = [
                    {"chunk_id": r["chunk_id"], "page_id": r["page_id"], "score": r["score"]}
                    for r in vis_results[:5]
                ]
            except Exception as e:
                logger.warning(f"Visual 检索跳过: {e}")

        if not routes:
            if owns_trace:
                obs_trace = tracer.finish_trace()
                if obs_trace:
                    collector.ingest_trace(obs_trace)
            result = {"results": [], "retrieval_trace": trace}
            if cache_on and self._cache is not None and cache_key is not None:
                self._cache.put(cache_key, result)
                collector.record_cache_event("retrieval", hit=False, config_label=config_label)
            return result

        fused = self.fusion.fuse(routes, k=min(k * 2, 40))
        # Phase B2：融合后、精排前按意图轻推 table/image chunk
        if boost_cfg.get("enabled"):
            fused = apply_modality_boost(
                fused,
                intent,
                table_bonus=float(boost_cfg.get("table_score_bonus") or 0.0),
                image_bonus=float(boost_cfg.get("image_score_bonus") or 0.0),
            )
            trace["query_intent"] = intent.label
            trace["modality_boost"] = True
        fused = self._maybe_expand(fused, stage="pre_rerank", k=k, trace=trace)

        # ── Rerank（支持双 reranker）────────────────────────
        if use_rerank and fused:
            reranker = self.zerank_reranker if reranker_type == "zerank" else self.reranker
            # Wrap fusion+rerank in a span
            with tracer.start_span("fusion_rerank") as fusion_span:
                reranked = reranker.rerank(query, fused, top_k=k)
                # Extract rerank score stats from results
                rerank_scores = [r.get("rerank_score", 0.0) for r in reranked]
                fusion_span.set_metadata({
                    "num_fused_input": len(fused),
                    "num_reranked_output": len(reranked),
                    "num_results": len(reranked),  # collector 据此追踪 fused hits
                    "max_rerank_score": round(max(rerank_scores), 4) if rerank_scores else 0.0,
                    "min_rerank_score": round(min(rerank_scores), 4) if rerank_scores else 0.0,
                    "mean_rerank_score": round(sum(rerank_scores) / len(rerank_scores), 4) if rerank_scores else 0.0,
                })
            reranked = self._maybe_expand(reranked, stage="post_rerank", k=k, trace=trace)
            if owns_trace:
                obs_trace = tracer.finish_trace()
                if obs_trace:
                    collector.ingest_trace(obs_trace)
            result = {"results": reranked, "retrieval_trace": trace}
            if cache_on and self._cache is not None and cache_key is not None:
                self._cache.put(cache_key, result)
                collector.record_cache_event("retrieval", hit=False, config_label=config_label)
            return result

        final = self._maybe_expand(fused[:k], stage="post_rerank", k=k, trace=trace)
        if owns_trace:
            obs_trace = tracer.finish_trace()
            if obs_trace:
                collector.ingest_trace(obs_trace)
        result = {"results": final, "retrieval_trace": trace}
        if cache_on and self._cache is not None and cache_key is not None:
            self._cache.put(cache_key, result)
            collector.record_cache_event("retrieval", hit=False, config_label=config_label)
        return result
