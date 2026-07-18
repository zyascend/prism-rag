"""检索缓存聚焦单测：LRU/TTL、collector 命中率、cache_key 归一化+版本盐、
search_with_trace 命中/未命中、全局开关关闭禁用缓存。

运行：cd <project> && .venv/bin/python tests/test_retrieval_cache.py
"""
import os
import sys

# 确保项目根在 sys.path（脚本直接运行时）
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.cache.store import InMemoryLRUCache
from src.observability import get_collector
from src.evaluation import vidore_adapter as va
from src.evaluation.vidore_adapter import PrismRAGRetriever


# ── 轻量 stub 检索组件 ──────────────────────────────────────────
class _StubBM25:
    def search(self, q, k=20):
        return [{"chunk_id": "c1", "page_id": 1, "score": 0.9}]


class _StubDense:
    def search(self, q, k=20):
        return [{"chunk_id": "c2", "page_id": 2, "score": 0.8}]


class _StubVisual:
    def search(self, q, k=20):
        return [{"chunk_id": "c3", "page_id": 3, "score": 0.7}]

    def search_with_embedding(self, emb, k=20):
        return [{"chunk_id": "c3", "page_id": 3, "score": 0.7}]


class _StubFusion:
    def fuse(self, routes, k=40):
        merged, seen, out = [], set(), []
        for r in routes:
            merged.extend(r)
        for r in merged:
            if r["chunk_id"] not in seen:
                seen.add(r["chunk_id"])
                out.append(r)
        return out[:k]


class _StubReranker:
    def rerank(self, q, fused, top_k=5):
        for i, r in enumerate(fused):
            r["rerank_score"] = round(1.0 - i * 0.1, 2)
        return fused[:top_k]


def _stub_retriever() -> PrismRAGRetriever:
    return PrismRAGRetriever(
        pg_store=None, faiss_store=None, bge=None, colpali=None, chunker=None,
        bm25=_StubBM25(), dense=_StubDense(), visual=_StubVisual(),
        fusion=_StubFusion(), reranker=_StubReranker(),
    )


# ── 1. LRU 基础 + 淘汰 ─────────────────────────────────────────
def test_lru_cache_basic():
    c = InMemoryLRUCache(max_size=2)
    c.put("a", 1)
    c.put("b", 2)
    assert c.get("a") == 1
    c.put("c", 3)  # 淘汰最久未用 b
    assert c.get("b") is None
    assert c.get("a") == 1
    assert c.get("c") == 3
    c.clear()
    assert c.get("a") is None


# ── 2. TTL（0/负数 = 不启用）────────────────────────────────────
def test_lru_cache_ttl_disabled():
    c = InMemoryLRUCache(max_size=10, ttl_seconds=0)
    c.put("x", 1)
    assert c.get("x") == 1
    c2 = InMemoryLRUCache(max_size=10, ttl_seconds=-5)
    c2.put("x", 1)
    assert c2.get("x") == 1


# ── 3. collector 缓存命中率聚合 ─────────────────────────────────
def test_collector_cache_hit_rate():
    col = get_collector()
    col.reset()
    col.record_cache_event("retrieval", hit=True, config_label="t1")
    col.record_cache_event("retrieval", hit=False, config_label="t1")
    col.record_cache_event("retrieval", hit=False, config_label="t1")
    # 模拟一次真实请求（带 latency），否则 get_config_metrics 因 num_queries=0 返回 None
    from src.observability import get_tracer
    get_tracer().start_trace(query="x", config_label="t1")
    fin = get_tracer().finish_trace()
    if fin:
        col.ingest_trace(fin)
    m = col.get_config_metrics("t1")
    assert m is not None
    assert abs(m.retrieval_cache_hit_rate - 1 / 3) < 1e-6
    assert "retrieval_cache_hit_rate" in m.to_dict()["cache"]


# ── 4. cache_key 归一化 + 版本盐 ────────────────────────────────
def test_cache_key_normalization_and_version():
    r = _stub_retriever()
    base = dict(
        k=5, use_bm25=True, use_dense=True, use_visual=True, use_rerank=True,
        visual_query_embedding=None, use_hyde=False, reranker_type="bge",
    )
    k1 = r._cache_key("How many pages?", **base)
    k2 = r._cache_key("how  many   pages?", **base)  # 归一化后等价
    assert k1 == k2
    k3 = r._cache_key("How many pages?", **{**base, "use_bm25": False})  # 开关不同
    assert k3 != k1
    r.invalidate_cache()  # 版本盐变化
    k4 = r._cache_key("How many pages?", **base)
    assert k4 != k1
    assert r.index_version == 1


# ── 5. search_with_trace 命中/未命中 ───────────────────────────
def test_search_with_trace_cache_hit_and_miss():
    r = _stub_retriever()
    col = get_collector()
    col.reset()
    res1 = r.search_with_trace("q-test-cache", k=5, config_label="api")
    res2 = r.search_with_trace("q-test-cache", k=5, config_label="api")
    assert res1 is res2  # 第二次命中缓存：同一对象
    m = col.get_config_metrics("api")
    assert m is not None
    assert abs(m.retrieval_cache_hit_rate - 0.5) < 1e-6


# ── 6. 全局开关关闭 → 不缓存 ───────────────────────────────────
def test_global_switch_disables_cache():
    class _FakeCacheCfg:
        enabled = False
        max_size = 2048
        ttl_seconds = 0

    class _FakeCfg:
        cache = _FakeCacheCfg()

    orig = va.cfg
    va.cfg = _FakeCfg()  # monkeypatch 模块级 cfg，search_with_trace 读取它
    try:
        r = _stub_retriever()
        res1 = r.search_with_trace("q-switch-off", k=5)
        res2 = r.search_with_trace("q-switch-off", k=5)
        assert res1 is not res2  # 关闭时每次重新检索，不复用缓存
    finally:
        va.cfg = orig


if __name__ == "__main__":
    tests = [
        test_lru_cache_basic,
        test_lru_cache_ttl_disabled,
        test_collector_cache_hit_rate,
        test_cache_key_normalization_and_version,
        test_search_with_trace_cache_hit_and_miss,
        test_global_switch_disables_cache,
    ]
    for t in tests:
        t()
        print(f"[OK] {t.__name__}")
    print(f"\nALL {len(tests)} CACHE TESTS PASSED")
