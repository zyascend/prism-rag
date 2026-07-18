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


# ── 7. K1 回归：fused 末路径 cache miss 必须计入正确 config_label ──
def test_retrieval_cache_miss_records_config_label():
    r = _stub_retriever()
    col = get_collector()
    col.reset()
    # use_rerank=False 命中 fused 末路径（K1 所在行），并传递明确 config_label
    r.search_with_trace("q-k1-miss", k=5, use_rerank=False, config_label="k1label")
    cd = col._cache_data
    # 关键回归：miss 事件落在传递的 "k1label"，而非默认 "api"
    assert "k1label" in cd and "retrieval" in cd["k1label"], "miss 事件未计入 k1label"
    assert cd["k1label"]["retrieval"]["misses"] == 1
    assert cd["k1label"]["retrieval"]["hits"] == 0
    # 默认 "api" 分组不应包含该事件（K1 修复前会错误落入）
    assert "api" not in cd or "retrieval" not in cd.get("api", {}), \
        "K1 未修复：miss 事件错误地落入默认 'api' 分组"


# ── 8. L4 answer_cache_key：确定性 + doc_id + 版本盐 ─────────────
def test_answer_cache_key_determinism_and_version():
    r = _stub_retriever()
    base = dict(model="gpt-4o-mini", k_context=5, doc_id=None)
    k1 = r.answer_cache_key("How many pages?", **base)
    k2 = r.answer_cache_key("how  many   pages?", **base)  # 归一化等价
    assert k1 == k2
    k3 = r.answer_cache_key("How many pages?", **{**base, "doc_id": "d1"})  # doc_id 不同
    assert k3 != k1
    r.invalidate_cache()  # 版本盐变化
    k4 = r.answer_cache_key("How many pages?", **base)
    assert k4 != k1
    assert r.index_version == 1


# ── 9. invalidate_cache 清理 L4 answer 缓存 ─────────────────────
def test_invalidate_clears_answer_cache():
    r = _stub_retriever()
    r._answer_cache = InMemoryLRUCache(max_size=10)
    r._answer_cache.put("ak", {"answer": "x"})
    assert r._answer_cache.get("ak") is not None
    r.invalidate_cache()
    assert r._answer_cache.get("ak") is None
    assert r.index_version == 1


# ── 10. collector answer 层命中率聚合 ───────────────────────────
def test_collector_answer_cache_hit_rate():
    col = get_collector()
    col.reset()
    col.record_cache_event("answer", hit=True, config_label="a1")
    col.record_cache_event("answer", hit=False, config_label="a1")
    # 模拟一次真实请求（带 latency），否则 get_config_metrics 因 num_queries=0 返回 None
    from src.observability import get_tracer
    get_tracer().start_trace(query="x", config_label="a1")
    fin = get_tracer().finish_trace()
    if fin:
        col.ingest_trace(fin)
    m = col.get_config_metrics("a1")
    assert m is not None
    assert abs(m.answer_cache_hit_rate - 0.5) < 1e-6
    assert "answer_cache_hit_rate" in m.to_dict()["cache"]


# ── 11. /ask L4 Answer 缓存命中（集成）：generator 仅调用一次 ──
def test_ask_l4_answer_cache():
    from fastapi.testclient import TestClient
    import src.api.routes as routes_mod

    # 复用真实 cfg（models.yaml 默认 cache.enabled=true），避免替换整个 Config 对象
    from src.config import cfg as _real_cfg
    assert _real_cfg.cache.enabled is True, "本测试依赖 cache.enabled=true（models.yaml 默认）"

    r = _stub_retriever()
    routes_mod.set_retriever(r)
    call_count = {"n": 0}

    class _StubGen:
        model = "stub-model"
        temperature = 0.0

        @property
        def cacheable(self):
            return self.temperature == 0.0

        def answer(self, query, retrieved, k_context=5):
            call_count["n"] += 1
            return {"answer": f"ans:{query}", "citations": [], "context": "ctx"}

    routes_mod.set_generator(_StubGen())
    try:
        get_collector().reset()
        client = TestClient(routes_mod.app)
        body = {"query": "q-l4", "k": 5, "use_rerank": True}
        resp1 = client.post("/ask", json=body)
        resp2 = client.post("/ask", json=body)
        assert resp1.status_code == 200 and resp2.status_code == 200
        assert resp1.json()["answer"] == resp2.json()["answer"] == "ans:q-l4"
        # L4 命中：第二次请求不再调用 generator
        assert call_count["n"] == 1, f"期望 L4 命中后 generator 仅调用 1 次，实际 {call_count['n']}"
        # 可观测：answer 层 1 miss + 1 hit
        cd = get_collector()._cache_data
        assert "" in cd and "answer" in cd[""], "answer 层事件未记录"
        assert cd[""]["answer"]["hits"] == 1 and cd[""]["answer"]["misses"] == 1
    finally:
        routes_mod.set_retriever(None)
        routes_mod.set_generator(None)


if __name__ == "__main__":
    tests = [
        test_lru_cache_basic,
        test_lru_cache_ttl_disabled,
        test_collector_cache_hit_rate,
        test_cache_key_normalization_and_version,
        test_search_with_trace_cache_hit_and_miss,
        test_global_switch_disables_cache,
        test_retrieval_cache_miss_records_config_label,
        test_answer_cache_key_determinism_and_version,
        test_invalidate_clears_answer_cache,
        test_collector_answer_cache_hit_rate,
        test_ask_l4_answer_cache,
    ]
    for t in tests:
        t()
        print(f"[OK] {t.__name__}")
    print(f"\nALL {len(tests)} CACHE TESTS PASSED")
