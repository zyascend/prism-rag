# tests/test_agent_supervise.py
"""Phase 2 supervisor：派单决策 + fallback + 配额接管 + arms 派发。"""
from src.agent.graph import (
    build_agent_graph,
    route_after_supervise,
)
from src.agent.state import empty_agent_state


def _cfg(**over):
    base = {
        "max_subqueries": 3,
        "max_total_searches": 3,
        "max_llm_calls": 6,
        "max_grade_cycles": 1,
        "grade": {"enabled": False},
        "hitl": {"review_subqueries": False},
        "checkpoint": {"enabled": False},
        "use_send": True,
        "supervise": {"enabled": True},
    }
    base.update(over)
    return base


def _sub_cfg():
    return {
        "max_subqueries": 3,
        "max_total_searches": 3,
        "max_llm_calls": 6,
        "max_grade_cycles": 1,
    }


def test_route_after_supervise_modes():
    # 有效 plan → 按 mode 路由
    assert route_after_supervise({"plan": {"mode": "atomic", "fallback": False}}) == "retrieve_one"
    assert route_after_supervise({"plan": {"mode": "multi", "fallback": False}}) == "retrieve_multi"
    # fallback plan → 用 decompose 的 strategy
    assert route_after_supervise({"plan": {"mode": "", "fallback": True}, "strategy": "multi"}) == "retrieve_multi"
    assert route_after_supervise({"plan": {"mode": "", "fallback": True}, "strategy": "atomic"}) == "retrieve_one"
    # 无 plan → 用 strategy
    assert route_after_supervise({"strategy": "atomic"}) == "retrieve_one"


def test_supervise_dispatch_valid_plan_arms_and_quota():
    """合法 plan：per-subquery arms 传给 knowledge_search，配额接管。"""
    arm_calls = {"bm25": [], "dense": [], "visual": []}

    def search_fn(q, k=5):
        # fallback 路径：全臂
        return [{"chunk_id": f"all-{q}", "text": q, "score": 1.0, "doc_id": "d"}]

    def make_arm(arm):
        def fn(q, k=5):
            arm_calls[arm].append(q)
            return [{"chunk_id": f"{arm}-{q}", "text": q, "score": 1.0, "doc_id": "d"}]
        return fn

    # 三臂独立注入：从「哪个臂被调了」验证派单
    search_fns = {"bm25": make_arm("bm25"), "dense": make_arm("dense"), "visual": make_arm("visual")}

    def complete_fn(p):
        if "subqueries" in p.lower():
            return (
                '{"subqueries": ["table q", "part q"], "strategy": "multi", '
                '"arm_hints": {"table q": "visual", "part q": "bm25"}}'
            )
        if "assignments" in p.lower():
            return (
                '{"mode": "multi", "assignments": ['
                '{"subquery": "table q", "arms": ["visual"], "searches": 1},'
                '{"subquery": "part q", "arms": ["bm25"], "searches": 1}'
                ']}'
            )
        return "{}"

    g = build_agent_graph(
        search_fn=search_fn,
        complete_fn=complete_fn,
        generate_fn=lambda q, h: {"answer": "ok", "citations": [], "rejected": False},
        cfg=_cfg(),
        search_fns=search_fns,
    )
    out = g.invoke(empty_agent_state("compare?", cfg=_sub_cfg()))
    assert out.get("answer") == "ok"
    # 派单：table q 只开 visual，part q 只开 bm25（dense 不应被调用）
    assert arm_calls["visual"] == ["table q"]
    assert arm_calls["bm25"] == ["part q"]
    assert arm_calls["dense"] == []
    # supervise trajectory 记录
    traj = out.get("trajectory") or []
    sup = [t for t in traj if t.get("node") == "supervise"]
    assert len(sup) == 1
    assert sup[0]["counts"]["fallback"] == 0
    assert sup[0]["counts"]["assignments"] == 2
    # plan 存进 state 供 trace 回放
    assert out["plan"]["fallback"] is False


def test_supervise_fallback_on_bad_json():
    """坏 JSON → fallback 规则计划：全臂检索、行为与关闭时一致。"""
    calls = []

    def search_fn(q, k=5):
        calls.append(q)
        return [{"chunk_id": "1", "text": q, "score": 1.0, "doc_id": "d"}]

    def complete_fn(p):
        if "subqueries" in p.lower():
            return '{"subqueries": ["a", "b"], "strategy": "multi"}'
        if "assignments" in p.lower():
            return "not json at all"  # supervisor 崩
        return "{}"

    g = build_agent_graph(
        search_fn=search_fn,
        complete_fn=complete_fn,
        generate_fn=lambda q, h: {"answer": "ok", "citations": [], "rejected": False},
        cfg=_cfg(),
    )
    out = g.invoke(empty_agent_state("multi?", cfg=_sub_cfg()))
    assert out.get("answer") == "ok"
    # fallback：plan.fallback=True，但全臂检索照常
    assert out.get("plan", {}).get("fallback") is True
    assert len(calls) == 2
    traj = out.get("trajectory") or []
    sup = [t for t in traj if t.get("node") == "supervise"]
    assert len(sup) == 1
    assert sup[0]["counts"]["fallback"] == 1


def test_supervise_quota_capped_by_budget():
    """plan 配额超预算 → 被 searches_left 硬截断（不超预算）。"""
    calls = []

    def search_fn(q, k=5, **kw):
        calls.append(q)
        return [{"chunk_id": "1", "text": q, "score": 1.0, "doc_id": "d"}]

    def complete_fn(p):
        if "subqueries" in p.lower():
            return '{"subqueries": ["a", "b", "c"], "strategy": "multi"}'
        if "assignments" in p.lower():
            return (
                '{"mode": "multi", "assignments": ['
                '{"subquery": "a", "arms": ["bm25"], "searches": 3},'
                '{"subquery": "b", "arms": ["bm25"], "searches": 3},'
                '{"subquery": "c", "arms": ["bm25"], "searches": 3}'
                ']}'
            )
        return "{}"

    g = build_agent_graph(
        search_fn=search_fn,
        complete_fn=complete_fn,
        generate_fn=lambda q, h: {"answer": "ok", "citations": [], "rejected": False},
        cfg=_cfg(max_total_searches=2, supervise={"enabled": True}),
    )
    out = g.invoke(
        empty_agent_state(
            "multi?",
            cfg={**_sub_cfg(), "max_total_searches": 2},
        )
    )
    # 总搜索数 ≤ 预算 2
    assert len(calls) <= 2
    assert (out.get("meta") or {}).get("searches", 0) <= 2


def test_supervise_skipped_when_disabled():
    """supervise.enabled=false → 无 supervise 节点、无 plan、行为与 Phase 1 一致。"""
    calls = []

    def search_fn(q, k=5):
        calls.append(q)
        return [{"chunk_id": "1", "text": q, "score": 1.0, "doc_id": "d"}]

    def complete_fn(p):
        if "subqueries" in p.lower():
            return '{"subqueries": ["a", "b"], "strategy": "multi"}'
        return "{}"

    g = build_agent_graph(
        search_fn=search_fn,
        complete_fn=complete_fn,
        generate_fn=lambda q, h: {"answer": "ok", "citations": [], "rejected": False},
        cfg=_cfg(supervise={"enabled": False}),
    )
    out = g.invoke(empty_agent_state("multi?", cfg=_sub_cfg()))
    traj = out.get("trajectory") or []
    assert not any(t.get("node") == "supervise" for t in traj)
    assert "plan" not in out or out.get("plan") in (None, {})


def test_supervise_respects_arm_hints_prior():
    """decompose 的 arm_hints 传给 supervise prompt（可被 LLM 覆盖，但作为先验）。"""
    seen_prompts = []

    def complete_fn(p):
        if "subqueries" in p.lower():
            return '{"subqueries": ["tbl"], "strategy": "multi", "arm_hints": {"tbl": "visual"}}'
        seen_prompts.append(p)
        return '{"mode": "multi", "assignments": [{"subquery": "tbl", "arms": ["visual"], "searches": 1}]}'

    def search_fn(q, k=5, **kw):
        return [{"chunk_id": "1", "text": q, "score": 1.0, "doc_id": "d"}]

    g = build_agent_graph(
        search_fn=search_fn,
        complete_fn=complete_fn,
        generate_fn=lambda q, h: {"answer": "ok", "citations": [], "rejected": False},
        cfg=_cfg(),
    )
    g.invoke(empty_agent_state("tbl?", cfg=_sub_cfg()))
    assert seen_prompts, "supervise 应被调用"
    assert '"tbl": "visual"' in seen_prompts[0] or "visual" in seen_prompts[0]


def test_arms_passthrough_opens_only_selected_arms():
    """arms 透传给 search_fn → retriever 只开被派的检索臂（根因修复验证）。

    这是云上 NO_GO 根因的修复：单臂注入下 supervise 的 arms 必须真实影响
    use_bm25/use_dense/use_visual，而不是被 _search_arms 架空成全臂。
    """
    from src.agent.tools import AgentToolBox

    calls = []

    class FakeRetriever:
        def search(self, q, k=5, use_bm25=True, use_dense=True, use_visual=True, use_rerank=True, **kw):
            calls.append(
                {
                    "q": q,
                    "use_bm25": use_bm25,
                    "use_dense": use_dense,
                    "use_visual": use_visual,
                }
            )
            return [{"chunk_id": "1", "text": q, "score": 1.0, "doc_id": "d"}]

    r = FakeRetriever()

    def search_fn(q, k=None, arms=None):
        # 与 eval.py agent_answer_for_eval.search_fn 同款映射
        kk = int(k) if k is not None else 5
        use_b = use_d = use_v = True
        if arms:
            use_b = "bm25" in arms
            use_d = "dense" in arms
            use_v = "visual" in arms
        return r.search(q, k=kk, use_bm25=use_b, use_dense=use_d, use_visual=use_v)

    box = AgentToolBox(
        search_fn=search_fn,
        complete_fn=lambda p: "{}",
        generate_fn=lambda q, h: {"answer": "", "citations": [], "rejected": True},
        cfg={},
    )
    # supervise 派 visual 臂 → 只开 visual
    box.knowledge_search("t", subquery_id=0, top_k=5, arms=["visual"])
    assert calls[-1] == {"q": "t", "use_bm25": False, "use_dense": False, "use_visual": True}
    # 派 bm25+dense → 关 visual
    box.knowledge_search("t", subquery_id=0, top_k=5, arms=["bm25", "dense"])
    assert calls[-1] == {"q": "t", "use_bm25": True, "use_dense": True, "use_visual": False}
    # 无 arms → 三路全开
    box.knowledge_search("t", subquery_id=0, top_k=5)
    assert calls[-1] == {"q": "t", "use_bm25": True, "use_dense": True, "use_visual": True}
