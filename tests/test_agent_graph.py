# tests/test_agent_graph.py
from src.agent.graph import build_agent_graph, route_after_decompose, route_after_grade
from src.agent.state import empty_agent_state


def test_route_atomic():
    assert route_after_decompose({"strategy": "atomic", "subqueries": ["q"]}) == "retrieve_one"
    assert route_after_decompose({"strategy": "multi", "subqueries": ["a", "b"]}) == "retrieve_multi"


def test_route_grade():
    st = {"grade": {"sufficient": True}, "budget": {"grade_cycles_left": 1, "searches_left": 2}}
    assert route_after_grade(st) == "synthesize"
    st2 = {
        "grade": {"sufficient": False},
        "budget": {"grade_cycles_left": 1, "searches_left": 2},
        "meta": {},
    }
    assert route_after_grade(st2) == "refine"
    st3 = {
        "grade": {"sufficient": False},
        "budget": {"grade_cycles_left": 0, "searches_left": 2},
    }
    assert route_after_grade(st3) in ("synthesize", "abstain_or_synthesize")


def test_graph_compile_and_invoke_atomic():
    def search_fn(q, k=5):
        return [{"chunk_id": "1", "text": f"fact about {q}", "score": 1.0, "doc_id": "d"}]

    def complete_fn(p):
        if "Split" in p or "sub-queries" in p or "subqueries" in p.lower():
            return '{"subqueries": ["what is torque?"], "strategy": "atomic", "reason": "simple"}'
        if "sufficient" in p.lower() or "Grade" in p or "evidence" in p.lower():
            return '{"sufficient": true, "missing": "", "score": 0.9}'
        return "{}"

    def generate_fn(q, hits):
        return {"answer": "42 Nm", "citations": [{"chunk_id": "1"}], "rejected": False}

    graph = build_agent_graph(
        search_fn=search_fn,
        complete_fn=complete_fn,
        generate_fn=generate_fn,
        cfg={
            "enabled": True,
            "max_subqueries": 3,
            "max_total_searches": 3,
            "max_search_per_subquery": 1,
            "max_llm_calls": 6,
            "max_grade_cycles": 1,
            "grade": {"enabled": True},
            "hitl": {"review_subqueries": False},
            "checkpoint": {"enabled": False},
        },
    )
    out = graph.invoke(empty_agent_state("what is torque?", cfg={
        "max_subqueries": 3, "max_total_searches": 3, "max_llm_calls": 6, "max_grade_cycles": 1,
    }))
    assert out["answer"]
    assert out["status"] in ("ok", "abstain")
    assert any(t.get("node") == "decompose" for t in out.get("trajectory") or [])


def test_search_budget_caps_calls():
    calls = {"n": 0}

    def search_fn(q, k=5):
        calls["n"] += 1
        return [{"chunk_id": str(calls["n"]), "text": q, "score": 1.0, "doc_id": "d"}]

    def complete_fn(p):
        return '{"subqueries": ["a", "b", "c"], "strategy": "multi", "reason": "x"}'

    def generate_fn(q, hits):
        return {"answer": "ok", "citations": [], "rejected": False}

    g = build_agent_graph(
        search_fn=search_fn,
        complete_fn=complete_fn,
        generate_fn=generate_fn,
        cfg={
            "max_subqueries": 3,
            "max_total_searches": 2,
            "max_search_per_subquery": 1,
            "max_llm_calls": 6,
            "max_grade_cycles": 0,
            "grade": {"enabled": False},
            "hitl": {"review_subqueries": False},
            "checkpoint": {"enabled": False},
        },
    )
    st = empty_agent_state(
        "q",
        cfg={
            "max_subqueries": 3,
            "max_total_searches": 2,
            "max_llm_calls": 6,
            "max_grade_cycles": 0,
        },
    )
    g.invoke(st)
    assert calls["n"] <= 2


def test_multi_uses_send_or_equivalent_n_searches():
    """N 个子问应触发 N 次 search（受 budget 限制）。"""
    calls = []

    def search_fn(q, k=5):
        calls.append(q)
        return [{"chunk_id": q, "text": q, "score": 1.0, "doc_id": "d"}]

    def complete_fn(p):
        if "subqueries" in p.lower() or "Split" in p or "sub-queries" in p:
            return '{"subqueries": ["q1", "q2"], "strategy": "multi", "reason": "m"}'
        return '{"sufficient": true, "missing": "", "score": 1}'

    g = build_agent_graph(
        search_fn=search_fn,
        complete_fn=complete_fn,
        generate_fn=lambda q, h: {"answer": "a", "citations": [], "rejected": False},
        cfg={
            "max_subqueries": 3,
            "max_total_searches": 3,
            "max_llm_calls": 6,
            "max_grade_cycles": 1,
            "grade": {"enabled": False},
            "hitl": {"review_subqueries": False},
            "checkpoint": {"enabled": False},
            "use_send": True,
        },
    )
    from src.agent.state import empty_agent_state

    out = g.invoke(
        empty_agent_state(
            "combo?",
            cfg={
                "max_subqueries": 3,
                "max_total_searches": 3,
                "max_llm_calls": 6,
                "max_grade_cycles": 1,
            },
        )
    )
    assert len(calls) == 2
    assert set(calls) == {"q1", "q2"}
    # evidence merged from both workers
    assert len(out.get("evidence") or []) == 2
    assert (out.get("meta") or {}).get("use_send") is True
    assert any(t.get("node") == "retrieval_worker" for t in out.get("trajectory") or [])
    assert any(t.get("node") == "prepare_multi" for t in out.get("trajectory") or [])
