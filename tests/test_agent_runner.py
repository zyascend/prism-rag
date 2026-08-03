# tests/test_agent_runner.py
from src.agent.runner import run_agent, AgentResult, merge_agent_cfg


def _base_cfg(**overrides):
    cfg = {
        "enabled": True,
        "grade": {"enabled": False},
        "max_total_searches": 3,
        "max_subqueries": 3,
        "max_llm_calls": 6,
        "max_grade_cycles": 0,
        "checkpoint": {"enabled": False},
        "hitl": {"review_subqueries": False},
        "on_error": "abstain",
        "return_trajectory": True,
    }
    cfg.update(overrides)
    return cfg


def test_run_agent_ok():
    res = run_agent(
        "what is X?",
        search_fn=lambda q, k=5: [
            {"chunk_id": "1", "text": "X is 1", "score": 1.0, "doc_id": "d"}
        ],
        complete_fn=lambda p: (
            '{"subqueries": ["what is X?"], "strategy": "atomic", "reason": "a"}'
        ),
        generate_fn=lambda q, hits: {
            "answer": "1",
            "citations": [{"chunk_id": "1"}],
            "rejected": False,
        },
        cfg=_base_cfg(),
    )
    assert isinstance(res, AgentResult)
    assert res.status in ("ok", "abstain")
    assert res.answer
    assert res.trajectory is not None
    assert "searches" in res.counts
    assert "llm_calls" in res.counts
    assert "evidence_n" in res.counts


def test_run_agent_return_trajectory_false():
    res = run_agent(
        "what is X?",
        search_fn=lambda q, k=5: [
            {"chunk_id": "1", "text": "X is 1", "score": 1.0, "doc_id": "d"}
        ],
        complete_fn=lambda p: (
            '{"subqueries": ["what is X?"], "strategy": "atomic", "reason": "a"}'
        ),
        generate_fn=lambda q, hits: {
            "answer": "1",
            "citations": [{"chunk_id": "1"}],
            "rejected": False,
        },
        cfg=_base_cfg(return_trajectory=False),
    )
    assert res.trajectory == []


def test_run_agent_on_error_abstain():
    def boom_search(q, k=5):
        raise RuntimeError("search exploded")

    res = run_agent(
        "q",
        search_fn=boom_search,
        complete_fn=lambda p: (
            '{"subqueries": ["q"], "strategy": "atomic", "reason": "a"}'
        ),
        generate_fn=lambda q, hits: {"answer": "x", "citations": [], "rejected": False},
        cfg=_base_cfg(on_error="abstain"),
    )
    assert res.status == "error"
    assert res.error and "exploded" in res.error
    assert res.answer  # abstain message


def test_run_agent_on_error_degrade_pipeline():
    def boom_search(q, k=5):
        raise RuntimeError("search exploded")

    res = run_agent(
        "q",
        search_fn=boom_search,
        complete_fn=lambda p: (
            '{"subqueries": ["q"], "strategy": "atomic", "reason": "a"}'
        ),
        generate_fn=lambda q, hits: {"answer": "x", "citations": [], "rejected": False},
        cfg=_base_cfg(on_error="degrade_pipeline"),
        pipeline_fallback_fn=lambda: {
            "answer": "fallback-ans",
            "citations": [{"chunk_id": "fb"}],
        },
    )
    assert res.status == "degraded"
    assert res.answer == "fallback-ans"
    assert res.citations and res.citations[0]["chunk_id"] == "fb"
    assert res.error and "exploded" in res.error


def test_run_agent_on_error_reraise():
    def boom_search(q, k=5):
        raise RuntimeError("search exploded")

    try:
        run_agent(
            "q",
            search_fn=boom_search,
            complete_fn=lambda p: (
                '{"subqueries": ["q"], "strategy": "atomic", "reason": "a"}'
            ),
            generate_fn=lambda q, hits: {
                "answer": "x",
                "citations": [],
                "rejected": False,
            },
            cfg=_base_cfg(on_error="raise"),
        )
        assert False, "expected raise"
    except RuntimeError as e:
        assert "exploded" in str(e)


def test_merge_agent_cfg_deep_nested():
    base = {
        "enabled": False,
        "max_llm_calls": 6,
        "grade": {"enabled": True},
        "hitl": {"review_subqueries": False},
        "checkpoint": {"enabled": True},
    }
    ov = {
        "enabled": True,
        "grade": {"enabled": False},
        "max_total_searches": 2,
    }
    m = merge_agent_cfg(base, ov)
    assert m["enabled"] is True
    assert m["grade"] == {"enabled": False}
    assert m["hitl"] == {"review_subqueries": False}
    assert m["checkpoint"] == {"enabled": True}
    assert m["max_total_searches"] == 2
    assert m["max_llm_calls"] == 6


def test_answer_cache_key_includes_agent_salt(monkeypatch):
    """L4 key must change when agent_cache_salt changes."""
    from src.evaluation.vidore_adapter import PrismRAGRetriever

    r = PrismRAGRetriever.__new__(PrismRAGRetriever)
    r.index_version = "v-test"

    salts = iter(["ag=off", "ag=on|msq=3|mts=3|mlc=6|mgc=1|gr=1|hitl=0"])

    def fake_salt():
        return next(salts)

    import src.agent.config as ac

    monkeypatch.setattr(ac, "agent_cache_salt", fake_salt)
    k1 = r.answer_cache_key("how many?", "m", 5, None)
    k2 = r.answer_cache_key("how many?", "m", 5, None)
    assert k1 != k2
    assert "ag=off" in k1
    assert "ag=on" in k2
