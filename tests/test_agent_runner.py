# tests/test_agent_runner.py
from src.agent.checkpoint import reset_memory_saver
from src.agent.runner import (
    AgentResult,
    merge_agent_cfg,
    new_thread_id,
    run_agent,
)


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


def test_new_thread_id_unique():
    ids = {new_thread_id() for _ in range(20)}
    assert len(ids) == 20
    assert all(i.startswith("agent-") for i in ids)


def _mock_search(q, k=5):
    # one hit per call — evidence_n should stay ~1 per isolated run
    return [
        {
            "chunk_id": f"c-{q}",
            "text": f"fact about {q}",
            "score": 1.0,
            "doc_id": "d1",
        }
    ]


def _mock_complete(_p):
    return '{"subqueries": ["atomic q"], "strategy": "atomic", "reason": "a"}'


def _mock_generate(q, hits):
    return {
        "answer": f"ans:{q}",
        "citations": [{"chunk_id": hits[0]["chunk_id"]}] if hits else [],
        "rejected": False,
    }


def test_batch_isolation_no_evidence_leak_with_checkpoint():
    """Regression: shared thread_id + MemorySaver merged evidence across questions.

    Default run_agent must isolate each invoke so evidence_n / trajectory
    do not grow across a dual-arm batch (Phase2 NO_GO root cause).
    """
    reset_memory_saver()
    cfg = _base_cfg(
        checkpoint={"enabled": True},  # the production default that leaked
        grade={"enabled": False},
        max_grade_cycles=0,
    )

    evidence_ns = []
    thread_ids = []
    for i in range(3):
        res = run_agent(
            f"query-{i}",
            search_fn=_mock_search,
            complete_fn=_mock_complete,
            generate_fn=_mock_generate,
            cfg=cfg,
            # no trace_id — must auto-allocate unique thread ids
        )
        assert res.status in ("ok", "abstain")
        evidence_ns.append(int(res.counts.get("evidence_n") or 0))
        thread_ids.append(res.thread_id)
        decomp = [t for t in res.trajectory if t.get("node") == "decompose"]
        assert len(decomp) == 1, f"trajectory polluted: {res.trajectory}"

    assert len(set(thread_ids)) == 3
    # each run: one search → one hit (not 1,2,3… cumulative)
    assert evidence_ns[0] == evidence_ns[1] == evidence_ns[2]
    assert evidence_ns[0] >= 1
    assert evidence_ns[0] <= 3  # budget headroom, never double-digit leak
    reset_memory_saver()


def test_shared_thread_id_still_leaks_trajectory_documenting_pitfall():
    """Same thread_id + checkpoint still accumulates trajectory — do not reuse in batch.

    Phase 2 (supervise branch): invoke_subgraph now strips the evidence echo, so
    evidence_n no longer grows (was the NO_GO root cause). Trajectory accumulation
    remains a LangGraph checkpoint behavior — the reason run_agent allocates a
    unique thread_id per request.
    """
    reset_memory_saver()
    cfg = _base_cfg(checkpoint={"enabled": True}, grade={"enabled": False})
    ns = []
    traj_lens = []
    for i in range(3):
        res = run_agent(
            f"shared-{i}",
            search_fn=_mock_search,
            complete_fn=_mock_complete,
            generate_fn=_mock_generate,
            cfg=cfg,
            trace_id="fixed-batch-id",  # intentional anti-pattern
        )
        ns.append(int(res.counts.get("evidence_n") or 0))
        traj_lens.append(len(res.trajectory))
    # evidence 泄漏已修复（invoke_subgraph 剥回显）——不再增长
    assert ns[0] == ns[1] == ns[2], f"evidence should not leak under shared thread_id, got {ns}"
    # trajectory 仍在累积（checkpoint reducer 叠加）——这才是必须唯一 thread_id 的原因
    assert traj_lens[-1] > traj_lens[0], f"expected trajectory accumulation, got {traj_lens}"
    reset_memory_saver()


def test_agent_answer_for_eval_batch_isolation():
    """agent_answer_for_eval must not leak evidence across sequential calls."""
    from src.agent.eval import agent_answer_for_eval

    class Ret:
        def search(self, query, k=5, **kwargs):
            return _mock_search(query, k=k)

    class Gen:
        def complete(self, prompt: str) -> str:
            return _mock_complete(prompt)

        def answer(self, query, hits, k_context=5):
            return _mock_generate(query, hits)

    reset_memory_saver()
    ret, gen = Ret(), Gen()
    ns = []
    for i in range(3):
        out = agent_answer_for_eval(
            f"eval-q-{i}",
            retriever=ret,
            generator=gen,
            k_context=5,
            cfg={"grade": {"enabled": False}, "max_grade_cycles": 0},
        )
        ns.append(int((out.get("agent") or {}).get("counts", {}).get("evidence_n") or 0))
        traj = (out.get("agent") or {}).get("trajectory") or []
        assert sum(1 for t in traj if t.get("node") == "decompose") == 1
    assert ns[0] == ns[1] == ns[2]
    reset_memory_saver()
