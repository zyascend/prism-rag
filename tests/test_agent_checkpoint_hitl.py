# tests/test_agent_checkpoint_hitl.py
"""P1c: MemorySaver checkpoint, HITL interrupt/resume, stream_agent."""
from __future__ import annotations

import pytest

from src.agent.checkpoint import get_memory_saver, reset_memory_saver
from src.agent.runner import run_agent, resume_agent, stream_agent


def _cfg(**overrides):
    base = {
        "enabled": True,
        "hitl": {"review_subqueries": True},
        "checkpoint": {"enabled": True},
        "grade": {"enabled": False},
        "max_subqueries": 3,
        "max_total_searches": 3,
        "max_llm_calls": 6,
        "max_grade_cycles": 0,
        "on_error": "error",
        "return_trajectory": True,
    }
    # shallow override + nested
    for k, v in overrides.items():
        if k in ("hitl", "checkpoint", "grade") and isinstance(v, dict):
            base[k] = {**base.get(k, {}), **v}
        else:
            base[k] = v
    return base


def _search(q, k=5):
    return [{"chunk_id": "1", "text": f"hit:{q}", "score": 1.0, "doc_id": "d"}]


def _complete_multi(p):
    return '{"subqueries": ["a", "b"], "strategy": "multi", "reason": "m"}'


def _generate(q, h):
    return {"answer": "done", "citations": [{"chunk_id": "1"}], "rejected": False}


@pytest.fixture(autouse=True)
def _clean_saver():
    """Isolate MemorySaver between tests (same process singleton)."""
    reset_memory_saver()
    yield
    reset_memory_saver()


def test_interrupt_and_resume():
    cfg = _cfg()
    res = run_agent(
        "multi hop?",
        search_fn=_search,
        complete_fn=_complete_multi,
        generate_fn=_generate,
        cfg=cfg,
        trace_id="thread-test-1",
    )
    assert res.status == "interrupted"
    assert res.thread_id == "thread-test-1"
    assert len(res.subqueries) == 2
    assert set(res.subqueries) == {"a", "b"}
    assert not res.answer  # paused before synthesize

    res2 = resume_agent(
        thread_id="thread-test-1",
        approved_subqueries=["a", "b"],
        search_fn=_search,
        complete_fn=_complete_multi,
        generate_fn=_generate,
        cfg=cfg,
    )
    assert res2.status in ("ok", "abstain")
    assert res2.answer
    assert res2.thread_id == "thread-test-1"
    assert any(t.get("node") == "hitl_review" for t in res2.trajectory)


def test_resume_with_revised_subqueries():
    cfg = _cfg()
    res = run_agent(
        "multi hop?",
        search_fn=_search,
        complete_fn=_complete_multi,
        generate_fn=_generate,
        cfg=cfg,
        trace_id="thread-revise-1",
    )
    assert res.status == "interrupted"

    searches = []

    def tracking_search(q, k=5):
        searches.append(q)
        return _search(q, k)

    res2 = resume_agent(
        thread_id="thread-revise-1",
        approved_subqueries=["revised-only"],
        search_fn=tracking_search,
        complete_fn=_complete_multi,
        generate_fn=_generate,
        cfg=cfg,
    )
    assert res2.status in ("ok", "abstain")
    assert res2.subqueries == ["revised-only"]
    assert searches == ["revised-only"]


def test_stream_agent_yields_events():
    cfg = _cfg(hitl={"review_subqueries": False}, checkpoint={"enabled": False})
    events = list(
        stream_agent(
            "what is X?",
            search_fn=_search,
            complete_fn=lambda p: (
                '{"subqueries": ["what is X?"], "strategy": "atomic", "reason": "a"}'
            ),
            generate_fn=_generate,
            cfg=cfg,
            trace_id="stream-1",
        )
    )
    assert len(events) >= 1
    # multi-mode yields (mode, payload) tuples
    modes = {e[0] for e in events if isinstance(e, tuple) and e}
    assert "updates" in modes or "values" in modes or any(
        isinstance(e, dict) for e in events
    )


def test_checkpoint_singleton_shared():
    a = get_memory_saver()
    b = get_memory_saver()
    assert a is b
    reset_memory_saver()
    c = get_memory_saver()
    assert c is not a


def test_export_graph_mermaid(tmp_path):
    from src.agent.graph import export_graph_mermaid, sanitize_langgraph_mermaid

    path = tmp_path / "agent-graph.mmd"
    mermaid = export_graph_mermaid(str(path))
    assert "decompose" in mermaid
    assert path.is_file()
    body = path.read_text(encoding="utf-8")
    assert "decompose" in body
    # Portable for VS Code / Cursor / common Mermaid SVG renderers
    assert "<p>" not in body
    assert "</p>" not in body
    assert "&nbsp;" not in body
    assert "line-height" not in body

    dirty = (
        "graph TD;\n"
        "__start__([<p>__start__</p>]):::first\n"
        "A -. &nbsp;label&nbsp; .-> B;\n"
        "classDef first fill-opacity:0\n"
        "classDef default fill:#f2f0ff,line-height:1.2\n"
    )
    clean = sanitize_langgraph_mermaid(dirty)
    assert "<p>" not in clean
    assert "&nbsp;" not in clean
    assert "line-height" not in clean
    assert "label" in clean


def test_hitl_off_no_interrupt():
    """Default HITL off: full run completes without interrupted status."""
    cfg = _cfg(hitl={"review_subqueries": False}, checkpoint={"enabled": False})
    res = run_agent(
        "multi hop?",
        search_fn=_search,
        complete_fn=_complete_multi,
        generate_fn=_generate,
        cfg=cfg,
        trace_id="no-hitl",
    )
    assert res.status != "interrupted"
    assert res.status in ("ok", "abstain")
    assert res.answer
