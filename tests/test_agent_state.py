# tests/test_agent_state.py
from src.agent.state import (
    empty_agent_state,
    merge_evidence,
    step_record,
)


def test_empty_state_shape():
    s = empty_agent_state("what is X?")
    assert s["query"] == "what is X?"
    assert s["subqueries"] == []
    assert s["evidence"] == []
    assert s["trajectory"] == []
    assert s["status"] == "ok"
    assert s["budget"]["searches_left"] >= 1


def test_merge_evidence_appends():
    a = [{"chunk_id": "1", "text": "a", "subquery_id": 0}]
    b = [{"chunk_id": "2", "text": "b", "subquery_id": 1}]
    assert merge_evidence(a, b) == a + b


def test_step_record_jsonable():
    rec = step_record(
        step=1,
        node="decompose",
        tool="decompose_query",
        input_summary="q",
        output_summary="2 subqs",
        ok=True,
        latency_ms=12.5,
        counts={"subqueries": 2},
    )
    import json
    json.dumps(rec)
    assert rec["node"] == "decompose"
