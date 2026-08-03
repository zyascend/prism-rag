# tests/test_agent_tools.py
from src.agent.tools import AgentToolBox, parse_json_object


def test_parse_json_object_fence():
    raw = '```json\n{"subqueries": ["a", "b"], "strategy": "multi"}\n```'
    data = parse_json_object(raw)
    assert data["strategy"] == "multi"
    assert len(data["subqueries"]) == 2


def test_decompose_fallback_on_bad_json():
    box = AgentToolBox(
        search_fn=lambda q, k=5: [],
        complete_fn=lambda p: "NOT JSON",
        generate_fn=lambda q, hits: {"answer": "x", "citations": [], "rejected": False},
        cfg={"max_subqueries": 3, "max_total_searches": 3},
    )
    out = box.decompose_query("simple question?")
    assert out["subqueries"] == ["simple question?"]
    assert out["strategy"] == "atomic"
    assert out.get("fallback") is True


def test_knowledge_search_tags_subquery_id():
    hits_in = [{"chunk_id": "c1", "text": "hello", "score": 0.9, "doc_id": "d1"}]
    box = AgentToolBox(
        search_fn=lambda q, k=5: hits_in,
        complete_fn=lambda p: "{}",
        generate_fn=lambda q, hits: {"answer": "", "citations": [], "rejected": True},
    )
    out = box.knowledge_search("q1", subquery_id=2, top_k=5)
    assert out["hits"][0]["subquery_id"] == 2
    assert out["hits"][0]["chunk_id"] == "c1"


def test_synthesize_empty_evidence_rejects():
    box = AgentToolBox(
        search_fn=lambda q, k=5: [],
        complete_fn=lambda p: "{}",
        generate_fn=lambda q, hits: {"answer": "should not call", "citations": [], "rejected": False},
    )
    out = box.synthesize_answer("q", evidence=[])
    assert out["rejected"] is True
    assert out["answer"]  # 拒答文案非空
