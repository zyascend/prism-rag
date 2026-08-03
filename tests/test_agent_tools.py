# tests/test_agent_tools.py
from src.agent.tools import (
    AgentToolBox,
    normalize_subquery_list,
    normalize_subquery_text,
    parse_json_object,
)


def test_parse_json_object_fence():
    raw = '```json\n{"subqueries": ["a", "b"], "strategy": "multi"}\n```'
    data = parse_json_object(raw)
    assert data["strategy"] == "multi"
    assert len(data["subqueries"]) == 2


def test_normalize_subquery_text_from_dict_and_stringified():
    assert normalize_subquery_text("plain q") == "plain q"
    assert (
        normalize_subquery_text({"query": "What is RAG?", "source": "paper"})
        == "What is RAG?"
    )
    # stringified JSON object (common model failure mode)
    s = '{"query": "What does RAG stand for?", "source": "RAG-Anything paper"}'
    assert normalize_subquery_text(s) == "What does RAG stand for?"
    # stringified Python dict (single quotes)
    s2 = "{'query': \"Who are the authors?\", 'strategy': 'atomic'}"
    assert normalize_subquery_text(s2) == "Who are the authors?"
    assert normalize_subquery_list(
        [
            {"query": "a"},
            "{'query': 'b', 'source': 'x'}",
            "c",
            "a",  # de-dupe
        ],
        max_n=5,
    ) == ["a", "b", "c"]


def test_decompose_cleans_object_shaped_subqueries():
    payload = (
        '{"subqueries": ['
        '{"query": "What does RAG stand for?", "source": "paper"}, '
        '{"query": "What is dual-graph construction?", "reason": "multi"}'
        '], "strategy": "multi", "reason": "x"}'
    )
    box = AgentToolBox(
        search_fn=lambda q, k=5: [],
        complete_fn=lambda p: payload,
        generate_fn=lambda q, hits: {"answer": "x", "citations": [], "rejected": False},
        cfg={"max_subqueries": 3},
        prompt_get_active=lambda pid: type(
            "P",
            (),
            {"render": staticmethod(lambda *a, **k: "prompt")},
        )(),
    )
    out = box.decompose_query("What is RAG dual-graph?")
    assert out["fallback"] is False
    assert out["strategy"] == "multi"
    assert out["subqueries"] == [
        "What does RAG stand for?",
        "What is dual-graph construction?",
    ]
    # none should look like a dumped dict
    for sq in out["subqueries"]:
        assert not sq.startswith("{")
        assert "source" not in sq


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
    seen = {}

    def search_fn(q, k=5):
        seen["q"] = q
        return hits_in

    box = AgentToolBox(
        search_fn=search_fn,
        complete_fn=lambda p: "{}",
        generate_fn=lambda q, hits: {"answer": "", "citations": [], "rejected": True},
    )
    dirty = "{'query': 'clean search text', 'source': 'x'}"
    out = box.knowledge_search(dirty, subquery_id=2, top_k=5)
    assert out["hits"][0]["subquery_id"] == 2
    assert out["hits"][0]["chunk_id"] == "c1"
    assert seen["q"] == "clean search text"
    assert out["query"] == "clean search text"


def test_synthesize_empty_evidence_rejects():
    box = AgentToolBox(
        search_fn=lambda q, k=5: [],
        complete_fn=lambda p: "{}",
        generate_fn=lambda q, hits: {"answer": "should not call", "citations": [], "rejected": False},
    )
    out = box.synthesize_answer("q", evidence=[])
    assert out["rejected"] is True
    assert out["answer"]  # 拒答文案非空

