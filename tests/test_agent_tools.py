# tests/test_agent_tools.py
from src.agent.tools import (
    AgentToolBox,
    diversify_evidence_for_synthesis,
    normalize_subquery_list,
    normalize_subquery_text,
    parse_json_object,
    synthesis_k_context,
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


def test_synthesis_k_context_scales_with_subqueries():
    assert synthesis_k_context(1, base_k=5, per_subquery=2, max_k=12) == 5
    assert synthesis_k_context(2, base_k=5, per_subquery=2, max_k=12) == 5
    assert synthesis_k_context(3, base_k=5, per_subquery=2, max_k=12) == 6
    assert synthesis_k_context(3, base_k=5, per_subquery=3, max_k=12) == 9
    assert synthesis_k_context(10, base_k=5, per_subquery=2, max_k=12) == 12


def test_diversify_evidence_round_robin_not_first_subquery_only():
    """Phase2 root cause: plain [:5] kept only subquery 0 hits."""
    evidence = []
    for sid in (0, 1, 2):
        for i in range(5):
            evidence.append(
                {
                    "chunk_id": f"s{sid}-{i}",
                    "subquery_id": sid,
                    "text": f"side{sid} hit{i}",
                    "score": 1.0 - i * 0.01,
                }
            )
    # Naive slice would be all subquery 0
    naive = evidence[:5]
    assert all(h["subquery_id"] == 0 for h in naive)

    picked = diversify_evidence_for_synthesis(evidence, k=6)
    assert len(picked) == 6
    sids = {h["subquery_id"] for h in picked}
    assert sids == {0, 1, 2}
    # at least one hit per side in first 3 of round-robin
    assert {h["subquery_id"] for h in picked[:3]} == {0, 1, 2}


def test_synthesize_passes_diversified_hits_and_k_context():
    evidence = []
    for sid in (0, 1):
        for i in range(5):
            evidence.append(
                {
                    "chunk_id": f"s{sid}-{i}",
                    "subquery_id": sid,
                    "text": f"side{sid} content {i}",
                    "score": 0.9 - i * 0.05,
                    "doc_id": "d",
                }
            )
    seen = {}

    def generate_fn(q, hits, k_context=None):
        seen["n"] = len(hits)
        seen["k"] = k_context
        seen["sids"] = {h.get("subquery_id") for h in hits}
        return {
            "answer": "both sides covered",
            "citations": [{"chunk_id": h["chunk_id"]} for h in hits],
            "rejected": False,
        }

    box = AgentToolBox(
        search_fn=lambda q, k=5: [],
        complete_fn=lambda p: "{}",
        generate_fn=generate_fn,
        cfg={
            "synthesize_k_context": 5,
            "synthesize_per_subquery": 2,
            "synthesize_max_k": 12,
        },
    )
    out = box.synthesize_answer("contrast A and B?", evidence=evidence)
    assert out["rejected"] is False
    assert seen["sids"] == {0, 1}
    assert seen["n"] >= 4  # 2 sides × at least 2
    assert seen["k"] == seen["n"]
    assert out.get("n_evidence_used") == seen["n"]

