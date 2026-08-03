"""Unit tests for agent eval helper (mocked retriever/generator — no GPU)."""
from __future__ import annotations

import json
from pathlib import Path

from src.agent.eval import agent_answer_for_eval, load_agent_eval_qa


class _FakeRetriever:
    def __init__(self, hits=None):
        self.hits = hits or [
            {
                "chunk_id": "c1",
                "page_id": "p1",
                "doc_id": "d1",
                "text": "X is 42. Shield braid clamp evenly distributed.",
                "score": 1.0,
            }
        ]
        self.calls = []

    def search(self, query, k=5, **kwargs):
        self.calls.append({"query": query, "k": k, **kwargs})
        return list(self.hits)[:k]


class _FakeGenerator:
    def __init__(self):
        self.complete_prompts = []
        self.answer_calls = []

    def complete(self, prompt: str) -> str:
        self.complete_prompts.append(prompt)
        # force atomic strategy for stable graph path
        return json.dumps(
            {
                "subqueries": ["what is X?"],
                "strategy": "atomic",
                "reason": "single fact",
            }
        )

    def answer(self, query, hits, k_context=5):
        self.answer_calls.append({"query": query, "n_hits": len(hits), "k": k_context})
        return {
            "answer": "42",
            "citations": [{"chunk_id": h.get("chunk_id")} for h in hits[:k_context]],
            "context": "\n".join(h.get("text") or "" for h in hits[:k_context]),
        }


def test_agent_answer_for_eval_shape():
    ret = _FakeRetriever()
    gen = _FakeGenerator()
    out = agent_answer_for_eval(
        "what is X?",
        retriever=ret,
        generator=gen,
        k_context=5,
        cfg={
            "grade": {"enabled": False},
            "max_grade_cycles": 0,
            "checkpoint": {"enabled": False},
            "on_error": "abstain",
            "return_trajectory": True,
        },
    )
    assert "answer" in out
    assert "citations" in out
    assert "context" in out
    assert "agent" in out
    ag = out["agent"]
    assert ag["status"] in ("ok", "abstain", "degraded", "error")
    assert isinstance(ag["subqueries"], list)
    assert isinstance(ag["trajectory"], list)
    assert "searches" in ag["counts"]
    assert out["answer"]
    assert ret.calls, "search_fn should hit retriever"
    assert gen.complete_prompts or gen.answer_calls


def test_agent_answer_for_eval_forces_hitl_off():
    ret = _FakeRetriever()
    gen = _FakeGenerator()
    out = agent_answer_for_eval(
        "q",
        retriever=ret,
        generator=gen,
        cfg={
            "hitl": {"review_subqueries": True},  # should be forced False
            "grade": {"enabled": False},
            "checkpoint": {"enabled": False},
            "on_error": "abstain",
        },
    )
    # if HITL stayed on without checkpointer resume, might interrupt — we force off
    assert out["agent"]["status"] != "interrupted"


def test_load_agent_eval_qa_skeleton():
    path = Path("data/agent_eval_qa.json")
    assert path.is_file()
    items = load_agent_eval_qa(path)
    assert 5 <= len(items) <= 10
    tags = {it.get("tag") for it in items}
    assert "multi_hop" in tags
    assert "atomic" in tags
    assert "reject" in tags
    rej = [it for it in items if it.get("tag") == "reject"]
    assert all(it.get("expect_reject") for it in rej)

    filtered = load_agent_eval_qa(path, tags=["atomic"], max_items=1)
    assert len(filtered) == 1
    assert filtered[0]["tag"] == "atomic"


def test_generator_complete_public():
    from src.generation.generator import Generator

    calls = []

    def _fn(prompt: str) -> str:
        calls.append(prompt)
        return "ok-resp"

    g = Generator(client=object(), complete_fn=_fn)
    assert g.complete("hello") == "ok-resp"
    assert calls == ["hello"]
