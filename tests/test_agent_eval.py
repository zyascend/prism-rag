"""Unit tests for agent eval helper (mocked retriever/generator — no GPU)."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from src.agent.eval import (
    agent_answer_for_eval,
    go_nogo_draft,
    load_agent_eval_qa,
    score_answer,
    summarize_dual_arm,
)


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


def test_load_agent_eval_qa_phase2_set():
    path = Path("data/agent_eval_qa.json")
    assert path.is_file()
    items = load_agent_eval_qa(path)
    assert 40 <= len(items) <= 50
    tags = Counter(it.get("tag") for it in items)
    assert tags["multi_hop"] >= 15
    assert tags["atomic"] >= 15
    assert tags["reject"] >= 8
    rej = [it for it in items if it.get("tag") == "reject"]
    assert all(it.get("expect_reject") for it in rej)
    assert all(it.get("query") for it in items)

    filtered = load_agent_eval_qa(path, tags=["atomic"], max_items=1)
    assert len(filtered) == 1
    assert filtered[0]["tag"] == "atomic"


def test_score_answer_heuristic_reject_and_overlap():
    from src.rejection import ABSTAIN_ANSWER

    r = score_answer(
        question="who won?",
        answer=ABSTAIN_ANSWER,
        gold=None,
        expect_reject=True,
        judge="heuristic",
    )
    assert r["correct"] is True
    assert r["is_rejected"] is True

    r2 = score_answer(
        question="q",
        answer=ABSTAIN_ANSWER,
        gold="The braid clamp must be evenly distributed around the shoulder.",
        expect_reject=False,
        judge="heuristic",
    )
    assert r2["correct"] is False
    assert r2["false_reject"] is True

    r3 = score_answer(
        question="q",
        answer="The braid clamp is evenly distributed around the shoulder flange.",
        gold="The braid clamp must be evenly distributed around the shoulder.",
        expect_reject=False,
        judge="heuristic",
    )
    assert r3["correct"] is True


def test_summarize_and_go_nogo_draft():
    rows = [
        {
            "id": "at_1",
            "tag": "atomic",
            "expect_reject": False,
            "pipeline": {
                "correct": True,
                "latency_ms": 1000,
                "false_reject": False,
                "is_rejected": False,
            },
            "agent": {
                "correct": True,
                "latency_ms": 5000,
                "false_reject": False,
                "is_rejected": False,
                "counts": {"searches": 1, "llm_calls": 3},
                "status": "ok",
                "degraded": False,
            },
        },
        {
            "id": "mh_1",
            "tag": "multi_hop",
            "expect_reject": False,
            "pipeline": {
                "correct": False,
                "latency_ms": 1200,
                "false_reject": False,
                "is_rejected": False,
            },
            "agent": {
                "correct": True,
                "latency_ms": 8000,
                "false_reject": False,
                "is_rejected": False,
                "counts": {"searches": 2, "llm_calls": 5},
                "status": "ok",
                "degraded": False,
            },
        },
        {
            "id": "rj_1",
            "tag": "reject",
            "expect_reject": True,
            "pipeline": {
                "correct": True,
                "latency_ms": 800,
                "false_reject": False,
                "is_rejected": True,
            },
            "agent": {
                "correct": True,
                "latency_ms": 2000,
                "false_reject": False,
                "is_rejected": True,
                "counts": {"searches": 1, "llm_calls": 2},
                "status": "ok",
                "degraded": False,
            },
        },
    ]
    s = summarize_dual_arm(rows)
    assert s["n_items"] == 3
    assert s["arms"]["pipeline"]["correct_rate"] is not None
    assert s["arms"]["agent"]["avg_searches"] == (1 + 2 + 1) / 3
    assert s["go_nogo"]["verdict"] == "GO_DRAFT"

    # multi_hop regression → NO_GO
    bad_agent = dict(s["arms"]["agent"])
    bad_agent["by_tag"] = {
        "atomic": {"n": 1, "correct_rate": 1.0},
        "multi_hop": {"n": 1, "correct_rate": 0.0},
        "reject": {"n": 1, "correct_rate": 1.0},
    }
    bad_pipe = dict(s["arms"]["pipeline"])
    bad_pipe["by_tag"] = {
        "atomic": {"n": 1, "correct_rate": 1.0},
        "multi_hop": {"n": 1, "correct_rate": 0.5},
        "reject": {"n": 1, "correct_rate": 1.0},
    }
    assert go_nogo_draft(bad_pipe, bad_agent)["verdict"] == "NO_GO_DRAFT"


def test_run_agent_eval_skeleton_cli(tmp_path):
    import scripts.run_agent_eval as rae

    out = tmp_path / "skel"
    rc = rae.main(
        [
            "--qa-file",
            "data/agent_eval_qa.json",
            "--max-queries",
            "3",
            "--dry-run",
            "--output-dir",
            str(out),
        ]
    )
    assert rc == 0
    assert (out / "results.json").is_file()
    data = json.loads((out / "results.json").read_text())
    assert data["phase"] == "skeleton"
    assert data["n_items"] == 3


def test_generator_complete_public():
    from src.generation.generator import Generator

    calls = []

    def _fn(prompt: str) -> str:
        calls.append(prompt)
        return "ok-resp"

    g = Generator(client=object(), complete_fn=_fn)
    assert g.complete("hello") == "ok-resp"
    assert calls == ["hello"]
