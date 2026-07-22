"""CRAG unit tests — mock judge / search，无真模型。"""
from __future__ import annotations

import json

from src.retrieval.crag import (
    CorrectiveRAG,
    apply_grades,
    crag_cache_salt,
    crag_config,
    grade_documents,
    reformulate_query,
)


def _docs():
    return [
        {
            "chunk_id": "c1",
            "page_id": 1,
            "doc_id": "d1",
            "text": "Pump service interval is 500 hours.",
            "chunk_type": "text",
        },
        {
            "chunk_id": "c2",
            "page_id": 2,
            "doc_id": "d1",
            "text": "Company history and warranty boilerplate.",
            "chunk_type": "text",
        },
        {
            "chunk_id": "c3",
            "page_id": 3,
            "doc_id": "d1",
            "text": "Torque spec for bolt M8 is 25 Nm.",
            "chunk_type": "text",
        },
    ]


def test_apply_crag_if_enabled_disabled():
    from src.retrieval.crag import apply_crag_if_enabled

    docs = _docs()
    out, meta = apply_crag_if_enabled(
        "q", docs, config={"enabled": False}
    )
    assert out == docs
    assert meta.get("applied") is False


def test_crag_cache_salt_off_by_default():
    # 默认 config 可能 enabled false
    salt = crag_cache_salt({"enabled": False})
    assert salt == "crag=off"
    salt_on = crag_cache_salt(
        {
            "enabled": True,
            "grade_top_n": 10,
            "min_relevant": 1,
            "reformulate": True,
            "max_retrieve_attempts": 2,
        }
    )
    assert salt_on.startswith("crag=on")


def test_apply_grades_filters():
    docs = _docs()
    grades = [
        {"chunk_id": "c1", "relevant": True, "reason": "interval"},
        {"chunk_id": "c2", "relevant": False, "reason": "noise"},
        {"chunk_id": "c3", "relevant": False, "reason": "other"},
    ]
    kept = apply_grades(docs, grades)
    assert [d["chunk_id"] for d in kept] == ["c1"]


def test_grade_documents_parses():
    docs = _docs()[:2]

    def complete(_p: str) -> str:
        return json.dumps(
            {
                "grades": [
                    {"chunk_id": "c1", "relevant": True, "reason": "yes"},
                    {"chunk_id": "c2", "relevant": False, "reason": "no"},
                ],
                "sufficient": True,
                "missing": "",
            }
        )

    g = grade_documents("interval?", docs, complete_fn=complete)
    assert g["sufficient"] is True
    assert g["grades"][0]["relevant"] is True
    assert g["grades"][1]["relevant"] is False


def test_reformulate_query():
    def complete(_p: str) -> str:
        return json.dumps(
            {
                "query": "hydraulic pump service interval hours",
                "rationale": "add keywords",
            }
        )

    r = reformulate_query("how often service?", complete_fn=complete, feedback="weak")
    assert "pump" in r["query"] or "interval" in r["query"]


def test_correct_disabled_passthrough():
    crag = CorrectiveRAG(
        search_fn=lambda q: [],
        complete_fn=lambda p: "{}",
        config={"enabled": False},
    )
    docs = _docs()
    out = crag.correct("q", docs)
    assert out["results"] == docs
    assert out["crag"]["applied"] is False


def test_correct_filters_when_sufficient():
    def complete(prompt: str) -> str:
        # grade path
        if "DOCUMENTS" in prompt or "documents" in prompt.lower() or "Grade" in prompt or "chunk_id" in prompt:
            return json.dumps(
                {
                    "grades": [
                        {"chunk_id": "c1", "relevant": True, "reason": "ok"},
                        {"chunk_id": "c2", "relevant": False, "reason": "noise"},
                        {"chunk_id": "c3", "relevant": False, "reason": "other"},
                    ],
                    "sufficient": True,
                    "missing": "",
                }
            )
        return json.dumps({"query": "should not reformulate", "rationale": ""})

    crag = CorrectiveRAG(
        search_fn=lambda q: (_ for _ in ()).throw(AssertionError("no re-search")),
        complete_fn=complete,
        config={
            "enabled": True,
            "grade_top_n": 10,
            "min_relevant": 1,
            "reformulate": True,
            "max_retrieve_attempts": 2,
            "on_grade_error": "pass_through",
            "regrade_after_retrieve": True,
            "chunk_text_max_chars": 600,
            "judge_timeout_ms": 8000,
        },
    )
    out = crag.correct("service interval?", _docs())
    assert [d["chunk_id"] for d in out["results"]] == ["c1"]
    assert out["crag"]["final_action"] == "accept"
    assert out["crag"]["num_relevant"] == 1


def test_correct_reformulate_and_research():
    calls = {"grade": 0, "search": 0}

    def complete(prompt: str) -> str:
        if "Rewrite" in prompt or "rewrite" in prompt or "ORIGINAL QUESTION" in prompt:
            return json.dumps(
                {
                    "query": "pump service interval 500 hours",
                    "rationale": "more keywords",
                }
            )
        calls["grade"] += 1
        if calls["grade"] == 1:
            return json.dumps(
                {
                    "grades": [
                        {"chunk_id": "c2", "relevant": False, "reason": "noise"},
                    ],
                    "sufficient": False,
                    "missing": "no interval",
                }
            )
        # second grade after re-search
        return json.dumps(
            {
                "grades": [
                    {"chunk_id": "c_new", "relevant": True, "reason": "interval"},
                ],
                "sufficient": True,
                "missing": "",
            }
        )

    def search(q: str):
        calls["search"] += 1
        assert "interval" in q or "pump" in q
        return [
            {
                "chunk_id": "c_new",
                "page_id": 9,
                "doc_id": "d1",
                "text": "Service interval: 500 hours.",
                "chunk_type": "text",
            }
        ]

    crag = CorrectiveRAG(
        search_fn=search,
        complete_fn=complete,
        config={
            "enabled": True,
            "grade_top_n": 5,
            "min_relevant": 1,
            "reformulate": True,
            "max_retrieve_attempts": 2,
            "on_grade_error": "pass_through",
            "regrade_after_retrieve": True,
            "chunk_text_max_chars": 600,
            "judge_timeout_ms": 8000,
        },
    )
    first = [
        {
            "chunk_id": "c2",
            "page_id": 2,
            "doc_id": "d1",
            "text": "warranty boilerplate only",
            "chunk_type": "text",
        }
    ]
    out = crag.correct("how often service?", first)
    assert calls["search"] == 1
    assert out["results"][0]["chunk_id"] == "c_new"
    assert out["query_used"] == "pump service interval 500 hours"
    assert out["crag"]["final_action"] == "accept"
    assert len(out["crag"]["attempts"]) == 2


def test_grade_error_pass_through():
    crag = CorrectiveRAG(
        search_fn=lambda q: [],
        complete_fn=lambda p: "NOT JSON",
        config={
            "enabled": True,
            "grade_top_n": 5,
            "min_relevant": 1,
            "reformulate": True,
            "max_retrieve_attempts": 2,
            "on_grade_error": "pass_through",
            "regrade_after_retrieve": True,
            "chunk_text_max_chars": 600,
            "judge_timeout_ms": 8000,
        },
    )
    docs = _docs()
    out = crag.correct("q", docs)
    assert out["results"] == docs
    assert out["crag"]["final_action"] == "pass_through_on_grade_error"
