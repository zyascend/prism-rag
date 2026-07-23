"""Phase B2: query intent + modality boost."""
from __future__ import annotations

from src.retrieval.query_intent import (
    apply_modality_boost,
    detect_query_intent,
)


def test_detect_table_intent():
    i = detect_query_intent("What is the maximum allowed nozzle pressure in PSI?")
    assert i.table is True


def test_detect_visual_intent():
    i = detect_query_intent("See figure 3 for the wiring diagram layout")
    assert i.visual is True


def test_detect_none():
    i = detect_query_intent("What is the purpose of a TC device?")
    assert i.table is False
    assert i.visual is False
    assert i.label == "none"


def test_boost_promotes_table_chunks():
    intent = detect_query_intent("table of pressure limits")
    results = [
        {"chunk_id": "t1", "chunk_type": "text", "score": 0.05, "text": "intro"},
        {"chunk_id": "tb", "chunk_type": "table", "score": 0.04, "text": "| a | b |"},
    ]
    out = apply_modality_boost(results, intent, table_bonus=0.02, image_bonus=0.0)
    assert out[0]["chunk_id"] == "tb"
    assert out[0]["score"] > results[1]["score"]
    assert out[0].get("modality_boost", 0) > 0


def test_boost_noop_when_no_intent():
    intent = detect_query_intent("hello world purpose")
    results = [
        {"chunk_id": "t1", "chunk_type": "text", "score": 0.05},
        {"chunk_id": "tb", "chunk_type": "table", "score": 0.04},
    ]
    out = apply_modality_boost(results, intent, table_bonus=0.02)
    assert [r["chunk_id"] for r in out] == ["t1", "tb"]
