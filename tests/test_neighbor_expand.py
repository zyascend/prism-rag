"""Phase B1: neighbor expand."""
from __future__ import annotations

from src.retrieval.expand import expand_neighbors


class _FakePG:
    def __init__(self):
        self.by_page = {
            10: [
                {
                    "chunk_id": "a", "page_id": 10, "doc_id": "d", "page_number": 1,
                    "chunk_type": "text", "text": "hit body",
                    "prev_chunk_id": "", "next_chunk_id": "b",
                    "section_path": "Sec", "caption": "",
                },
                {
                    "chunk_id": "b", "page_id": 10, "doc_id": "d", "page_number": 1,
                    "chunk_type": "table", "text": "| x | 1 |",
                    "prev_chunk_id": "a", "next_chunk_id": "c",
                    "section_path": "Sec", "caption": "T1",
                },
                {
                    "chunk_id": "c", "page_id": 10, "doc_id": "d", "page_number": 1,
                    "chunk_type": "text", "text": "after table",
                    "prev_chunk_id": "b", "next_chunk_id": "",
                    "section_path": "Sec", "caption": "",
                },
            ]
        }
        self.by_id = {c["chunk_id"]: c for cs in self.by_page.values() for c in cs}

    def get_chunks_by_page_ids(self, page_ids):
        out = []
        for p in page_ids:
            out.extend(self.by_page.get(p, []))
        return out

    def get_chunks_by_ids(self, ids):
        return [self.by_id[i] for i in ids if i in self.by_id]


def test_expand_page_adds_neighbors():
    hits = [
        {
            "chunk_id": "a", "page_id": 10, "score": 1.0, "rerank_score": 0.9,
            "text": "hit body", "chunk_type": "text",
        }
    ]
    out, trace = expand_neighbors(hits, _FakePG(), mode="page", max_extra=2)
    ids = [r["chunk_id"] for r in out]
    assert ids[0] == "a"
    assert "b" in ids
    assert "c" in ids
    assert trace["added"] == 2
    assert any(r.get("retrieval_type") == "neighbor_expand" for r in out[1:])


def test_expand_prev_next():
    hits = [
        {
            "chunk_id": "b", "page_id": 10, "score": 1.0, "rerank_score": 0.8,
            "prev_chunk_id": "a", "next_chunk_id": "c",
            "text": "table", "chunk_type": "table",
        }
    ]
    out, trace = expand_neighbors(hits, _FakePG(), mode="prev_next", max_extra=2)
    ids = {r["chunk_id"] for r in out}
    assert ids == {"a", "b", "c"}
    assert trace["added"] == 2


def test_expand_disabled_path_empty_max():
    hits = [{"chunk_id": "a", "page_id": 10, "score": 1.0}]
    out, trace = expand_neighbors(hits, _FakePG(), mode="page", max_extra=0)
    assert out == hits
    assert trace["added"] == 0


def test_expand_dedupes_existing():
    hits = [
        {"chunk_id": "a", "page_id": 10, "score": 1.0, "rerank_score": 1.0},
        {"chunk_id": "b", "page_id": 10, "score": 0.9, "rerank_score": 0.9},
    ]
    out, trace = expand_neighbors(hits, _FakePG(), mode="page", max_extra=5)
    ids = [r["chunk_id"] for r in out]
    assert ids.count("a") == 1
    assert ids.count("b") == 1
    assert "c" in ids
