"""P1-B crossref expand 单测。"""
from __future__ import annotations

from src.retrieval.expand import expand_crossrefs, extract_crossrefs


class _FakePG:
    def __init__(self, chunks):
        self.chunks = chunks

    def find_chunks_by_ref(self, doc_id, needles, limit=5):
        out = []
        for ch in self.chunks:
            if doc_id and ch.get("doc_id") != doc_id:
                continue
            blob = " ".join(
                [
                    ch.get("caption") or "",
                    ch.get("text") or "",
                    ch.get("table_summary") or "",
                ]
            ).lower()
            if any(n.lower() in blob for n in needles):
                out.append(ch)
            if len(out) >= limit:
                break
        return out


def test_extract_crossrefs_en_table_and_fig():
    refs = extract_crossrefs("See Table 2-1 and Fig. 3 for details.")
    kinds = {k for k, _ in refs}
    ids = {i for _, i in refs}
    assert "table" in kinds
    assert "figure" in kinds
    assert "2-1" in ids
    assert "3" in ids


def test_extract_crossrefs_zh():
    refs = extract_crossrefs("详细参数见表 3.2，接线图见图 12。")
    assert any(k == "table" and "3.2" in i for k, i in refs)
    assert any(k == "figure" and "12" in i for k, i in refs)


def test_expand_crossrefs_adds_table():
    pg = _FakePG(
        [
            {
                "chunk_id": "tbl",
                "doc_id": "d1",
                "page_id": 2,
                "page_number": 2,
                "chunk_type": "table",
                "text": "| A | 1 |",
                "caption": "Table 2-1 Limits",
                "table_summary": "limits table",
                "section_path": "",
                "prev_chunk_id": "",
                "next_chunk_id": "",
            }
        ]
    )
    hits = [
        {
            "chunk_id": "body",
            "doc_id": "d1",
            "page_id": 1,
            "score": 1.0,
            "rerank_score": 0.9,
            "chunk_type": "text",
            "text": "See Table 2-1 for the pressure limits.",
            "caption": "",
            "table_summary": "",
            "section_path": "",
        }
    ]
    out, trace = expand_crossrefs(hits, pg, max_extra=3, max_per_hit=1)
    ids = [r["chunk_id"] for r in out]
    assert ids[0] == "body"
    assert "tbl" in ids
    assert trace["added"] == 1
    assert any(r.get("retrieval_type") == "crossref_expand" for r in out[1:])


def test_expand_disabled_via_max_extra_zero():
    hits = [{"chunk_id": "a", "doc_id": "d", "text": "See Table 1", "score": 1.0}]
    out, trace = expand_crossrefs(hits, _FakePG([]), max_extra=0)
    assert out == hits
    assert trace["added"] == 0


def test_expand_same_doc_only():
    pg = _FakePG(
        [
            {
                "chunk_id": "other",
                "doc_id": "d2",
                "page_id": 1,
                "page_number": 1,
                "chunk_type": "table",
                "text": "x",
                "caption": "Table 1",
                "table_summary": "",
                "section_path": "",
                "prev_chunk_id": "",
                "next_chunk_id": "",
            }
        ]
    )
    hits = [
        {
            "chunk_id": "body",
            "doc_id": "d1",
            "text": "See Table 1",
            "score": 1.0,
            "caption": "",
            "table_summary": "",
            "section_path": "",
        }
    ]
    out, trace = expand_crossrefs(hits, pg, same_doc_only=True)
    assert "other" not in [r["chunk_id"] for r in out]
    assert trace["added"] == 0


def test_expand_dedupes_existing_hit():
    pg = _FakePG(
        [
            {
                "chunk_id": "tbl",
                "doc_id": "d1",
                "page_id": 1,
                "page_number": 1,
                "chunk_type": "table",
                "text": "|a|",
                "caption": "Table 1",
                "table_summary": "",
                "section_path": "",
                "prev_chunk_id": "",
                "next_chunk_id": "",
            }
        ]
    )
    hits = [
        {
            "chunk_id": "tbl",
            "doc_id": "d1",
            "text": "See Table 1",
            "score": 1.0,
            "caption": "Table 1",
            "table_summary": "",
            "section_path": "",
        }
    ]
    out, trace = expand_crossrefs(hits, pg)
    assert len(out) == 1
    assert trace["added"] == 0
