from src.generation.generator import Generator, GenerationError


class _FakeCompletions:
    def create(self, **_):
        return type("R", (), {
            "choices": [type("C", (), {"message": type("M", (), {"content": "Answer here."})()})()]
        })()


class _FakeClient:
    def __init__(self): self.chat = type("Chat", (), {"completions": _FakeCompletions()})()


def _retrieved():
    return [
        {"chunk_id": "pg1_ch001", "page_id": 1, "doc_id": "d1",
         "page_number": 1, "text": "hydraulic pump interval 500 hours", "doc_ref": ""},
        {"chunk_id": "pg2_ch001", "page_id": 2, "doc_id": "d1",
         "page_number": 2, "text": "filter every 250 hours", "doc_ref": ""},
    ]


def test_answer_returns_citations_from_chunks():
    g = Generator(client=_FakeClient(), bge_embedder=None)
    out = g.answer("pump interval?", _retrieved(), k_context=5)
    assert out["answer"] == "Answer here."
    assert len(out["citations"]) == 2
    assert out["citations"][0]["chunk_id"] == "pg1_ch001"
    assert out["citations"][0]["page_id"] == 1


def test_empty_retrieval_honest_reject():
    g = Generator(client=_FakeClient(), bge_embedder=None)
    out = g.answer("x", [], k_context=5)
    assert out["answer"]
    assert out["citations"] == []
