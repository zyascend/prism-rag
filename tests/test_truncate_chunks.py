"""Text re-ingest: truncate_chunks helper."""
from __future__ import annotations

from src.store.pgvector_store import PgVectorStore


def test_truncate_chunks_method_exists_and_doc():
    assert hasattr(PgVectorStore, "truncate_chunks")
    doc = PgVectorStore.truncate_chunks.__doc__ or ""
    assert "re-ingest" in doc.lower() or "清空" in doc
    # 实现里必须真 truncate
    import inspect
    src = inspect.getsource(PgVectorStore.truncate_chunks)
    assert "TRUNCATE" in src
