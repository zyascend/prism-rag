# tests/e2e_local.py
"""本地端到端：需 pgvector 容器 (make db) + OPENAI_API_KEY + BGE 模型。
不可达时自动 skip，避免 CI/纯单元环境误跑。"""
import os
import pytest
from pathlib import Path
from fastapi.testclient import TestClient
from src.config import cfg
from src.api import routes

FIXTURE = Path(__file__).parent / "fixtures" / "sample.pdf"


def _pg_reachable() -> bool:
    try:
        from src.store.pgvector_store import PgVectorStore
        PgVectorStore().conn  # 触发连接
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _pg_reachable(), reason="pgvector 不可达（先 make db）")
@pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY"), reason="需 OPENAI_API_KEY")
def test_ingest_and_ask_e2e():
    cfg.load()
    c = TestClient(routes.app)
    with open(FIXTURE, "rb") as f:
        r = c.post("/ingest", files={"file": ("sample.pdf", f, "application/pdf")})
    assert r.status_code == 200, r.text
    doc_id = r.json()["doc_id"]
    a = c.post("/ask", json={"query": "pump maintenance interval?", "doc_id": doc_id, "k": 5})
    assert a.status_code == 200, a.text
    body = a.json()
    assert body["answer"]
    assert any("pump" in (ci["snippet"] or "").lower() or ci["page_id"] for ci in body["citations"])
