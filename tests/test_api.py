from fastapi.testclient import TestClient
from src.api import routes


def _fake_retriever():
    class R:
        pg = type("PG", (), {"delete_by_doc_id": lambda self, d: 0})()
        faiss = type("F", (), {"save": lambda self: None})()
        bge = None
        colpali = None
        chunker = None
        bm25 = type("B", (), {"fit_from_pgvector": lambda self, pg: None})()
        # L4 Answer 缓存所需成员（PR #28）
        _answer_cache = None
        index_version = 0

        def answer_cache_key(self, query, model, k, doc_id):
            return f"{query}|{model}|{k}|{doc_id}|v{self.index_version}"

        def _hit(self):
            return {
                "chunk_id": "c1", "page_id": 1, "doc_id": "d",
                "page_number": 1, "text": "pump interval", "doc_ref": "x",
                "score": 0.9, "retrieval_type": "dense", "rerank_score": 0.9,
            }

        def search(self, query, k=10, use_visual=True, use_rerank=True):
            return [self._hit()]

        def search_with_trace(self, query, k=10, use_visual=True, use_rerank=True):
            item = {"chunk_id": "c1", "page_id": 1, "score": 0.9}
            return {
                "results": [self._hit()],
                "retrieval_trace": {
                    "bm25_top5": [item],
                    "dense_top5": [item],
                    "visual_top5": [],
                },
            }
    return R()


def _fake_generator():
    class G:
        # L4 Answer 缓存守卫：/ask 读取 gen.cacheable / gen.model 构造缓存键（PR #28）
        cacheable = True
        model = "fake-model"

        def answer(self, q, retrieved, k_context=5):
            return {"answer": "ok", "citations": [{"chunk_id": "c1", "page_id": 1,
                    "doc_id": "d", "page_number": 1, "snippet": "s"}], "context": ""}
    return G()


def test_ingest_rejects_non_pdf():
    routes.set_retriever(_fake_retriever())
    c = TestClient(routes.app)
    r = c.post("/ingest", files={"file": ("x.txt", b"hi", "text/plain")})
    assert r.status_code == 422


def test_ask_returns_answer_and_citations():
    routes.set_retriever(_fake_retriever())
    routes.set_generator(_fake_generator())
    c = TestClient(routes.app)
    r = c.post("/ask", json={"query": "pump interval?", "k": 5})
    assert r.status_code == 200
    body = r.json()
    assert body["answer"] == "ok"
    assert body["citations"][0]["chunk_id"] == "c1"
