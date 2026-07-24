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


def test_ask_with_doc_id_overfetches_then_filters():
    """有 doc_id 时应放大 k 再过滤，避免新上传文档被大库挤出 top-k。"""
    seen = {}

    class R:
        pg = type("PG", (), {"delete_by_doc_id": lambda self, d: 0})()
        faiss = type("F", (), {"save": lambda self: None})()
        bge = None
        colpali = None
        chunker = None
        bm25 = type("B", (), {"ready": True, "fit_from_pgvector": lambda self, pg: None})()
        _answer_cache = None
        index_version = 0

        def answer_cache_key(self, query, model, k, doc_id):
            return f"{query}|{model}|{k}|{doc_id}|v{self.index_version}"

        def invalidate_cache(self):
            self.index_version += 1

        def search_with_trace(self, query, k=10, use_visual=True, use_rerank=True):
            seen["k"] = k
            hits = [
                {
                    "chunk_id": "other", "page_id": 1, "doc_id": "other-doc",
                    "page_number": 1, "text": "noise", "score": 0.99,
                    "retrieval_type": "dense", "rerank_score": 0.99,
                },
                {
                    "chunk_id": "mine", "page_id": 2, "doc_id": "mine-doc",
                    "page_number": 2, "text": "pump interval", "score": 0.5,
                    "retrieval_type": "dense", "rerank_score": 0.5,
                },
            ]
            item = {"chunk_id": "c1", "page_id": 1, "score": 0.9}
            return {
                "results": hits,
                "retrieval_trace": {
                    "bm25_top5": [item],
                    "dense_top5": [item],
                    "visual_top5": [],
                },
            }

    routes.set_retriever(R())
    routes.set_generator(_fake_generator())
    c = TestClient(routes.app)
    r = c.post("/ask", json={"query": "pump?", "k": 5, "doc_id": "mine-doc"})
    assert r.status_code == 200
    assert seen["k"] >= 50  # over-fetch
    # generator 仍被调用；若过滤后非空即可
    assert r.json()["answer"] == "ok"


def test_demo_static_index_served():
    """Demo 静态页由 StaticFiles 挂载；不依赖真实 retriever 重模型。"""
    routes.set_retriever(_fake_retriever())
    c = TestClient(routes.app)
    r = c.get("/demo/")
    # StaticFiles html=True 可能 200 于 /demo/ 或需 /demo/index.html
    if r.status_code == 404:
        r = c.get("/demo/index.html")
    assert r.status_code == 200
    assert "PrismRAG" in r.text
    # 附属资源
    r2 = c.get("/demo/app.js")
    assert r2.status_code == 200
    r3 = c.get("/demo/fixtures.json")
    assert r3.status_code == 200
    for path in ("/demo/documents.html", "/demo/embed.html", "/demo/common.js"):
        assert c.get(path).status_code == 200, path


def test_list_documents_endpoint():
    class R:
        faiss = type("F", (), {"num_pages": 3, "index_size_mb": 1.2})()
        bm25 = type("B", (), {"ready": True})()
        pg = type(
            "PG",
            (),
            {
                "list_documents": lambda self: [
                    {
                        "doc_id": "abc",
                        "content_hash": "h",
                        "source_path": "data/uploads/abc.pdf",
                        "created_at": "2026-07-24T00:00:00",
                        "num_chunks": 2,
                        "num_pages": 1,
                        "num_tables": 0,
                        "num_text": 2,
                        "page_from": 1,
                        "page_to": 1,
                    }
                ],
                "corpus_stats": lambda self: {
                    "num_documents": 1,
                    "num_document_rows": 1,
                    "num_pages": 1,
                    "num_chunks": 2,
                    "num_table_chunks": 0,
                },
            },
        )()

    routes.set_retriever(R())
    c = TestClient(routes.app)
    r = c.get("/documents")
    assert r.status_code == 200
    body = r.json()
    assert body["stats"]["num_documents"] == 1
    assert body["stats"]["bm25_ready"] is True
    assert body["documents"][0]["doc_id"] == "abc"


def test_ingest_job_not_found():
    routes.set_retriever(_fake_retriever())
    c = TestClient(routes.app)
    r = c.get("/ingest/jobs/does-not-exist")
    assert r.status_code == 404
