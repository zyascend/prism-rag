"""API 契约：/ask mode=agent + /ask/resume（mock TestClient，不碰真模型）。"""
from __future__ import annotations

from unittest.mock import patch

from fastapi.testclient import TestClient

from src.agent.runner import AgentResult
from src.api import routes


def _fake_retriever():
    class R:
        pg = type("PG", (), {"delete_by_doc_id": lambda self, d: 0})()
        faiss = type("F", (), {"save": lambda self: None})()
        bge = None
        colpali = None
        chunker = None
        bm25 = type("B", (), {"fit_from_pgvector": lambda self, pg: None})()
        _answer_cache = None
        index_version = 0
        last_cache_key = None

        def answer_cache_key(self, query, model, k, doc_id, mode=None):
            m = mode or "pipeline"
            base = f"{query}|{model}|{k}|{doc_id}|v{self.index_version}"
            key = base if m == "pipeline" else f"{base}|mode={m}"
            self.last_cache_key = key
            return key

        def _hit(self):
            return {
                "chunk_id": "c1",
                "page_id": 1,
                "doc_id": "d",
                "page_number": 1,
                "text": "pump interval",
                "doc_ref": "x",
                "score": 0.9,
                "retrieval_type": "dense",
                "rerank_score": 0.9,
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
        cacheable = True
        model = "fake-model"
        client = object()  # present for complete_fn path (mocked away in agent tests)
        _complete_fn = staticmethod(lambda prompt: '{"subqueries":["q"],"strategy":"atomic"}')

        def answer(self, q, retrieved, k_context=5):
            return {
                "answer": "pipeline-ok",
                "citations": [
                    {
                        "chunk_id": "c1",
                        "page_id": 1,
                        "doc_id": "d",
                        "page_number": 1,
                        "snippet": "s",
                    }
                ],
                "context": "ctx-for-llm",
            }

    return G()


def _setup():
    routes.set_retriever(_fake_retriever())
    routes.set_generator(_fake_generator())
    return TestClient(routes.app)


def test_ask_pipeline_agent_null_by_default():
    c = _setup()
    r = c.post("/ask", json={"query": "hi", "k": 3})
    assert r.status_code == 200
    body = r.json()
    assert body["answer"] == "pipeline-ok"
    assert body.get("agent") is None


def test_ask_mode_pipeline_explicit_agent_null():
    c = _setup()
    r = c.post("/ask", json={"query": "hi", "k": 3, "mode": "pipeline"})
    assert r.status_code == 200
    assert r.json().get("agent") is None


def test_ask_mode_agent_ignored_when_disabled():
    c = _setup()
    with patch("src.agent.config.agent_config", return_value={"enabled": False}):
        r = c.post("/ask", json={"query": "hi", "mode": "agent", "k": 3})
    assert r.status_code == 200
    body = r.json()
    # pipeline still runs
    assert body["answer"] == "pipeline-ok"
    ag = body.get("agent")
    assert ag is not None
    assert ag.get("used") is False
    assert "enabled" in (ag.get("ignored_reason") or "")


def test_ask_mode_agent_when_enabled_returns_trajectory():
    c = _setup()
    fake_result = AgentResult(
        answer="agent-ans",
        citations=[
            {
                "chunk_id": "c1",
                "page_id": 1,
                "doc_id": "d",
                "page_number": 1,
                "snippet": "s",
            }
        ],
        status="ok",
        subqueries=["sub-q1"],
        trajectory=[
            {
                "step": 1,
                "node": "decompose",
                "tool": None,
                "input_summary": "q",
                "output_summary": "1 sub",
                "ok": True,
                "error": None,
                "latency_ms": 1.0,
                "counts": {},
            }
        ],
        counts={"subqueries": 1, "searches": 1, "llm_calls": 2, "evidence_n": 1},
        thread_id="tid-1",
        context="agent-ctx",
        evidence=[{"chunk_id": "c1", "page_id": 1, "score": 0.9, "text": "x"}],
    )
    acfg = {
        "enabled": True,
        "return_trajectory": True,
        "checkpoint": {"enabled": True},
        "hitl": {"review_subqueries": False},
    }
    with patch("src.agent.config.agent_config", return_value=acfg), patch(
        "src.agent.runner.run_agent", return_value=fake_result
    ) as mock_run:
        r = c.post("/ask", json={"query": "hi", "mode": "agent", "k": 3})
    assert r.status_code == 200
    body = r.json()
    assert body["answer"] == "agent-ans"
    ag = body["agent"]
    assert ag["used"] is True
    assert ag["status"] == "ok"
    assert ag["subqueries"] == ["sub-q1"]
    assert len(ag["trajectory"]) == 1
    assert ag["trajectory"][0]["node"] == "decompose"
    assert ag["counts"]["searches"] == 1
    assert ag["degraded_to_pipeline"] is False
    assert ag["thread_id"] == "tid-1"
    assert body["context"] == "agent-ctx"
    mock_run.assert_called_once()
    # pipeline_fallback_fn must be provided for degrade_pipeline default
    kwargs = mock_run.call_args.kwargs
    assert callable(kwargs.get("pipeline_fallback_fn"))


def test_ask_cache_key_includes_mode_agent():
    """mode=agent must not share L4 keys with pipeline."""
    retriever = _fake_retriever()
    routes.set_retriever(retriever)
    routes.set_generator(_fake_generator())
    c = TestClient(routes.app)

    acfg = {
        "enabled": True,
        "return_trajectory": True,
        "checkpoint": {"enabled": True},
        "hitl": {"review_subqueries": False},
    }
    fake = AgentResult(
        answer="a",
        status="ok",
        trajectory=[],
        counts={},
        thread_id="t",
    )
    with patch("src.agent.config.agent_config", return_value=acfg), patch(
        "src.agent.runner.run_agent", return_value=fake
    ):
        r = c.post("/ask", json={"query": "same-q", "mode": "agent", "k": 3})
    assert r.status_code == 200
    assert retriever.last_cache_key is not None
    assert "mode=agent" in retriever.last_cache_key

    # pipeline key for same query differs
    pipe_key = retriever.answer_cache_key("same-q", "fake-model", 3, None, mode="pipeline")
    agent_key = retriever.answer_cache_key("same-q", "fake-model", 3, None, mode="agent")
    assert pipe_key != agent_key
    assert "mode=agent" in agent_key
    assert "mode=" not in pipe_key or "mode=pipeline" not in pipe_key


def test_ask_resume_400_when_agent_disabled():
    c = _setup()
    with patch("src.agent.config.agent_config", return_value={"enabled": False}):
        r = c.post(
            "/ask/resume",
            json={"thread_id": "t1", "subqueries": ["a"]},
        )
    assert r.status_code == 400
    assert "enabled" in r.json()["detail"].lower() or "agent" in r.json()["detail"].lower()


def test_ask_resume_400_when_checkpoint_disabled():
    c = _setup()
    with patch(
        "src.agent.config.agent_config",
        return_value={
            "enabled": True,
            "checkpoint": {"enabled": False},
            "return_trajectory": True,
        },
    ):
        r = c.post(
            "/ask/resume",
            json={"thread_id": "t1", "subqueries": ["a"]},
        )
    assert r.status_code == 400
    assert "checkpoint" in r.json()["detail"].lower()


def test_ask_resume_ok_when_enabled():
    c = _setup()
    acfg = {
        "enabled": True,
        "checkpoint": {"enabled": True},
        "return_trajectory": True,
        "hitl": {"review_subqueries": True},
    }
    fake = AgentResult(
        answer="resumed",
        citations=[{"chunk_id": "c1", "page_id": 1, "snippet": "s"}],
        status="ok",
        subqueries=["a", "b"],
        trajectory=[{"step": 2, "node": "synthesize", "ok": True}],
        counts={"searches": 2, "llm_calls": 3, "subqueries": 2, "evidence_n": 1},
        thread_id="thread-x",
        context="ctx",
    )
    with patch("src.agent.config.agent_config", return_value=acfg), patch(
        "src.agent.runner.resume_agent", return_value=fake
    ) as mock_resume:
        r = c.post(
            "/ask/resume",
            json={"thread_id": "thread-x", "subqueries": ["a", "b"], "k": 3},
        )
    assert r.status_code == 200
    body = r.json()
    assert body["answer"] == "resumed"
    assert body["agent"]["used"] is True
    assert body["agent"]["thread_id"] == "thread-x"
    assert body["agent"]["subqueries"] == ["a", "b"]
    mock_resume.assert_called_once()
    assert mock_resume.call_args.kwargs["approved_subqueries"] == ["a", "b"]
    assert callable(mock_resume.call_args.kwargs.get("pipeline_fallback_fn"))


def test_answer_cache_key_mode_param_on_real_method():
    """PrismRAGRetriever.answer_cache_key accepts mode and separates keys."""
    from src.evaluation.vidore_adapter import PrismRAGRetriever

    r = PrismRAGRetriever.__new__(PrismRAGRetriever)
    r.index_version = "v-test"
    k_pipe = r.answer_cache_key("q", "m", 5, None, mode="pipeline")
    k_agent = r.answer_cache_key("q", "m", 5, None, mode="agent")
    assert k_pipe != k_agent
    assert "mode=agent" in k_agent
    # default / pipeline leaves historical key shape (no mode= suffix)
    k_default = r.answer_cache_key("q", "m", 5, None)
    assert k_default == k_pipe
