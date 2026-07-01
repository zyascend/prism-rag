"""PrismRAGRetriever 适配器测试（mock 所有依赖）"""

from unittest.mock import MagicMock

import torch

from src.evaluation.vidore_adapter import PrismRAGRetriever


def _make_mock_retriever():
    """构造 mock 化的 PrismRAGRetriever 实例"""
    return PrismRAGRetriever(
        pg_store=MagicMock(),
        faiss_store=MagicMock(),
        bge=MagicMock(),
        colpali=MagicMock(),
        chunker=MagicMock(),
        bm25=MagicMock(),
        dense=MagicMock(),
        visual=MagicMock(),
        fusion=MagicMock(),
        reranker=MagicMock(),
    )


def test_search_with_visual_embedding_uses_search_with_embedding():
    """传入 visual_query_embedding 时，visual route 走 search_with_embedding 而不是 search"""
    retriever = _make_mock_retriever()
    retriever.bm25.search.return_value = []
    retriever.dense.search.return_value = []
    retriever.visual.search_with_embedding.return_value = [
        {"chunk_id": "c1", "page_id": 1, "score": 0.85, "retrieval_type": "visual"},
    ]
    retriever.fusion.fuse.return_value = [
        {"chunk_id": "c1", "page_id": 1, "score": 0.85, "retrieval_type": "visual"},
    ]

    q_emb = torch.randn(1, 10, 128)
    result = retriever.search(
        query="test", k=10,
        use_bm25=False, use_dense=False, use_visual=True, use_rerank=False,
        visual_query_embedding=q_emb,
    )

    retriever.visual.search_with_embedding.assert_called_once()
    retriever.visual.search.assert_not_called()

    assert len(result) == 1


def test_search_without_visual_embedding_uses_search():
    """不传 visual_query_embedding 时，visual route 走原来的 search()"""
    retriever = _make_mock_retriever()
    retriever.bm25.search.return_value = []
    retriever.dense.search.return_value = []
    retriever.visual.search.return_value = [
        {"chunk_id": "c1", "page_id": 1, "score": 0.85, "retrieval_type": "visual"},
    ]
    retriever.fusion.fuse.return_value = [
        {"chunk_id": "c1", "page_id": 1, "score": 0.85, "retrieval_type": "visual"},
    ]

    result = retriever.search(
        query="test", k=10,
        use_bm25=False, use_dense=False, use_visual=True, use_rerank=False,
    )

    retriever.visual.search.assert_called_once()
    retriever.visual.search_with_embedding.assert_not_called()

    assert len(result) == 1


def test_search_visual_false_ignores_embedding():
    """use_visual=False 时不会误用 visual_query_embedding"""
    retriever = _make_mock_retriever()
    retriever.bm25.search.return_value = []
    retriever.dense.search.return_value = [
        {"chunk_id": "c1", "page_id": 1, "score": 0.85, "retrieval_type": "dense"},
    ]
    retriever.fusion.fuse.return_value = [
        {"chunk_id": "c1", "page_id": 1, "score": 0.85, "retrieval_type": "dense"},
    ]

    q_emb = torch.randn(1, 10, 128)
    retriever.search(
        query="test", k=10,
        use_bm25=False, use_dense=True, use_visual=False, use_rerank=False,
        visual_query_embedding=q_emb,
    )

    retriever.visual.search.assert_not_called()
    retriever.visual.search_with_embedding.assert_not_called()


def test_search_with_trace_passes_visual_embedding():
    """search_with_trace 也应该透传 visual_query_embedding"""
    retriever = _make_mock_retriever()
    retriever.bm25.search.return_value = []
    retriever.dense.search.return_value = []
    retriever.visual.search_with_embedding.return_value = [
        {"chunk_id": "c1", "page_id": 1, "score": 0.85, "retrieval_type": "visual"},
    ]
    retriever.fusion.fuse.return_value = [
        {"chunk_id": "c1", "page_id": 1, "score": 0.85, "retrieval_type": "visual"},
    ]

    q_emb = torch.randn(1, 10, 128)
    result = retriever.search_with_trace(
        query="test", k=10,
        use_bm25=False, use_dense=False, use_visual=True, use_rerank=False,
        visual_query_embedding=q_emb,
    )

    retriever.visual.search_with_embedding.assert_called_once()
    assert "results" in result
    assert "retrieval_trace" in result