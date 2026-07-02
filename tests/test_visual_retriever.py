"""Visual 检索器测试（mock FAISS + pgvector）"""

from unittest.mock import MagicMock

import torch

from src.retrieval.visual_retriever import VisualRetriever


def test_visual_retriever_search():
    mock_faiss = MagicMock()
    mock_faiss.maxsim_search.return_value = [
        {"page_id": 1, "score": 0.85},
        {"page_id": 2, "score": 0.72},
    ]

    mock_pg = MagicMock()
    mock_pg.get_chunks_by_page_ids.return_value = [
        {"chunk_id": "ch1", "page_id": 1, "text": "Page 1 text", "chunk_type": "text"},
        {"chunk_id": "ch2", "page_id": 2, "text": "Page 2 text", "chunk_type": "text"},
    ]

    mock_colpali = MagicMock()
    mock_colpali.encode_query.return_value = torch.randn(1, 10, 128)

    retriever = VisualRetriever(
        faiss_store=mock_faiss,
        pg_store=mock_pg,
        colpali_embedder=mock_colpali,
    )

    results = retriever.search("test query", k=2)
    assert len(results) == 2
    assert all(r["retrieval_type"] == "visual" for r in results)
    assert results[0]["page_id"] == 1
    assert results[0]["score"] == 0.85
    mock_faiss.maxsim_search.assert_called_once()
    mock_colpali.encode_query.assert_called_once_with("test query")


def test_visual_retriever_search_with_embedding():
    """search_with_embedding() 跳过 encode_query()，直接调用 faiss.maxsim_search"""
    mock_faiss = MagicMock()
    mock_faiss.maxsim_search.return_value = [
        {"page_id": 1, "score": 0.85},
        {"page_id": 2, "score": 0.72},
    ]

    mock_pg = MagicMock()
    mock_pg.get_chunks_by_page_ids.return_value = [
        {"chunk_id": "ch1", "page_id": 1, "text": "Page 1 text", "chunk_type": "text"},
        {"chunk_id": "ch2", "page_id": 2, "text": "Page 2 text", "chunk_type": "text"},
    ]

    mock_colpali = MagicMock()

    retriever = VisualRetriever(
        faiss_store=mock_faiss,
        pg_store=mock_pg,
        colpali_embedder=mock_colpali,
    )

    q_emb = torch.randn(1, 10, 128)
    results = retriever.search_with_embedding(q_emb, k=2)

    assert len(results) == 2
    assert all(r["retrieval_type"] == "visual" for r in results)
    assert results[0]["page_id"] == 1
    assert results[0]["score"] == 0.85
    # 验证没有调用 encode_query
    mock_colpali.encode_query.assert_not_called()
    # 验证调用了 faiss.maxsim_search 且传入了 q_emb
    mock_faiss.maxsim_search.assert_called_once()
    call_args = mock_faiss.maxsim_search.call_args
    assert call_args[0][0] is q_emb  # 同一个 tensor 对象
    assert call_args[1]["k"] == 2


def test_search_with_embedding_returns_same_structure_as_search():
    """search_with_embedding() 与 search() 返回相同结构"""
    mock_faiss = MagicMock()
    mock_faiss.maxsim_search.return_value = [
        {"page_id": 1, "score": 0.85},
    ]
    mock_pg = MagicMock()
    mock_pg.get_chunks_by_page_ids.return_value = [
        {"chunk_id": "ch1", "page_id": 1, "text": "Page 1 text", "chunk_type": "text"},
    ]

    retriever = VisualRetriever(
        faiss_store=mock_faiss,
        pg_store=mock_pg,
        colpali_embedder=MagicMock(),
    )

    q_emb = torch.randn(1, 10, 128)
    results = retriever.search_with_embedding(q_emb, k=1)

    assert len(results) == 1
    assert "chunk_id" in results[0]
    assert "page_id" in results[0]
    assert "score" in results[0]
    assert "retrieval_type" in results[0]
    assert results[0]["retrieval_type"] == "visual"