"""Dense 检索器测试（mock pgvector）"""

from unittest.mock import MagicMock

import numpy as np
import torch

from src.retrieval.dense_retriever import DenseRetriever


def test_dense_retriever_search():
    mock_pg = MagicMock()
    mock_pg.search_by_vector.return_value = [
        {"chunk_id": "ch1", "page_id": 1, "text": "doc text", "score": 0.92},
    ]

    mock_bge = MagicMock()
    mock_bge.encode.return_value = torch.randn(1, 768)

    retriever = DenseRetriever(pg_store=mock_pg, embedder=mock_bge)
    results = retriever.search("test query", k=10)

    assert len(results) == 1
    assert results[0]["retrieval_type"] == "dense"
    mock_bge.encode.assert_called_once()
    mock_pg.search_by_vector.assert_called_once()