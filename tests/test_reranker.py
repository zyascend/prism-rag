"""Reranker 测试（mock model）"""

from unittest.mock import MagicMock, patch

import numpy as np

from src.retrieval.reranker import Reranker


@patch("src.retrieval.reranker.CrossEncoder")
def test_reranker_basic(MockCrossEncoder):
    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([0.95, 0.80, 0.70])
    MockCrossEncoder.return_value = mock_model

    reranker = Reranker(device="cpu")
    candidates = [
        {"chunk_id": "ch1", "text": "document about conveyor belt"},
        {"chunk_id": "ch2", "text": "safety guidelines unrelated"},
        {"chunk_id": "ch3", "text": "load capacity specs"},
    ]

    results = reranker.rerank("conveyor belt load capacity", candidates, top_k=2)
    assert len(results) == 2
    assert results[0]["retrieval_type"] == "reranked"
    assert "rerank_score" in results[0]


@patch("src.retrieval.reranker.CrossEncoder")
def test_reranker_empty_candidates(MockCrossEncoder):
    reranker = Reranker(device="cpu")
    results = reranker.rerank("test query", [], top_k=5)
    assert results == []
