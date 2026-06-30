"""RRF 融合测试"""

from src.retrieval.fusion import RRFFusion


def test_rrf_single_route():
    fusion = RRFFusion(rrf_k=60)
    route_a = [
        {"chunk_id": "ch1", "score": 0.9},
        {"chunk_id": "ch2", "score": 0.8},
    ]
    result = fusion.fuse([route_a], k=2)
    assert len(result) == 2
    assert result[0]["chunk_id"] == "ch1"
    assert result[0]["retrieval_type"] == "rrf_fused"


def test_rrf_two_routes():
    fusion = RRFFusion(rrf_k=60)
    route_a = [
        {"chunk_id": "ch1", "score": 0.9},
        {"chunk_id": "ch2", "score": 0.8},
    ]
    route_b = [
        {"chunk_id": "ch2", "score": 0.85},
        {"chunk_id": "ch3", "score": 0.75},
    ]
    result = fusion.fuse([route_a, route_b], k=3)
    assert len(result) == 3
    assert result[0]["chunk_id"] == "ch2"


def test_rrf_empty_input():
    fusion = RRFFusion()
    result = fusion.fuse([[]], k=10)
    assert result == []


def test_rrf_deduplication():
    fusion = RRFFusion()
    route = [
        {"chunk_id": "ch1", "score": 0.9},
        {"chunk_id": "ch1", "score": 0.8},
    ]
    result = fusion.fuse([route], k=5)
    assert len(result) == 1
