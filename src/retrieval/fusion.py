"""融合策略接口 + RRF 融合"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List


class FusionStrategy(ABC):
    """融合策略抽象接口"""

    @abstractmethod
    def fuse(self, results_list: List[List[dict]], k: int) -> List[dict]:
        ...


class RRFFusion(FusionStrategy):
    """RRF 融合: score = Σ 1/(k + rank)"""

    def __init__(self, rrf_k: int = 60):
        self.rrf_k = rrf_k

    def fuse(self, results_list: List[List[dict]], k: int = 20) -> List[dict]:
        rrf_scores: Dict[str, float] = {}
        chunk_map: Dict[str, dict] = {}

        for results in results_list:
            for rank, result in enumerate(results, start=1):
                chunk_id = result["chunk_id"]
                if chunk_id not in rrf_scores:
                    rrf_scores[chunk_id] = 0.0
                    chunk_map[chunk_id] = result
                rrf_scores[chunk_id] += 1.0 / (self.rrf_k + rank)

        sorted_chunks = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

        fused = []
        for chunk_id, score in sorted_chunks[:k]:
            result = dict(chunk_map[chunk_id])
            result["score"] = score
            result["retrieval_type"] = "rrf_fused"
            fused.append(result)

        return fused
