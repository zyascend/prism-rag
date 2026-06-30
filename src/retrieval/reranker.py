"""Cross-encoder 重排器"""

from __future__ import annotations

from typing import List

import torch
from sentence_transformers import CrossEncoder

from src.config import cfg


class Reranker:
    """Cross-encoder 重排器"""

    def __init__(self, device: str | None = None):
        self.device = device or cfg.get("embedding.colpali_device", "cpu")
        self.model = CrossEncoder(
            cfg.reranker_model_id,
            device=self.device,
        )

    @torch.no_grad()
    def rerank(self, query: str, candidates: List[dict], top_k: int = 5) -> List[dict]:
        if not candidates:
            return []

        pairs = [(query, c["text"]) for c in candidates]
        scores = self.model.predict(pairs, convert_to_tensor=True)

        scored = list(zip(candidates, scores.tolist()))
        scored.sort(key=lambda x: x[1], reverse=True)

        results = []
        for cand, score in scored[:top_k]:
            result = dict(cand)
            result["rerank_score"] = float(score)
            result["retrieval_type"] = "reranked"
            results.append(result)

        return results
