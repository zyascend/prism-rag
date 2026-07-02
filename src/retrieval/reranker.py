"""Cross-encoder 重排器"""

from __future__ import annotations

from typing import List

import torch
from sentence_transformers import CrossEncoder

from src.config import cfg


class Reranker:
    """Cross-encoder 重排器

    Args:
        device: 推理设备，默认取 config 中 colpali_device
        model_id: 模型 ID，默认取 config 中 bge_reranker。
                  传入其他 ID（如 zerank-2）可创建不同 reranker 实例。
        automodel_args: 传递给 AutoModel.from_pretrained() 的额外参数，
                       如 {"torch_dtype": torch.bfloat16} 控制加载精度。
    """

    def __init__(self, device: str | None = None, model_id: str | None = None,
                 model_kwargs: dict | None = None):
        self.device = device or cfg.get("embedding.colpali_device", "cpu")
        self.model_id = model_id or cfg.reranker_model_id
        self.model = CrossEncoder(
            self.model_id,
            device=self.device,
            model_kwargs=model_kwargs,
        )

    @torch.no_grad()
    def rerank(self, query: str, candidates: List[dict], top_k: int = 5) -> List[dict]:
        if not candidates:
            return []

        pairs = [(query, c["text"]) for c in candidates]

        # 逐条预测，兼容不支持 batch>1 的模型（如 zerank-2 无 padding token）
        scores = []
        for pair in pairs:
            score = self.model.predict([pair], convert_to_tensor=True)
            scores.append(score.item() if hasattr(score, "item") else float(score))

        scored = list(zip(candidates, scores))
        scored.sort(key=lambda x: x[1], reverse=True)

        results = []
        for cand, score in scored[:top_k]:
            result = dict(cand)
            result["rerank_score"] = float(score)
            result["retrieval_type"] = "reranked"
            results.append(result)

        return results
