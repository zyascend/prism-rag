"""FAISS ColPali 多向量存储封装

存储结构: page_id → [n_patches, 128] 多向量
查询时执行 MaxSim: score(q_emb, page_emb) = mean(max_j(q_i · p_j))
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import faiss
import numpy as np
import torch

from src.config import cfg


class FaissColPaliStore:
    """FAISS ColPali 多向量存储 + MaxSim 查询"""

    def __init__(self, index_path: str | None = None, id_map_path: str | None = None):
        self.index_path = index_path or cfg.get("storage.faiss.index_path", "indexes/colpali-vidore-industrial.faiss")
        self.id_map_path = id_map_path or cfg.get("storage.faiss.id_map_path", "indexes/colpali-vidore-industrial-ids.npy")
        self._index: Optional[faiss.Index] = None
        self._page_ids: Optional[np.ndarray] = None
        self._page_boundaries: Optional[List[Tuple[int, int]]] = None

    def build_index(self, page_embeddings: Dict[int, torch.Tensor]):
        """从 page_embeddings 构建 FAISS 索引"""
        all_vectors: List[np.ndarray] = []
        all_ids: List[int] = []
        boundaries: List[Tuple[int, int]] = []

        start = 0
        for page_id in sorted(page_embeddings.keys()):
            emb = page_embeddings[page_id].float().numpy().astype(np.float32)
            n = emb.shape[0]
            all_vectors.append(emb)
            all_ids.extend([page_id] * n)
            boundaries.append((start, start + n))
            start += n

        vectors = np.vstack(all_vectors)
        self._page_ids = np.array(all_ids, dtype=np.int64)
        self._page_boundaries = boundaries

        dim = vectors.shape[1]
        self._index = faiss.IndexFlatIP(dim)
        self._index.add(vectors)

        self._num_pages = len(page_embeddings)
        self._num_patches = vectors.shape[0]

    def save(self):
        """保存索引到磁盘"""
        Path(self.index_path).parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, self.index_path)
        np.save(self.id_map_path, self._page_ids)
        print(f"  FAISS 索引已保存: {self.index_path} ({self._num_patches:,} patches, {self._num_pages:,} pages)")

    def load(self) -> bool:
        """从磁盘加载索引，成功返回 True"""
        if not Path(self.index_path).exists():
            return False
        self._index = faiss.read_index(self.index_path)
        self._page_ids = np.load(self.id_map_path)
        # 重建 page_boundaries
        boundaries = []
        start = 0
        cur_id = self._page_ids[0]
        for i, pid in enumerate(self._page_ids):
            if pid != cur_id:
                boundaries.append((start, i))
                start = i
                cur_id = pid
        boundaries.append((start, len(self._page_ids)))
        self._page_boundaries = boundaries
        self._num_pages = len(boundaries)
        self._num_patches = len(self._page_ids)
        print(f"  FAISS 索引已加载: {self.index_path} ({self._num_patches:,} patches, {self._num_pages:,} pages)")
        return True

    def maxsim_search(self, query_embedding: torch.Tensor, k: int = 20) -> List[dict]:
        """MaxSim 搜索"""
        assert self._index is not None, "索引未加载/构建"
        assert self._page_boundaries is not None

        q = query_embedding.numpy().astype(np.float32)
        n_q = q.shape[1]
        q_flat = q.reshape(n_q, -1)

        # 全表矩阵乘: [n_q, 128] @ [total_patches, 128].T → [n_q, total_patches]
        all_vectors = faiss.rev_swig_ptr(self._index.x, self._index.ntotal * self._index.d).reshape(
            self._index.ntotal, self._index.d
        )

        scores = q_flat @ all_vectors.T

        page_scores: Dict[int, float] = {}
        for page_idx, (start, end) in enumerate(self._page_boundaries):
            page_patch_scores = scores[:, start:end]
            max_per_query = page_patch_scores.max(axis=1)
            page_score = float(max_per_query.mean())
            page_id = int(self._page_ids[start])
            page_scores[page_id] = page_score

        sorted_pages = sorted(page_scores.items(), key=lambda x: x[1], reverse=True)
        return [
            {"page_id": page_id, "score": score}
            for page_id, score in sorted_pages[:k]
        ]

    @property
    def index_size_mb(self) -> float:
        if self._index is None:
            return 0.0
        return self._index.ntotal * self._index.d * 4 / (1024 * 1024)

    @property
    def num_pages(self) -> int:
        return getattr(self, "_num_pages", 0)
