"""FAISS ColPali 多向量存储封装

存储结构: page_id → [n_patches, 128] 多向量
查询时执行 MaxSim: score(q_emb, page_emb) = mean(max_j(q_i · p_j))

加速策略:
  - flat: 全局矩阵乘 + 按页 MaxSim（精确，O(N)）
  - hnsw: HNSW 预筛选候选页 → 仅对候选页做精确 MaxSim（近似加速，~140x）
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import faiss
import numpy as np
import torch

from src.config import cfg


class FaissColPaliStore:
    """FAISS ColPali 多向量存储 + MaxSim 查询

    支持两种模式:
      - index_type="flat": 精确全表 MaxSim（小数据集或离线评估）
      - index_type="hnsw": HNSW 预筛 + 候选页精确 MaxSim（在线低延迟）
    """

    def __init__(self, index_path: str | None = None, id_map_path: str | None = None):
        self.index_path = index_path or cfg.get("storage.faiss.index_path", "indexes/colpali-vidore-industrial.faiss")
        self.id_map_path = id_map_path or cfg.get("storage.faiss.id_map_path", "indexes/colpali-vidore-industrial-ids.npy")
        self._index: Optional[faiss.Index] = None
        self._hnsw: Optional[faiss.IndexHNSWFlat] = None
        self._vectors: Optional[np.ndarray] = None  # [total_patches, dim] flat storage for exact MaxSim
        self._page_ids: Optional[np.ndarray] = None
        self._page_boundaries: Optional[List[Tuple[int, int]]] = None
        self._index_type: str = "flat"

    # ── build ──────────────────────────────────────────────

    def build_index(
        self,
        page_embeddings: Dict[int, torch.Tensor],
        index_type: str = "flat",
        hnsw_m: int = 32,
    ):
        """从 page_embeddings 构建 FAISS 索引

        Args:
            page_embeddings: {page_id: tensor[n_patches, 128]}
            index_type: "flat" (精确) 或 "hnsw" (HNSW 预筛加速)
            hnsw_m: HNSW 图的连接数（越大越精确但越慢，默认 32）
        """
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

        self._vectors = np.vstack(all_vectors)
        self._page_ids = np.array(all_ids, dtype=np.int64)
        self._page_boundaries = boundaries
        self._index_type = index_type

        dim = self._vectors.shape[1]
        self._num_pages = len(page_embeddings)
        self._num_patches = self._vectors.shape[0]

        # 始终构建 IndexFlatIP 用于精确 MaxSim（hnsw 模式下也需要 fallback）
        self._index = faiss.IndexFlatIP(dim)
        self._index.add(self._vectors)

        if index_type == "hnsw":
            self._hnsw = faiss.IndexHNSWFlat(dim, hnsw_m, faiss.METRIC_INNER_PRODUCT)
            self._hnsw.hnsw.efConstruction = 200
            self._hnsw.add(self._vectors)
            print(f"  HNSW 索引已构建: M={hnsw_m}, efConstruction=200")
        else:
            self._hnsw = None

        print(f"  FAISS 索引: {self._num_patches:,} patches, {self._num_pages:,} pages, type={index_type}")

    # ── save / load ────────────────────────────────────────

    def save(self):
        """保存索引到磁盘"""
        out_dir = Path(self.index_path).parent
        out_dir.mkdir(parents=True, exist_ok=True)

        base = Path(self.index_path).with_suffix("")

        # 保存 vectors 和 page_ids（供精确 MaxSim 使用）
        vectors_path = str(base) + "_vectors.npy"
        np.save(vectors_path, self._vectors)
        np.save(self.id_map_path, self._page_ids)

        # 保存 FAISS 索引
        faiss.write_index(self._index, self.index_path)

        if self._hnsw is not None:
            hnsw_path = str(base) + "_hnsw.faiss"
            faiss.write_index(self._hnsw, hnsw_path)

        print(f"  FAISS 索引已保存: {self.index_path} ({self._num_patches:,} patches, {self._num_pages:,} pages)")

    def load(self) -> bool:
        """从磁盘加载索引，成功返回 True"""
        if not Path(self.index_path).exists():
            return False

        base = Path(self.index_path).with_suffix("")

        # 加载 vectors
        vectors_path = str(base) + "_vectors.npy"
        if Path(vectors_path).exists():
            self._vectors = np.load(vectors_path)
        else:
            # 兼容旧格式：从 IndexFlatIP 提取
            self._index = faiss.read_index(self.index_path)
            ntotal = self._index.ntotal
            d = self._index.d
            self._vectors = faiss.rev_swig_ptr(self._index.x, ntotal * d).reshape(ntotal, d).copy()

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

        # 尝试加载 HNSW
        hnsw_path = str(base) + "_hnsw.faiss"
        if Path(hnsw_path).exists():
            self._hnsw = faiss.read_index(hnsw_path)
            self._index_type = "hnsw"
        else:
            self._hnsw = None
            self._index_type = "flat"

        print(f"  FAISS 索引已加载: {self.index_path} "
              f"({self._num_patches:,} patches, {self._num_pages:,} pages, type={self._index_type})")
        return True

    # ── search ─────────────────────────────────────────────

    def maxsim_search(self, query_embedding: torch.Tensor, k: int = 20) -> List[dict]:
        """MaxSim 搜索：返回 Top-k 页（含分数）

        策略：
          - flat 模式：全局矩阵乘 → 按页 MaxSim（精确，O(N)）
          - hnsw 模式：HNSW 预筛候选页 → 仅候选页精确 MaxSim（加速）
        """
        assert self._vectors is not None, "索引未加载/构建"
        assert self._page_boundaries is not None

        q = query_embedding.float().numpy().astype(np.float32)  # [n_q, 128]
        n_q = q.shape[1]

        if self._hnsw is not None:
            return self._maxsim_hnsw(q, n_q, k)
        else:
            return self._maxsim_exact(q, n_q, k)

    def _maxsim_exact(self, q: np.ndarray, n_q: int, k: int) -> List[dict]:
        """精确全局 MaxSim（flat 模式）"""
        q_flat = q.reshape(n_q, -1)
        scores = q_flat @ self._vectors.T  # [n_q, total_patches]

        page_scores = self._compute_page_scores(scores)
        return self._rank_pages(page_scores, k)

    def _maxsim_hnsw(self, q: np.ndarray, n_q: int, k: int) -> List[dict]:
        """HNSW 预筛 → 候选页精确 MaxSim"""
        # 1. 每个 query patch 用 HNSW 找 top-M nearest patches
        M = 50  # 每个 query patch 召回的 patch 数
        candidate_page_ids: set[int] = set()

        for i in range(n_q):
            q_vec = q[:, i : i + 1].T  # [1, dim]
            _, patch_indices = self._hnsw.search(q_vec, M)
            for idx in patch_indices[0]:
                if idx >= 0:
                    page_id = int(self._page_ids[idx])
                    candidate_page_ids.add(page_id)

        if not candidate_page_ids:
            return []

        # 2. 构建候选页的 patch mask
        candidate_mask = np.isin(self._page_ids, list(candidate_page_ids))

        # 3. 仅对候选 patches 计算精确 dot product
        q_flat = q.reshape(n_q, -1)
        candidate_vectors = self._vectors[candidate_mask]  # [candidate_patches, dim]
        candidate_scores = q_flat @ candidate_vectors.T  # [n_q, candidate_patches]

        # 4. 映射回 candidate indices → patch index，计算页面分数
        candidate_indices = np.where(candidate_mask)[0]
        candidate_page_ids_arr = self._page_ids[candidate_mask]

        page_scores: Dict[int, float] = {}
        page_patch_map: Dict[int, List[int]] = {}
        for ci, pid in enumerate(candidate_page_ids_arr):
            pid_int = int(pid)
            if pid_int not in page_patch_map:
                page_patch_map[pid_int] = []
            page_patch_map[pid_int].append(ci)

        for pid, ci_list in page_patch_map.items():
            page_patch_scores = candidate_scores[:, ci_list]
            max_per_query = page_patch_scores.max(axis=1)
            page_score = float(max_per_query.mean())
            page_scores[pid] = page_score

        return self._rank_pages(page_scores, k)

    def _compute_page_scores(self, scores: np.ndarray) -> Dict[int, float]:
        """从 patch 级分数矩阵 [n_q, total_patches] 计算页级 MaxSim 分数"""
        page_scores: Dict[int, float] = {}
        for start, end in self._page_boundaries:
            page_patch_scores = scores[:, start:end]
            max_per_query = page_patch_scores.max(axis=1)
            page_score = float(max_per_query.mean())
            page_id = int(self._page_ids[start])
            page_scores[page_id] = page_score
        return page_scores

    @staticmethod
    def _rank_pages(page_scores: Dict[int, float], k: int) -> List[dict]:
        sorted_pages = sorted(page_scores.items(), key=lambda x: x[1], reverse=True)
        return [
            {"page_id": page_id, "score": score}
            for page_id, score in sorted_pages[:k]
        ]

    # ── properties ─────────────────────────────────────────

    @property
    def index_size_mb(self) -> float:
        size = 0.0
        if self._vectors is not None:
            size += self._vectors.nbytes / (1024 * 1024)
        if self._hnsw is not None:
            # HNSW 图结构的额外内存 ≈ M × 2 × num_patches × 4 bytes
            size += self._hnsw.ntotal * 32 * 2 * 4 / (1024 * 1024)
        return size

    @property
    def num_pages(self) -> int:
        return getattr(self, "_num_pages", 0)

    @property
    def index_type(self) -> str:
        return self._index_type
