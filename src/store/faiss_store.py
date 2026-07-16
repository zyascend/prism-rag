"""FAISS ColPali 多向量存储封装

存储结构: page_id → [n_patches, 128] 多向量
查询时执行 MaxSim: score(q_emb, page_emb) = mean(max_j(q_i · p_j))

加速策略:
  - flat: 全局矩阵乘 + 按页 MaxSim（精确，O(N)）
  - hnsw: HNSW 预筛选候选页 → 仅对候选页做精确 MaxSim（近似加速，~140x）
  - GPU: 自动检测 CUDA，MaxSim 走 torch GPU 矩阵乘（50-100x vs numpy CPU）
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import faiss
import numpy as np
import torch

from src.config import cfg

logger = logging.getLogger(__name__)


class FaissColPaliStore:
    """FAISS ColPali 多向量存储 + MaxSim 查询

    支持三种加速:
      - flat: 精确全表 MaxSim（评测用）
      - hnsw: HNSW 预筛 + 候选页精确 MaxSim（在线低延迟）
      - GPU: CUDA 可用时自动将 MaxSim 矩阵乘切到 torch GPU
    """

    def __init__(self, index_path: str | None = None, id_map_path: str | None = None):
        self.index_path = index_path or cfg.get("storage.faiss.index_path", "indexes/colpali-vidore-industrial.faiss")
        self.id_map_path = id_map_path or cfg.get("storage.faiss.id_map_path", "indexes/colpali-vidore-industrial-ids.npy")
        self._index: Optional[faiss.Index] = None
        self._hnsw: Optional[faiss.IndexHNSWFlat] = None
        self._vectors: Optional[np.ndarray] = None          # numpy (FAISS 兼容)
        self._vectors_torch: Optional[torch.Tensor] = None  # torch GPU（MaxSim 用）
        self._page_ids: Optional[np.ndarray] = None
        self._page_boundaries: Optional[List[Tuple[int, int]]] = None
        self._index_type: str = "flat"
        self._device: Optional[torch.device] = None
        # P1: 逻辑删除（墓碑）与 page→doc / page→hash 映射
        self._deleted_page_ids: Set[int] = set()        # 墓碑：被删除但物理仍在的 page_id
        self._page_doc_ids: Dict[int, str] = {}         # page_id -> doc_id（删除编排用）
        self._page_hashes: Dict[int, str] = {}          # page_id -> content_hash（P2 diff 用）

    @property
    def _gpu_available(self) -> bool:
        return torch.cuda.is_available()

    # ── build ──────────────────────────────────────────────

    def build_index(
        self,
        page_embeddings: Dict[int, torch.Tensor],
        index_type: str = "flat",
        hnsw_m: int = 32,
        page_doc_ids: Optional[Dict[int, str]] = None,
        page_hashes: Optional[Dict[int, str]] = None,
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
        self._page_doc_ids = dict(page_doc_ids) if page_doc_ids else {}
        self._page_hashes = dict(page_hashes) if page_hashes else {}

        dim = self._vectors.shape[1]
        self._num_pages = len(page_embeddings)
        self._num_patches = self._vectors.shape[0]

        # GPU 加速：将向量也存为 torch tensor
        if self._gpu_available:
            self._device = torch.device("cuda")
            self._vectors_torch = torch.from_numpy(self._vectors).to(self._device)
            vram_mb = self._vectors_torch.element_size() * self._vectors_torch.numel() / (1024 * 1024)
            print(f"  GPU MaxSim 已启用: {vram_mb:.0f} MB VRAM ({self._num_patches:,} patches)")
        else:
            self._device = torch.device("cpu")
            self._vectors_torch = None

        # 始终构建 IndexFlatIP（FAISS 兼容、save/load）
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

        # 保存 P1 新增的映射（墓碑 / page→doc / page→hash），保证重启后删除一致性
        self._save_auxiliary(base)

        print(f"  FAISS 索引已保存: {self.index_path} ({self._num_patches:,} patches, {self._num_pages:,} pages)")

    def _save_auxiliary(self, base: Path):
        """保存墓碑 / page→doc / page→hash 映射（JSON + npy）"""
        doc_map_path = str(base) + "_pagedoc.json"
        with open(doc_map_path, "w", encoding="utf-8") as f:
            json.dump({str(k): v for k, v in self._page_doc_ids.items()}, f)
        hash_path = str(base) + "_pagehash.json"
        with open(hash_path, "w", encoding="utf-8") as f:
            json.dump({str(k): v for k, v in self._page_hashes.items()}, f)
        del_path = str(base) + "_deleted.npy"
        np.save(del_path, np.array(sorted(self._deleted_page_ids), dtype=np.int64))

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
            old_index = faiss.read_index(self.index_path)
            ntotal = old_index.ntotal
            d = old_index.d
            self._vectors = faiss.rev_swig_ptr(old_index.x, ntotal * d).reshape(ntotal, d).copy()

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

        # 恢复 P1 映射（墓碑 / page→doc / page→hash）
        self._load_auxiliary(base)

        # GPU 加速
        if self._gpu_available:
            self._device = torch.device("cuda")
            self._vectors_torch = torch.from_numpy(self._vectors).to(self._device)
            vram_mb = self._vectors_torch.element_size() * self._vectors_torch.numel() / (1024 * 1024)
            print(f"  GPU MaxSim 已启用: {vram_mb:.0f} MB VRAM")
        else:
            self._device = torch.device("cpu")
            self._vectors_torch = None

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

    def _load_auxiliary(self, base: Path):
        """从磁盘恢复 P1 映射；旧索引缺文件时安全降级为空"""
        doc_map_path = str(base) + "_pagedoc.json"
        if Path(doc_map_path).exists():
            with open(doc_map_path, encoding="utf-8") as f:
                self._page_doc_ids = {int(k): v for k, v in json.load(f).items()}
        else:
            self._page_doc_ids = {}

        hash_path = str(base) + "_pagehash.json"
        if Path(hash_path).exists():
            with open(hash_path, encoding="utf-8") as f:
                self._page_hashes = {int(k): v for k, v in json.load(f).items()}
        else:
            self._page_hashes = {}

        del_path = str(base) + "_deleted.npy"
        if Path(del_path).exists():
            self._deleted_page_ids = set(int(x) for x in np.load(del_path))
        else:
            self._deleted_page_ids = set()

    # ── P2-C: 大批量原子快照切换（零停机刷新）──────────────

    @staticmethod
    def _aux_paths(base: Path) -> Dict[str, Path]:
        """返回某索引基名对应的辅助文件路径映射（vectors / 映射 / 墓碑 / hnsw）"""
        return {
            "vectors": base.with_name(base.name + "_vectors.npy"),
            "pagedoc": base.with_name(base.name + "_pagedoc.json"),
            "pagehash": base.with_name(base.name + "_pagehash.json"),
            "deleted": base.with_name(base.name + "_deleted.npy"),
            "hnsw": base.with_name(base.name + "_hnsw.faiss"),
        }

    def apply_snapshot(self, snapshot_index_path: str, snapshot_id_map_path: str):
        """将「完整一致的快照索引」原子覆盖到当前 live 路径（§4.6 FAISS 部分）。

        调用方应先以 snapshot 路径 `build_index` + `save` 出一个完整快照集，
        再调用本方法：把快照的全部文件 `os.replace` 到 live 路径。
        检索进程内存中持有旧 vectors（不受影响）；新查询 `load()` 后读到新文件，
        全程无需下线。覆盖后需调用 `load()` 生效。
        """
        snap_idx = Path(snapshot_index_path)
        snap_ids = Path(snapshot_id_map_path)
        live_base = Path(self.index_path).with_suffix("")
        snap_base = snap_idx.with_suffix("")
        live_aux = self._aux_paths(live_base)
        snap_aux = self._aux_paths(snap_base)

        # 主索引 + id_map 原子替换
        pairs = [(snap_idx, Path(self.index_path)), (snap_ids, Path(self.id_map_path))]
        # 辅助文件按 key 对齐替换（仅替换快照中存在的）
        for key in ("vectors", "pagedoc", "pagehash", "deleted"):
            if snap_aux[key].exists():
                pairs.append((snap_aux[key], live_aux[key]))
        if snap_aux["hnsw"].exists():
            pairs.append((snap_aux["hnsw"], live_aux["hnsw"]))

        for src, dst in pairs:
            if src.exists():
                os.replace(src, dst)
        logger.info("FAISS 快照已原子切换：%d 个文件覆盖到 %s", len(pairs), self.index_path)

    # ── search ─────────────────────────────────────────────

    def maxsim_search(self, query_embedding: torch.Tensor, k: int = 20) -> List[dict]:
        """MaxSim 搜索：返回 Top-k 页（含分数）

        策略（按优先级）:
          1. GPU 可用 → torch GPU 矩阵乘（最快）
          2. HNSW 预筛 → 候选页精确 MaxSim
          3. flat → numpy CPU 全量 MaxSim
        """
        assert self._vectors is not None, "索引未加载/构建"
        assert self._page_boundaries is not None

        # GPU 路径
        if self._vectors_torch is not None:
            return self._maxsim_torch(query_embedding, k)

        # CPU 路径
        q = query_embedding.float().numpy().astype(np.float32)
        n_q = q.shape[1]

        if self._hnsw is not None:
            return self._maxsim_hnsw(q, n_q, k)
        else:
            return self._maxsim_exact(q, n_q, k)

    # ── GPU MaxSim ─────────────────────────────────────────

    def _maxsim_torch(self, query_embedding: torch.Tensor, k: int) -> List[dict]:
        """torch GPU 加速 MaxSim（分页批处理避免 OOM）

        ColPali query 可产出 ~1000+ patches，全量 matmul 的中间矩阵
        [n_q, total_patches] ≈ 1050 × 5.3M × 4B ≈ 21 GB。按页 batch 计算，
        每个 batch 的中间矩阵 [n_q, batch_patches] ≤ 几百 MB，安全可控。
        """
        q = query_embedding.float().to(self._device)         # [1, n_q, 128]
        n_q = q.shape[1]
        q_flat = q.reshape(n_q, -1)                          # [n_q, 128]

        page_scores: Dict[int, float] = {}

        # 按页 batch 计算，每个 batch 最多 200 页（~200k patches → ~800 MB intermediate）
        PAGE_BATCH = 200
        boundaries = self._page_boundaries
        for b_start in range(0, len(boundaries), PAGE_BATCH):
            b_end = min(b_start + PAGE_BATCH, len(boundaries))
            batch_start_patch = boundaries[b_start][0]
            batch_end_patch = boundaries[b_end - 1][1]

            # 该 batch 的所有 patches → [n_q, batch_patches]
            batch_vectors = self._vectors_torch[batch_start_patch:batch_end_patch]
            batch_scores = q_flat @ batch_vectors.T           # [n_q, batch_patches]

            # 对该 batch 内每页计算 MaxSim
            for i in range(b_start, b_end):
                start, end = boundaries[i]
                local_start = start - batch_start_patch
                local_end = end - batch_start_patch
                page_patch_scores = batch_scores[:, local_start:local_end]
                max_per_query = page_patch_scores.max(dim=1).values
                page_score = float(max_per_query.mean().cpu())
                page_id = int(self._page_ids[start])
                page_scores[page_id] = page_score

        return self._rank_pages(page_scores, k)

    # ── CPU MaxSim (fallback) ──────────────────────────────

    def _maxsim_exact(self, q: np.ndarray, n_q: int, k: int) -> List[dict]:
        """精确全局 MaxSim（numpy CPU）"""
        q_flat = q.reshape(n_q, -1)
        scores = q_flat @ self._vectors.T

        page_scores = self._compute_page_scores(scores)
        return self._rank_pages(page_scores, k)

    def _maxsim_hnsw(self, q: np.ndarray, n_q: int, k: int) -> List[dict]:
        """HNSW 预筛 → 候选页精确 MaxSim"""
        M = 50
        candidate_page_ids: set[int] = set()

        for i in range(n_q):
            q_vec = q[:, i : i + 1].T
            _, patch_indices = self._hnsw.search(q_vec, M)
            for idx in patch_indices[0]:
                if idx >= 0:
                    candidate_page_ids.add(int(self._page_ids[idx]))

        if not candidate_page_ids:
            return []

        candidate_mask = np.isin(self._page_ids, list(candidate_page_ids))
        q_flat = q.reshape(n_q, -1)
        candidate_vectors = self._vectors[candidate_mask]
        candidate_scores = q_flat @ candidate_vectors.T

        page_scores: Dict[int, float] = {}
        page_patch_map: Dict[int, List[int]] = {}
        candidate_page_ids_arr = self._page_ids[candidate_mask]
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

    def _rank_pages(self, page_scores: Dict[int, float], k: int) -> List[dict]:
        """页级打分排序，自动过滤墓碑页（P1 删除一致性核心）"""
        deleted = self._deleted_page_ids
        filtered = [(pid, s) for pid, s in page_scores.items() if pid not in deleted]
        sorted_pages = sorted(filtered, key=lambda x: x[1], reverse=True)
        return [
            {"page_id": page_id, "score": score}
            for page_id, score in sorted_pages[:k]
        ]

    # ── incremental add ────────────────────────────────────

    def add_pages(
        self,
        page_embeddings: Dict[int, torch.Tensor],
        page_doc_ids: Optional[Dict[int, str]] = None,
        page_hashes: Optional[Dict[int, str]] = None,
    ):
        """增量写入多向量页面。首次调用（无索引）时等价于 build_index。

        Args:
            page_embeddings: {page_id: tensor[n_patches, 128]}
            page_doc_ids: {page_id: doc_id}，供 delete_by_doc_id 按文档删除（P1）
            page_hashes: {page_id: content_hash}，供 P2 按页 diff（P1 仅存储）
        """
        dim = 128
        if self._vectors is None:
            self._vectors = np.empty((0, dim), dtype=np.float32)
            self._page_ids = np.empty((0,), dtype=np.int64)
            self._page_boundaries = []
            self._index_type = cfg.get("storage.faiss.index_type", "flat")
            self._index = faiss.IndexFlatIP(dim)
            self._device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
            self._vectors_torch = None

        start = self._vectors.shape[0]
        new_vecs, new_ids, new_bounds = [], [], []
        for pid in sorted(page_embeddings.keys()):
            emb = page_embeddings[pid].float().numpy().astype(np.float32)
            n = emb.shape[0]
            new_vecs.append(emb)
            new_ids.extend([int(pid)] * n)
            new_bounds.append((start, start + n))
            start += n
        if not new_vecs:
            return
        nv = np.vstack(new_vecs)
        self._vectors = np.vstack([self._vectors, nv])
        self._page_ids = np.concatenate([self._page_ids, np.array(new_ids, dtype=np.int64)])
        self._page_boundaries.extend(new_bounds)
        self._index.add(nv)
        if self._index_type == "hnsw":
            if self._hnsw is None:
                hnsw_m = cfg.get("storage.faiss.hnsw_m", 32)
                self._hnsw = faiss.IndexHNSWFlat(self._vectors.shape[1], hnsw_m, faiss.METRIC_INNER_PRODUCT)
                self._hnsw.hnsw.efConstruction = 200
            self._hnsw.add(nv)
        # 记录映射（P1）
        self._page_doc_ids.update(page_doc_ids or {})
        self._page_hashes.update(page_hashes or {})
        self._num_pages = len(self._page_boundaries)
        self._num_patches = self._vectors.shape[0]
        if self._device.type == "cuda":
            self._vectors_torch = torch.from_numpy(self._vectors).to(self._device)

    # ── P1: 逻辑删除（墓碑）+ 异步压缩 ────────────────────

    def delete_by_page_ids(self, page_ids: List[int]) -> int:
        """墓碑删除：把 page 加入已删集合（不立即物理删，避免破坏 HNSW 图结构）。

        返回本次新增墓碑数。
        """
        before = len(self._deleted_page_ids)
        self._deleted_page_ids.update(int(p) for p in page_ids)
        return len(self._deleted_page_ids) - before

    def delete_by_doc_id(self, doc_id: str) -> int:
        """按 doc_id 删除（墓碑）。依赖 _page_doc_ids 映射（add_pages 时写入）。

        若索引来自 P1 之前的旧版本（缺 _page_doc_ids），返回 0 并告警，
        不崩溃——此时应走删旧索引重建的路径。
        """
        target = [pid for pid, d in self._page_doc_ids.items() if d == doc_id]
        if not target:
            logger.warning(
                "FAISS 无 doc_id=%s 的 page 映射（旧索引或未记录），跳过删除", doc_id
            )
            return 0
        return self.delete_by_page_ids(target)

    @property
    def tombstone_ratio(self) -> float:
        """墓碑页占比，用于决定是否触发压缩"""
        if self._num_pages == 0:
            return 0.0
        return len(self._deleted_page_ids) / self._num_pages

    def maybe_compact(self, threshold: float = 0.2) -> bool:
        """墓碑占比超过阈值则压缩（当前同步执行；异步线程可后续叠加）。

        返回本次是否触发了压缩。
        """
        if self.tombstone_ratio >= threshold and self._deleted_page_ids:
            self.compact()
            return True
        return False

    def compact(self):
        """墓碑物理回收：重建索引仅保留未删除的页，并清理相关映射。

        HNSW 不支持高效中间行物理删除（删节点会破坏图结构），
        因此采用「重建」而非「in-place 删」，与 Milvus/Weaviate 内部一致。
        """
        if not self._deleted_page_ids:
            return
        deleted = self._deleted_page_ids
        kept_bounds = [
            (s, e) for (s, e) in self._page_boundaries
            if int(self._page_ids[s]) not in deleted
        ]

        if not kept_bounds:
            # 全部删除 → 清空索引为空
            dim = self._vectors.shape[1]
            self._vectors = np.empty((0, dim), dtype=np.float32)
            self._page_ids = np.empty((0,), dtype=np.int64)
            self._page_boundaries = []
            self._index = faiss.IndexFlatIP(dim)
            self._hnsw = None
        else:
            kept_idx = np.concatenate([np.arange(s, e) for (s, e) in kept_bounds])
            self._vectors = self._vectors[kept_idx]
            new_page_ids: List[int] = []
            new_bounds: List[Tuple[int, int]] = []
            offset = 0
            for (s, e) in kept_bounds:
                n = e - s
                new_page_ids.extend([int(self._page_ids[s])] * n)
                new_bounds.append((offset, offset + n))
                offset += n
            self._page_ids = np.array(new_page_ids, dtype=np.int64)
            self._page_boundaries = new_bounds
            self._index = faiss.IndexFlatIP(self._vectors.shape[1])
            self._index.add(self._vectors)
            if self._index_type == "hnsw":
                hnsw_m = cfg.get("storage.faiss.hnsw_m", 32)
                self._hnsw = faiss.IndexHNSWFlat(self._vectors.shape[1], hnsw_m, faiss.METRIC_INNER_PRODUCT)
                self._hnsw.hnsw.efConstruction = 200
                self._hnsw.add(self._vectors)
            else:
                self._hnsw = None
            if self._device.type == "cuda":
                self._vectors_torch = torch.from_numpy(self._vectors).to(self._device)

        # 同步清理映射
        self._page_doc_ids = {pid: d for pid, d in self._page_doc_ids.items() if pid not in deleted}
        self._page_hashes = {pid: h for pid, h in self._page_hashes.items() if pid not in deleted}
        self._deleted_page_ids = set()
        self._num_pages = len(self._page_boundaries)
        self._num_patches = self._vectors.shape[0]
        logger.info("FAISS compact 完成：回收 %d 个墓碑页，剩余 %d 页", len(deleted), self._num_pages)

    # ── properties ─────────────────────────────────────────

    @property
    def index_size_mb(self) -> float:
        size = 0.0
        if self._vectors is not None:
            size += self._vectors.nbytes / (1024 * 1024)
        if self._hnsw is not None:
            size += self._hnsw.ntotal * 32 * 2 * 4 / (1024 * 1024)
        return size

    @property
    def num_pages(self) -> int:
        return getattr(self, "_num_pages", 0)

    @property
    def index_type(self) -> str:
        return self._index_type
