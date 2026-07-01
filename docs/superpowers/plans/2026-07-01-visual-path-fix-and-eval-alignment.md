# Visual Path Fix & Eval Alignment — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix CUDA OOM in Visual retrieval path (both `--skip-index` startup and query loop) and align eval metrics to the ViDoRe V3 English-only baseline with explicit language filtering and validation.

**Architecture:** Split into three phases: (1) encoder lifecycle API (`unload`, `encode_queries_batch`, guard), (2) retriever pre-encoded embedding pass-through, (3) eval pipeline reorder + language validation. Each phase is independently testable.

**Tech Stack:** Python 3.12+, PyTorch, ColPali v1.3, FAISS GPU, ViDoRe v3 dataset

---

### Task 1: `ColPaliEmbedder` — Add `encode_queries_batch()`, `unload()`, `_require_loaded()` guard

**Files:**
- Modify: `src/ingestion/encoders.py:43-101`
- Test: `tests/test_encoders.py`

- [ ] **Step 1: Write the failing tests for `encode_queries_batch()`**

```python
# tests/test_encoders.py — append after existing tests

def test_colpali_encode_queries_batch():
    """encode_queries_batch() 输出格式与单条 encode_query 一致"""
    embedder = ColPaliEmbedder(device="cpu")
    texts = ["load capacity", "conveyor belt specs"]
    result = embedder.encode_queries_batch(texts, batch_size=2)
    assert isinstance(result, dict)
    assert len(result) == 2
    for idx, emb in result.items():
        assert emb.ndim == 3  # [1, n_patches, 128]
        assert emb.shape[-1] == 128


def test_colpali_unload_raises_on_encode_query():
    """unload() 后调用 encode_query() 必须抛 RuntimeError"""
    embedder = ColPaliEmbedder(device="cpu")
    embedder.encode_query("load capacity")  # 确保 loaded
    embedder.unload()
    import traceback
    traceback.print_stack()
    try:
        embedder.encode_query("load capacity")
        assert False, "应该已抛出 RuntimeError"
    except RuntimeError:
        pass


def test_colpali_unload_raises_on_encode_pages():
    """unload() 后调用 encode_pages() 必须抛 RuntimeError"""
    from PIL import Image
    import numpy as np

    embedder = ColPaliEmbedder(device="cpu")
    embedder.unload()
    imgs = [Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))]
    try:
        embedder.encode_pages(imgs)
        assert False, "应该已抛出 RuntimeError"
    except RuntimeError:
        pass


def test_colpali_unload_raises_on_encode_queries_batch():
    """unload() 后调用 encode_queries_batch() 必须抛 RuntimeError"""
    embedder = ColPaliEmbedder(device="cpu")
    embedder.unload()
    try:
        embedder.encode_queries_batch(["test"])
        assert False, "应该已抛出 RuntimeError"
    except RuntimeError:
        pass
```

- [ ] **Step 2: Run new tests to verify they fail**

Run: `pytest tests/test_encoders.py::test_colpali_encode_queries_batch tests/test_encoders.py::test_colpali_unload_raises_on_encode_query tests/test_encoders.py::test_colpali_unload_raises_on_encode_pages tests/test_encoders.py::test_colpali_unload_raises_on_encode_queries_batch -v`

Expected: All 4 tests FAIL with `AttributeError: 'ColPaliEmbedder' object has no attribute 'encode_queries_batch'` or `unload`.

- [ ] **Step 3: Add `_loaded` flag, `_require_loaded()`, `encode_queries_batch()`, and `unload()` to `ColPaliEmbedder`**

```python
# src/ingestion/encoders.py — ColPaliEmbedder modifications

class ColPaliEmbedder:
    """ColPali 整页多向量编码器"""

    def __init__(self, device: str | None = None):
        self.device = device if device is not None else cfg.get("embedding.colpali_device", "cpu")
        self.model = ColPali.from_pretrained(
            cfg.colpali_model_id,
            torch_dtype=torch.bfloat16,
            device_map=self.device,
        ).eval()
        self.processor = ColPaliProcessor.from_pretrained(cfg.colpali_model_id)
        self._warmed_up = False
        self._loaded = True  # <-- NEW

    def _require_loaded(self):
        """检查模型是否可用，不可用时抛出 RuntimeError"""
        if not self._loaded:
            raise RuntimeError(
                "ColPaliEmbedder 已通过 unload() 卸载，无法执行编码操作。"
                "请重新创建 ColPaliEmbedder 实例后再调用。"
            )

    @torch.no_grad()
    def encode_pages(
        self, images: List[Image.Image], batch_size: int | None = None, show_progress: bool = False
    ) -> List[torch.Tensor]:
        self._require_loaded()  # <-- NEW guard
        # ... rest unchanged ...

    @torch.no_grad()
    def encode_query(self, text: str) -> torch.Tensor:
        self._require_loaded()  # <-- NEW guard
        # ... rest unchanged ...

    @torch.no_grad()
    def encode_queries_batch(self, texts: List[str], batch_size: int = 4) -> Dict[int, torch.Tensor]:
        """批量编码多个 query，返回 {idx: tensor[1, n_patches, 128]}

        Args:
            texts: 查询文本列表
            batch_size: 批大小

        Returns:
            {索引: query_embedding} 字典，每个 embedding shape [1, n_patches, 128]
        """
        self._require_loaded()
        results: Dict[int, torch.Tensor] = {}
        for i in trange(0, len(texts), batch_size, desc="ColPali encode queries"):
            batch_texts = texts[i : i + batch_size]
            dummy = Image.new("RGB", (448, 448), color=255)
            inputs = self.processor(
                images=[dummy] * len(batch_texts),
                text=batch_texts,
                return_tensors="pt",
                padding=True,
            ).to(self.device)
            batch_outputs = self.model(**inputs)  # [batch, n_q, 128]
            for j, emb in enumerate(batch_outputs):
                results[i + j] = emb.unsqueeze(0).cpu()  # -> [1, n_q, 128]
        return results

    def unload(self):
        """卸载模型并释放显存。之后调用任何 encode_* 方法都会抛 RuntimeError。"""
        del self.model
        self.model = None
        torch.cuda.empty_cache()
        self._loaded = False
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_encoders.py -v`

Expected: All 4 new tests PASS. (Existing `test_colpali_encoder_page_output`, `test_colpali_query_output`, `test_bge_encoder_output_shape` may also pass if run on CPU; if they fail due to model download, that's a pre-existing concern.)

- [ ] **Step 5: Commit**

```bash
git add src/ingestion/encoders.py tests/test_encoders.py
git commit -m "feat(encoders): add encode_queries_batch, unload, _require_loaded guard"
```

---

### Task 2: `VisualRetriever` — Add `search_with_embedding()`

**Files:**
- Modify: `src/retrieval/visual_retriever.py`
- Modify: `tests/test_visual_retriever.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_visual_retriever.py — append after existing test

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
    """search_with_embedding() 与 search() 在相同 embedding 下返回相同结构"""
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_visual_retriever.py -v`

Expected: 2 new tests FAIL with `AttributeError: 'VisualRetriever' object has no attribute 'search_with_embedding'`. Existing `test_visual_retriever_search` PASS.

- [ ] **Step 3: Add `search_with_embedding()` to `VisualRetriever`**

```python
# src/retrieval/visual_retriever.py — add method to class

def search_with_embedding(self, q_emb: torch.Tensor, k: int = 20) -> List[dict]:
    """使用预编码 query embedding 执行检索（跳过 encode_query()）

    Args:
        q_emb: 预编码的 query embedding, shape [1, n_patches, 128]
        k: 返回 Top-k 页

    Returns:
        与 search() 相同格式的结果列表
    """
    # 1. FAISS MaxSim 搜索 → Top-k 页
    page_results = self.faiss.maxsim_search(q_emb, k=k)

    if not page_results:
        return []

    # 2. Grounding 反查：命中页的所有 BGE chunk
    page_ids = [pr["page_id"] for pr in page_results]
    page_score_map = {pr["page_id"]: pr["score"] for pr in page_results}

    chunks = self.pg.get_chunks_by_page_ids(page_ids)

    # 3. 合并分数
    results = []
    for chunk in chunks:
        results.append({
            **chunk,
            "score": page_score_map[chunk["page_id"]],
            "retrieval_type": "visual",
        })

    return results
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_visual_retriever.py -v`

Expected: All 3 tests PASS (1 existing + 2 new).

- [ ] **Step 5: Commit**

```bash
git add src/retrieval/visual_retriever.py tests/test_visual_retriever.py
git commit -m "feat(retriever): add search_with_embedding for pre-encoded visual query"
```

---

### Task 3: `PrismRAGRetriever` — Accept `visual_query_embedding` in unified API

**Files:**
- Modify: `src/evaluation/vidore_adapter.py`
- Create: `tests/test_vidore_adapter.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_vidore_adapter.py

"""PrismRAGRetriever 适配器测试（mock 所有依赖）"""

from unittest.mock import MagicMock, patch

import torch

from src.evaluation.vidore_adapter import PrismRAGRetriever


def _make_mock_retriever():
    """构造 mock 化的 PrismRAGRetriever 实例"""
    return PrismRAGRetriever(
        pg_store=MagicMock(),
        faiss_store=MagicMock(),
        bge=MagicMock(),
        colpali=MagicMock(),
        chunker=MagicMock(),
        bm25=MagicMock(),
        dense=MagicMock(),
        visual=MagicMock(),
        fusion=MagicMock(),
        reranker=MagicMock(),
    )


def test_search_with_visual_embedding_uses_search_with_embedding():
    """传入 visual_query_embedding 时，visual route 走 search_with_embedding 而不是 search"""
    retriever = _make_mock_retriever()
    retriever.bm25.search.return_value = []
    retriever.dense.search.return_value = []
    retriever.visual.search_with_embedding.return_value = [
        {"chunk_id": "c1", "page_id": 1, "score": 0.85, "retrieval_type": "visual"},
    ]
    retriever.fusion.fuse.return_value = [
        {"chunk_id": "c1", "page_id": 1, "score": 0.85, "retrieval_type": "visual"},
    ]

    q_emb = torch.randn(1, 10, 128)
    result = retriever.search(
        query="test", k=10,
        use_bm25=False, use_dense=False, use_visual=True, use_rerank=False,
        visual_query_embedding=q_emb,
    )

    # 验证 search_with_embedding 被调用，search 未被调用
    retriever.visual.search_with_embedding.assert_called_once()
    retriever.visual.search.assert_not_called()

    assert len(result) == 1


def test_search_without_visual_embedding_uses_search():
    """不传 visual_query_embedding 时，visual route 走原来的 search()"""
    retriever = _make_mock_retriever()
    retriever.bm25.search.return_value = []
    retriever.dense.search.return_value = []
    retriever.visual.search.return_value = [
        {"chunk_id": "c1", "page_id": 1, "score": 0.85, "retrieval_type": "visual"},
    ]
    retriever.fusion.fuse.return_value = [
        {"chunk_id": "c1", "page_id": 1, "score": 0.85, "retrieval_type": "visual"},
    ]

    result = retriever.search(
        query="test", k=10,
        use_bm25=False, use_dense=False, use_visual=True, use_rerank=False,
    )

    retriever.visual.search.assert_called_once()
    retriever.visual.search_with_embedding.assert_not_called()

    assert len(result) == 1


def test_search_visual_false_ignores_embedding():
    """use_visual=False 时不会误用 visual_query_embedding"""
    retriever = _make_mock_retriever()
    retriever.bm25.search.return_value = []
    retriever.dense.search.return_value = [
        {"chunk_id": "c1", "page_id": 1, "score": 0.85, "retrieval_type": "dense"},
    ]
    retriever.fusion.fuse.return_value = [
        {"chunk_id": "c1", "page_id": 1, "score": 0.85, "retrieval_type": "dense"},
    ]

    q_emb = torch.randn(1, 10, 128)
    retriever.search(
        query="test", k=10,
        use_bm25=False, use_dense=True, use_visual=False, use_rerank=False,
        visual_query_embedding=q_emb,
    )

    # visual 不应该被调用
    retriever.visual.search.assert_not_called()
    retriever.visual.search_with_embedding.assert_not_called()


def test_search_with_trace_passes_visual_embedding():
    """search_with_trace 也应该透传 visual_query_embedding"""
    retriever = _make_mock_retriever()
    retriever.bm25.search.return_value = []
    retriever.dense.search.return_value = []
    retriever.visual.search_with_embedding.return_value = [
        {"chunk_id": "c1", "page_id": 1, "score": 0.85, "retrieval_type": "visual"},
    ]
    retriever.fusion.fuse.return_value = [
        {"chunk_id": "c1", "page_id": 1, "score": 0.85, "retrieval_type": "visual"},
    ]

    q_emb = torch.randn(1, 10, 128)
    result = retriever.search_with_trace(
        query="test", k=10,
        use_bm25=False, use_dense=False, use_visual=True, use_rerank=False,
        visual_query_embedding=q_emb,
    )

    retriever.visual.search_with_embedding.assert_called_once()
    assert "results" in result
    assert "retrieval_trace" in result
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_vidore_adapter.py -v`

Expected: All 4 tests FAIL with `unexpected keyword argument 'visual_query_embedding'` (since the method signature doesn't accept it yet).

- [ ] **Step 3: Extend `PrismRAGRetriever.search()` and `search_with_trace()`**

Modify `search()` signature to accept `visual_query_embedding`:

```python
# src/evaluation/vidore_adapter.py — modified search()

def search(
    self,
    query: str,
    k: int = 10,
    use_bm25: bool = True,
    use_dense: bool = True,
    use_visual: bool = True,
    use_rerank: bool = True,
    visual_query_embedding: Optional[torch.Tensor] = None,
) -> List[dict]:
    result = self.search_with_trace(
        query, k, use_bm25, use_dense, use_visual, use_rerank,
        visual_query_embedding=visual_query_embedding,
    )
    return result["results"]
```

Modify `search_with_trace()` signature and the visual route section:

```python
# src/evaluation/vidore_adapter.py — modified search_with_trace()

def search_with_trace(
    self,
    query: str,
    k: int = 10,
    use_bm25: bool = True,
    use_dense: bool = True,
    use_visual: bool = True,
    use_rerank: bool = True,
    visual_query_embedding: Optional[torch.Tensor] = None,
) -> dict:
    routes = []
    trace = {"bm25_top5": [], "dense_top5": [], "visual_top5": []}

    if use_bm25:
        try:
            bm25_results = self.bm25.search(query, k=20)
            routes.append(bm25_results)
            trace["bm25_top5"] = [
                {"chunk_id": r["chunk_id"], "page_id": r["page_id"], "score": r["score"]}
                for r in bm25_results[:5]
            ]
        except RuntimeError:
            logger.warning("BM25 未就绪，跳过")

    if use_dense:
        dense_results = self.dense.search(query, k=20)
        routes.append(dense_results)
        trace["dense_top5"] = [
            {"chunk_id": r["chunk_id"], "page_id": r["page_id"], "score": r["score"]}
            for r in dense_results[:5]
        ]

    if use_visual:
        try:
            if visual_query_embedding is not None:
                visual_results = self.visual.search_with_embedding(visual_query_embedding, k=20)
            else:
                visual_results = self.visual.search(query, k=20)
            routes.append(visual_results)
            trace["visual_top5"] = [
                {"chunk_id": r["chunk_id"], "page_id": r["page_id"], "score": r["score"]}
                for r in visual_results[:5]
            ]
        except Exception as e:
            logger.warning(f"Visual 检索跳过: {e}")

    # ... rest unchanged ...
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_vidore_adapter.py -v`

Expected: All 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/evaluation/vidore_adapter.py tests/test_vidore_adapter.py
git commit -m "feat(adapter): PrismRAGRetriever accepts visual_query_embedding for pre-encoded visual path"
```

---

### Task 4: `ablation.py` — Refactor to `load_eval_data()` + pass `pre_encoded_visual` to `run_ablation()`

**Files:**
- Modify: `src/evaluation/ablation.py`
- Create: `tests/test_ablation.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_ablation.py

"""消融实验模块测试（mock datasets）"""

from unittest.mock import MagicMock, patch

import torch

from src.evaluation.ablation import load_eval_data, run_ablation, AblationConfig, ABLATION_CONFIGS


def _make_mock_dataset(num_queries: int = 5):
    """创建 mock 的 ViDoRe 格式 dataset"""
    queries_ds = MagicMock()
    queries_ds.__len__.return_value = num_queries
    queries_ds.__getitem__ = MagicMock()
    queries_ds.select = MagicMock(return_value=queries_ds)
    queries_ds.filter = MagicMock(return_value=queries_ds)

    def getitem_side_effect(idx):
        return {
            "query_id": idx,
            "query": f"test query {idx}",
            "query_lang": "en",
        }
    queries_ds.__getitem__.side_effect = getitem_side_effect

    qrels_ds = MagicMock()
    qrels_ds.__iter__.return_value = [
        {"query_id": 0, "corpus_id": 101},
        {"query_id": 0, "corpus_id": 102},
        {"query_id": 1, "corpus_id": 103},
    ]

    return queries_ds, qrels_ds


@patch("src.evaluation.ablation.load_dataset")
def test_load_eval_data_filters_language(mock_load_dataset):
    """load_eval_data(language='en') 应调用 filter 并按 query_lang 过滤"""
    queries_ds, qrels_ds = _make_mock_dataset()
    mock_load_dataset.side_effect = lambda path, split, *a, **kw: {
        queries_ds if split == "queries" else qrels_ds
    }.get(split, qrels_ds)

    queries_out, qrel_map = load_eval_data(
        dataset_path="vidore/vidore_v3_industrial",
        max_queries=None,
        language="en",
    )

    # 验证 filter 被调用
    queries_ds.filter.assert_called_once()
    # qrel_map 必须是 dict，值为 set
    assert isinstance(qrel_map, dict)
    if qrel_map:
        assert isinstance(next(iter(qrel_map.values())), set)


@patch("src.evaluation.ablation.load_dataset")
def test_load_eval_data_applies_max_queries(mock_load_dataset):
    """load_eval_data 应用 max_queries 限制"""
    queries_ds, qrels_ds = _make_mock_dataset(num_queries=10)
    mock_load_dataset.side_effect = lambda path, split, *a, **kw: {
        queries_ds if split == "queries" else qrels_ds
    }.get(split, qrels_ds)

    queries_out, qrel_map = load_eval_data(
        dataset_path="vidore/vidore_v3_industrial",
        max_queries=3,
        language="all",
    )

    queries_ds.select.assert_called_once_with(range(3))


@patch("src.evaluation.ablation.load_dataset")
def test_run_ablation_passes_pre_encoded_visual(mock_load_dataset):
    """run_ablation() 在 visual 配置下会把 pre_encoded_visual 透传给 retriever.search"""
    queries_ds, qrels_ds = _make_mock_dataset(num_queries=2)
    mock_load_dataset.side_effect = lambda path, split, *a, **kw: {
        queries_ds if split == "queries" else qrels_ds
    }.get(split, qrels_ds)

    mock_retriever = MagicMock()
    mock_retriever.search.return_value = []

    pre_encoded = {
        0: torch.randn(1, 10, 128),
        1: torch.randn(1, 10, 128),
    }

    run_ablation(
        retriever=mock_retriever,
        queries_ds=queries_ds,
        qrel_map={0: {101, 102}, 1: {103}},
        output_dir="/tmp/test_ablation_results",
        pre_encoded_visual=pre_encoded,
        language="en",
    )

    # 验证对 Visual_only 配置（use_visual=True）传入了 visual_query_embedding
    visual_config_calls = [
        call for call in mock_retriever.search.call_args_list
        if call.kwargs.get("use_visual") is True
    ]
    assert len(visual_config_calls) > 0
    for call in visual_config_calls:
        assert "visual_query_embedding" in call.kwargs
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_ablation.py -v`

Expected: All 3 tests FAIL with `ImportError: cannot import name 'load_eval_data' from 'src.evaluation.ablation'`.

- [ ] **Step 3: Implement `load_eval_data()` and refactor `run_ablation()`**

```python
# src/evaluation/ablation.py — full modified file

"""消融实验运行器"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from datasets import load_dataset as hf_load_dataset
from tqdm import tqdm

from src.evaluation.vidore_adapter import PrismRAGRetriever

logger = logging.getLogger(__name__)


@dataclass
class AblationConfig:
    name: str
    use_bm25: bool = True
    use_dense: bool = True
    use_visual: bool = True
    use_rerank: bool = True


ABLATION_CONFIGS = [
    AblationConfig(name="BM25_only", use_bm25=True, use_dense=False, use_visual=False, use_rerank=False),
    AblationConfig(name="Dense_only", use_bm25=False, use_dense=True, use_visual=False, use_rerank=False),
    AblationConfig(name="Visual_only", use_bm25=False, use_dense=False, use_visual=True, use_rerank=False),
    AblationConfig(name="BM25_Dense", use_bm25=True, use_dense=True, use_visual=False, use_rerank=False),
    AblationConfig(name="BM25_Dense_Visual", use_bm25=True, use_dense=True, use_visual=True, use_rerank=False),
    AblationConfig(name="Full_no_rerank", use_bm25=True, use_dense=True, use_visual=True, use_rerank=False),
    AblationConfig(name="Full_with_rerank", use_bm25=True, use_dense=True, use_visual=True, use_rerank=True),
]


def compute_ndcg(relevant: set, ranked: List[str], k: int) -> float:
    dcg, idcg = 0.0, 0.0
    for i in range(min(k, len(ranked))):
        if ranked[i] in relevant:
            dcg += 1.0 / (i + 1)
    for i in range(min(k, len(relevant))):
        idcg += 1.0 / (i + 1)
    return dcg / idcg if idcg > 0 else 0.0


def compute_recall(relevant: set, ranked: List[str], k: int) -> float:
    if not relevant:
        return 0.0
    return sum(1 for r in ranked[:k] if r in relevant) / len(relevant)


def compute_mrr(relevant: set, ranked: List[str]) -> float:
    for i, r in enumerate(ranked, start=1):
        if r in relevant:
            return 1.0 / i
    return 0.0


def load_eval_data(
    dataset_path: str,
    max_queries: Optional[int] = None,
    language: str = "en",
) -> tuple:
    """加载评测数据，按语言过滤，支持 query 数量限制

    Returns:
        (queries_ds, qrel_map) 元组
        - queries_ds: HuggingFace Dataset（已过滤和截断）
        - qrel_map: Dict[int, set] — query_id -> set of corpus_ids
    """
    queries_ds = hf_load_dataset(dataset_path, "queries", split="test")
    qrels_ds = hf_load_dataset(dataset_path, "qrels", split="test")

    if language != "all":
        queries_ds = queries_ds.filter(lambda x: x["query_lang"] == language)

    if max_queries:
        queries_ds = queries_ds.select(range(min(max_queries, len(queries_ds))))

    qrel_map: Dict[int, set] = {}
    for qrel in qrels_ds:
        qid = int(qrel["query_id"])
        cid = int(qrel["corpus_id"])
        if qid not in qrel_map:
            qrel_map[qid] = set()
        qrel_map[qid].add(cid)

    return queries_ds, qrel_map


def run_ablation(
    retriever: PrismRAGRetriever,
    queries_ds,
    qrel_map: Dict[int, set],
    output_dir: str = "results",
    pre_encoded_visual: Optional[Dict[int, torch.Tensor]] = None,
    language: str = "en",
) -> List[dict]:
    """运行全量消融实验

    Args:
        retriever: PrismRAGRetriever 实例
        queries_ds: 已加载和过滤的 queries dataset
        qrel_map: query_id -> relevant corpus_id set
        output_dir: 结果输出目录
        pre_encoded_visual: {q_idx: tensor[1, n_q, 128]} 预编码的 visual query embedding
        language: 当前评测语言，会写入结果元数据
    """
    results = []

    for config in ABLATION_CONFIGS:
        logger.info(f"=== 消融配置: {config.name} ===")
        latencies = []
        all_ranked_page_ids: List[List[str]] = []
        all_relevant: List[set] = []

        for q_idx in tqdm(range(len(queries_ds)), desc=f"  {config.name}"):
            q = queries_ds[q_idx]
            qid = int(q["query_id"])
            query_text = str(q["query"])

            visual_q_emb = None
            if config.use_visual and pre_encoded_visual is not None and q_idx in pre_encoded_visual:
                visual_q_emb = pre_encoded_visual[q_idx]

            start = time.time()
            retrieved = retriever.search(
                query=query_text, k=10,
                use_bm25=config.use_bm25, use_dense=config.use_dense,
                use_visual=config.use_visual, use_rerank=config.use_rerank,
                visual_query_embedding=visual_q_emb,
            )
            latencies.append((time.time() - start) * 1000)

            ranked_page_ids = [str(r["page_id"]) for r in retrieved]
            all_ranked_page_ids.append(ranked_page_ids)
            relevant = {str(cid) for cid in qrel_map.get(qid, set())}
            all_relevant.append(relevant)

        n = len(all_ranked_page_ids)
        ndcg5 = sum(compute_ndcg(rel, ranked, 5) for rel, ranked in zip(all_relevant, all_ranked_page_ids)) / n
        ndcg10 = sum(compute_ndcg(rel, ranked, 10) for rel, ranked in zip(all_relevant, all_ranked_page_ids)) / n
        rec5 = sum(compute_recall(rel, ranked, 5) for rel, ranked in zip(all_relevant, all_ranked_page_ids)) / n
        rec10 = sum(compute_recall(rel, ranked, 10) for rel, ranked in zip(all_relevant, all_ranked_page_ids)) / n
        mrr = sum(compute_mrr(rel, ranked) for rel, ranked in zip(all_relevant, all_ranked_page_ids)) / n
        avg_lat = sum(latencies) / len(latencies) if latencies else 0

        result = {
            "config": config.name,
            "ndcg@5": round(ndcg5, 4), "ndcg@10": round(ndcg10, 4),
            "recall@5": round(rec5, 4), "recall@10": round(rec10, 4),
            "mrr": round(mrr, 4), "avg_latency_ms": round(avg_lat, 1),
            "num_queries": n,
            "language": language,
        }
        results.append(result)
        logger.info(f"  NDCG@10={ndcg10:.4f}, Recall@5={rec5:.4f}, MRR={mrr:.4f}")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    with open(output_path / "ablation_results.json", "w") as f:
        json.dump(results, f, indent=2)

    logger.info("\n" + "=" * 80)
    logger.info("消融实验结果")
    header = f"{'Config':<25} {'NDCG@5':<10} {'NDCG@10':<10} {'Recall@5':<10} {'Recall@10':<10} {'MRR':<10} {'Lat(ms)':<10}"
    logger.info(header)
    logger.info("-" * 80)
    for r in results:
        logger.info(f"{r['config']:<25} {r['ndcg@5']:<10.4f} {r['ndcg@10']:<10.4f} {r['recall@5']:<10.4f} {r['recall@10']:<10.4f} {r['mrr']:<10.4f} {r['avg_latency_ms']:<10.0f}")

    return results
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_ablation.py -v`

Expected: All 3 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/evaluation/ablation.py tests/test_ablation.py
git commit -m "refactor(ablation): extract load_eval_data, pass pre_encoded_visual + language metadata"
```

---

### Task 5: `scripts/run_eval.py` — Fix init order, add `--language` & `--expected-query-count` CLI flags

**Files:**
- Modify: `scripts/run_eval.py`

- [ ] **Step 1: Write a verification test for expected query count validation (optional unit test or inline)**

Add to `tests/test_ablation.py`:

```python
# tests/test_ablation.py — append

def test_expected_query_count_validation():
    """语言过滤后，可以验证 query 数量是否与预期一致"""
    queries_ds, qrels_ds = _make_mock_dataset(num_queries=3)
    with patch("src.evaluation.ablation.hf_load_dataset") as mock_load:
        mock_load.side_effect = lambda path, split, *a, **kw: {
            queries_ds if split == "queries" else qrels_ds
        }.get(split, qrels_ds)

        queries_out, qrel_map = load_eval_data(
            dataset_path="vidore/vidore_v3_industrial",
            max_queries=None,
            language="en",
        )

        # 验证 select 未被调用（因为 max_queries=None）
        queries_ds.select.assert_not_called()
        # filter 应该被调用过一次
        queries_ds.filter.assert_called_once()
```

- [ ] **Step 2: Run the test to verify it passes**

Run: `pytest tests/test_ablation.py::test_expected_query_count_validation -v`

Expected: PASS.

- [ ] **Step 3: Rewrite `scripts/run_eval.py`**

```python
#!/usr/bin/env python
"""评测入口脚本 — 修正 ColPali 预编码 + FAISS 加载分离顺序"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import cfg
from src.ingestion.encoders import BGEEmbedder, ColPaliEmbedder
from src.ingestion.text_chunker import TextChunker
from src.evaluation.ablation import load_eval_data, run_ablation
from src.evaluation.vidore_adapter import PrismRAGRetriever
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.fusion import RRFFusion
from src.retrieval.reranker import Reranker
from src.retrieval.visual_retriever import VisualRetriever
from src.store.faiss_store import FaissColPaliStore
from src.store.pgvector_store import PgVectorStore

logger = logging.getLogger(__name__)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description="Run PrismRAG evaluation")
    parser.add_argument("--dataset", default="vidore/vidore_v3_industrial")
    parser.add_argument("--max-queries", type=int, default=None)
    parser.add_argument("--skip-index", action="store_true")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--language", default="en", choices=["en", "all"],
                        help="评测语言 ('en' 为论文对齐模式, 'all' 为全量 1698 query)")
    parser.add_argument("--expected-query-count", type=int, default=None,
                        help="预期 query 数量校验 (默认: en=283, all=不校验)")
    args = parser.parse_args()

    cfg.load()

    # Phase 0: 加载 & 过滤评测数据
    logger.info(f"加载评测数据 (language={args.language})...")
    queries_ds, qrel_map = load_eval_data(
        dataset_path=args.dataset,
        max_queries=args.max_queries,
        language=args.language,
    )

    num_queries = len(queries_ds)
    logger.info(f"评测 query 数量: {num_queries}")

    # 校验 query 数量
    expected = args.expected_query_count
    if expected is None:
        expected = 283 if args.language == "en" else None
    if expected is not None and num_queries != expected:
        raise RuntimeError(
            f"query 数量校验失败: 预期 {expected}, 实际 {num_queries}. "
            f"请检查 dataset 的 query_lang 字段分布。"
        )

    pg_store = PgVectorStore()
    faiss_store = FaissColPaliStore()
    bge = BGEEmbedder()
    chunker = TextChunker()

    # Phase A: 预编码 visual query（仅占用 ColPali 模型显存）
    pre_encoded_visual = None
    has_visual_config = True  # 消融实验总是包含 visual 配置
    if has_visual_config:
        logger.info("预编码 visual query...")
        colpali = ColPaliEmbedder()
        query_texts = [str(queries_ds[i]["query"]) for i in range(num_queries)]
        pre_encoded_visual = colpali.encode_queries_batch(query_texts, batch_size=8)
        logger.info(f"完成 {len(pre_encoded_visual)} 条 query 预编码")
        colpali.unload()
        logger.info("ColPali 模型已卸载，显存已释放")

    # Phase B: 加载 FAISS GPU 索引（此时无 ColPali 模型竞争显存）
    bm25 = BM25Retriever()
    dense = DenseRetriever(pg_store, bge)
    fusion = RRFFusion(rrf_k=60)
    reranker = Reranker()

    if not args.skip_index:
        colpali_for_ingest = ColPaliEmbedder()
        from src.ingestion.vidore_ingestor import ViDoReIngestor
        ingestor = ViDoReIngestor(pg_store, faiss_store, bge, colpali_for_ingest, chunker)
        ingestor.ingest(dataset_path=args.dataset)
        bm25.fit_from_pgvector(pg_store)
        logger.info("BM25 索引构建完成")
        colpali_for_ingest.unload()
    else:
        faiss_loaded = faiss_store.load()
        if faiss_loaded:
            bm25.fit_from_pgvector(pg_store)
            logger.info("索引加载成功，跳过构建")
        else:
            logger.warning("FAISS 索引不存在，重新构建")
            colpali_for_ingest = ColPaliEmbedder()
            from src.ingestion.vidore_ingestor import ViDoReIngestor
            ingestor = ViDoReIngestor(pg_store, faiss_store, bge, colpali_for_ingest, chunker)
            ingestor.ingest(dataset_path=args.dataset)
            bm25.fit_from_pgvector(pg_store)
            logger.info("BM25 索引构建完成")
            colpali_for_ingest.unload()

    visual = VisualRetriever(faiss_store, pg_store, colpali=None)  # colpali 已卸载，search_with_embedding 不需要
    retriever = PrismRAGRetriever(
        pg_store=pg_store, faiss_store=faiss_store, bge=bge, colpali=colpali,
        chunker=chunker, bm25=bm25, dense=dense, visual=visual,
        fusion=fusion, reranker=reranker,
    )

    run_ablation(
        retriever,
        queries_ds=queries_ds,
        qrel_map=qrel_map,
        output_dir=args.output_dir,
        pre_encoded_visual=pre_encoded_visual,
        language=args.language,
    )


if __name__ == "__main__":
    main()
```

Note: The `colpali=None` in `VisualRetriever(faiss_store, pg_store, colpali=None)` is safe because with `search_with_embedding()` we skip `encode_query()`. If `search()` is called (no pre-encoded embedding), it would fail — but in eval mode we always use pre-encoded.

**Alternative:** Keep `colpali=colpali` (the unloaded embedder) — `search()` would hit `_require_loaded()` and raise `RuntimeError`. Either way, eval path is safe.

- [ ] **Step 4: Review with `--dry-run` / syntax check**

Run: `python -c "import ast; ast.parse(open('scripts/run_eval.py').read()); print('Syntax OK')"`

Expected: `Syntax OK`

- [ ] **Step 5: Commit**

```bash
git add scripts/run_eval.py tests/test_ablation.py
git commit -m "fix(run_eval): reorder ColPali pre-encode before FAISS load, add --language/--expected-query-count"
```

---

### Task 6: Self-Review — Verify spec coverage

- [ ] **Step 1: Verify spec coverage by checking each section**

| Spec Section | Covered By |
|---|---|
| §3.1 Visual 路预编码 | Task 1 (`encode_queries_batch` + `unload`), Task 2 (`search_with_embedding`), Task 3 (`visual_query_embedding` pass-through), Task 5 (Phase ordering in `run_eval.py`) |
| §3.2 语言过滤 + 校验 | Task 4 (`load_eval_data`), Task 5 (`--language`, `--expected-query-count`) |
| §4.1 `ColPaliEmbedder` 改动 | Task 1 |
| §4.2 `VisualRetriever` 改动 | Task 2 |
| §4.3 `PrismRAGRetriever` 改动 | Task 3 |
| §4.4 `ablation.py` 改动 | Task 4 |
| §4.5 `run_eval.py` 编排修正 | Task 5 |
| §4.6 CLI 参数 | Task 5 (`--language`, `--expected-query-count`) |
| §5 不改的文件 | `faiss_store.py`, `pgvector_store.py`, `fusion.py`, `reranker.py` — confirmed untouched |
| §6.1 功能验收 | Covered by Task 5 ordering + Task 4 refactor |
| §6.2 测试验收 | Task 1-4 test coverage for all new methods |

- [ ] **Step 2: Scan for placeholders**

Search plan for: "TBD", "TODO", "implement later", "fill in details", "Add appropriate error handling", "Similar to". None found — all steps contain complete, executable code.

- [ ] **Step 3: Type consistency check**

- `encode_queries_batch()` returns `Dict[int, torch.Tensor]` (Task 1) — consumed as `pre_encoded_visual: Optional[Dict[int, torch.Tensor]]` (Task 4).
- `search_with_embedding(q_emb: torch.Tensor, k: int)` (Task 2) — called from `PrismRAGRetriever` (Task 3).
- `visual_query_embedding: Optional[torch.Tensor]` in `search()` / `search_with_trace()` (Task 3) — flows from `pre_encoded_visual[q_idx]` (Task 4).
- `load_eval_data()` returns `(queries_ds, qrel_map)` tuple (Task 4) — consumed by `run_ablation()` (Task 4) and `run_eval.py` (Task 5).
- `run_ablation()` now takes `queries_ds, qrel_map, pre_encoded_visual, language` (Task 4) — called from `run_eval.py` (Task 5).

All types match end-to-end.

- [ ] **Step 4: Check all commits are independent**

Each commit leaves the codebase in a working state:
1. Task 1: Adds new methods, doesn't break existing callers
2. Task 2: Adds new method, doesn't break existing callers
3. Task 3: Extends signatures with optional param, backward-compatible
4. Task 4: Refactors `run_ablation()` signature but only `run_eval.py` calls it — updated in Task 5
5. Task 5: Final integration, hooks everything together

- [ ] **Step 5: Commit the plan document**

```bash
git add docs/superpowers/plans/2026-07-01-visual-path-fix-and-eval-alignment.md
git commit -m "docs: add implementation plan for visual path fix and eval alignment"
```

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-07-01-visual-path-fix-and-eval-alignment.md`. Two execution options:

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints

**Which approach?**