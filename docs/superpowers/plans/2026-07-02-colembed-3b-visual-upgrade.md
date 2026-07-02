# Colembed-3B Visual 路升级 — 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 以最小侵入方式将 Visual 路 embedding 模型从 ColPali v1.3 替换为 `nvidia/llama-nemoretriever-colembed-3b-v1`，保持 FAISS / MaxSim / pgvector grounding 不变。

**Architecture:** 新增 `ColembedEncoder` 类（与 `ColPaliEmbedder` 接口对齐）→ 工厂函数 `create_visual_encoder()` 按模型名分发 → CLI 加 `--visual-model` 参数 → FAISS 索引路径按模型名隔离。

**Tech Stack:** PyTorch, transformers>=4.49.0, flash-attn==2.6.3, FAISS, pgvector

---

### Task 1: 配置变更 — models.yaml + config.py

**Files:**
- Modify: `config/models.yaml`
- Modify: `src/config.py:75-97`

- [ ] **Step 1: 在 models.yaml 中新增 colembed 模型配置和 FAISS 路径**

```yaml
# config/models.yaml — 在 models: 块中新增一行
models:
  colpali: "vidore/colpali-v1.3"
  bge_embedding: "BAAI/bge-large-en-v1.5"
  bge_reranker: "BAAI/bge-reranker-large"
  zerank_reranker: "zeroentropy/zerank-2-reranker"
  llm: "qwen2:7b"
  colembed: "nvidia/llama-nemoretriever-colembed-3b-v1"          # ← 新增

# 在 embedding: 块中新增
embedding:
  bge_dim: 1024
  bge_device: "auto"
  colpali_device: "auto"
  colpali_batch_size: 4
  colembed_batch_size: 4                                          # ← 新增
  colembed_max_input_tiles: 2                                     # ← 新增

# 在 storage.faiss: 块中新增
storage:
  faiss:
    index_path: "indexes/colpali-v1.3-vidore-industrial.faiss"
    id_map_path: "indexes/colpali-v1.3-vidore-industrial-ids.npy"
    colembed_index_path: "indexes/colembed-3b-vidore-industrial.faiss"       # ← 新增
    colembed_id_map_path: "indexes/colembed-3b-vidore-industrial-ids.npy"     # ← 新增
    index_type: "flat"
    hnsw_m: 32
```

- [ ] **Step 2: 在 config.py 中新增 colembed_model_id 属性**

```python
# src/config.py — 在现有 @property 块末尾（llm_model_id 之后）新增

@property
def colembed_model_id(self) -> str:
    return self._data["models"]["colembed"]
```

- [ ] **Step 3: 验证配置加载**

```bash
cd /Users/theyang/Documents/ai/pdf-rag && python -c "
from src.config import cfg; cfg.load()
print('colembed_model_id:', cfg.colembed_model_id)
print('colembed_batch_size:', cfg.get('embedding.colembed_batch_size'))
print('colembed_index_path:', cfg.get('storage.faiss.colembed_index_path'))
"
```

Expected: 输出 colembed 模型 ID 和路径，无报错。

- [ ] **Step 4: Commit**

```bash
git add config/models.yaml src/config.py
git commit -m "feat: colembed-3b 模型配置 — models.yaml + config 属性"
```

---

### Task 2: ColembedEncoder 类

**Files:**
- Modify: `src/ingestion/encoders.py`（在 ColPaliEmbedder 类之后新增）

- [ ] **Step 1: 新增 ColembedEncoder 类**

在 `src/ingestion/encoders.py` 文件末尾（`ColPaliEmbedder` 类之后）新增以下代码：

```python
class ColembedEncoder:
    """NVIDIA llama-nemoretriever-colembed-3b-v1 multi-vector encoder.

    接口与 ColPaliEmbedder 对齐，下游零改动：
      encode_pages(images, batch_size, show_progress) → List[torch.Tensor]
      encode_query(text)                              → torch.Tensor [1, n_q, 128]
      encode_queries_batch(texts, batch_size)          → Dict[int, torch.Tensor]
      unload()                                         → None
    """

    def __init__(
        self,
        model_id: str | None = None,
        device: str | None = None,
        max_input_tiles: int | None = None,
    ):
        from transformers import AutoModel

        self._model_id = model_id or cfg.colembed_model_id
        self.device = device or cfg.get("embedding.colpali_device", "cuda")
        self.max_input_tiles = (
            max_input_tiles
            if max_input_tiles is not None
            else cfg.get("embedding.colembed_max_input_tiles", 2)
        )

        self.model = AutoModel.from_pretrained(
            self._model_id,
            device_map=self.device,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        ).eval()
        self._loaded = True

    @torch.no_grad()
    def encode_pages(
        self,
        images: List[Image.Image],
        batch_size: int | None = None,
        show_progress: bool = False,
    ) -> List[torch.Tensor]:
        """编码页面列表，每页返回 [n_patches, 128] 多向量"""
        self._require_loaded()
        if batch_size is None:
            batch_size = cfg.get("embedding.colembed_batch_size", 4)

        all_embs: List[torch.Tensor] = []
        rng = trange(0, len(images), batch_size, disable=not show_progress, desc="Colembed encode")
        for i in rng:
            batch_imgs = images[i : i + batch_size]
            batch_embs = self.model.forward_passages(
                batch_imgs,
                batch_size=len(batch_imgs),
                max_input_tiles=self.max_input_tiles,
            )
            # batch_embs: List[torch.Tensor] each [n_patches, 128]
            all_embs.extend([emb.cpu() for emb in batch_embs])

        return all_embs

    @torch.no_grad()
    def encode_query(self, text: str) -> torch.Tensor:
        """编码单条查询文本 → [1, n_q, 128]"""
        self._require_loaded()
        emb = self.model.forward_queries([text], batch_size=1)
        # emb: List[torch.Tensor], each [n_q, 128]
        return emb[0].unsqueeze(0).cpu()  # → [1, n_q, 128]

    @torch.no_grad()
    def encode_queries_batch(
        self, texts: List[str], batch_size: int = 4
    ) -> Dict[int, torch.Tensor]:
        """批量编码多个 query → {idx: tensor[1, n_q, 128]}"""
        self._require_loaded()
        results: Dict[int, torch.Tensor] = {}
        for i in trange(0, len(texts), batch_size, desc="Colembed encode queries"):
            batch_texts = texts[i : i + batch_size]
            batch_embs = self.model.forward_queries(batch_texts, batch_size=len(batch_texts))
            # batch_embs: List[torch.Tensor] each [n_q, 128]
            for j, emb in enumerate(batch_embs):
                results[i + j] = emb.unsqueeze(0).cpu()  # → [1, n_q, 128]
        return results

    def unload(self):
        """卸载模型释放显存"""
        if self._loaded:
            del self.model
            self.model = None
            torch.cuda.empty_cache()
            self._loaded = False

    def _require_loaded(self):
        if not self._loaded:
            raise RuntimeError(
                "ColembedEncoder 已通过 unload() 卸载，无法执行编码操作。"
                "请重新创建 ColembedEncoder 实例后再调用。"
            )
```

- [ ] **Step 2: 验证 ColembedEncoder 可导入（本地语法检查）**

```bash
cd /Users/theyang/Documents/ai/pdf-rag && python -c "
from src.ingestion.encoders import ColembedEncoder
print('ColembedEncoder imported successfully')
"
```

Expected: 成功导入（本地无 CUDA/flash-attn 也能通过 import，因为 `from transformers import AutoModel` 只在 `__init__` 中执行）。

- [ ] **Step 3: Commit**

```bash
git add src/ingestion/encoders.py
git commit -m "feat: ColembedEncoder — llama-nemoretriever-colembed-3b-v1 编码器"
```

---

### Task 3: 工厂函数 + 解耦硬编码的 ColPaliEmbedder 引用

**Files:**
- Modify: `src/ingestion/encoders.py`（新增工厂函数）
- Modify: `src/retrieval/visual_retriever.py:1-23`（类型注解泛化）
- Modify: `src/evaluation/vidore_adapter.py:16,30-40`（类型注解泛化）
- Modify: `src/ingestion/vidore_ingestor.py:20,39-49`（类型注解泛化）

- [ ] **Step 1: 在 encoders.py 末尾新增工厂函数**

```python
# src/ingestion/encoders.py — 文件末尾追加

def create_visual_encoder(
    model_name: str = "colpali",
    device: str | None = None,
) -> "ColPaliEmbedder | ColembedEncoder":
    """工厂函数：按模型名前缀创建视觉编码器。

    model_name 前缀匹配规则:
      - "colpali" → ColPaliEmbedder (vidore/colpali-v1.3)
      - "colembed" → ColembedEncoder (nvidia/llama-nemoretriever-colembed-3b-v1)
    """
    if model_name.startswith("colembed"):
        return ColembedEncoder(device=device)
    else:
        return ColPaliEmbedder(device=device)
```

- [ ] **Step 2: 泛化 VisualRetriever 的类型注解**

```python
# src/retrieval/visual_retriever.py — 修改 import 和 __init__ 签名

# 原:
# from src.ingestion.encoders import ColPaliEmbedder
# ...
# def __init__(self, faiss_store, pg_store, colpali_embedder: ColPaliEmbedder):

# 改为:
from __future__ import annotations
from typing import List

import torch

from src.store.faiss_store import FaissColPaliStore
from src.store.pgvector_store import PgVectorStore


class VisualRetriever:
    """Visual 检索器：Visual encode → FAISS MaxSim → pgvector grounding 反查"""

    def __init__(
        self,
        faiss_store: FaissColPaliStore,
        pg_store: PgVectorStore,
        colpali_embedder,  # ColPaliEmbedder | ColembedEncoder (duck typing)
    ):
        self.faiss = faiss_store
        self.pg = pg_store
        self.colpali = colpali_embedder

    # 其余方法不变
```

- [ ] **Step 3: 泛化 PrismRAGRetriever 的类型注解**

```python
# src/evaluation/vidore_adapter.py — 修改 import 和 __init__ 签名

# 原:
# from src.ingestion.encoders import BGEEmbedder, ColPaliEmbedder
# ...
# def __init__(self, ..., colpali: ColPaliEmbedder, ...):

# 改为:
from src.ingestion.encoders import BGEEmbedder  # 移除 ColPaliEmbedder

class PrismRAGRetriever:
    def __init__(
        self,
        pg_store: PgVectorStore,
        faiss_store: FaissColPaliStore,
        bge: BGEEmbedder,
        colpali,  # ColPaliEmbedder | ColembedEncoder (duck typing)
        chunker: TextChunker,
        bm25: BM25Retriever,
        dense: DenseRetriever,
        visual: VisualRetriever,
        fusion: RRFFusion,
        reranker: Reranker,
        hyde: Optional[HyDEGenerator] = None,
        zerank_reranker: Optional[Reranker] = None,
    ):
```

- [ ] **Step 4: 泛化 ViDoReIngestor 的类型注解**

```python
# src/ingestion/vidore_ingestor.py — 修改 import 和 __init__ 签名

# 原:
# from src.ingestion.encoders import BGEEmbedder, ColPaliEmbedder
# ...
# def __init__(self, ..., colpali_embedder: ColPaliEmbedder, ...):

# 改为:
from src.ingestion.encoders import BGEEmbedder  # 移除 ColPaliEmbedder

class ViDoReIngestor:
    def __init__(
        self,
        pg_store: PgVectorStore,
        faiss_store: FaissColPaliStore,
        bge_embedder: BGEEmbedder,
        colpali_embedder,  # ColPaliEmbedder | ColembedEncoder (duck typing)
        chunker: TextChunker,
    ):
```

- [ ] **Step 5: 运行现有测试确保未破坏**

```bash
cd /Users/theyang/Documents/ai/pdf-rag && python -m pytest tests/ -x -q
```

Expected: 所有现有测试通过（类型注解宽松化不影响运行时行为）。

- [ ] **Step 6: Commit**

```bash
git add src/ingestion/encoders.py src/retrieval/visual_retriever.py src/evaluation/vidore_adapter.py src/ingestion/vidore_ingestor.py
git commit -m "feat: create_visual_encoder 工厂函数 + 解耦 ColPaliEmbedder 硬编码"
```

---

### Task 4: 评分一致性验证方法

**Files:**
- Modify: `src/ingestion/encoders.py`（在 ColembedEncoder 类中新增静态方法）

- [ ] **Step 1: 在 ColembedEncoder 中新增 verify_scoring_equivalence 静态方法**

在 `ColembedEncoder` 类中（`_require_loaded` 方法之后）新增：

```python
    @staticmethod
    def verify_scoring_equivalence(
        encoder: "ColembedEncoder",
        test_pages: List[Image.Image],
        test_queries: List[str],
        k: int = 5,
    ) -> dict:
        """验证 FAISS MaxSim 与模型原生 get_scores() 排名一致性。

        Returns:
            {"passed": bool, "match_rate": float, "details": str}
        """
        import numpy as np
        import faiss

        # 1. 编码
        page_embs = encoder.encode_pages(test_pages, batch_size=len(test_pages))
        query_embs = encoder.encode_queries_batch(test_queries, batch_size=len(test_queries))

        # 2. 构建临时 FAISS IndexFlatIP + 手动 MaxSim
        all_vecs = []
        boundaries = []
        start = 0
        for emb in page_embs:
            arr = emb.float().numpy().astype(np.float32)
            n = arr.shape[0]
            all_vecs.append(arr)
            boundaries.append((start, start + n))
            start += n
        vecs_flat = np.vstack(all_vecs)
        dim = vecs_flat.shape[1]

        index = faiss.IndexFlatIP(dim)
        index.add(vecs_flat)

        # 3. 比较每对 (query, 所有 page) 的排名
        n_queries = len(test_queries)
        n_pages = len(test_pages)
        matches = 0
        total_pairs = n_queries

        for qi in range(n_queries):
            q = query_embs[qi].float().numpy().astype(np.float32)  # [1, n_q, 128]
            n_q = q.shape[1]
            q_flat = q.reshape(n_q, -1)  # [n_q, 128]

            # 方法 A: FAISS MaxSim
            scores_a = q_flat @ vecs_flat.T  # [n_q, total_patches]
            faiss_scores = []
            for s, e in boundaries:
                page_max = scores_a[:, s:e].max(axis=1)
                faiss_scores.append(float(page_max.mean()))
            faiss_ranking = np.argsort(faiss_scores)[::-1][:k].tolist()

            # 方法 B: 模型原生 get_scores()
            native_scores = encoder.model.get_scores(
                query_embs[qi].to(encoder.device),
                [emb.to(encoder.device) for emb in page_embs],
            )
            # native_scores: List[float] of length n_pages
            if isinstance(native_scores, torch.Tensor):
                native_scores = native_scores.cpu().tolist()
            native_ranking = sorted(
                range(len(native_scores)), key=lambda i: native_scores[i], reverse=True
            )[:k]

            if faiss_ranking == native_ranking:
                matches += 1

        match_rate = matches / total_pairs
        passed = match_rate >= 0.95

        return {
            "passed": passed,
            "match_rate": round(match_rate, 4),
            "details": (
                f"Top-{k} ranking match: {matches}/{total_pairs} ({match_rate:.1%})"
            ),
        }
```

- [ ] **Step 2: 本地语法验证**

```bash
cd /Users/theyang/Documents/ai/pdf-rag && python -c "
from src.ingestion.encoders import ColembedEncoder
print('verify_scoring_equivalence method present:', hasattr(ColembedEncoder, 'verify_scoring_equivalence'))
"
```

Expected: `True`

- [ ] **Step 3: Commit**

```bash
git add src/ingestion/encoders.py
git commit -m "feat: ColembedEncoder.verify_scoring_equivalence() 评分一致性验证"
```

---

### Task 5: CLI 参数 — ingest_vidore.py + run_eval.py

**Files:**
- Modify: `scripts/ingest_vidore.py`
- Modify: `scripts/run_eval.py`

- [ ] **Step 1: ingest_vidore.py 新增 --visual-model 参数**

```python
# scripts/ingest_vidore.py — 完整替换

#!/usr/bin/env python
"""ViDoRe 数据导入入口脚本"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import cfg
from src.ingestion.encoders import BGEEmbedder, create_visual_encoder
from src.ingestion.text_chunker import TextChunker
from src.ingestion.vidore_ingestor import ViDoReIngestor
from src.store.pgvector_store import PgVectorStore
from src.store.faiss_store import FaissColPaliStore


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description="Import ViDoRe dataset into stores")
    parser.add_argument("--dataset", default="vidore/vidore_v3_industrial", help="HF dataset path")
    parser.add_argument("--max-pages", type=int, default=None, help="Limit pages (for testing)")
    parser.add_argument("--skip-faiss", action="store_true", help="Skip ColPali + FAISS (BGE only)")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    parser.add_argument("--visual-model", default="colpali",
                        choices=["colpali", "colembed-3b"],
                        help="Visual embedding model (default: colpali)")
    args = parser.parse_args()

    cfg.load()
    pg_store = PgVectorStore()

    # 按模型选择 FAISS 路径
    if args.visual_model == "colembed-3b":
        index_path = cfg.get("storage.faiss.colembed_index_path")
        id_map_path = cfg.get("storage.faiss.colembed_id_map_path")
    else:
        index_path = cfg.get("storage.faiss.index_path")
        id_map_path = cfg.get("storage.faiss.id_map_path")

    faiss_store = FaissColPaliStore(index_path=index_path, id_map_path=id_map_path)
    bge = BGEEmbedder()
    visual_encoder = create_visual_encoder(model_name=args.visual_model)
    chunker = TextChunker()

    ingestor = ViDoReIngestor(pg_store, faiss_store, bge, visual_encoder, chunker)
    ingestor.ingest(
        dataset_path=args.dataset,
        max_pages=args.max_pages,
        skip_faiss=args.skip_faiss,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: run_eval.py 新增 --visual-model 参数 + 工厂创建编码器**

修改 `scripts/run_eval.py` 的关键部分：

```python
# scripts/run_eval.py — 在 import 区域修改

# 原:
# from src.ingestion.encoders import BGEEmbedder, ColPaliEmbedder

# 改为:
from src.ingestion.encoders import BGEEmbedder, create_visual_encoder

# — 在 main() 中，parser 区域新增参数 —
parser.add_argument("--visual-model", default="colpali",
                    choices=["colpali", "colembed-3b"],
                    help="Visual embedding model (default: colpali)")

# — 在 FaissColPaliStore 初始化部分，替换 —
# 原:
# faiss_store = FaissColPaliStore()

# 改为:
if args.visual_model == "colembed-3b":
    faiss_store = FaissColPaliStore(
        index_path=cfg.get("storage.faiss.colembed_index_path"),
        id_map_path=cfg.get("storage.faiss.colembed_id_map_path"),
    )
else:
    faiss_store = FaissColPaliStore()

# — Phase A 预编码部分，替换 —
# 原:
# colpali = ColPaliEmbedder()

# 改为:
visual_encoder = create_visual_encoder(model_name=args.visual_model)

# 原:
# pre_encoded_visual = colpali.encode_queries_batch(query_texts, batch_size=8)
# colpali.unload()

# 改为:
pre_encoded_visual = visual_encoder.encode_queries_batch(query_texts, batch_size=8)
visual_encoder.unload()

# — Phase B ingest 部分，当需要重建索引时 —
# 原:
# colpali_for_ingest = ColPaliEmbedder()

# 改为:
visual_for_ingest = create_visual_encoder(model_name=args.visual_model)

# （后续 references to colpali_for_ingest 替换为 visual_for_ingest）

# — 构造检索器部分 —
# 原:
# visual = VisualRetriever(faiss_store, pg_store, colpali)

# 改为:
visual = VisualRetriever(faiss_store, pg_store, visual_encoder)

# 原:
# retriever = PrismRAGRetriever(..., colpali=colpali, ...)

# 改为:
retriever = PrismRAGRetriever(..., colpali=visual_encoder, ...)
```

完整替换后的 `scripts/run_eval.py`:

```python
#!/usr/bin/env python
"""评测入口脚本 — 支持多模型 Visual 路"""

import argparse
import logging
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import cfg
from src.ingestion.encoders import BGEEmbedder, create_visual_encoder
from src.ingestion.text_chunker import TextChunker
from src.evaluation.ablation import load_eval_data, run_ablation
from src.evaluation.vidore_adapter import PrismRAGRetriever
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.fusion import RRFFusion
from src.retrieval.hyde import HyDEGenerator
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
    parser.add_argument("--language", default="en", choices=["en", "all"])
    parser.add_argument("--expected-query-count", type=int, default=None)
    parser.add_argument("--quick", action="store_true",
                        help="仅跑新增配置")
    parser.add_argument("--visual-model", default="colpali",
                        choices=["colpali", "colembed-3b"],
                        help="Visual embedding model")
    args = parser.parse_args()

    cfg.load()

    # ── Phase 0: 加载 & 过滤评测数据 ──────────────────────────
    logger.info(f"加载评测数据 (language={args.language}, visual={args.visual_model})...")
    queries_ds, qrel_map = load_eval_data(
        dataset_path=args.dataset,
        max_queries=args.max_queries,
        language=args.language,
    )
    num_queries = len(queries_ds)
    logger.info(f"评测 query 数量: {num_queries}")

    expected = args.expected_query_count
    if expected is None:
        expected = 283 if args.language == "en" else None
    if expected is not None and args.max_queries is None and num_queries != expected:
        raise RuntimeError(
            f"query 数量校验失败: 预期 {expected}, 实际 {num_queries}。"
        )

    # ── 基础设施初始化 ────────────────────────────────────────
    pg_store = PgVectorStore()
    if args.visual_model == "colembed-3b":
        faiss_store = FaissColPaliStore(
            index_path=cfg.get("storage.faiss.colembed_index_path"),
            id_map_path=cfg.get("storage.faiss.colembed_id_map_path"),
        )
    else:
        faiss_store = FaissColPaliStore()

    bge = BGEEmbedder()
    chunker = TextChunker()
    bm25 = BM25Retriever()
    dense = DenseRetriever(pg_store, bge)
    fusion = RRFFusion(rrf_k=60)
    reranker = Reranker()
    hyde = HyDEGenerator()

    # ── Phase A: 预编码 visual query ──────────────────────────
    logger.info(f"预编码 visual query ({args.visual_model})...")
    visual_encoder = create_visual_encoder(model_name=args.visual_model)
    query_texts = [str(queries_ds[i]["query"]) for i in range(num_queries)]
    pre_encoded_visual = visual_encoder.encode_queries_batch(query_texts, batch_size=8)
    logger.info(f"完成 {len(pre_encoded_visual)} 条 query 预编码")
    visual_encoder.unload()
    logger.info("Visual encoder 已卸载，显存已释放")
    torch.cuda.empty_cache()

    # ── Phase A2: HyDE 预计算 ─────────────────────────────────
    logger.info("HyDE 预计算 283 条 query（Ollama GPU 加速）...")
    import subprocess, time
    result = subprocess.run(["pgrep", "-f", "ollama serve"], capture_output=True)
    if result.returncode != 0:
        subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(5)
    hyde.precompute(query_texts)
    logger.info(f"HyDE 预计算完成，缓存 {len(hyde._cache)} 条")
    subprocess.run(["pkill", "-f", "ollama serve"], capture_output=True)
    time.sleep(3)
    torch.cuda.empty_cache()
    logger.info("Ollama 已关闭，显存已释放")

    # ── 加载 zerank-2 ─────────────────────────────────────────
    logger.info("加载 zerank-2 reranker (bf16)...")
    zerank_reranker = Reranker(
        model_id=cfg.zerank_reranker_model_id,
        model_kwargs={"torch_dtype": torch.bfloat16},
    )

    # ── Phase B: Ingest / Load FAISS ──────────────────────────
    if not args.skip_index:
        visual_for_ingest = create_visual_encoder(model_name=args.visual_model)
        from src.ingestion.vidore_ingestor import ViDoReIngestor
        ingestor = ViDoReIngestor(pg_store, faiss_store, bge, visual_for_ingest, chunker)
        ingestor.ingest(dataset_path=args.dataset)
        bm25.fit_from_pgvector(pg_store)
        logger.info("BM25 索引构建完成")
        visual_for_ingest.unload()
    else:
        faiss_loaded = faiss_store.load()
        if faiss_loaded:
            bm25.fit_from_pgvector(pg_store)
            logger.info("索引加载成功，跳过构建")
        else:
            logger.warning("FAISS 索引不存在，重新构建")
            visual_for_ingest = create_visual_encoder(model_name=args.visual_model)
            from src.ingestion.vidore_ingestor import ViDoReIngestor
            ingestor = ViDoReIngestor(pg_store, faiss_store, bge, visual_for_ingest, chunker)
            ingestor.ingest(dataset_path=args.dataset)
            bm25.fit_from_pgvector(pg_store)
            logger.info("BM25 索引构建完成")
            visual_for_ingest.unload()

    # ── 构造检索器 ────────────────────────────────────────────
    visual = VisualRetriever(faiss_store, pg_store, visual_encoder)
    retriever = PrismRAGRetriever(
        pg_store=pg_store, faiss_store=faiss_store, bge=bge, colpali=visual_encoder,
        chunker=chunker, bm25=bm25, dense=dense, visual=visual,
        fusion=fusion, reranker=reranker, hyde=hyde, zerank_reranker=zerank_reranker,
    )

    # ── 执行消融实验 ──────────────────────────────────────────
    run_ablation(
        retriever,
        queries_ds=queries_ds,
        qrel_map=qrel_map,
        output_dir=args.output_dir,
        pre_encoded_visual=pre_encoded_visual,
        language=args.language,
        quick=args.quick,
        visual_model=args.visual_model,
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: 语法验证**

```bash
cd /Users/theyang/Documents/ai/pdf-rag && python -c "
from scripts.ingest_vidore import main
from scripts.run_eval import main
print('Both CLI entry points importable')
"
```

- [ ] **Step 4: Commit**

```bash
git add scripts/ingest_vidore.py scripts/run_eval.py
git commit -m "feat: --visual-model CLI 参数 — ingest_vidore + run_eval 支持 colembed-3b"
```

---

### Task 6: 消融配置 — ablation.py

**Files:**
- Modify: `src/evaluation/ablation.py`

- [ ] **Step 1: 在 ablation.py 中新增 colembed 消融配置**

在 `ABLATION_CONFIGS` 列表末尾（`Full_zerank2_HyDE` 之后）新增：

```python
# src/evaluation/ablation.py — 在 ABLATION_CONFIGS 末尾新增

# ── colembed-3b 消融（仅跑 colembed 相关配置）──
# 基线对照 ColPali Visual_only=0.1302 来自此前消融记录，无需重跑
ABLATION_CONFIGS_COLEMBED = [
    # Colembed Visual_only
    AblationConfig(name="Visual_only_colembed",
        use_bm25=False, use_dense=False, use_visual=True, use_rerank=False),
    # Colembed Full + zerank-2
    AblationConfig(name="Full_zerank2_colembed",
        use_bm25=True, use_dense=True, use_visual=True, use_rerank=True,
        reranker_type="zerank", use_hyde=False),
]
```

在 `run_ablation()` 函数中，增加 `visual_model` 参数并据此选择配置列表：

```python
# src/evaluation/ablation.py — run_ablation() 签名和 config 选择逻辑

def run_ablation(
    retriever: PrismRAGRetriever,
    queries_ds,
    qrel_map: Dict[int, set],
    output_dir: str = "results",
    pre_encoded_visual: Optional[Dict[int, "torch.Tensor"]] = None,
    language: str = "en",
    quick: bool = False,
    visual_model: str = "colpali",
) -> List[dict]:
    # ...
    # ── 选择消融配置 ──
    if visual_model.startswith("colembed"):
        configs = ABLATION_CONFIGS_COLEMBED
        logger.info(f"Colembed 模式: 仅跑 {len(configs)} 组配置")
    elif quick:
        configs = [c for c in ABLATION_CONFIGS if c.name in (
            "Full_BGE_HyDE", "Full_zerank2_HyDE"
        )]
        logger.info(f"Quick 模式: 仅跑 {len(configs)} 组新配置")
    else:
        configs = ABLATION_CONFIGS
```
```


- [ ] **Step 2: 验证消融配置可导入**

```bash
cd /Users/theyang/Documents/ai/pdf-rag && python -c "
from src.evaluation.ablation import ABLATION_CONFIGS_COLEMBED
for c in ABLATION_CONFIGS_COLEMBED:
    print(c.name)
"
```

Expected: 输出 2 行配置名。

- [ ] **Step 3: Commit**

```bash
git add src/evaluation/ablation.py
git commit -m "feat: colembed-3b 消融配置 — Visual_only_colembed + Full_zerank2_colembed"
```

---

### Task 7: 云端依赖 — requirements-cloud.txt + cloud_setup.sh

**Files:**
- Modify: `requirements-cloud.txt`
- Modify: `scripts/cloud_setup.sh`

- [ ] **Step 1: requirements-cloud.txt 新增版本约束**

```txt
# requirements-cloud.txt — 新增最后两行

torch>=2.0
colpali-engine>=0.3.0
datasets>=2.15.0
psycopg2-binary>=2.9
pgvector>=0.3
faiss-gpu>=1.14
rank-bm25>=0.2
sentence-transformers>=3.0
vidore-benchmark>=0.1
ragas>=0.2
fastapi>=0.115
uvicorn[standard]>=0.34
pyyaml>=6.0
pillow>=10.0
numpy>=1.24
tqdm>=4.66
pytest>=8.0
ruff>=0.8
transformers>=4.49.0
flash-attn>=2.6.3
```

- [ ] **Step 2: cloud_setup.sh 新增 flash-attn 安装检查**

在 `scripts/cloud_setup.sh` 的 pip install 区域后新增 flash-attn 安装步骤。找到 `pip install` 的段落，在其后添加：

```bash
# ============================================================
# Flash Attention 2 安装（Colembed-3B 依赖，仅 GPU 环境）
# ============================================================
echo ""
echo ">>> 检查 flash-attn..."
if python -c "import flash_attn" 2>/dev/null; then
    echo "  ✅ flash-attn 已安装"
else
    echo "  📦 安装 flash-attn==2.6.3..."
    pip install flash-attn==2.6.3 --no-build-isolation
    echo "  ✅ flash-attn 安装完成"
fi
```

- [ ] **Step 3: 验证文件语法**

```bash
cd /Users/theyang/Documents/ai/pdf-rag && bash -n scripts/cloud_setup.sh && echo "shell syntax OK"
```

- [ ] **Step 4: Commit**

```bash
git add requirements-cloud.txt scripts/cloud_setup.sh
git commit -m "feat: 云端依赖 — flash-attn 2.6.3 + transformers>=4.49.0"
```

---

### Task 8: 端到端集成测试

**Files:**
- 无新文件，集成验证

- [ ] **Step 1: 本地语法全量检查**

```bash
cd /Users/theyang/Documents/ai/pdf-rag && python -c "
from src.config import cfg; cfg.load()
from src.ingestion.encoders import BGEEmbedder, ColPaliEmbedder, ColembedEncoder, create_visual_encoder
from src.retrieval.visual_retriever import VisualRetriever
from src.evaluation.vidore_adapter import PrismRAGRetriever
from src.evaluation.ablation import ABLATION_CONFIGS_COLEMBED
from src.ingestion.vidore_ingestor import ViDoReIngestor
print('All imports OK')
e = create_visual_encoder('colpali')
print('ColPali factory OK')
# Colembed 工厂在本地会因无 flash-attn 而失败，仅验证不 crash 在 import 阶段
try:
    e2 = create_visual_encoder('colembed-3b')
except Exception as ex:
    print(f'Colembed factory deferred (expected on macOS): {ex}')
"
```

Expected: ColPali 工厂成功，Colembed 在 macOS 上延迟失败（无 CUDA/flash-attn）。

- [ ] **Step 2: 运行现有测试套件**

```bash
cd /Users/theyang/Documents/ai/pdf-rag && python -m pytest tests/ -x -q --timeout=60
```

Expected: 所有测试通过。

- [ ] **Step 3: Commit（如有残余改动）**

```bash
git add -A && git diff --cached --stat
# 如有改动则提交
```

---

## Execution Order

任务按编号顺序执行：1 → 2 → 3 → 4 → 5 → 6 → 7 → 8

- Task 1-4 可在本地完成（纯代码 + macOS 语法验证）
- Task 5-6 可在本地完成（CLI + 配置代码）
- Task 7 是云端准备
- Task 8 是本地集成验证
- **实际运行**（评分验证 + 全量编码 + 消融评测）需在云端 GPU 环境进行
