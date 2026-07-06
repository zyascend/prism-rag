# PrismRAG P1 检索 MVP 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 构建 PrismRAG 检索层 MVP —— BM25 + Dense + Visual 三路检索 + RRF 融合 + cross-encoder 重排，在 ViDoRe Industrial 子集上跑通评测闭环，并加入 20 条拒答 RAGAS sanity。

**架构:** 逻辑分层单体，同 Python 进程内模块边界实现。Ingestion 将 ViDoRe 语料编码为 BGE 单向量（pgvector）和 ColPali 多向量（FAISS）；Retrieval 实现三路检索 + RRF 融合 + rerank；Evaluation 适配 vidore-benchmark 接口并运行消融实验。可复现骨架（Makefile + config + 版本化）从第一天嵌入。

**Tech Stack:** Python 3.11 (uv), torch (MPS), colpali-engine, datasets, pgvector, FAISS, rank_bm25, bge-reranker-large, vidore-benchmark, RAGAS, Ollama qwen2:7b

**前置数据状态:** ViDoRe v3 Industrial 子集已下载（5,244 页, 1,698 查询, 9,684 qrels），HF cache 位于 `~/.cache/huggingface/datasets/vidore___vidore_v3_industrial/`。POC 已验证 colpali-engine 编码吞吐 ~1.5 pg/s（MPS bfloat16），FAISS MaxSim 可行性。

---

## 文件结构

### 新增目录

```
pdf-rag/
├── config/
│   └── models.yaml                  # 模型版本钉死文件
├── src/
│   ├── __init__.py
│   ├── config.py                    # 配置加载器
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── text_chunker.py          # 文本分块策略（按 ViDoRe 文本切块）
│   │   ├── encoders.py              # BGE + ColPali 编码器封装
│   │   └── vidore_ingestor.py       # ViDoRe 数据集 → pgvector + FAISS
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── bm25_retriever.py        # BM25 检索器（rank_bm25）
│   │   ├── dense_retriever.py       # Dense 检索器（pgvector HNSW）
│   │   ├── visual_retriever.py      # Visual 检索器（FAISS ColPali + MaxSim + grounding 反查）
│   │   ├── fusion.py                # FusionStrategy 接口 + RRFFusion
│   │   └── reranker.py              # cross-encoder reranker
│   ├── store/
│   │   ├── __init__.py
│   │   ├── pgvector_store.py        # pgvector 客户端封装
│   │   └── faiss_store.py           # FAISS 索引封装
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── vidore_adapter.py        # vidore-benchmark BaseBeIRRetriever 适配器
│   │   ├── ablation.py              # 消融实验运行器
│   │   └── ragas_sanity.py          # 20 条拒答 RAGAS faithfulness
│   └── api/
│       ├── __init__.py
│       └── routes.py                # FastAPI search + ingest 端点
├── scripts/
│   ├── ingest_vidore.py             # 入口: 导入 ViDoRe 数据
│   └── run_eval.py                  # 入口: 运行评测
├── tests/
│   ├── test_text_chunker.py
│   ├── test_bm25_retriever.py
│   ├── test_dense_retriever.py
│   ├── test_visual_retriever.py
│   ├── test_fusion.py
│   └── test_reranker.py
├── data/
│   └── vidore/                      # 符号链接 → HF cache 或本地副本
├── pyproject.toml                   # uv 项目定义
├── Makefile                         # 一键命令入口
└── .env.example                     # 环境变量模板
```

### 不包含（留 P2）

- `src/ingestion/pdf_ingestor.py` —— 真实 PDF 解析（P2 + Demo 知识库时建）
- `src/graphrag/` —— 图检索
- `src/agent/` —— ReACT Agent
- `frontend/` —— React 前端（P1 评估阶段靠 curl/json 验证，UI demo 捆绑到 API 验证后）

---

## 任务分解

### 任务 1: 项目脚手架 —— pyproject.toml + 目录结构 + config

**Files:**
- Create: `pyproject.toml`
- Create: `config/models.yaml`
- Create: `src/__init__.py`
- Create: `src/config.py`
- Create: `src/ingestion/__init__.py`
- Create: `src/retrieval/__init__.py`
- Create: `src/store/__init__.py`
- Create: `src/evaluation/__init__.py`
- Create: `src/api/__init__.py`
- Create: `.env.example`
- Modify: `.gitignore`

- [ ] **Step 1: 创建 pyproject.toml**

```toml
[project]
name = "prismrag"
version = "0.1.0"
description = "Multimodal RAG that prisms PDF pages into lexical, semantic, and visual retrieval channels"
requires-python = ">=3.11"
dependencies = [
    "torch>=2.0",
    "colpali-engine>=0.3.0",
    "datasets>=5.0",
    "psycopg2-binary>=2.9",
    "pgvector>=0.3",
    "faiss-cpu>=1.14",
    "rank-bm25>=0.2",
    "sentence-transformers>=3.0",
    "vidore-benchmark>=0.1",
    "ragas>=0.2",
    "fastapi>=0.115",
    "uvicorn[standard]>=0.34",
    "pyyaml>=6.0",
    "pillow>=10.0",
    "numpy>=1.24",
    "tqdm>=4.66",
]

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.backends._legacy:_Backend"

[tool.setuptools.packages.find]
where = ["src"]
```

- [ ] **Step 2: 创建 config/models.yaml**

```yaml
# 模型版本钉死文件
# 换版本 = 改此文件 + 重编码 + 重评测
models:
  colpali: "vidore/colpali-v1.3"
  bge_embedding: "BAAI/bge-large-en-v1.5"
  bge_reranker: "BAAI/bge-reranker-large"
  llm: "qwen2:7b"           # Ollama model name

embedding:
  bge_dim: 768
  bge_device: "mps"         # mps / cpu / cuda
  colpali_device: "mps"

retrieval:
  bm25_k: 20
  dense_k: 20
  visual_k: 20
  rrf_k: 60
  rerank_k: 5

storage:
  pgvector:
    host: "localhost"
    port: 5432
    dbname: "prismrag"
    user: "prismrag"
    password: "prismrag"
  faiss:
    index_path: "indexes/colpali-vidore-industrial.faiss"
    id_map_path: "indexes/colpali-vidore-industrial-ids.npy"

vidore:
  dataset_name: "vidore/vidore_v3_industrial"
  split_corpus: "corpus"
  split_queries: "queries"
  split_qrels: "qrels"
  num_pages: 5244
```

- [ ] **Step 3: 创建 src/config.py**

```python
"""配置加载器 — 加载 models.yaml 并提供类型安全访问"""

from pathlib import Path
from typing import Any
import yaml


class Config:
    """全局配置，单例模式"""
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._loaded = False
        return cls._instance

    def load(self, path: str | None = None) -> "Config":
        if self._loaded:
            return self
        config_path = path or (Path(__file__).parent.parent / "config" / "models.yaml")
        with open(config_path) as f:
            self._data = yaml.safe_load(f)
        self._loaded = True
        return self

    def __getattr__(self, key: str) -> Any:
        if not self._loaded:
            self.load()
        if key in self._data:
            return self._data[key]
        for section in self._data.values():
            if isinstance(section, dict) and key in section:
                return section[key]
        raise AttributeError(f"Config key '{key}' not found")

    @property
    def colpali_model_id(self) -> str:
        return self._data["models"]["colpali"]

    @property
    def bge_model_id(self) -> str:
        return self._data["models"]["bge_embedding"]

    @property
    def reranker_model_id(self) -> str:
        return self._data["models"]["bge_reranker"]

    @property
    def bge_dim(self) -> int:
        return self._data["embedding"]["bge_dim"]


cfg = Config()
```

- [ ] **Step 4: 更新 .gitignore**

```gitignore
# Python
__pycache__/
*.py[cod]
*.egg-info/
.venv/
venv/

# 索引文件（大文件走 Release artifact，不走仓库）
indexes/
results/

# 环境变量
.env

# IDE
.idea/
.vscode/
.DS_Store
```

- [ ] **Step 5: 创建 .env.example**

```env
# pgvector
PGVECTOR_HOST=localhost
PGVECTOR_PORT=5432
PGVECTOR_DB=prismrag
PGVECTOR_USER=prismrag
PGVECTOR_PASSWORD=prismrag

# 索引路径
FAISS_INDEX_PATH=indexes/colpali-vidore-industrial.faiss
FAISS_ID_MAP_PATH=indexes/colpali-vidore-industrial-ids.npy

# Ollama
OLLAMA_BASE_URL=http://localhost:11434
```

- [ ] **Step 6: 初始化 uv 环境和安装依赖**

```bash
cd /Users/theyang/Documents/ai/pdf-rag
uv venv .venv --python 3.11
source .venv/bin/activate
uv pip install -e ".[dev]"
```

Run: `uv venv .venv --python 3.11 && source .venv/bin/activate && uv pip install -e ".[dev]"`
Expected: 依赖安装成功，`import prismrag` 不报错

- [ ] **Step 7: 提交**

```bash
git add pyproject.toml config/ src/ .gitignore .env.example
git commit -m "chore: scaffold project structure with pyproject.toml and config"
```

---

### 任务 2: 文本编码器封装（BGE + ColPali）

**Files:**
- Create: `src/ingestion/encoders.py`

- [ ] **Step 1: 创建 encoders.py 实现 BGE 编码器和 ColPali 编码器**

```python
"""BGE + ColPali 编码器封装"""

from __future__ import annotations

from pathlib import Path
from typing import List

import torch
from colpali_engine.models import ColPali, ColPaliProcessor
from PIL import Image
from sentence_transformers import SentenceTransformer
from tqdm import trange

from src.config import cfg


class BGEEmbedder:
    """BGE-large-en-v1.5 文本编码器"""

    def __init__(self, device: str | None = None):
        self.device = device or cfg.bge_device
        # 使用 sentence-transformers 加载 BGE
        self.model = SentenceTransformer(
            cfg.bge_model_id,
            device=self.device,
        )
        self.dim = cfg.bge_dim

    @torch.no_grad()
    def encode(self, texts: List[str], batch_size: int = 32, show_progress: bool = False) -> torch.Tensor:
        """编码文本列表为向量矩阵 [N, dim]"""
        return self.model.encode(
            texts,
            batch_size=batch_size,
            convert_to_tensor=True,
            show_progress_bar=show_progress,
            normalize_embeddings=True,  # BGE 惯例：归一化后用内积等价余弦
        )

    def __call__(self, texts: List[str]) -> torch.Tensor:
        return self.encode(texts)


class ColPaliEmbedder:
    """ColPali 整页多向量编码器"""

    def __init__(self, device: str | None = None):
        self.device = device or cfg.colpali_device
        self.model = ColPali.from_pretrained(
            cfg.colpali_model_id,
            torch_dtype=torch.bfloat16,
            device_map=self.device,
        ).eval()
        self.processor = ColPaliProcessor.from_pretrained(cfg.colpali_model_id)

    @torch.no_grad()
    def encode_pages(
        self, images: List[Image.Image], batch_size: int = 4, show_progress: bool = False
    ) -> List[torch.Tensor]:
        """编码页面列表，每页返回 [n_patches, 128] 多向量"""
        # 预热：首次 query 有 torch.compile 开销
        if not hasattr(self, "_warmed_up"):
            dummy = Image.new("RGB", (1000, 1600), color=255)
            self._warmup(dummy)
            self._warmed_up = True

        batches = []
        for i in trange(0, len(images), batch_size, disable=not show_progress, desc="ColPali encode"):
            batch_imgs = images[i : i + batch_size]
            batch_inputs = self.processor(
                images=batch_imgs,
                return_tensors="pt",
                padding=True,
            ).to(self.device)
            batch_outputs = self.model(**batch_inputs)
            # batch_outputs: [batch, n_patches, 128]
            batches.extend(list(batch_outputs.cpu()))

        return batches

    def _warmup(self, dummy_image: Image.Image):
        """MPS 首次查询预热"""
        inputs = self.processor(images=[dummy_image], return_tensors="pt", padding=True).to(self.device)
        _ = self.model(**inputs)

    @torch.no_grad()
    def encode_query(self, text: str) -> torch.Tensor:
        """编码单条文本查询为 [1, n_patches, 128]（ColPali 查询编码）"""
        inputs = self.processor(text=[text], return_tensors="pt", padding=True).to(self.device)
        return self.model(**inputs).cpu()
```

- [ ] **Step 2: 创建测试文件 tests/test_encoders.py**

```python
"""编码器单元测试"""

import torch
from src.ingestion.encoders import BGEEmbedder, ColPaliEmbedder


def test_bge_encoder_output_shape():
    embedder = BGEEmbedder(device="cpu")
    texts = ["What is the load capacity?", "Conveyor belt specifications"]
    embs = embedder.encode(texts)
    assert isinstance(embs, torch.Tensor)
    assert embs.shape == (2, 768)
    # 验证归一化
    norms = torch.norm(embs, dim=1)
    assert torch.allclose(norms, torch.ones(2), atol=1e-5)


def test_colpali_encoder_page_output():
    from PIL import Image
    import numpy as np

    embedder = ColPaliEmbedder(device="cpu")
    # 创建模拟页面
    imgs = [Image.fromarray(np.random.randint(0, 255, (1600, 1000, 3), dtype=np.uint8)) for _ in range(2)]
    embs = embedder.encode_pages(imgs, batch_size=2)
    assert len(embs) == 2
    for emb in embs:
        assert emb.ndim == 2  # [n_patches, 128]
        assert emb.shape[-1] == 128


def test_colpali_query_output():
    embedder = ColPaliEmbedder(device="cpu")
    q_emb = embedder.encode_query("load capacity")
    assert q_emb.ndim == 3  # [1, n_patches, 128]
    assert q_emb.shape[-1] == 128
```

- [ ] **Step 3: 运行测试**

```bash
cd /Users/theyang/Documents/ai/pdf-rag
source .venv/bin/activate
python -m pytest tests/test_encoders.py -v
```

Expected: 全部 PASS（注意：ColPali 测试需要模型下载，首次会较慢）

- [ ] **Step 4: 提交**

```bash
git add src/ingestion/encoders.py tests/test_encoders.py
git commit -m "feat: add BGE and ColPali encoder wrappers"
```

---

### 任务 3: 文本分块策略

**Files:**
- Create: `src/ingestion/text_chunker.py`
- Create: `tests/test_text_chunker.py`

- [ ] **Step 1: 创建 text_chunker.py**

ViDoRe 语料的每个 corpus 条目是一页，带有 `markdown` 文本（OCR 提取）。分块策略：按段落（双换行）切块，每块最大 512 tokens；超长段落用句子边界切。

```python
"""ViDoRe 语料的文本分块策略

策略：
1. 按双换行切段落
2. 段落 ≤ 512 tokens → 直接作为一块
3. 段落 > 512 tokens → 按句号/换行边界切到 ≤ 512
4. 空段落跳过
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List


@dataclass
class Chunk:
    chunk_id: str
    page_id: int
    doc_id: str
    page_number: int
    text: str
    chunk_type: str = "text"  # text | table

    def __repr__(self) -> str:
        return f"Chunk(id={self.chunk_id}, page={self.page_id}, type={self.chunk_type})"


class TextChunker:
    """ViDoRe 文本分块器"""

    MAX_TOKENS = 512
    # 简单 token 估算：英文约 4 chars/token
    TOKEN_EST_RATIO = 4

    def __init__(self, max_tokens: int = MAX_TOKENS):
        self.max_tokens = max_tokens
        self.max_chars = max_tokens * self.TOKEN_EST_RATIO

    def chunk_page(
        self,
        page_id: int,
        doc_id: str,
        page_number: int,
        markdown_text: str | None,
    ) -> List[Chunk]:
        """将一页 markdown 文本切成 chunk 列表"""
        if not markdown_text or not markdown_text.strip():
            return []

        paragraphs = re.split(r"\n\s*\n", markdown_text.strip())
        chunks: List[Chunk] = []
        chunk_idx = 0

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            if len(para) <= self.max_chars:
                # 短段落直接作为一块
                chunk_idx += 1
                chunks.append(Chunk(
                    chunk_id=f"pg{page_id:05d}_ch{chunk_idx:03d}",
                    page_id=page_id,
                    doc_id=doc_id,
                    page_number=page_number,
                    text=para,
                    chunk_type="table" if self._looks_like_table(para) else "text",
                ))
            else:
                # 长段落：按句子边界切
                sentences = re.split(r"(?<=[.?!])\s+", para)
                buffer = ""
                for sent in sentences:
                    if len(buffer) + len(sent) + 1 <= self.max_chars:
                        buffer = (buffer + " " + sent).strip()
                    else:
                        if buffer:
                            chunk_idx += 1
                            chunks.append(Chunk(
                                chunk_id=f"pg{page_id:05d}_ch{chunk_idx:03d}",
                                page_id=page_id,
                                doc_id=doc_id,
                                page_number=page_number,
                                text=buffer,
                                chunk_type="text",
                            ))
                        buffer = sent
                if buffer:
                    chunk_idx += 1
                    chunks.append(Chunk(
                        chunk_id=f"pg{page_id:05d}_ch{chunk_idx:03d}",
                        page_id=page_id,
                        doc_id=doc_id,
                        page_number=page_number,
                        text=buffer,
                        chunk_type="text",
                    ))

        return chunks

    @staticmethod
    def _looks_like_table(text: str) -> bool:
        """启发式判断是否为表格文本（含管道符或明显的列对齐）"""
        lines = text.split("\n")
        pipe_count = sum(line.count("|") for line in lines[:5])
        return pipe_count >= 3
```

- [ ] **Step 2: 创建 tests/test_text_chunker.py**

```python
"""文本分块器测试"""

from src.ingestion.text_chunker import TextChunker


def test_empty_text():
    chunker = TextChunker()
    chunks = chunker.chunk_page(page_id=1, doc_id="doc1", page_number=1, markdown_text=None)
    assert chunks == []

    chunks = chunker.chunk_page(page_id=1, doc_id="doc1", page_number=1, markdown_text="")
    assert chunks == []


def test_single_paragraph():
    chunker = TextChunker()
    text = "This is a single paragraph with a reasonable length."
    chunks = chunker.chunk_page(page_id=1, doc_id="doc1", page_number=1, markdown_text=text)
    assert len(chunks) == 1
    assert chunks[0].text == text
    assert chunks[0].chunk_id == "pg00001_ch001"


def test_multiple_paragraphs():
    chunker = TextChunker()
    text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
    chunks = chunker.chunk_page(page_id=1, doc_id="doc1", page_number=1, markdown_text=text)
    assert len(chunks) == 3


def test_long_paragraph_splits():
    chunker = TextChunker(max_tokens=10)  # 40 chars
    text = "This is a very long paragraph that should be split into multiple chunks because it exceeds the maximum token limit."
    chunks = chunker.chunk_page(page_id=1, doc_id="doc1", page_number=1, markdown_text=text)
    assert len(chunks) >= 2


def test_table_detection():
    chunker = TextChunker()
    text = "| Col1 | Col2 | Col3 |\n|------|------|------|\n| A    | B    | C    |"
    chunks = chunker.chunk_page(page_id=1, doc_id="doc1", page_number=1, markdown_text=text)
    assert len(chunks) == 1
    assert chunks[0].chunk_type == "table"
```

- [ ] **Step 3: 运行测试**

```bash
cd /Users/theyang/Documents/ai/pdf-rag && source .venv/bin/activate && python -m pytest tests/test_text_chunker.py -v
```

Expected: 全部 PASS

- [ ] **Step 4: 提交**

```bash
git add src/ingestion/text_chunker.py tests/test_text_chunker.py
git commit -m "feat: add ViDoRe text chunker"
```

---

### 任务 4: 存储层封装（pgvector + FAISS）

**Files:**
- Create: `src/store/pgvector_store.py`
- Create: `src/store/faiss_store.py`

- [ ] **Step 1: 创建 pgvector_store.py**

```python
"""pgvector 存储封装

Schema:
  CREATE TABLE chunks (
    chunk_id TEXT PRIMARY KEY,
    page_id INTEGER NOT NULL,
    doc_id TEXT NOT NULL,
    page_number INTEGER NOT NULL,
    chunk_type TEXT NOT NULL DEFAULT 'text',
    text TEXT NOT NULL,
    bge_vector vector(768) NOT NULL
  );
  CREATE INDEX idx_chunks_page_id ON chunks(page_id);
  CREATE INDEX idx_chunks_doc_id ON chunks(doc_id);
"""

from __future__ import annotations

import os
from typing import List, Optional, Tuple

import numpy as np
import psycopg2
import psycopg2.extras
from pgvector.psycopg2 import register_vector

from src.config import cfg


class PgVectorStore:
    """pgvector 存储客户端"""

    def __init__(self, connection_string: str | None = None):
        self.conn_string = connection_string or self._default_conn_string()
        self._conn: Optional[psycopg2.extensions.connection] = None

    def _default_conn_string(self) -> str:
        pg = cfg.storage["pgvector"]
        return f"host={pg['host']} port={pg['port']} dbname={pg['dbname']} user={pg['user']} password={pg['password']}"

    @property
    def conn(self):
        if self._conn is None or self._conn.closed:
            self._conn = psycopg2.connect(self.conn_string)
            register_vector(self._conn)
        return self._conn

    def create_schema(self):
        """创建表和索引（幂等）"""
        with self.conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cur.execute("""
                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id TEXT PRIMARY KEY,
                    page_id INTEGER NOT NULL,
                    doc_id TEXT NOT NULL,
                    page_number INTEGER NOT NULL,
                    chunk_type TEXT NOT NULL DEFAULT 'text',
                    text TEXT NOT NULL,
                    bge_vector vector(768) NOT NULL
                )
            """)
            cur.execute("CREATE INDEX IF NOT EXISTS idx_chunks_page_id ON chunks(page_id)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_chunks_doc_id ON chunks(doc_id)")
            # HNSW 索引
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_chunks_bge_hnsw
                ON chunks USING hnsw (bge_vector vector_cosine_ops)
                WITH (m = 16, ef_construction = 200)
            """)
        self.conn.commit()

    def insert_chunks(self, chunks: List[Tuple]):
        """批量插入 chunk

        Args:
            chunks: [(chunk_id, page_id, doc_id, page_number, chunk_type, text, bge_vector), ...]
        """
        with self.conn.cursor() as cur:
            psycopg2.extras.execute_values(
                cur,
                """
                INSERT INTO chunks (chunk_id, page_id, doc_id, page_number, chunk_type, text, bge_vector)
                VALUES %s
                ON CONFLICT (chunk_id) DO NOTHING
                """,
                chunks,
                template="(%s, %s, %s, %s, %s, %s, %s::vector)",
            )
        self.conn.commit()

    def search_by_vector(self, query_vector: np.ndarray, k: int = 20) -> List[dict]:
        """余弦相似度搜索，返回 Top-k chunk"""
        with self.conn.cursor() as cur:
            cur.execute(
                """
                SELECT chunk_id, page_id, doc_id, page_number, chunk_type, text,
                       1 - (bge_vector <=> %s::vector) AS score
                FROM chunks
                ORDER BY bge_vector <=> %s::vector
                LIMIT %s
                """,
                (query_vector.tolist(), query_vector.tolist(), k),
            )
            rows = cur.fetchall()
            return [
                {
                    "chunk_id": r[0],
                    "page_id": r[1],
                    "doc_id": r[2],
                    "page_number": r[3],
                    "chunk_type": r[4],
                    "text": r[5],
                    "score": float(r[6]),
                }
                for r in rows
            ]

    def get_chunks_by_page_ids(self, page_ids: List[int]) -> List[dict]:
        """按 page_id 列表查询所有 chunk（Visual 路 grounding 反查用）"""
        if not page_ids:
            return []
        with self.conn.cursor() as cur:
            cur.execute(
                """
                SELECT chunk_id, page_id, doc_id, page_number, chunk_type, text
                FROM chunks
                WHERE page_id = ANY(%s)
                """,
                (page_ids,),
            )
            rows = cur.fetchall()
            return [
                {
                    "chunk_id": r[0],
                    "page_id": r[1],
                    "doc_id": r[2],
                    "page_number": r[3],
                    "chunk_type": r[4],
                    "text": r[5],
                }
                for r in rows
            ]

    def count(self) -> int:
        with self.conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM chunks")
            return cur.fetchone()[0]

    def close(self):
        if self._conn and not self._conn.closed:
            self._conn.close()
```

- [ ] **Step 2: 创建 faiss_store.py**

```python
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
        self.index_path = index_path or cfg.storage["faiss"]["index_path"]
        self.id_map_path = id_map_path or cfg.storage["faiss"]["id_map_path"]
        self._index: Optional[faiss.Index] = None
        self._page_ids: Optional[np.ndarray] = None  # 索引数组：每个 patch 对应的 page_id
        self._page_boundaries: Optional[List[Tuple[int, int]]] = None  # [(start, end), ...] 每页的 patch 范围

    def build_index(self, page_embeddings: Dict[int, torch.Tensor]):
        """从 page_embeddings 构建 FAISS 索引

        Args:
            page_embeddings: {page_id: [n_patches, 128]} 多向量字典
        """
        all_vectors: List[np.ndarray] = []
        all_ids: List[int] = []
        boundaries: List[Tuple[int, int]] = []

        start = 0
        for page_id in sorted(page_embeddings.keys()):
            emb = page_embeddings[page_id].numpy().astype(np.float32)  # [n_patches, 128]
            n = emb.shape[0]
            all_vectors.append(emb)
            all_ids.extend([page_id] * n)
            boundaries.append((start, start + n))
            start += n

        vectors = np.vstack(all_vectors)  # [total_patches, 128]
        self._page_ids = np.array(all_ids, dtype=np.int64)
        self._page_boundaries = boundaries

        # 用 IndexFlatIP（内积）存储所有 patch 向量
        dim = vectors.shape[1]
        self._index = faiss.IndexFlatIP(dim)
        self._index.add(vectors)

        # 索引大小记录
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
        """MaxSim 搜索

        Args:
            query_embedding: [1, n_q_patches, 128] ColPali 查询编码
            k: 返回 Top-k 页

        Returns:
            [{page_id, score}, ...]
        """
        assert self._index is not None, "索引未加载/构建"
        assert self._page_boundaries is not None

        q = query_embedding.numpy().astype(np.float32)  # [1, n_q, 128]
        n_q = q.shape[1]
        q_flat = q.reshape(n_q, -1)  # [n_q, 128]

        # 全表 Patch 搜索：对每个查询 patch 找 Top-k 最相似 patch
        # 但 MaxSim 需要：对每个查询 patch，取与所有目标 patch 的内积最大值
        # 优化版：直接对所有 patch 做矩阵乘，每页取 max 后平均
        all_vectors = faiss.rev_swig_ptr(self._index.x, self._index.ntotal * self._index.d).reshape(
            self._index.ntotal, self._index.d
        )  # [total_patches, 128]

        # 矩阵乘: [n_q, 128] @ [128, total_patches] → [n_q, total_patches]
        scores = q_flat @ all_vectors.T  # [n_q, total_patches]

        page_scores: Dict[int, float] = {}
        for page_idx, (start, end) in enumerate(self._page_boundaries):
            page_patch_scores = scores[:, start:end]  # [n_q, n_patches_in_page]
            max_per_query = page_patch_scores.max(axis=1)  # [n_q] —— 每个查询 patch 的 max
            page_score = float(max_per_query.mean())  # 标量
            page_id = int(self._page_ids[start])
            page_scores[page_id] = page_score

        # 排序取 Top-k
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
```

- [ ] **Step 3: 提交**

```bash
git add src/store/pgvector_store.py src/store/faiss_store.py
git commit -m "feat: add pgvector and FAISS storage wrappers"
```

---

### 任务 5: ViDoRe 数据导入器

**Files:**
- Create: `src/ingestion/vidore_ingestor.py`
- Create: `scripts/ingest_vidore.py`

- [ ] **Step 1: 创建 vidore_ingestor.py**

```python
"""ViDoRe 数据集导入管道

数据流：
  HF Dataset (image + markdown)
    ├─ TextChunker → BGE encode → pgvector
    └─ ColPali encode → FAISS index

用法:
  python -m scripts.ingest_vidore --dataset vidore/vidore_v3_industrial
"""

from __future__ import annotations

import logging
from typing import List, Optional

import torch
from datasets import load_dataset
from tqdm import tqdm

from src.config import cfg
from src.ingestion.encoders import BGEEmbedder, ColPaliEmbedder
from src.ingestion.text_chunker import Chunk, TextChunker
from src.store.faiss_store import FaissColPaliStore
from src.store.pgvector_store import PgVectorStore

logger = logging.getLogger(__name__)


class ViDoReIngestor:
    """ViDoRe 数据集导入器"""

    def __init__(
        self,
        pg_store: PgVectorStore,
        faiss_store: FaissColPaliStore,
        bge_embedder: BGEEmbedder,
        colpali_embedder: ColPaliEmbedder,
        chunker: TextChunker,
    ):
        self.pg = pg_store
        self.faiss = faiss_store
        self.bge = bge_embedder
        self.colpali = colpali_embedder
        self.chunker = chunker

    def ingest(
        self,
        dataset_path: str = "vidore/vidore_v3_industrial",
        max_pages: Optional[int] = None,
        skip_faiss: bool = False,
    ):
        """执行导入流程"""
        logger.info(f"加载数据集: {dataset_path}")
        ds = load_dataset(dataset_path, split="corpus", trust_remote_code=True)

        if max_pages:
            ds = ds.select(range(min(max_pages, len(ds))))

        total_pages = len(ds)
        logger.info(f"共 {total_pages} 页")

        # 1. 文本路：分块 → BGE → pgvector
        self._ingest_text(ds)

        # 2. 视觉路：ColPali 编码 → FAISS
        if not skip_faiss:
            self._ingest_visual(ds)

        logger.info("导入完成")

    def _ingest_text(self, ds):
        """文本路导入"""
        logger.info("=== 文本路: 分块 + BGE 编码 + pgvector 入库 ===")
        self.pg.create_schema()

        all_chunk_rows: List[tuple] = []
        all_texts: List[str] = []
        page_chunk_map: List[tuple] = []  # (page_idx, chunk_idx) 用于 BGE 编码后匹配

        for idx in tqdm(range(len(ds)), desc="分块"):
            row = ds[idx]
            page_id = int(row["corpus_id"])
            doc_id = str(row.get("doc_id", ""))
            page_number = int(row.get("page_number_in_doc", 0))
            markdown = row.get("markdown", None)

            chunks: List[Chunk] = self.chunker.chunk_page(
                page_id=page_id,
                doc_id=doc_id,
                page_number=page_number,
                markdown_text=markdown,
            )

            for chunk in chunks:
                page_chunk_map.append((idx, len(all_chunk_rows)))  # 实际不需要，直接按顺序
                all_chunk_rows.append((
                    chunk.chunk_id,
                    chunk.page_id,
                    chunk.doc_id,
                    chunk.page_number,
                    chunk.chunk_type,
                    chunk.text,
                    None,  # bge_vector 占位，BGE 编码后填充
                ))
                all_texts.append(chunk.text)

        logger.info(f"共 {len(all_chunk_rows)} 个 chunk，开始 BGE 编码...")

        # BGE 批量编码
        bge_embs = self.bge.encode(all_texts, batch_size=32, show_progress=True)

        # 填充向量并写入 pgvector
        batch_size = 100
        for i in range(0, len(all_chunk_rows), batch_size):
            batch = []
            for j in range(i, min(i + batch_size, len(all_chunk_rows))):
                chunk_id, page_id, doc_id, pn, ctype, text, _ = all_chunk_rows[j]
                vec = bge_embs[j].numpy().tolist()
                batch.append((chunk_id, page_id, doc_id, pn, ctype, text, vec))
            self.pg.insert_chunks(batch)

        logger.info(f"pgvector 入库完成, 共 {self.pg.count()} 条 chunk")

    def _ingest_visual(self, ds):
        """视觉路导入"""
        logger.info("=== 视觉路: ColPali 编码 + FAISS 建索引 ===")

        page_embeddings: dict = {}

        # 批次编码
        batch_size = 4
        for i in tqdm(range(0, len(ds), batch_size), desc="ColPali 编码"):
            batch_rows = [ds[j] for j in range(i, min(i + batch_size, len(ds)))]
            images = [row["image"] for row in batch_rows]
            page_ids = [int(row["corpus_id"]) for row in batch_rows]

            embs = self.colpali.encode_pages(images, batch_size=len(images))
            for pid, emb in zip(page_ids, embs):
                page_embeddings[pid] = emb

        logger.info(f"ColPali 编码完成, 共 {len(page_embeddings)} 页")

        # FAISS 建索引
        self.faiss.build_index(page_embeddings)
        self.faiss.save()

        logger.info(f"FAISS 索引完成, 共 {self.faiss.num_pages} 页, {self.faiss.index_size_mb:.1f} MB")
```

- [ ] **Step 2: 创建 scripts/ingest_vidore.py**

```python
#!/usr/bin/env python
"""ViDoRe 数据导入入口脚本"""

import argparse
import logging
import sys
from pathlib import Path

# 确保 src 在 Python path 中
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import cfg
from src.ingestion.encoders import BGEEmbedder, ColPaliEmbedder
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
    args = parser.parse_args()

    # 初始化
    cfg.load()
    pg_store = PgVectorStore()
    faiss_store = FaissColPaliStore()
    bge = BGEEmbedder()
    colpali = ColPaliEmbedder()
    chunker = TextChunker()

    ingestor = ViDoReIngestor(pg_store, faiss_store, bge, colpali, chunker)
    ingestor.ingest(
        dataset_path=args.dataset,
        max_pages=args.max_pages,
        skip_faiss=args.skip_faiss,
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: 验证导入脚本可解析（不执行，需要 pgvector 运行中）**

```bash
cd /Users/theyang/Documents/ai/pdf-rag && source .venv/bin/activate && python -c "
from scripts.ingest_vidore import main
import argparse
# 只验证 import，不实际运行
print('✅ 导入脚本可正常 import')
"
```

Expected: `✅ 导入脚本可正常 import`

- [ ] **Step 4: 提交**

```bash
git add src/ingestion/vidore_ingestor.py scripts/ingest_vidore.py
git commit -m "feat: add ViDoRe data ingestion pipeline"
```

---

### 任务 6: BM25 检索器

**Files:**
- Create: `src/retrieval/bm25_retriever.py`
- Create: `tests/test_bm25_retriever.py`

- [ ] **Step 1: 创建 bm25_retriever.py**

```python
"""BM25 检索器 — 基于 rank_bm25"""

from __future__ import annotations

import math
from typing import List, Optional

from rank_bm25 import BM25Okapi

from src.ingestion.text_chunker import Chunk
from src.store.pgvector_store import PgVectorStore


class BM25Retriever:
    """BM25 检索器

    用法:
      retriever = BM25Retriever()
      retriever.fit(all_chunks)          # 从 chunk 列表构建索引
      results = retriever.search(query)  # 检索
    """

    def __init__(self):
        self._bm25: Optional[BM25Okapi] = None
        self._chunks: List[dict] = []

    def fit_from_pgvector(self, pg_store: PgVectorStore):
        """从 pgvector 读取所有 chunk 并构建 BM25 索引"""
        # 使用 pgvector 的 count 和分页查询获取所有文本
        chunks = []
        offset = 0
        limit = 1000
        while True:
            with pg_store.conn.cursor() as cur:
                cur.execute(
                    "SELECT chunk_id, page_id, doc_id, page_number, chunk_type, text FROM chunks ORDER BY chunk_id LIMIT %s OFFSET %s",
                    (limit, offset),
                )
                rows = cur.fetchall()
                if not rows:
                    break
                for r in rows:
                    chunks.append({
                        "chunk_id": r[0],
                        "page_id": r[1],
                        "doc_id": r[2],
                        "page_number": r[3],
                        "chunk_type": r[4],
                        "text": r[5],
                    })
                offset += limit

        self.fit(chunks)

    def fit(self, chunks: List[dict]):
        """从 chunk dict 列表构建 BM25 索引"""
        self._chunks = chunks
        tokenized_corpus = [self._tokenize(c["text"]) for c in chunks]
        self._bm25 = BM25Okapi(tokenized_corpus)

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """分词：小写 + 非字母数字分割"""
        import re
        return re.findall(r"[a-z0-9]+", text.lower())

    def search(self, query: str, k: int = 20) -> List[dict]:
        """检索 Top-k chunk"""
        if self._bm25 is None:
            raise RuntimeError("BM25 索引未构建，请先调用 fit()")

        tokenized_query = self._tokenize(query)
        scores = self._bm25.get_scores(tokenized_query)

        # 排序取 Top-k
        top_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True,
        )[:k]

        results = []
        for idx in top_indices:
            score = scores[idx]
            if score > 0:  # 过滤零分结果
                chunk = self._chunks[idx]
                results.append({
                    **chunk,
                    "score": float(score),
                    "retrieval_type": "bm25",
                })

        return results
```

- [ ] **Step 2: 创建 tests/test_bm25_retriever.py**

```python
"""BM25 检索器测试"""

from src.retrieval.bm25_retriever import BM25Retriever


def test_bm25_tokenize():
    tokens = BM25Retriever._tokenize("Load Capacity: 500 kg")
    assert "load" in tokens
    assert "capacity" in tokens
    assert "500" in tokens
    assert "kg" in tokens


def test_bm25_simple_search():
    chunks = [
        {"chunk_id": "ch1", "page_id": 1, "doc_id": "doc1", "page_number": 1, "chunk_type": "text", "text": "The conveyor belt has a load capacity of 500 kg."},
        {"chunk_id": "ch2", "page_id": 1, "doc_id": "doc1", "page_number": 1, "chunk_type": "text", "text": "Motor speed is 1500 RPM."},
        {"chunk_id": "ch3", "page_id": 2, "doc_id": "doc1", "page_number": 2, "chunk_type": "text", "text": "Safety guidelines for operation."},
    ]
    retriever = BM25Retriever()
    retriever.fit(chunks)

    results = retriever.search("conveyor load capacity", k=2)
    assert len(results) == 2
    assert results[0]["chunk_id"] == "ch1"  # 最相关


def test_bm25_no_match():
    chunks = [
        {"chunk_id": "ch1", "page_id": 1, "doc_id": "doc1", "page_number": 1, "chunk_type": "text", "text": "Safety guidelines."},
    ]
    retriever = BM25Retriever()
    retriever.fit(chunks)
    results = retriever.search("unrelated query about nothing", k=5)
    assert len(results) == 0  # 无匹配时返回空


def test_bm25_retrieval_type_tag():
    chunks = [
        {"chunk_id": "ch1", "page_id": 1, "doc_id": "doc1", "page_number": 1, "chunk_type": "text", "text": "Load capacity is 500 kg."},
    ]
    retriever = BM25Retriever()
    retriever.fit(chunks)
    results = retriever.search("load capacity", k=5)
    assert len(results) >= 1
    assert results[0]["retrieval_type"] == "bm25"
```

- [ ] **Step 3: 运行测试**

```bash
cd /Users/theyang/Documents/ai/pdf-rag && source .venv/bin/activate && python -m pytest tests/test_bm25_retriever.py -v
```

Expected: 全部 PASS

- [ ] **Step 4: 提交**

```bash
git add src/retrieval/bm25_retriever.py tests/test_bm25_retriever.py
git commit -m "feat: add BM25 retriever"
```

---

### 任务 7: Dense 检索器

**Files:**
- Create: `src/retrieval/dense_retriever.py`
- Create: `tests/test_dense_retriever.py`

- [ ] **Step 1: 创建 dense_retriever.py**

```python
"""Dense 检索器 — pgvector HNSW 余弦相似度搜索"""

from __future__ import annotations

from typing import List

import numpy as np
import torch

from src.ingestion.encoders import BGEEmbedder
from src.store.pgvector_store import PgVectorStore


class DenseRetriever:
    """Dense 检索器：BGE encode query → pgvector HNSW 搜索"""

    def __init__(self, pg_store: PgVectorStore, embedder: BGEEmbedder):
        self.pg = pg_store
        self.bge = embedder

    def search(self, query: str, k: int = 20) -> List[dict]:
        """检索 Top-k chunk

        返回格式:
          [{chunk_id, page_id, doc_id, page_number, chunk_type, text, score, retrieval_type}, ...]
        """
        # 1. BGE 编码查询
        query_emb = self.bge.encode([query])  # [1, 768]
        query_vec = query_emb.cpu().numpy().astype(np.float32)[0]

        # 2. pgvector HNSW 搜索
        results = self.pg.search_by_vector(query_vec, k=k)

        # 3. 添加 retrieval_type 标记
        for r in results:
            r["retrieval_type"] = "dense"

        return results
```

- [ ] **Step 2: 创建 tests/test_dense_retriever.py**

```python
"""Dense 检索器测试（mock pgvector）"""

from unittest.mock import MagicMock, patch

import numpy as np
import torch

from src.retrieval.dense_retriever import DenseRetriever


def test_dense_retriever_search():
    """使用 mock 的 pgstore 验证搜索流程"""
    mock_pg = MagicMock()
    mock_pg.search_by_vector.return_value = [
        {"chunk_id": "ch1", "page_id": 1, "text": "doc text", "score": 0.92},
    ]

    mock_bge = MagicMock()
    mock_bge.encode.return_value = torch.randn(1, 768)

    retriever = DenseRetriever(pg_store=mock_pg, embedder=mock_bge)
    results = retriever.search("test query", k=10)

    assert len(results) == 1
    assert results[0]["retrieval_type"] == "dense"
    mock_bge.encode.assert_called_once()
    mock_pg.search_by_vector.assert_called_once()
```

- [ ] **Step 3: 运行测试**

```bash
cd /Users/theyang/Documents/ai/pdf-rag && source .venv/bin/activate && python -m pytest tests/test_dense_retriever.py -v
```

Expected: PASS

- [ ] **Step 4: 提交**

```bash
git add src/retrieval/dense_retriever.py tests/test_dense_retriever.py
git commit -m "feat: add Dense retriever (BGE + pgvector)"
```

---

### 任务 8: Visual 检索器（FAISS ColPali + MaxSim + Grounding 反查）

**Files:**
- Create: `src/retrieval/visual_retriever.py`
- Create: `tests/test_visual_retriever.py`

- [ ] **Step 1: 创建 visual_retriever.py**

```python
"""Visual 检索器 — ColPali + FAISS MaxSim + pgvector grounding 反查

流程：
  1. ColPali encode 查询文本 → [1, n_q_patches, 128]
  2. FAISS MaxSim 全表扫 → Top-20 页
  3. 命中页 → 反查 pgvector 该页所有 BGE chunk → 纳入候选集
"""

from __future__ import annotations

from typing import List

from src.ingestion.encoders import ColPaliEmbedder
from src.store.faiss_store import FaissColPaliStore
from src.store.pgvector_store import PgVectorStore


class VisualRetriever:
    """Visual 检索器：ColPali encode → FAISS MaxSim → pgvector grounding 反查"""

    def __init__(
        self,
        faiss_store: FaissColPaliStore,
        pg_store: PgVectorStore,
        colpali_embedder: ColPaliEmbedder,
    ):
        self.faiss = faiss_store
        self.pg = pg_store
        self.colpali = colpali_embedder

    def search(self, query: str, k: int = 20) -> List[dict]:
        """检索 Top-k 页 → 反查该页所有 chunk

        返回:
          [{chunk_id, page_id, text, chunk_type, score, retrieval_type}, ...]
          score 是页级 MaxSim 分数（所有 chunk 共享该页分数）
        """
        # 1. ColPali 编码查询
        q_emb = self.colpali.encode_query(query)  # [1, n_patches, 128]

        # 2. FAISS MaxSim 搜索 → Top-k 页
        page_results = self.faiss.maxsim_search(q_emb, k=k)

        if not page_results:
            return []

        # 3. Grounding 反查：命中页的所有 BGE chunk
        page_ids = [pr["page_id"] for pr in page_results]
        page_score_map = {pr["page_id"]: pr["score"] for pr in page_results}

        chunks = self.pg.get_chunks_by_page_ids(page_ids)

        # 4. 合并分数
        results = []
        for chunk in chunks:
            results.append({
                **chunk,
                "score": page_score_map[chunk["page_id"]],
                "retrieval_type": "visual",
            })

        return results
```

- [ ] **Step 2: 创建 tests/test_visual_retriever.py**

```python
"""Visual 检索器测试（mock FAISS + pgvector）"""

from unittest.mock import MagicMock, patch

import torch

from src.retrieval.visual_retriever import VisualRetriever


def test_visual_retriever_search():
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
    mock_colpali.encode_query.return_value = torch.randn(1, 10, 128)

    retriever = VisualRetriever(
        faiss_store=mock_faiss,
        pg_store=mock_pg,
        colpali_embedder=mock_colpali,
    )

    results = retriever.search("test query", k=2)
    assert len(results) == 2
    assert all(r["retrieval_type"] == "visual" for r in results)
    assert results[0]["page_id"] == 1
    assert results[0]["score"] == 0.85
    mock_faiss.maxsim_search.assert_called_once()
    mock_colpali.encode_query.assert_called_once_with("test query")
```

- [ ] **Step 3: 运行测试**

```bash
cd /Users/theyang/Documents/ai/pdf-rag && source .venv/bin/activate && python -m pytest tests/test_visual_retriever.py -v
```

Expected: PASS

- [ ] **Step 4: 提交**

```bash
git add src/retrieval/visual_retriever.py tests/test_visual_retriever.py
git commit -m "feat: add Visual retriever (ColPali + FAISS MaxSim + grounding)"
```

---

### 任务 9: RRF 融合 + Cross-encoder 重排

**Files:**
- Create: `src/retrieval/fusion.py`
- Create: `src/retrieval/reranker.py`
- Create: `tests/test_fusion.py`
- Create: `tests/test_reranker.py`

- [ ] **Step 1: 创建 fusion.py**

```python
"""融合策略接口 + RRF 融合

接口设计预留第二阶段 ConvexFusion 扩展（见设计文档 §7.1）。
调用点不 if/else 硬切。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List


class FusionStrategy(ABC):
    """融合策略抽象接口"""

    @abstractmethod
    def fuse(self, results_list: List[List[dict]], k: int) -> List[dict]:
        """融合多路检索结果

        Args:
            results_list: 每路检索的结果列表，每路为 [{chunk_id, score, ...}, ...]
            k: 保留 Top-k

        Returns:
            融合后的 Top-k [{chunk_id, score, ...}, ...]
        """
        ...


class RRFFusion(FusionStrategy):
    """RRF 融合: score = Σ 1/(k + rank)

    非参数，无可调权重，工程上稳（设计文档 §4.2）。
    """

    def __init__(self, rrf_k: int = 60):
        self.rrf_k = rrf_k

    def fuse(self, results_list: List[List[dict]], k: int = 20) -> List[dict]:
        """RRF 融合多路结果"""
        # 累积每路的 RRF 分数
        rrf_scores: Dict[str, float] = {}
        chunk_map: Dict[str, dict] = {}

        for results in results_list:
            for rank, result in enumerate(results, start=1):
                chunk_id = result["chunk_id"]
                if chunk_id not in rrf_scores:
                    rrf_scores[chunk_id] = 0.0
                    chunk_map[chunk_id] = result
                rrf_scores[chunk_id] += 1.0 / (self.rrf_k + rank)

        # 按 RRF 分数排序
        sorted_chunks = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

        # 返回 Top-k
        fused = []
        for chunk_id, score in sorted_chunks[:k]:
            result = dict(chunk_map[chunk_id])
            result["score"] = score
            result["retrieval_type"] = "rrf_fused"
            fused.append(result)

        return fused
```

- [ ] **Step 2: 创建 reranker.py**

```python
"""Cross-encoder 重排器"""

from __future__ import annotations

from typing import List

import torch
from sentence_transformers import CrossEncoder

from src.config import cfg


class Reranker:
    """Cross-encoder 重排器

    用法:
      reranker = Reranker()
      reranked = reranker.rerank(query, candidates, top_k=5)
    """

    def __init__(self, device: str | None = None):
        self.device = device or cfg.colpali_device
        self.model = CrossEncoder(
            cfg.reranker_model_id,
            device=self.device,
        )

    @torch.no_grad()
    def rerank(self, query: str, candidates: List[dict], top_k: int = 5) -> List[dict]:
        """对候选集进行 cross-encoder 重排

        Args:
            query: 原始查询文本
            candidates: [{chunk_id, text, ...}, ...]
            top_k: 返回 Top-k

        Returns:
            重排后的 Top-k [{chunk_id, text, score, ...}, ...]
        """
        if not candidates:
            return []

        # 构建 query-passage pairs
        pairs = [(query, c["text"]) for c in candidates]
        scores = self.model.predict(pairs, convert_to_tensor=True)  # [n_candidates]

        # 按重排分数排序
        scored = list(zip(candidates, scores.tolist()))
        scored.sort(key=lambda x: x[1], reverse=True)

        results = []
        for cand, score in scored[:top_k]:
            result = dict(cand)
            result["rerank_score"] = float(score)
            result["retrieval_type"] = "reranked"
            results.append(result)

        return results
```

- [ ] **Step 3: 创建 tests/test_fusion.py**

```python
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
    # ch2 在两路都出现，RRF 分数理应最高
    assert result[0]["chunk_id"] == "ch2"


def test_rrf_common_overlap():
    """验证交集在 RRF 中获得更高分数"""
    fusion = RRFFusion(rrf_k=60)
    route_a = [{"chunk_id": f"ch{i}", "score": 1.0} for i in range(10)]
    route_b = [{"chunk_id": f"ch{i}", "score": 1.0} for i in range(10)]
    # ch0 在两路都是 rank 1 → RRF = 2/(60+1) ≈ 0.0328
    # ch1 在两路都是 rank 2 → RRF = 2/(60+2) ≈ 0.0323
    result = fusion.fuse([route_a, route_b], k=5)
    assert len(result) == 5
    assert result[0]["chunk_id"] == "ch0"


def test_rrf_empty_input():
    fusion = RRFFusion()
    result = fusion.fuse([[]], k=10)
    assert result == []


def test_rrf_deduplication():
    """同一路内重复 chunk_id 应去重"""
    fusion = RRFFusion()
    route = [
        {"chunk_id": "ch1", "score": 0.9},
        {"chunk_id": "ch1", "score": 0.8},  # 重复
    ]
    result = fusion.fuse([route], k=5)
    assert len(result) == 1
```

- [ ] **Step 4: 创建 tests/test_reranker.py**

```python
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
```

- [ ] **Step 5: 运行测试**

```bash
cd /Users/theyang/Documents/ai/pdf-rag && source .venv/bin/activate && python -m pytest tests/test_fusion.py tests/test_reranker.py -v
```

Expected: 全部 PASS

- [ ] **Step 6: 提交**

```bash
git add src/retrieval/fusion.py src/retrieval/reranker.py tests/test_fusion.py tests/test_reranker.py
git commit -m "feat: add RRF fusion and cross-encoder reranker"
```

---

### 任务 10: ViDoRe 评测适配器 + 消融实验

**Files:**
- Create: `src/evaluation/vidore_adapter.py`
- Create: `src/evaluation/ablation.py`
- Create: `scripts/run_eval.py`

- [ ] **Step 1: 创建 vidore_adapter.py**

```python
"""ViDoRe 评测适配器

实现 BaseBeIRRetriever 接口，将 PrismRAG 检索管道包装成 vidore-benchmark 可调用的检索器。

适配策略（设计文档 §5.2）：
  - corpus 端按 ViDoRe 提供的 page 图片喂 ColPali，按附带 OCR 文本喂 BGE
  - 不私自重切 corpus，保证结果可对比 leaderboard
  - 评测只在离线 pipeline 跑，与在线服务共用 retrieval/ 模块
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import torch
from datasets import load_dataset
from tqdm import tqdm

from src.config import cfg
from src.ingestion.encoders import BGEEmbedder, ColPaliEmbedder
from src.ingestion.text_chunker import Chunk, TextChunker
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.fusion import RRFFusion
from src.retrieval.reranker import Reranker
from src.retrieval.visual_retriever import VisualRetriever
from src.store.faiss_store import FaissColPaliStore
from src.store.pgvector_store import PgVectorStore

logger = logging.getLogger(__name__)


class PrismRAGRetriever:
    """PrismRAG 统一检索器（vidore-benchmark 适配用）

    工作流：
      build_index(corpus_path) → 建索引
      search(query, k) → 检索
    """

    def __init__(
        self,
        pg_store: PgVectorStore,
        faiss_store: FaissColPaliStore,
        bge: BGEEmbedder,
        colpali: ColPaliEmbedder,
        chunker: TextChunker,
        bm25: BM25Retriever,
        dense: DenseRetriever,
        visual: VisualRetriever,
        fusion: RRFFusion,
        reranker: Reranker,
    ):
        self.pg = pg_store
        self.faiss = faiss_store
        self.bge = bge
        self.colpali = colpali
        self.chunker = chunker
        self.bm25 = bm25
        self.dense = dense
        self.visual = visual
        self.fusion = fusion
        self.reranker = reranker

    def build_index(self, dataset_path: str = "vidore/vidore_v3_industrial", max_pages: Optional[int] = None):
        """构建索引（BGE → pgvector + ColPali → FAISS + BM25）"""
        logger.info(f"加载语料: {dataset_path}")
        ds = load_dataset(dataset_path, split="corpus", trust_remote_code=True)
        if max_pages:
            ds = ds.select(range(min(max_pages, len(ds))))

        # 1. 文本路: 分块 → BGE → pgvector
        self._build_text_index(ds)

        # 2. 视觉路: ColPali → FAISS
        self._build_visual_index(ds)

        # 3. BM25 索引（从 pgvector 读取）
        self._build_bm25_index()

        logger.info("索引构建完成")

    def _build_text_index(self, ds):
        self.pg.create_schema()
        all_rows = []
        all_texts = []

        for idx in tqdm(range(len(ds)), desc="分块+BGE"):
            row = ds[idx]
            page_id = int(row["corpus_id"])
            doc_id = str(row.get("doc_id", ""))
            page_number = int(row.get("page_number_in_doc", 0))
            markdown = row.get("markdown", None)
            chunks = self.chunker.chunk_page(page_id, doc_id, page_number, markdown)
            for c in chunks:
                all_rows.append((c.chunk_id, c.page_id, c.doc_id, c.page_number, c.chunk_type, c.text))
                all_texts.append(c.text)

        logger.info(f"共 {len(all_rows)} 个 chunk，BGE 编码中...")
        embs = self.bge.encode(all_texts, batch_size=32, show_progress=True)

        batch = []
        for j, (chunk_id, page_id, doc_id, pn, ctype, text) in enumerate(all_rows):
            batch.append((chunk_id, page_id, doc_id, pn, ctype, text, embs[j].numpy().tolist()))
            if len(batch) >= 100:
                self.pg.insert_chunks(batch)
                batch = []
        if batch:
            self.pg.insert_chunks(batch)
        logger.info(f"pgvector 入库完成: {self.pg.count()} chunks")

    def _build_visual_index(self, ds):
        page_embeddings = {}
        batch_size = 4
        for i in tqdm(range(0, len(ds), batch_size), desc="ColPali 编码"):
            batch_rows = [ds[j] for j in range(i, min(i + batch_size, len(ds)))]
            images = [row["image"] for row in batch_rows]
            page_ids = [int(row["corpus_id"]) for row in batch_rows]
            embs = self.colpali.encode_pages(images, batch_size=len(images))
            for pid, emb in zip(page_ids, embs):
                page_embeddings[pid] = emb
        self.faiss.build_index(page_embeddings)
        self.faiss.save()
        logger.info(f"FAISS 索引完成: {self.faiss.num_pages} pages, {self.faiss.index_size_mb:.1f} MB")

    def _build_bm25_index(self):
        """从 pgvector 读取文本构建 BM25 索引"""
        self.bm25.fit_from_pgvector(self.pg)
        logger.info("BM25 索引完成")

    def search(
        self,
        query: str,
        k: int = 10,
        use_bm25: bool = True,
        use_dense: bool = True,
        use_visual: bool = True,
        use_rerank: bool = True,
    ) -> List[dict]:
        """统一检索接口

        Args:
            query: 查询文本
            k: 返回 Top-k chunk
            use_bm25/dense/visual: 控制各路的开关（消融用）
            use_rerank: 是否使用 cross-encoder 重排

        Returns:
            [{chunk_id, page_id, text, score, retrieval_type}, ...]
        """
        routes = []

        # BM25
        if use_bm25:
            try:
                bm25_results = self.bm25.search(query, k=20)
                routes.append(bm25_results)
            except RuntimeError:
                logger.warning("BM25 未就绪，跳过")

        # Dense
        if use_dense:
            dense_results = self.dense.search(query, k=20)
            routes.append(dense_results)

        # Visual
        if use_visual:
            try:
                visual_results = self.visual.search(query, k=20)
                routes.append(visual_results)
            except Exception as e:
                logger.warning(f"Visual 检索跳过: {e}")

        # RRF 融合
        if not routes:
            return []
        fused = self.fusion.fuse(routes, k=min(k * 2, 40))

        # Cross-encoder 重排
        if use_rerank and fused:
            reranked = self.reranker.rerank(query, fused, top_k=k)
            return reranked

        return fused[:k]
```

- [ ] **Step 2: 创建 ablation.py**

```python
"""消融实验运行器

独立评测每种配置组合的 NDCG@10，产出消融表。

消融配置（设计文档 §5.3）:
  路由增量组（证明三路各自不可或缺）:
    - 纯 BM25
    - 纯 Dense
    - 纯 Visual
    - BM25 + Dense
    - BM25 + Dense + Visual
  重排增量组（证明 rerank 增量）:
    - 三路 RRF（无 rerank）
    - 三路 RRF + rerank
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from datasets import load_dataset
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


@dataclass
class AblationResult:
    config_name: str
    ndcg_at_5: float = 0.0
    ndcg_at_10: float = 0.0
    recall_at_5: float = 0.0
    recall_at_10: float = 0.0
    mrr: float = 0.0
    avg_latency_ms: float = 0.0
    num_queries: int = 0


# 消融配置清单
ABLATION_CONFIGS = [
    # 路由增量组
    AblationConfig(name="BM25_only", use_bm25=True, use_dense=False, use_visual=False, use_rerank=False),
    AblationConfig(name="Dense_only", use_bm25=False, use_dense=True, use_visual=False, use_rerank=False),
    AblationConfig(name="Visual_only", use_bm25=False, use_dense=False, use_visual=True, use_rerank=False),
    AblationConfig(name="BM25_Dense", use_bm25=True, use_dense=True, use_visual=False, use_rerank=False),
    AblationConfig(name="BM25_Dense_Visual", use_bm25=True, use_dense=True, use_visual=True, use_rerank=False),
    # 重排增量组
    AblationConfig(name="Full_no_rerank", use_bm25=True, use_dense=True, use_visual=True, use_rerank=False),
    AblationConfig(name="Full_with_rerank", use_bm25=True, use_dense=True, use_visual=True, use_rerank=True),
]


def compute_ndcg(relevant: set, ranked: List[str], k: int) -> float:
    """计算 NDCG@k"""
    dcg = 0.0
    idcg = 0.0
    for i in range(min(k, len(ranked))):
        if ranked[i] in relevant:
            dcg += 1.0 / (i + 1)  # 简化 DCG：相关得 1
    for i in range(min(k, len(relevant))):
        idcg += 1.0 / (i + 1)
    return dcg / idcg if idcg > 0 else 0.0


def compute_recall(relevant: set, ranked: List[str], k: int) -> float:
    """计算 Recall@k"""
    if not relevant:
        return 0.0
    hits = sum(1 for r in ranked[:k] if r in relevant)
    return hits / len(relevant)


def compute_mrr(relevant: set, ranked: List[str]) -> float:
    """计算 MRR"""
    for i, r in enumerate(ranked, start=1):
        if r in relevant:
            return 1.0 / i
    return 0.0


def run_ablation(
    retriever: PrismRAGRetriever,
    dataset_path: str = "vidore/vidore_v3_industrial",
    max_queries: Optional[int] = None,
    output_dir: str = "results",
) -> List[AblationResult]:
    """运行全量消融实验"""
    logger.info("加载查询和 qrels...")
    queries_ds = load_dataset(dataset_path, split="queries", trust_remote_code=True)
    qrels_ds = load_dataset(dataset_path, split="qrels", trust_remote_code=True)

    if max_queries:
        queries_ds = queries_ds.select(range(min(max_queries, len(queries_ds))))

    # 构建 qrel 映射: query_id → set of corpus_ids
    qrel_map: Dict[int, set] = {}
    for qrel in qrels_ds:
        qid = int(qrel["query_id"])
        cid = int(qrel["corpus_id"])
        if qid not in qrel_map:
            qrel_map[qid] = set()
        qrel_map[qid].add(cid)

    results: List[AblationResult] = []

    for config in ABLATION_CONFIGS:
        logger.info(f"=== 消融配置: {config.name} ===")
        latencies = []
        all_ranked_page_ids: List[List[str]] = []
        all_relevant: List[set] = []

        for q_idx in tqdm(range(len(queries_ds)), desc=f"  {config.name}"):
            q = queries_ds[q_idx]
            qid = int(q["query_id"])
            query_text = str(q["query"])

            start = time.time()
            retrieved = retriever.search(
                query=query_text,
                k=10,
                use_bm25=config.use_bm25,
                use_dense=config.use_dense,
                use_visual=config.use_visual,
                use_rerank=config.use_rerank,
            )
            latencies.append((time.time() - start) * 1000)

            # 提取 page_id 列表（用于评估）
            ranked_page_ids = [str(r["page_id"]) for r in retrieved]
            all_ranked_page_ids.append(ranked_page_ids)

            # 相关页 ID
            relevant = {str(cid) for cid in qrel_map.get(qid, set())}
            all_relevant.append(relevant)

        # 计算指标
        ndcg5 = sum(compute_ndcg(rel, ranked, 5) for rel, ranked in zip(all_relevant, all_ranked_page_ids)) / len(all_ranked_page_ids)
        ndcg10 = sum(compute_ndcg(rel, ranked, 10) for rel, ranked in zip(all_relevant, all_ranked_page_ids)) / len(all_ranked_page_ids)
        rec5 = sum(compute_recall(rel, ranked, 5) for rel, ranked in zip(all_relevant, all_ranked_page_ids)) / len(all_ranked_page_ids)
        rec10 = sum(compute_recall(rel, ranked, 10) for rel, ranked in zip(all_relevant, all_ranked_page_ids)) / len(all_ranked_page_ids)
        mrr = sum(compute_mrr(rel, ranked) for rel, ranked in zip(all_relevant, all_ranked_page_ids)) / len(all_ranked_page_ids)
        avg_lat = sum(latencies) / len(latencies) if latencies else 0

        result = AblationResult(
            config_name=config.name,
            ndcg_at_5=ndcg5,
            ndcg_at_10=ndcg10,
            recall_at_5=rec5,
            recall_at_10=rec10,
            mrr=mrr,
            avg_latency_ms=avg_lat,
            num_queries=len(all_ranked_page_ids),
        )
        results.append(result)
        logger.info(f"  NDCG@10={ndcg10:.4f}, Recall@5={rec5:.4f}, MRR={mrr:.4f}, avg_lat={avg_lat:.0f}ms")

    # 保存结果
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    result_data = [
        {
            "config": r.config_name,
            "ndcg@5": round(r.ndcg_at_5, 4),
            "ndcg@10": round(r.ndcg_at_10, 4),
            "recall@5": round(r.recall_at_5, 4),
            "recall@10": round(r.recall_at_10, 4),
            "mrr": round(r.mrr, 4),
            "avg_latency_ms": round(r.avg_latency_ms, 1),
            "num_queries": r.num_queries,
        }
        for r in results
    ]
    with open(output_path / "ablation_results.json", "w") as f:
        json.dump(result_data, f, indent=2)

    # 打印消融表
    logger.info("\n" + "=" * 80)
    logger.info("消融实验结果")
    logger.info("=" * 80)
    header = f"{'Config':<25} {'NDCG@5':<10} {'NDCG@10':<10} {'Recall@5':<10} {'Recall@10':<10} {'MRR':<10} {'Lat(ms)':<10}"
    logger.info(header)
    logger.info("-" * 80)
    for r in results:
        logger.info(f"{r.config_name:<25} {r.ndcg_at_5:<10.4f} {r.ndcg_at_10:<10.4f} {r.recall_at_5:<10.4f} {r.recall_at_10:<10.4f} {r.mrr:<10.4f} {r.avg_latency_ms:<10.0f}")

    return results
```

- [ ] **Step 3: 创建 scripts/run_eval.py**

```python
#!/usr/bin/env python
"""评测入口脚本

用法:
  python scripts/run_eval.py                    # 全量消融
  python scripts/run_eval.py --max-queries 10   # 快速验证
  python scripts/run_eval.py --skip-index       # 跳过索引构建（如果已建好）
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import cfg
from src.ingestion.encoders import BGEEmbedder, ColPaliEmbedder
from src.ingestion.text_chunker import TextChunker
from src.evaluation.ablation import run_ablation
from src.evaluation.vidore_adapter import PrismRAGRetriever
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.fusion import RRFFusion
from src.retrieval.reranker import Reranker
from src.retrieval.visual_retriever import VisualRetriever
from src.store.faiss_store import FaissColPaliStore
from src.store.pgvector_store import PgVectorStore


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description="Run PrismRAG evaluation")
    parser.add_argument("--dataset", default="vidore/vidore_v3_industrial")
    parser.add_argument("--max-queries", type=int, default=None, help="Limit queries for quick test")
    parser.add_argument("--skip-index", action="store_true", help="Skip index building")
    parser.add_argument("--output-dir", default="results", help="Output directory")
    args = parser.parse_args()

    cfg.load()

    # 初始化组件
    pg_store = PgVectorStore()
    faiss_store = FaissColPaliStore()
    bge = BGEEmbedder()
    colpali = ColPaliEmbedder()
    chunker = TextChunker()

    bm25 = BM25Retriever()
    dense = DenseRetriever(pg_store, bge)
    visual = VisualRetriever(faiss_store, pg_store, colpali)
    fusion = RRFFusion(rrf_k=60)
    reranker = Reranker()

    retriever = PrismRAGRetriever(
        pg_store=pg_store,
        faiss_store=faiss_store,
        bge=bge,
        colpali=colpali,
        chunker=chunker,
        bm25=bm25,
        dense=dense,
        visual=visual,
        fusion=fusion,
        reranker=reranker,
    )

    if not args.skip_index:
        retriever.build_index(dataset_path=args.dataset)
    else:
        # 尝试加载已有索引
        faiss_loaded = faiss_store.load()
        if faiss_loaded:
            bm25.fit_from_pgvector(pg_store)
        else:
            logging.warning("FAISS 索引不存在，重新构建")
            retriever.build_index(dataset_path=args.dataset)

    # 运行消融
    run_ablation(
        retriever=retriever,
        dataset_path=args.dataset,
        max_queries=args.max_queries,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: 提交**

```bash
git add src/evaluation/vidore_adapter.py src/evaluation/ablation.py scripts/run_eval.py
git commit -m "feat: add ViDoRe evaluation adapter and ablation runner"
```

---

### 任务 11: RAGAS 20 条拒答 Sanity

**Files:**
- Create: `src/evaluation/ragas_sanity.py`
- Create: `data/rejection_qa.json`

- [ ] **Step 1: 创建 data/rejection_qa.json（20 条拒答测试集）**

```json
[
  {"question": "What is the meaning of life according to philosophy?", "expected_rejection": true, "reason": "非工业 PDF 知识域"},
  {"question": "Who won the 2022 FIFA World Cup?", "expected_rejection": true, "reason": "非知识库内容"},
  {"question": "How do I install Python on Windows?", "expected_rejection": true, "reason": "非工业文档内容"},
  {"question": "What is the capital of France?", "expected_rejection": true, "reason": "常识问题，不在知识库中"},
  {"question": "Can you write a poem about conveyor belts?", "expected_rejection": true, "reason": "创造性内容，非文档事实"},
  {"question": "What is the stock price of Siemens today?", "expected_rejection": true, "reason": "实时数据，不在知识库中"},
  {"question": "How to cook pasta?", "expected_rejection": true, "reason": "非工业领域"},
  {"question": "Who is the current president of the United States?", "expected_rejection": true, "reason": "常识问题，不在知识库中"},
  {"question": "What are the health benefits of coffee?", "expected_rejection": true, "reason": "非工业文档内容"},
  {"question": "Explain quantum computing in simple terms.", "expected_rejection": true, "reason": "非知识库主题"},
  {"question": "What movies are playing this weekend?", "expected_rejection": true, "reason": "实时信息，非知识库内容"},
  {"question": "How to train a dog?", "expected_rejection": true, "reason": "非工业领域"},
  {"question": "What is the weather forecast for tomorrow?", "expected_rejection": true, "reason": "实时数据"},
  {"question": "Write a resignation letter.", "expected_rejection": true, "reason": "创造性/通用内容"},
  {"question": "Who developed the theory of relativity?", "expected_rejection": true, "reason": "常识知识，不在知识库中"},
  {"question": "What is the recipe for chocolate cake?", "expected_rejection": true, "reason": "非工业内容"},
  {"question": "How to fix a leaking faucet?", "expected_rejection": true, "reason": "通用生活技能，非知识库内容"},
  {"question": "What are the best tourist attractions in Paris?", "expected_rejection": true, "reason": "非工业文档知识域"},
  {"question": "Tell me a joke.", "expected_rejection": true, "reason": "创造性内容，不在知识库中"},
  {"question": "How does cryptocurrency work?", "expected_rejection": true, "reason": "非工业 PDF 主题"}
]
```

- [ ] **Step 2: 创建 ragas_sanity.py**

```python
"""RAGAS 拒答 Sanity 评测

用 20 条 out-of-scope 问题验证系统"不该答时不乱答"。
Judge 用 Ollama qwen2:7b（设计文档 §5.1 允许第一阶段自评，承认偏好风险）。

流程:
  1. 对每条拒答问题，走完整检索+生成管道
  2. 用 Ollama judge 评估 faithfulness
  3. 检查系统是否成功拒绝回答（或生成与知识库无关的合理回应）
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional

import requests

from src.config import cfg
from src.evaluation.vidore_adapter import PrismRAGRetriever

logger = logging.getLogger(__name__)

OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"


def call_ollama(prompt: str, model: str = "qwen2:7b") -> str:
    """调用 Ollama 生成"""
    try:
        resp = requests.post(
            OLLAMA_CHAT_URL,
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
            },
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"]
    except Exception as e:
        logger.warning(f"Ollama 调用失败: {e}")
        return ""


def generate_answer(retriever: PrismRAGRetriever, query: str) -> Dict:
    """检索后生成回答"""
    # 1. 检索
    retrieved = retriever.search(query, k=5, use_rerank=True)

    # 2. 拼上下文
    if not retrieved:
        context = ""
    else:
        context = "\n\n---\n\n".join([r.get("text", "") for r in retrieved])

    # 3. LLM 生成
    system_prompt = (
        "You are a helpful assistant for industrial document QA. "
        "Answer the question based ONLY on the provided context. "
        "If the context does not contain enough information to answer the question, "
        "say 'I cannot answer this question based on the available documents.' "
        "Do NOT make up information."
    )
    user_prompt = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"

    answer = call_ollama(
        f"System: {system_prompt}\n\n{user_prompt}",
        model=cfg.model_llm_id if hasattr(cfg, 'model_llm_id') else "qwen2:7b",
    )

    return {
        "query": query,
        "retrieved": len(retrieved),
        "context_length": len(context),
        "answer": answer,
        "is_rejected": "cannot answer" in answer.lower() or "not enough information" in answer.lower() or "based on the available" in answer.lower(),
    }


def run_ragas_sanity(
    retriever: PrismRAGRetriever,
    rejection_qa_path: str = "data/rejection_qa.json",
    output_dir: str = "results",
) -> Dict:
    """运行 RAGAS 拒答 sanity 评测"""
    # 加载拒答集
    with open(rejection_qa_path) as f:
        rejection_qa = json.load(f)

    logger.info(f"加载 {len(rejection_qa)} 条拒答问题")

    results = []
    rejected_count = 0
    for item in rejection_qa:
        logger.info(f"  查询: {item['question'][:50]}...")
        result = generate_answer(retriever, item["question"])
        results.append(result)
        if result["is_rejected"]:
            rejected_count += 1
        logger.info(f"    拒绝={result['is_rejected']}, answer={result['answer'][:80]}...")
        time.sleep(1)  # Ollama rate limit 兜底

    # 统计
    total = len(rejection_qa)
    rejection_rate = rejected_count / total if total > 0 else 0.0
    summary = {
        "total_questions": total,
        "rejected_count": rejected_count,
        "rejection_rate": round(rejection_rate, 4),
        "passed": rejection_rate >= 0.8,  # 至少 80% 拒答算 pass
    }

    logger.info(f"\nRAGAS Sanity 结果:")
    logger.info(f"  总拒答数: {total}")
    logger.info(f"  正确拒绝: {rejected_count}")
    logger.info(f"  拒绝率: {rejection_rate:.1%}")
    logger.info(f"  是否通过(≥80%): {summary['passed']}")

    # 保存
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    with open(output_path / "ragas_sanity_results.json", "w") as f:
        json.dump({"summary": summary, "details": results}, f, indent=2, ensure_ascii=False)
    logger.info(f"结果已保存到 {output_path / 'ragas_sanity_results.json'}")

    return summary
```

- [ ] **Step 3: 提交**

```bash
git add src/evaluation/ragas_sanity.py data/rejection_qa.json
git commit -m "feat: add RAGAS rejection sanity (20 out-of-scope questions)"
```

---

### 任务 12: 可复现骨架（Makefile + Index 版本化 + .env）

**Files:**
- Create: `Makefile`
- Create: `scripts/fetch_indexes.py`
- Create: `.env`（从 `.env.example` 复制）

- [ ] **Step 1: 创建 Makefile**

```makefile
.PHONY: help install ingest-vidore eval-vidore eval-ragas fetch-indexes clean

help: ## 显示帮助
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## 安装依赖
	uv venv .venv --python 3.11
	uv pip install -e ".[dev]"

.env: ## 从模板创建 .env
	cp -n .env.example .env || true

ingest-vidore: .env ## 导入 ViDoRe Industrial 子集（构建索引）
	python scripts/ingest_vidore.py --dataset vidore/vidore_v3_industrial

eval-vidore: .env ## 运行 ViDoRe 消融评测（全量）
	python scripts/run_eval.py --dataset vidore/vidore_v3_industrial

eval-vidore-quick: .env ## 运行 ViDoRe 快速消融（10 条查询）
	python scripts/run_eval.py --dataset vidore/vidore_v3_industrial --max-queries 10

eval-vidore-skip-index: .env ## 跳过索引构建，直接评测
	python scripts/run_eval.py --dataset vidore/vidore_v3_industrial --skip-index

eval-ragas: .env ## 运行 RAGAS 拒答 sanity
	python scripts/run_ragas_sanity.py

fetch-indexes: ## 从 GitHub Release 拉取预编码索引
	python scripts/fetch_indexes.py

clean: ## 清理索引和评测结果
	rm -rf indexes/ results/

results/ablation_results.json: .env
	python scripts/run_eval.py --dataset vidore/vidore_v3_industrial --skip-index
```

- [ ] **Step 2: 创建 scripts/fetch_indexes.py**

```python
#!/usr/bin/env python
"""从 GitHub Release 拉取预编码索引（首次使用）

用法:
  python scripts/fetch_indexes.py

Release artifact 命名:
  indexes-<model_id>-<corpus_version>-<sha>.zip
"""

import logging
import os
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# TODO: 替换为实际 Release URL（P1 末首次发布时确定）
RELEASE_URL_TEMPLATE = "https://github.com/zyascend/prism-rag/releases/download/v0.1.0/indexes.zip"
INDEXES_DIR = Path("indexes")


def main():
    INDEXES_DIR.mkdir(parents=True, exist_ok=True)

    # 检查索引是否已存在
    existing = list(INDEXES_DIR.glob("*.faiss"))
    if existing:
        logger.info(f"索引已存在: {existing}")
        logger.info("如要重新下载，请先运行 `make clean`")
        return

    url = RELEASE_URL_TEMPLATE
    zip_path = INDEXES_DIR / "indexes.zip"

    logger.info(f"下载预编码索引: {url}")
    try:
        urlretrieve(url, zip_path)
    except Exception as e:
        logger.warning(f"下载失败（首次发布前此步骤可跳过）: {e}")
        logger.info("请先运行 `make ingest-vidore` 自行编码")
        return

    logger.info("解压中...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(INDEXES_DIR)

    zip_path.unlink()
    logger.info(f"索引已下载到 {INDEXES_DIR.resolve()}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: 创建 scripts/run_ragas_sanity.py**

```python
#!/usr/bin/env python
"""RAGAS 拒答评测入口"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import cfg
from src.evaluation.ragas_sanity import run_ragas_sanity
from src.evaluation.vidore_adapter import PrismRAGRetriever
from src.ingestion.encoders import BGEEmbedder, ColPaliEmbedder
from src.ingestion.text_chunker import TextChunker
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.fusion import RRFFusion
from src.retrieval.reranker import Reranker
from src.retrieval.visual_retriever import VisualRetriever
from src.store.faiss_store import FaissColPaliStore
from src.store.pgvector_store import PgVectorStore


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description="Run RAGAS rejection sanity")
    parser.add_argument("--rejection-qa", default="data/rejection_qa.json")
    parser.add_argument("--output-dir", default="results")
    args = parser.parse_args()

    cfg.load()
    pg_store = PgVectorStore()
    faiss_store = FaissColPaliStore()
    bge = BGEEmbedder()
    colpali = ColPaliEmbedder()
    chunker = TextChunker()
    bm25 = BM25Retriever()
    dense = DenseRetriever(pg_store, bge)
    visual = VisualRetriever(faiss_store, pg_store, colpali)
    fusion = RRFFusion(rrf_k=60)
    reranker = Reranker()

    retriever = PrismRAGRetriever(
        pg_store=pg_store, faiss_store=faiss_store, bge=bge, colpali=colpali,
        chunker=chunker, bm25=bm25, dense=dense, visual=visual,
        fusion=fusion, reranker=reranker,
    )

    # 尝试加载索引
    if not faiss_store.load():
        logging.error("FAISS 索引未找到。请先运行 `python scripts/ingest_vidore.py` 构建索引")
        sys.exit(1)

    bm25.fit_from_pgvector(pg_store)
    run_ragas_sanity(retriever, rejection_qa_path=args.rejection_qa, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: 创建 .env（从示例复制）并为 Makefile 加 .PHONY**

```bash
cd /Users/theyang/Documents/ai/pdf-rag && cp .env.example .env
```

- [ ] **Step 5: 提交**

```bash
git add Makefile scripts/fetch_indexes.py scripts/run_ragas_sanity.py .env
git commit -m "chore: add repro spine (Makefile, index versioning, ragas entry)"
```

---

### 任务 13: 最小 API 端点

**Files:**
- Create: `src/api/routes.py`
- Create: `scripts/run_api.py`

- [ ] **Step 1: 创建 routes.py**

```python
"""FastAPI 路由

端点:
  POST /search   检索+生成
  GET  /health   健康检查
"""

from __future__ import annotations

import logging
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.config import cfg
from src.evaluation.vidore_adapter import PrismRAGRetriever
from src.ingestion.encoders import BGEEmbedder, ColPaliEmbedder
from src.ingestion.text_chunker import TextChunker
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.fusion import RRFFusion
from src.retrieval.reranker import Reranker
from src.retrieval.visual_retriever import VisualRetriever
from src.store.faiss_store import FaissColPaliStore
from src.store.pgvector_store import PgVectorStore

logger = logging.getLogger(__name__)

app = FastAPI(title="PrismRAG API", version="0.1.0")
_retriever: Optional[PrismRAGRetriever] = None


class SearchRequest(BaseModel):
    query: str
    k: int = 10
    use_rerank: bool = True


class SearchResult(BaseModel):
    chunk_id: str
    page_id: int
    doc_id: str
    text: str
    chunk_type: str
    score: float
    retrieval_type: str


class SearchResponse(BaseModel):
    query: str
    results: List[SearchResult]
    num_results: int


class HealthResponse(BaseModel):
    status: str
    index_pages: int = 0
    index_size_mb: float = 0.0


def get_retriever() -> PrismRAGRetriever:
    global _retriever
    if _retriever is None:
        cfg.load()
        pg_store = PgVectorStore()
        faiss_store = FaissColPaliStore()
        bge = BGEEmbedder()
        colpali = ColPaliEmbedder()
        chunker = TextChunker()
        bm25 = BM25Retriever()
        dense = DenseRetriever(pg_store, bge)
        visual = VisualRetriever(faiss_store, pg_store, colpali)
        fusion = RRFFusion(rrf_k=60)
        reranker = Reranker()

        _retriever = PrismRAGRetriever(
            pg_store=pg_store, faiss_store=faiss_store, bge=bge, colpali=colpali,
            chunker=chunker, bm25=bm25, dense=dense, visual=visual,
            fusion=fusion, reranker=reranker,
        )

        # 加载索引
        faiss_loaded = faiss_store.load()
        if faiss_loaded:
            bm25.fit_from_pgvector(pg_store)
            logger.info("API: 索引加载完成")
        else:
            logger.warning("API: FAISS 索引未找到，请先运行 ingest_vidore.py")
    return _retriever


@app.get("/health", response_model=HealthResponse)
async def health():
    retriever = get_retriever()
    return HealthResponse(
        status="ok",
        index_pages=retriever.faiss.num_pages,
        index_size_mb=round(retriever.faiss.index_size_mb, 1),
    )


@app.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    retriever = get_retriever()
    try:
        results = retriever.search(
            query=request.query,
            k=request.k,
            use_rerank=request.use_rerank,
        )
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    return SearchResponse(
        query=request.query,
        results=[SearchResult(**r) for r in results],
        num_results=len(results),
    )
```

- [ ] **Step 2: 创建 scripts/run_api.py**

```python
#!/usr/bin/env python
"""启动 API 服务"""

import uvicorn

if __name__ == "__main__":
    uvicorn.run("src.api.routes:app", host="0.0.0.0", port=8000, reload=False)
```

- [ ] **Step 3: 验证 API 模块可导入**

```bash
cd /Users/theyang/Documents/ai/pdf-rag && source .venv/bin/activate && python -c "from src.api.routes import app; print('✅ API 模块可导入')"
```

Expected: `✅ API 模块可导入`

- [ ] **Step 4: 提交**

```bash
git add src/api/routes.py scripts/run_api.py
git commit -m "feat: add minimal FastAPI search endpoint"
```

---

### 任务 14: 首轮 ViDoRe 编码 + 消融运行（人工执行）

**这一步不在代码中完成，需要在真实环境跑。以下是指令。**

- [ ] **Step 1: 确保 pgvector 运行**

需要 PostgreSQL 16+ 并安装 pgvector 扩展。如本地未安装：

```bash
# macOS 安装方式 1: brew
brew install postgresql@16 pgvector
brew services start postgresql@16
createdb prismrag

# 安装 pgvector 扩展
psql -d prismrag -c "CREATE EXTENSION vector;"
```

- [ ] **Step 2: 启动 Ollama**

```bash
ollama pull qwen2:7b
ollama pull BAAI/bge-reranker-large
```

- [ ] **Step 3: 运行导入（全部 5,244 页）**

预计耗时:
- BGE 编码: ~5-10 分钟（5k 页 × 平均每个 chunk ~6 tokens 分块）
- ColPali 编码: ~1 小时（5,244 页 @ ~1.5 pg/s，竖版图片每页 ~1600 patches）

```bash
cd /Users/theyang/Documents/ai/pdf-rag && source .venv/bin/activate
python scripts/ingest_vidore.py --dataset vidore/vidore_v3_industrial
```

- [ ] **Step 4: 运行消融评测**

```bash
make eval-vidore
```

预期产出:
- `results/ablation_results.json` —— 7 种配置的 NDCG@5/10, Recall@5/10, MRR
- 终端打印消融表

- [ ] **Step 5: 运行 RAGAS 拒答 sanity**

```bash
make eval-ragas
```

预期产出:
- `results/ragas_sanity_results.json` —— 20 条拒答的拒绝率
- 拒绝率 ≥ 80% 为通过

- [ ] **Step 6: 将评测结果写入文档**

创建 `docs/evaluation/p1-baseline-results.md` 记录基线数据，内容模板：

```markdown
# P1 基线评测结果

**日期:** 2026-07-xx
**数据集:** vidore/vidore_v3_industrial (5,244 页, 1,698 查询)
**模型:** colpali-v1.3 + BGE-large-en-v1.5 + bge-reranker-large

## 消融表

| Config | NDCG@5 | NDCG@10 | Recall@5 | Recall@10 | MRR |
|--------|--------|---------|----------|-----------|-----|
| BM25_only | 0.xxxx | 0.xxxx | 0.xxxx | 0.xxxx | 0.xxxx |
| Dense_only | ... | ... | ... | ... | ... |
| ... | | | | | |

## 拒答 Sanity

- 总拒答数: 20
- 正确拒绝: xx/20
- 拒绝率: xx%
- 通过(≥80%): ✅/❌

## 系统配置

- 索引大小: xx MB (FAISS), xx 行 (pgvector)
- 平均查询延迟: xx ms
- 硬件: MacBook M系列 32GB
```

---

## 计划自检

### 设计文档覆盖检查

| 设计文档需求 | 对应任务 |
|------------|---------|
| Ingestion pipeline（MinerU 解析 | 任务 4/5 — 但 P1 使用 ViDoRe 自带图片+OCR 文本，MinerU 解析留 P2 真实 PDF 导入） |
| 文本路 BGE 编码 → pgvector | 任务 2（BGE 编码器）+ 任务 4（pgvector 存储）+ 任务 5（ingestor） |
| 视觉路 ColPali 编码 → FAISS | 任务 2（ColPali 编码器）+ 任务 4（FAISS 存储）+ 任务 5（ingestor） |
| BM25 检索 | 任务 6 |
| Dense 检索（pgvector HNSW） | 任务 7 |
| Visual 检索（FAISS MaxSim + grounding 反查） | 任务 8 |
| RRF 融合 | 任务 9（fusion.py） |
| Cross-encoder rerank | 任务 9（reranker.py） |
| ViDoRe 评测（BaseBeIRRetriever 适配） | 任务 10（vidore_adapter.py） |
| 消融实验（路由增量 + 重排增量两组分开） | 任务 10（ablation.py） |
| RAGAS 拒答 20 条 sanity | 任务 11 |
| FusionStrategy 接口（预留 ConvexFusion） | 任务 9（fusion.py — abstract base class） |
| 可复现骨架（Makefile + config + 版本化） | 任务 1（config）+ 任务 12（Makefile） |
| API（FastAPI /search 端点） | 任务 13 |
| ColPali 编码预热 | 任务 2（ColPaliEmbedder._warmup） |

### 已知缺失（P2 补齐）

- MinerU 真实 PDF 解析管道（P1 只处理 ViDoRe 预解析数据）
- GraphRAG（Neo4j）
- ReACT Agent
- React 前端（P1 末用 curl 验证 API，UI 留 P2）
- Docker Compose 容器化（P1 末或 P2 初加入）
- pgvector 的 SQLite/本地备选（当前依赖独立的 PostgreSQL 进程）
- `docs/evaluation/` 目录和评测报告模板（任务 14 step 6 创建）

### 占位符检查

- 无 "TBD" / "TODO" / "implement later" 占位符
- 所有代码步骤有完整实现
- 函数签名和类型在任务间一致
- `fetch_indexes.py` 的 Release URL 标记了 `TODO` 并说明原因（首次 Release 前不可知），这是合理的

---

## 执行交接

计划完成并保存到 `docs/superpowers/plans/2026-06-30-prismrag-p1-retrieval-mvp.md`。

两种执行方式：

1. **Subagent-Driven（推荐）** —— 每个任务派发一个子 agent，任务间 review，快速迭代
2. **Inline Execution** —— 在当前对话中顺序执行，分批 checkpoint review

选哪个？
