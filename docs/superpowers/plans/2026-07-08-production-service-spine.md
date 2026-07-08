# PrismRAG 生产服务骨架（MVP 切片）Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让 PrismRAG 成为本地可跑、可点的生产服务骨架——真实 PDF 上传入库 + 三路检索 + LLM 生成带引用回链的问答，容器化一条命令起。

**Architecture:** 新增 `Parser` 抽象（MinerU 生产 / SimplePDFParser 本地兜底）、`PDFIngestor`（复用现有 chunker+encoders+stores）、`Generator`（OpenAI SDK，复用 compress_context）。FastAPI 加 `/ingest` + `/ask`，docker-compose 起 pgvector+api。本地 dev 用 `config/models.local-dev.yaml`（`use_visual:false` 免 ColPali）。

**Tech Stack:** FastAPI, OpenAI Python SDK, PyMuPDF(fitz), MinerU, pgvector(psycopg2), FAISS, Docker Compose。

## Global Constraints

- 生成 LLM 一律经 **OpenAI SDK**：`OpenAI(base_url=cfg.get("llm.base_url"), api_key=cfg.get("llm.api_key"))`，`model=cfg.get("llm.model","gpt-4o-mini")`。不用 Ollama。
- 本地 dev profile（`config/models.local-dev.yaml`）默认 `ingestion.parser: simple` + `retrieval.use_visual: false`，以免 MinerU/ColPali 下载。
- 端到端必须在本地 macOS 可跑：pgvector 走 `docker compose up db`；模型缺失时脚本**提示用户手动下载**，不自动下载（遵循 AGENTS.md 本地禁下大模型）。
- 本切片**不含** `make eval-vidore` / Repro Spine / GraphRAG / MinIO / CI（已与用户确认拿掉）。
- 复用而非重写：`TextChunker`、`BGEEmbedder`、`ColPaliEmbedder`、`PgVectorStore`、`FaissColPaliStore`、`PrismRAGRetriever.search()`、`compress_context()` 全部直接复用。
- 验证分层：`make test` 纯单元（无 PG/无模型，用 fixture + mock）；`make e2e-local` 需 pgvector 容器 + 模型。
- 分支：`feat/production-spine`（已建）。每个 task 独立 commit。

---

## File Structure

| 文件 | 责任 |
|------|------|
| `src/ingestion/parser.py` | 新增：`Page` dataclass + `Parser` ABC + `SimplePDFParser`(PyMuPDF) + `MinerUParser`(MinerU CLI) |
| `src/ingestion/pdf_ingestor.py` | 新增：`PDFIngestor.ingest(pdf_path, doc_id)` → 解析→分块→BGE→pgvector + ColPali→FAISS 增量 |
| `src/store/faiss_store.py` | 改：新增 `add_pages(page_embeddings)` 增量写入 |
| `src/store/pgvector_store.py` | 改：新增 `delete_by_doc_id(doc_id)` 失败清理 |
| `src/config.py` | 改：`load()` 支持 `CONFIG_PROFILE` 环境变量合并 `config/models.<profile>.yaml` |
| `config/models.local-dev.yaml` | 新增：本地 dev profile |
| `src/generation/generator.py` | 新增：`Generator`(OpenAI SDK) + `GenerationError` |
| `src/api/routes.py` | 改：新增 `/ingest`、`/ask` 及 pydantic 模型；`get_generator()` 缓存 |
| `src/api/errors.py` | 新增（可选）：`IngestError`；或直接用内置异常 |
| `tests/test_parser.py` | 新增：SimplePDFParser 解析样例 PDF |
| `tests/test_pdf_ingestor.py` | 新增：用 fake stores/encoders 验证入库逻辑（无 PG/模型） |
| `tests/test_faiss_add_pages.py` | 新增：add_pages 增量正确性 |
| `tests/test_generator.py` | 新增：mock OpenAI client 验证答案+引用结构 |
| `tests/test_api.py` | 新增：TestClient 测 422/503/正常（monkeypatch retriever+generator） |
| `tests/e2e_local.py` | 新增：真·端到端，需 pgvector 容器+模型，不可达时 skip |
| `tests/fixtures/sample.pdf` | 新增：1 页最小 PDF（ASCII 文本） |
| `Dockerfile` | 新增 |
| `docker-compose.yml` | 新增 |
| `Makefile` | 改：加 `db`/`up`/`e2e-local`/`ingest-pdf` |
| `scripts/ingest_pdf.py` | 新增：CLI 包装 PDFIngestor |
| `.env.example` | 改：加 LLM 变量 |

---

### Task 1: Parser 抽象（SimplePDFParser + MinerUParser）

**Files:**
- Create: `src/ingestion/parser.py`
- Create: `tests/fixtures/sample.pdf`
- Test: `tests/test_parser.py`

**Interfaces:**
- Produces: `Page(page_number:int, markdown:str, image:PIL.Image.Image)`、`Parser.parse(pdf_path:Path)->List[Page]`、`SimplePDFParser`、`MinerUParser`、`build_parser(name:str="simple"|"mineru")->Parser`

- [ ] **Step 1: 生成 fixture 样例 PDF（1 页，含一行可检索文本）**

```bash
python - <<'PY'
import fitz
doc = fitz.open()
page = doc.new_page()
page.insert_text((72, 72), "PrismRAG hydraulic pump maintenance interval is 500 hours.")
page.insert_text((72, 100), "Replace the filter every 250 hours of operation.")
doc.save("tests/fixtures/sample.pdf")
doc.close()
PY
```

- [ ] **Step 2: 写失败测试**

```python
# tests/test_parser.py
from pathlib import Path
from src.ingestion.parser import SimplePDFParser, build_parser

FIXTURE = Path(__file__).parent / "fixtures" / "sample.pdf"

def test_simple_parser_returns_pages():
    pages = SimplePDFParser().parse(FIXTURE)
    assert len(pages) == 1
    assert "hydraulic pump" in pages[0].markdown
    assert pages[0].image is not None
    assert pages[0].page_number == 1

def test_build_parser_default_simple():
    p = build_parser()
    assert isinstance(p, SimplePDFParser)
```

- [ ] **Step 3: 运行测试确认失败**

Run: `python -m pytest tests/test_parser.py -v`
Expected: FAIL（模块不存在）

- [ ] **Step 4: 写实现**

```python
# src/ingestion/parser.py
"""PDF 解析抽象：生产用 MinerU，本地兜底用 PyMuPDF"""
from __future__ import annotations
import io
import shutil
import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import List
from PIL import Image
import fitz  # PyMuPDF


@dataclass
class Page:
    page_number: int
    markdown: str
    image: Image.Image


class Parser(ABC):
    @abstractmethod
    def parse(self, pdf_path: Path) -> List[Page]:
        ...


class SimplePDFParser(Parser):
    """本地兜底：PyMuPDF 抽文本 + 渲染页面图。零外部依赖。"""

    def parse(self, pdf_path: Path) -> List[Page]:
        doc = fitz.open(pdf_path)
        pages: List[Page] = []
        try:
            for i, page in enumerate(doc):
                markdown = page.get_text("text") or ""
                pix = page.get_pixmap(dpi=150)
                image = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
                pages.append(Page(page_number=i + 1, markdown=markdown, image=image))
        finally:
            doc.close()
        return pages


class MinerUParser(Parser):
    """生产用：MinerU CLI 解析，质量最高。best-effort 逐页切分。"""

    def __init__(self, output_dir: Path | None = None):
        self.output_dir = output_dir or Path("data/mineru_output")

    def parse(self, pdf_path: Path) -> List[Page]:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if shutil.which("mineru") is None:
            raise RuntimeError("mineru CLI 未安装；本地 dev 请改用 SimplePDFParser（parser=simple）")
        subprocess.run(
            ["mineru", "-p", str(pdf_path), "-o", str(self.output_dir), "--device", "cpu"],
            check=True,
        )
        stem = pdf_path.stem
        base = self.output_dir / stem / stem
        md_path = base / f"{stem}.md"
        images_dir = base / "images"
        markdown = md_path.read_text() if md_path.exists() else ""
        # 按图片引用切分 markdown 为逐页片段，并与 images 目录按顺序配对
        import re
        parts = re.split(r"(!\[[^\]]*\]\(images/[^)]+\))", markdown)
        pages: List[Page] = []
        img_files = sorted(images_dir.glob("*.png")) if images_dir.exists() else []
        text_acc, idx = "", 0
        for part in parts:
            if re.match(r"!\[[^\]]*\]\(images/[^)]+\)", part or ""):
                image = Image.open(img_files[idx]) if idx < len(img_files) else Image.new("RGB", (1000, 1600), 255)
                pages.append(Page(page_number=idx + 1, markdown=text_acc.strip(), image=image))
                idx += 1
                text_acc = ""
            else:
                text_acc += part or ""
        if idx == 0:  # 未拆分出图片：整篇作为单页
            pages.append(Page(page_number=1, markdown=markdown.strip(),
                              image=Image.new("RGB", (1000, 1600), 255)))
        return pages


def build_parser(name: str | None = None) -> Parser:
    from src.config import cfg
    name = name or cfg.get("ingestion.parser", "mineru")
    if name == "simple":
        return SimplePDFParser()
    return MinerUParser()
```

- [ ] **Step 5: 运行测试确认通过**

Run: `python -m pytest tests/test_parser.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/ingestion/parser.py tests/test_parser.py tests/fixtures/sample.pdf
git commit -m "feat(ingestion): add Parser abstraction (SimplePDFParser + MinerUParser)"
```

---

### Task 2: PgVectorStore.delete_by_doc_id + FaissColPaliStore.add_pages

**Files:**
- Modify: `src/store/pgvector_store.py` (末尾追加 `delete_by_doc_id`)
- Modify: `src/store/faiss_store.py` (末尾追加 `add_pages`)
- Test: `tests/test_faiss_add_pages.py`

**Interfaces:**
- Produces: `PgVectorStore.delete_by_doc_id(doc_id:str)->int`（返回删除行数）、`FaissColPaliStore.add_pages(page_embeddings:Dict[int,torch.Tensor])->None`（增量；首次调用等价于 build）

- [ ] **Step 1: 写失败测试（faiss add_pages 增量）**

```python
# tests/test_faiss_add_pages.py
import numpy as np
import torch
from src.store.faiss_store import FaissColPaliStore


def _emb(n_patches=10):
    return torch.rand(n_patches, 128)


def test_add_pages_incremental():
    store = FaissColPaliStore(index_path="indexes/_test.faiss",
                              id_map_path="indexes/_test-ids.npy")
    store.add_pages({1: _emb(10), 2: _emb(12)})
    assert store.num_pages == 2
    assert store._vectors.shape[0] == 22
    store.add_pages({3: _emb(8)})
    assert store.num_pages == 3
    assert store._vectors.shape[0] == 30
    # MaxSim 仍可跑
    q = torch.rand(1, 5, 128)
    res = store.maxsim_search(q, k=3)
    assert len(res) == 3
```

- [ ] **Step 2: 运行确认失败**

Run: `python -m pytest tests/test_faiss_add_pages.py -v`
Expected: FAIL（add_pages 不存在）

- [ ] **Step 3: 实现 pgvector delete_by_doc_id（在 pgvector_store.py 末尾 `close()` 前追加）**

```python
    def delete_by_doc_id(self, doc_id: str) -> int:
        """删除某 doc_id 的全部 chunk，返回删除行数（失败清理用）"""
        with self.conn.cursor() as cur:
            cur.execute("DELETE FROM chunks WHERE doc_id = %s", (doc_id,))
            deleted = cur.rowcount
        self.conn.commit()
        return deleted
```

- [ ] **Step 4: 实现 faiss add_pages（在 faiss_store.py 末尾 `index_type` property 前追加）**

```python
    def add_pages(self, page_embeddings: Dict[int, torch.Tensor]):
        """增量写入多向量页面。首次调用（无索引）时等价于 build_index。"""
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
        self._num_pages = len(self._page_boundaries)
        self._num_patches = self._vectors.shape[0]
        if self._device.type == "cuda":
            self._vectors_torch = torch.from_numpy(self._vectors).to(self._device)
```

- [ ] **Step 5: 运行测试确认通过**

Run: `python -m pytest tests/test_faiss_add_pages.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/store/pgvector_store.py src/store/faiss_store.py tests/test_faiss_add_pages.py
git commit -m "feat(store): add pgvector delete_by_doc_id + faiss incremental add_pages"
```

---

### Task 3: PDFIngestor

**Files:**
- Create: `src/ingestion/pdf_ingestor.py`
- Test: `tests/test_pdf_ingestor.py`

**Interfaces:**
- Consumes: `build_parser()`（Task 1）、`TextChunker.chunk_page`、`BGEEmbedder.encode`、`ColPaliEmbedder.encode_pages`、`PgVectorStore.insert_chunks`+`create_schema`、`FaissColPaliStore.add_pages`+`save`
- Produces: `PDFIngestor(pg, faiss, bge, colpali, chunker, parser=None).ingest(pdf_path, doc_id=None)->dict{doc_id,num_pages,num_chunks}`

- [ ] **Step 1: 写失败测试（用 fake stores/encoders，无 PG/模型）**

```python
# tests/test_pdf_ingestor.py
from pathlib import Path
from src.ingestion.pdf_ingestor import PDFIngestor

FIXTURE = Path(__file__).parent / "fixtures" / "sample.pdf"


class _FakeChunker:
    def chunk_page(self, page_id, doc_id, page_number, markdown_text):
        text = (markdown_text or "").strip()
        if not text:
            return []
        return [type("C", (), {"chunk_id": f"pg{page_id:05d}_ch001",
                               "page_id": page_id, "doc_id": doc_id,
                               "page_number": page_number, "text": text,
                               "chunk_type": "text", "doc_ref": ""})()]


class _FakeBGE:
    def encode(self, texts, **_):
        import torch
        return torch.zeros((len(texts), 1024))


class _FakeColPali:
    def encode_pages(self, images, **_):
        import torch
        return [torch.rand(10, 128) for _ in images]


class _FakePG:
    def __init__(self): self.rows = []
    def create_schema(self): pass
    def insert_chunks(self, chunks): self.rows.extend(chunks)
    def count(self): return len(self.rows)


class _FakeFAISS:
    def __init__(self): self.added = 0
    def add_pages(self, embs): self.added += len(embs)
    def save(self): pass


def test_ingest_builds_chunks_and_index():
    ing = PDFIngestor(_FakePG(), _FakeFAISS(), _FakeBGE(), _FakeColPali(), _FakeChunker())
    res = ing.ingest(FIXTURE, doc_id="docX")
    assert res["doc_id"] == "docX"
    assert res["num_pages"] == 1
    assert res["num_chunks"] >= 1
```

- [ ] **Step 2: 运行确认失败**

Run: `python -m pytest tests/test_pdf_ingestor.py -v`
Expected: FAIL

- [ ] **Step 3: 实现**

```python
# src/ingestion/pdf_ingestor.py
"""真实 PDF 导入管道：Parser → 分块 → BGE→pgvector + ColPali→FAISS 增量"""
from __future__ import annotations
import logging
import random
from pathlib import Path
from typing import Dict, List, Optional

import torch
from src.config import cfg
from src.ingestion.parser import build_parser

logger = logging.getLogger(__name__)


class PDFIngestor:
    def __init__(self, pg, faiss, bge, colpali, chunker, parser=None):
        self.pg = pg
        self.faiss = faiss
        self.bge = bge
        self.colpali = colpali
        self.chunker = chunker
        self.parser = parser or build_parser()

    def ingest(self, pdf_path: Path, doc_id: Optional[str] = None) -> Dict:
        doc_id = doc_id or _rand_doc_id()
        use_visual = cfg.get("retrieval.use_visual", True)
        pages = self.parser.parse(Path(pdf_path))
        if not pages:
            raise RuntimeError("解析未产出任何页面")

        self.pg.create_schema()
        all_rows, all_texts = [], []
        page_images = []
        page_id_for_image = []
        for p in pages:
            page_id = _rand_page_id()
            chunks = self.chunker.chunk_page(
                page_id=page_id, doc_id=doc_id,
                page_number=p.page_number, markdown_text=p.markdown,
            )
            for c in chunks:
                all_rows.append((c.chunk_id, c.page_id, c.doc_id, c.page_number,
                                 c.chunk_type, c.text, None, c.doc_ref))
                all_texts.append(c.text)
            if use_visual:
                page_images.append(p.image)
                page_id_for_image.append(page_id)

        if all_texts:
            embs = self.bge.encode(all_texts, batch_size=32)
            for i in range(0, len(all_rows), 100):
                batch = all_rows[i:i + 100]
                vecs = embs[i:i + 100].cpu().numpy().tolist()
                self.pg.insert_chunks([
                    (r[0], r[1], r[2], r[3], r[4], r[5], v, r[7]) for r, v in zip(batch, vecs)
                ])

        if use_visual and page_images:
            page_embs = self.colpali.encode_pages(page_images)
            self.faiss.add_pages({pid: e for pid, e in zip(page_id_for_image, page_embs)})
            self.faiss.save()

        return {"doc_id": doc_id, "num_pages": len(pages), "num_chunks": len(all_rows)}


def _rand_doc_id() -> str:
    return f"doc_{random.getrandbits(31):08x}"


def _rand_page_id() -> int:
    return random.getrandbits(31)
```

- [ ] **Step 4: 运行测试确认通过**

Run: `python -m pytest tests/test_pdf_ingestor.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/ingestion/pdf_ingestor.py tests/test_pdf_ingestor.py
git commit -m "feat(ingestion): add PDFIngestor for real PDF upload"
```

---

### Task 4: Generator（OpenAI SDK）

**Files:**
- Create: `src/generation/generator.py`
- Test: `tests/test_generator.py`

**Interfaces:**
- Consumes: `compress_context(query, chunks, bge_embedder, ratio)` from `src.evaluation.ragas_metrics`
- Produces: `Generator(client=None, bge_embedder=None)`、`Generator.answer(query, retrieved:List[dict], k_context=5)->dict{answer,citations,context}`、`GenerationError`

- [ ] **Step 1: 写失败测试（mock OpenAI client）**

```python
# tests/test_generator.py
from src.generation.generator import Generator, GenerationError


class _FakeCompletions:
    def create(self, **_):
        return type("R", (), {
            "choices": [type("C", (), {"message": type("M", (), {"content": "Answer here."})()})()]
        })()


class _FakeClient:
    def __init__(self): self.chat = type("Chat", (), {"completions": _FakeCompletions()})()


def _retrieved():
    return [
        {"chunk_id": "pg1_ch001", "page_id": 1, "doc_id": "d1",
         "page_number": 1, "text": "hydraulic pump interval 500 hours", "doc_ref": ""},
        {"chunk_id": "pg2_ch001", "page_id": 2, "doc_id": "d1",
         "page_number": 2, "text": "filter every 250 hours", "doc_ref": ""},
    ]


def test_answer_returns_citations_from_chunks():
    g = Generator(client=_FakeClient(), bge_embedder=None)
    out = g.answer("pump interval?", _retrieved(), k_context=5)
    assert out["answer"] == "Answer here."
    assert len(out["citations"]) == 2
    assert out["citations"][0]["chunk_id"] == "pg1_ch001"
    assert out["citations"][0]["page_id"] == 1


def test_empty_retrieval_honest_reject():
    g = Generator(client=_FakeClient(), bge_embedder=None)
    out = g.answer("x", [], k_context=5)
    assert out["answer"]
    assert out["citations"] == []
```

- [ ] **Step 2: 运行确认失败**

Run: `python -m pytest tests/test_generator.py -v`
Expected: FAIL

- [ ] **Step 3: 实现**

```python
# src/generation/generator.py
"""LLM 生成（OpenAI SDK）+ 引用回链。引用以检索 chunk 为准，不依赖模型自报。"""
from __future__ import annotations
import logging
import os
from typing import List, Optional

from src.config import cfg
from src.evaluation.ragas_metrics import compress_context

logger = logging.getLogger(__name__)


class GenerationError(RuntimeError):
    pass


class Generator:
    def __init__(self, client=None, bge_embedder=None):
        if client is None:
            from openai import OpenAI
            client = OpenAI(
                base_url=cfg.get("llm.base_url", "https://api.openai.com/v1"),
                api_key=cfg.get("llm.api_key", "") or os.environ.get("OPENAI_API_KEY", ""),
            )
        self.client = client
        self.model = cfg.get("llm.model", "gpt-4o-mini")
        self.bge = bge_embedder

    def answer(self, query: str, retrieved: List[dict], k_context: int = 5) -> dict:
        top = retrieved[:k_context]
        if not top:
            return {"answer": "I don't have enough information to answer that question.",
                    "citations": [], "context": ""}
        contexts = [r["text"] for r in top]
        if self.bge is not None:
            context = compress_context(
                query, contexts, self.bge,
                ratio=cfg.get("retrieval.context_compression_ratio", 0.4),
            )
        else:
            context = "\n\n".join(contexts)

        prompt = [
            {"role": "system", "content":
             "You are a precise assistant. Answer ONLY from the provided context. "
             "If the context lacks the answer, say you don't know."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
        ]
        try:
            resp = self.client.chat.completions.create(
                model=self.model, messages=prompt, temperature=0.0,
            )
        except Exception as e:
            raise GenerationError(f"LLM call failed: {e}") from e

        answer_text = resp.choices[0].message.content
        citations = [
            {"chunk_id": r["chunk_id"], "page_id": r["page_id"],
             "doc_id": r.get("doc_id"), "page_number": r.get("page_number"),
             "snippet": (r.get("text") or "")[:200]}
            for r in top
        ]
        return {"answer": answer_text, "citations": citations, "context": context}
```

- [ ] **Step 4: 运行测试确认通过**

Run: `python -m pytest tests/test_generator.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/generation/generator.py tests/test_generator.py
git commit -m "feat(generation): add OpenAI-SDK Generator with citation extraction"
```

---

### Task 5: API 端点 /ingest + /ask

**Files:**
- Modify: `src/api/routes.py`（追加模型 + 两个端点 + `get_generator`）
- Test: `tests/test_api.py`

**Interfaces:**
- Consumes: `PDFIngestor`（Task 3）、`Generator`（Task 4）、`PrismRAGRetriever.search()`（已有）、`PgVectorStore.delete_by_doc_id`（Task 2）、`cfg.get("retrieval.use_visual")`
- Produces: `POST /ingest`（UploadFile→入库）、`POST /ask`、模块级 `_generator` 可被 `set_generator()` 覆盖（供测试）

- [ ] **Step 1: 写失败测试（monkeypatch retriever + generator）**

```python
# tests/test_api.py
import uuid
from fastapi.testclient import TestClient
from src.api import routes


def _fake_retriever():
    class R:
        pg = type("PG", (), {"delete_by_doc_id": lambda self, d: 0})()
        faiss = type("F", (), {"save": lambda self: None})()
        bge = None
        chunker = None
        bm25 = type("B", (), {"fit_from_pgvector": lambda self, pg: None})()
    return R()


def _fake_generator():
    class G:
        def answer(self, q, retrieved, k_context=5):
            return {"answer": "ok", "citations": [{"chunk_id": "c1", "page_id": 1,
                    "doc_id": "d", "page_number": 1, "snippet": "s"}], "context": ""}
    return G()


def test_ingest_rejects_non_pdf():
    routes.set_retriever(_fake_retriever())
    c = TestClient(routes.app)
    r = c.post("/ingest", files={"file": ("x.txt", b"hi", "text/plain")})
    assert r.status_code == 422


def test_ask_returns_answer_and_citations():
    routes.set_retriever(_fake_retriever())
    routes.set_generator(_fake_generator())
    c = TestClient(routes.app)
    r = c.post("/ask", json={"query": "pump interval?", "k": 5})
    assert r.status_code == 200
    body = r.json()
    assert body["answer"] == "ok"
    assert body["citations"][0]["chunk_id"] == "c1"
```

- [ ] **Step 2: 运行确认失败**

Run: `python -m pytest tests/test_api.py -v`
Expected: FAIL（端点/函数不存在）

- [ ] **Step 3: 实现（在 routes.py 顶部 import 区追加，并在文件末尾追加端点）**

在 import 区追加：
```python
import uuid
from fastapi import File, UploadFile
from src.ingestion.pdf_ingestor import PDFIngestor
from src.generation.generator import Generator, GenerationError
from src.generation.generator import GenerationError as _GE
```

在 `get_retriever()` 之后追加：
```python
_generator: Optional[Generator] = None


def get_generator(bge=None) -> Generator:
    global _generator
    if _generator is None:
        _generator = Generator(bge_embedder=bge)
    return _generator


def set_retriever(r):
    global _retriever
    _retriever = r


def set_generator(g):
    global _generator
    _generator = g
```

在文件末尾追加 pydantic 模型与端点：
```python
class IngestResponse(BaseModel):
    doc_id: str
    num_pages: int
    num_chunks: int


class Citation(BaseModel):
    chunk_id: str
    page_id: int
    doc_id: Optional[str] = None
    page_number: Optional[int] = None
    snippet: str


class AskRequest(BaseModel):
    query: str
    doc_id: Optional[str] = None
    k: int = 5
    use_rerank: bool = True


class AskResponse(BaseModel):
    query: str
    answer: str
    citations: List[Citation] = []
    retrieval_trace: RetrievalTrace = RetrievalTrace()


UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


@app.post("/ingest", response_model=IngestResponse)
async def ingest(file: UploadFile = File(...)):
    if not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(status_code=422, detail="仅支持 PDF 文件")
    doc_id = uuid.uuid4().hex[:12]
    pdf_path = UPLOAD_DIR / f"{doc_id}.pdf"
    pdf_path.write_bytes(await file.read())
    retriever = get_retriever()
    try:
        result = PDFIngestor(
            retriever.pg, retriever.faiss, retriever.bge,
            retriever.colpali, retriever.chunker,
        ).ingest(pdf_path, doc_id=doc_id)
    except Exception as e:
        retriever.pg.delete_by_doc_id(doc_id)
        logger.error(f"ingest failed: {e}")
        raise HTTPException(status_code=500, detail=f"入库失败: {e}")
    retriever.bm25.fit_from_pgvector(retriever.pg)
    if cfg.get("retrieval.use_visual", True):
        retriever.faiss.save()
    return IngestResponse(**result)


@app.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest):
    retriever = get_retriever()
    use_visual = cfg.get("retrieval.use_visual", True)
    try:
        results = retriever.search(
            request.query, k=request.k,
            use_visual=use_visual, use_rerank=request.use_rerank,
        )
    except Exception as e:
        logger.error(f"search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    if request.doc_id:
        results = [r for r in results if r.get("doc_id") == request.doc_id]
    if not results:
        return AskResponse(query=request.query,
                           answer="I don't have enough information to answer that question.",
                           citations=[])
    try:
        gen = get_generator(retriever.bge).answer(request.query, results, k_context=request.k)
    except GenerationError as e:
        raise HTTPException(status_code=502, detail=str(e))
    return AskResponse(
        query=request.query, answer=gen["answer"],
        citations=[Citation(**c) for c in gen["citations"]],
    )
```

- [ ] **Step 4: 运行测试确认通过**

Run: `python -m pytest tests/test_api.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/api/routes.py tests/test_api.py
git commit -m "feat(api): add /ingest and /ask endpoints with citation return"
```

---

### Task 6: 配置 profile + local-dev.yaml

**Files:**
- Modify: `src/config.py`（`load()` 末尾合并 profile）
- Create: `config/models.local-dev.yaml`
- Modify: `.env.example`

**Interfaces:**
- Produces: `CONFIG_PROFILE` 环境变量触发合并 `config/models.<profile>.yaml`；`cfg.get("ingestion.parser")` / `cfg.get("retrieval.use_visual")`

- [ ] **Step 1: 写失败测试**

```python
# tests/test_config_profile.py
import importlib, os
from src import config as config_mod


def test_local_dev_profile_merges():
    os.environ["CONFIG_PROFILE"] = "local-dev"
    config_mod.cfg._loaded = False
    config_mod.cfg._data = None
    config_mod.cfg.load()
    assert config_mod.cfg.get("ingestion.parser") == "simple"
    assert config_mod.cfg.get("retrieval.use_visual") is False
    del os.environ["CONFIG_PROFILE"]
    config_mod.cfg._loaded = False
    config_mod.cfg._data = None
```

- [ ] **Step 2: 运行确认失败**

Run: `python -m pytest tests/test_config_profile.py -v`
Expected: FAIL

- [ ] **Step 3: 实现（config.py load 末尾，return self 之前插入）**

```python
        # 合并 profile（如 local-dev）：CONFIG_PROFILE=local-dev -> config/models.local-dev.yaml
        profile = os.environ.get("CONFIG_PROFILE")
        if profile:
            profile_path = Path(__file__).parent.parent / "config" / f"models.{profile}.yaml"
            if profile_path.exists():
                with open(profile_path) as pf:
                    deep_merge(self._data, yaml.safe_load(pf) or {})
```

并在 config.py 顶部 `import os` 已存在；追加辅助函数：
```python
def deep_merge(base: dict, override: dict) -> None:
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_merge(base[k], v)
        else:
            base[k] = v
```

- [ ] **Step 4: 创建 local-dev.yaml**

```yaml
# config/models.local-dev.yaml
ingestion:
  parser: simple
retrieval:
  use_visual: false
  use_rerank: true
llm:
  base_url: https://api.openai.com/v1
  api_key: ${OPENAI_API_KEY}
  model: gpt-4o-mini
```

- [ ] **Step 5: 更新 .env.example（追加 LLM 段）**

```
LLM_BASE_URL=https://api.openai.com/v1
LLM_API_KEY=sk-...
LLM_MODEL=gpt-4o-mini
```

- [ ] **Step 6: 运行测试确认通过**

Run: `python -m pytest tests/test_config_profile.py -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add src/config.py config/models.local-dev.yaml .env.example tests/test_config_profile.py
git commit -m "feat(config): support CONFIG_PROFILE merge + local-dev profile"
```

---

### Task 7: 容器化 + Makefile + CLI

**Files:**
- Create: `Dockerfile`
- Create: `docker-compose.yml`
- Modify: `Makefile`
- Create: `scripts/ingest_pdf.py`

**Interfaces:**
- Produces: `docker compose up db`（本地 pgvector）、`make up`、`make e2e-local`、`make ingest-pdf PDF=path`、`python scripts/ingest_pdf.py --pdf <path>`

- [ ] **Step 1: 写 Dockerfile**

```dockerfile
FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 poppler-utils \
    && pip install --no-cache-dir uv
COPY pyproject.toml ./
RUN uv pip install --system -e ".[default]" || pip install --no-cache-dir -e ".[default]"
COPY . .
EXPOSE 8000
CMD ["uvicorn", "src.api.routes:app", "--host", "0.0.0.0", "--port", "8000"]
```

- [ ] **Step 2: 写 docker-compose.yml**

```yaml
services:
  db:
    image: pgvector/pgvector:pg16
    environment:
      POSTGRES_USER: ${PGVECTOR_USER:-prismrag}
      POSTGRES_PASSWORD: ${PGVECTOR_PASSWORD:-prismrag}
      POSTGRES_DB: ${PGVECTOR_DB:-prismrag}
    ports: ["5432:5432"]
    volumes: ["pgdata:/var/lib/postgresql/data"]
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U ${PGVECTOR_USER:-prismrag}"]
      interval: 5s
      timeout: 3s
      retries: 10
  api:
    build: .
    depends_on:
      db:
        condition: service_healthy
    environment:
      PGVECTOR_HOST: db
      PGVECTOR_PORT: 5432
      PGVECTOR_DB: ${PGVECTOR_DB:-prismrag}
      PGVECTOR_USER: ${PGVECTOR_USER:-prismrag}
      PGVECTOR_PASSWORD: ${PGVECTOR_PASSWORD:-prismrag}
      CONFIG_PROFILE: ${CONFIG_PROFILE:-local-dev}
      LLM_BASE_URL: ${LLM_BASE_URL}
      LLM_API_KEY: ${LLM_API_KEY}
      LLM_MODEL: ${LLM_MODEL:-gpt-4o-mini}
    ports: ["8000:8000"]
    volumes: ["screenshots:/app/data/screenshots"]
volumes:
  pgdata:
  screenshots:
```

- [ ] **Step 3: 更新 Makefile（在 `clean:` 前追加）**

```makefile
db: ## 起 pgvector 容器（本地 dev 用）
	docker compose up -d db

up: ## 全栈起服务
	docker compose up -d --build

e2e-local: ## 本地端到端（需 pgvector 容器 + 模型）
	pytest tests/e2e_local.py -v

ingest-pdf: ## 入库本地 PDF: make ingest-pdf PDF=path/to.pdf
	python scripts/ingest_pdf.py --pdf $(PDF)
```

- [ ] **Step 4: 写 scripts/ingest_pdf.py**

```python
"""CLI: 入库本地 PDF（容器内或本地，需 PG 可达 + 模型已下载）"""
import argparse
from pathlib import Path
from src.config import cfg
from src.evaluation.vidore_adapter import PrismRAGRetriever
from src.ingestion.encoders import BGEEmbedder, ColPaliEmbedder
from src.ingestion.pdf_ingestor import PDFIngestor
from src.ingestion.text_chunker import TextChunker
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.fusion import RRFFusion
from src.retrieval.reranker import Reranker
from src.retrieval.visual_retriever import VisualRetriever
from src.store.faiss_store import FaissColPaliStore
from src.store.pgvector_store import PgVectorStore


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", required=True)
    args = ap.parse_args()
    cfg.load()
    pg = PgVectorStore(); faiss = FaissColPaliStore()
    bge = BGEEmbedder(); colpali = ColPaliEmbedder()
    chunker = TextChunker()
    bm25 = BM25Retriever(); dense = DenseRetriever(pg, bge)
    visual = VisualRetriever(faiss, pg, colpali)
    fusion = RRFFusion(rrf_k=cfg.get("retrieval.rrf_k", 60))
    reranker = Reranker()
    retriever = PrismRAGRetriever(pg, faiss, bge, colpali, chunker,
                                  bm25, dense, visual, fusion, reranker)
    faiss.load()
    bm25.fit_from_pgvector(pg)
    res = PDFIngestor(pg, faiss, bge, colpali, chunker).ingest(Path(args.pdf))
    print(f"入库完成: {res}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: 语法/导入自检**

Run: `python -c "import ast; ast.parse(open('Dockerfile').read())" 2>/dev/null; python scripts/ingest_pdf.py --help`
Expected: 显示 argparse 帮助（无 import 崩溃说明依赖可解析；若本地缺 torch 等会报错，属正常，容器/云端才跑）

- [ ] **Step 6: Commit**

```bash
git add Dockerfile docker-compose.yml Makefile scripts/ingest_pdf.py
git commit -m "feat(infra): docker compose + Makefile targets + ingest_pdf CLI"
```

---

### Task 8: 本地端到端测试 tests/e2e_local.py

**Files:**
- Create: `tests/e2e_local.py`

**Interfaces:**
- Consumes: 全部前述组件；需 `DATABASE_URL` 可达（pgvector 容器）+ `OPENAI_API_KEY` 设置

- [ ] **Step 1: 写 e2e（不可达时 skip）**

```python
# tests/e2e_local.py
"""本地端到端：需 pgvector 容器 (make db) + OPENAI_API_KEY + BGE 模型。
不可达时自动 skip，避免 CI/纯单元环境误跑。"""
import os
import pytest
from pathlib import Path
from fastapi.testclient import TestClient
from src.config import cfg
from src.api import routes

FIXTURE = Path(__file__).parent / "fixtures" / "sample.pdf"


def _pg_reachable() -> bool:
    try:
        from src.store.pgvector_store import PgVectorStore
        PgVectorStore().conn  # 触发连接
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _pg_reachable(), reason="pgvector 不可达（先 make db）")
@pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY"), reason="需 OPENAI_API_KEY")
def test_ingest_and_ask_e2e():
    cfg.load()
    c = TestClient(routes.app)
    with open(FIXTURE, "rb") as f:
        r = c.post("/ingest", files={"file": ("sample.pdf", f, "application/pdf")})
    assert r.status_code == 200, r.text
    doc_id = r.json()["doc_id"]
    a = c.post("/ask", json={"query": "pump maintenance interval?", "doc_id": doc_id, "k": 5})
    assert a.status_code == 200, a.text
    body = a.json()
    assert body["answer"]
    assert any("pump" in (ci["snippet"] or "").lower() or ci["page_id"] for ci in body["citations"])
```

- [ ] **Step 2: 本地无 PG 时确认 skip**

Run: `python -m pytest tests/e2e_local.py -v`
Expected: SKIPPED（本地无 PG）

- [ ] **Step 3: Commit**

```bash
git add tests/e2e_local.py
git commit -m "test: add local e2e (skips when pgvector/OPENAI unavailable)"
```

---

### Task 9: 全量 lint + test 回归

**Files:**
- 无新增；运行 `make lint` + `make test`

- [ ] **Step 1: 运行 lint**

Run: `python -m ruff check src/ tests/`
Expected: 无错误（若有，修复后重跑）

- [ ] **Step 2: 运行全部单元测试**

Run: `python -m pytest tests/ -v --tb=short -k "not e2e_local"`
Expected: PASS（e2e_local 显式排除；它单独 skip 验证）

- [ ] **Step 3: Commit（若有 lint 修复）**

```bash
git add -A
git commit -m "chore: lint fixes for production-spine"  # 仅在确有修复时
```

---

## Self-Review

**1. Spec 覆盖**：§2.1 Parser✓(Task1) / PDFIngestor✓(Task3) / §2.2 Generator✓(Task4) / §2.3 /ingest+/ask✓(Task5) / §2.4 compose✓(Task7) / §2.6 profile✓(Task6) / §4 错误处理 delete_by_doc_id+502✓(Task2,Task5) / §5 测试分层✓(Task1-6 单元, Task8 e2e) / §7 验收点：本地 e2e✓、docker up✓、make test+lint✓。eval-vidore 已按用户要求移除，无对应 task。

**2. Placeholder 扫描**：无 TBD/TODO；每个 code step 均含完整实现或明确命令；测试均含真实断言。

**3. 类型一致性**：`PDFIngestor.ingest` 返回 dict 含 `doc_id/num_pages/num_chunks`，Task5 `IngestResponse(**result)` 对齐；`Generator.answer` 返回 `citations:[{chunk_id,page_id,doc_id,page_number,snippet}]`，Task5 `Citation(**c)` 字段对齐；`delete_by_doc_id`、`add_pages` 签名跨 Task2/Task3/Task5 一致。
