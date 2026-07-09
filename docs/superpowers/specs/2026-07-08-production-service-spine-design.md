# PrismRAG 生产服务骨架（MVP 切片）— 设计文档

> brainstorm 日期: 2026-07-08。
> 分支: 从 `chore/analyze-bottlehole` 切出 `feat/production-spine`（遵循 AGENTS.md 分支保护）。
> 目标: 把架构图 §1 ON 路径真正落地 + 路线图 §2 Repro Spine 最小面，作为"往实际生产贴近"的第一刀。
> 配套阅读: `handoff.md`、`docs/industrial-pdf-rag-architecture.md`、`docs/superpowers/specs/2026-06-30-prismrag-longterm-roadmap-design.md`。

---

## 0. 背景与决策（已确认）

| 维度 | 结论 |
|------|------|
| 用户诉求 | 不再刷 benchmark 分数，往"真实生产系统"贴近 |
| 选定方向 | 生产服务骨架（薄垂直切片，Approach C） |
| 落地形态 | 真实 PDF → 入库 → 问答 最小闭环 + Repro Spine 最小面 + 容器化 |
| 生成 LLM | **OpenAI SDK** 调用 LLM API（`gpt-4o-mini` / `DeepSeek` 经 `base_url`+`api_key` 配置化），不用本地 Ollama |
| 切片外（留下一轮） | GraphRAG/ReACT(P2)、多文档库多租户、MinIO 全量接入、CI/leaderboard 自动更新 |
| **本地可跑** | 端到端必须在本地 macOS M 系列可起可点（硬要求）。三道本地拦路虎逐一化解：无本地 PG→pgvector 容器；MinerU 重→轻量 Parser 兜底；ColPali/BGE 模型→一次性下载进 HF cache 后常驻，且本地 smoke 可关 visual 路免 ColPali |
| 验证分层 | `make test` 纯单元（无模型/无 PG，用 fixture）；`make e2e-local` 真·端到端（需 pgvector 容器 + 模型，首次需一次性下载，遵循 AGENTS.md 不主动触发） |

---

## 1. 范围

这一刀只做：

1. **真实 PDF 入库闭环**:上传 1 份 PDF → Parser（生产 MinerU / 本地 `SimplePDFParser` 兜底）解析 → 逐页 (markdown + 页面图) → 复用现有 `TextChunker`/`BGEEmbedder`/`ColPaliEmbedder`/`PgVectorStore`/`FaissColPaliStore` 入库。
2. **问答闭环 `/ask`**:检索(三路+RRF+rerank, Top-5) → `compress_context` 句级过滤 → `Generator`(OpenAI SDK) → 答案 + 引用回链 `{page_id, chunk_id, snippet}`。
3. **容器化**:`docker-compose` 一条命令起 `pgvector/postgres` + `api`(uvicorn)，截图存命名卷。

不做（明确边界，防 scope 蔓延）：GraphRAG、ReACT Agent、MinIO 全量、CI、leaderboard 自动更新、多租户、**Repro Spine / `make eval-vidore`（留到下一轮，本切片聚焦服务闭环）**。

---

## 2. 组件设计

### 2.1 `src/ingestion/pdf_ingestor.py`（新增）

```python
class PDFIngestor:
    def __init__(self, pg, faiss, bge, colpali, chunker): ...
    def ingest(self, pdf_path: Path, doc_id: str | None = None) -> IngestResult:
        # 1. doc_id = doc_id or uuid4().hex
        # 2. MinerU 解析 pdf_path -> 逐页 (markdown_text, page_image, page_number)
        # 3. 复用 TextChunker.chunk_page(page_id, doc_id, page_number, markdown_text)
        #    -> BGE encode -> pgvector (幂等 ON CONFLICT DO NOTHING)
        # 4. ColPali encode_pages([page_image]) -> FAISS 增量 add
        # 5. 返回 {doc_id, num_pages, num_chunks}
```

- **MinerU 调用**:走 `mineru` CLI（`mineru -p <pdf> -o <out>`）或 `mineru-api` Python 包；解析产物取每页 markdown + 页面截图。封装在 `src/ingestion/mineru_parser.py`，便于替换/降级（解析失败可降级纯文本提取）。
- **复用而非重写**:`TextChunker`、`BGEEmbedder`、`ColPaliEmbedder`、`PgVectorStore`、`FaissColPaliStore` 全部复用现有实现；`PDFIngestor` 只是把"HF dataset 逐页"换成"解析器逐页"的新入口。
- **Parser 抽象（本地可跑关键）**:新增 `src/ingestion/parser.py` 定义 `Parser` 接口 `parse(pdf_path) -> List[Page](markdown, image, page_number)`，两个实现：
  - `MinerUParser`：生产用，调用 MinerU CLI/包，质量最高。
  - `SimplePDFParser`：本地兜底，用 `pymupdf`(fitz) 提取每页文本 + 渲染页面图为 PIL Image。**零外部依赖、pip 即装**，本地开发/CI 不装 MinerU 也能跑通真实 PDF 入库。
  - `PDFIngestor` 按 `cfg.get("ingestion.parser", "mineru")` 选择；本地 dev profile 默认 `simple`。
- **增量入库**:文本路用 pgvector `ON CONFLICT DO NOTHING`(已支持)；FAISS 需新增 `FaissColPaliStore.add_pages(page_embs)` 增量 add（当前只有 `build_index` 全量，切片内补一个增量方法）。
- **失败清理**:MinerU 崩溃/加密 PDF → 抛 `IngestError`，API 层回 4xx 并清理该 `doc_id` 的半成品 chunk（pgvector 按 `doc_id` 删）。

### 2.2 `src/generation/generator.py`（新增）

```python
class Generator:
    def __init__(self, client=None):  # client: OpenAI SDK 实例, 配置化 base_url/api_key/model
        ...
    def answer(self, query: str, chunks: List[Chunk]) -> GenerationResult:
        # 1. context = compress_context(chunks, ratio=0.4)   # 复用现有句级过滤
        # 2. prompt = build_qa_prompt(query, context)        # 系统提示 + 引用要求
        # 3. resp = client.chat.completions.create(model=..., messages=[...])
        # 4. 解析答案 + 从 chunks 抽取引用 {page_id, chunk_id, snippet}
        # 5. 检索无结果 / context 为空 -> 诚实拒答
```

- **LLM 客户端**:用 `openai.OpenAI(base_url=cfg.get("llm.base_url"), api_key=cfg.get("llm.api_key"))`，`model=cfg.get("llm.model", "gpt-4o-mini")`。DeepSeek 只需改 `base_url`+`api_key`+`model`，代码不变。
- **引用回链**:prompt 要求模型标注引用来源；`Generator` 从传入 `chunks` 反向映射 `chunk_id`→`page_id`/`snippet`，返回结构化引用，不依赖模型自报的可靠性（以检索 chunk 为准）。
- **复用**:`compress_context`（来自 `src/evaluation/ragas_metrics.py:415`）抽成共享工具（`src/retrieval/context_compressor.py` 或保留原位由 generator import），避免 duplication。

### 2.3 `src/api/routes.py`（改）

新增两个端点，保留现有 `/search`、`/health`：

| 端点 | 方法 | 说明 |
|------|------|------|
| `POST /ingest` | `UploadFile` → 落盘 `data/uploads/<doc_id>.pdf` → `PDFIngestor.ingest` → 返回 `{doc_id, num_pages, num_chunks}` | 入库 |
| `POST /ask` | `{query, doc_id?, k=5, use_rerank=True}` → 检索 → `Generator.answer` → `{answer, citations:[{page_id,chunk_id,snippet}], retrieval_trace}` | 问答 |

- `doc_id` 可选：不传则跨全部已入库文档检索；传入则限定该文档（真实产品多文档库的前身）。
- 全局 `_retriever` 懒加载逻辑复用（已支持 FAISS + pgvector 加载）。

### 2.4 `Dockerfile` + `docker-compose.yml`（新增）

```yaml
services:
  db:
    image: pgvector/pgvector:pg16
    environment: {POSTGRES_USER, POSTGRES_PASSWORD, POSTGRES_DB}
    ports: ["5432:5432"]
    volumes: [pgdata:/var/lib/postgresql/data]
  api:
    build: .
    depends_on: [db]
    environment: [DATABASE_URL, LLM_BASE_URL, LLM_API_KEY, LLM_MODEL]
    ports: ["8000:8000"]
    volumes: [screenshots:/app/data/screenshots]   # 页面截图命名卷
```

- **`db` 即本地开发 backbone**：本机无 PG，`docker compose up db` 起一个 pgvector 容器，`DATABASE_URL` 指向 `localhost:5432`。本地开发时 api 可直接在宿主机 `uvicorn` 跑（不必须进容器），连这个 PG 容器即可——标准 dev 模式。
- 截图本轮回卷（不接 MinIO），`/ask` 引用里的 `page_id` 对应卷内截图路径，前端可静态访问（前端留待下一轮）。
- `Dockerfile` 基于 `python:3.11-slim`，装 `.[default]`，`CMD uvicorn src.api.routes:app --host 0.0.0.0 --port 8000`。MinerU 体积大，切片内 compose 的 api 镜像含 MinerU；若镜像过大，下一轮拆 MinerU 独立 sidecar。
- 纯本地 dev 也可只起 `db`，api 用 `config/models.local-dev.yaml`（见 §2.6）在宿主机跑。

### 2.6 配置 Profile（新增）

`config/models.yaml` 现有全局配置保留；新增 `config/models.local-dev.yaml` 供本地 e2e：

```yaml
ingestion:
  parser: simple          # 用 SimplePDFParser，免 MinerU
retrieval:
  use_visual: false       # 本地 smoke 关 visual 路，免 ColPali 下载（BM25+Dense+OpenAI 即可跑通）
  use_rerank: true
llm:
  base_url: https://api.openai.com/v1
  api_key: ${OPENAI_API_KEY}
  model: gpt-4o-mini
```

- `use_visual: false` 时 `VisualRetriever` 跳过，FAISS 不加载——本地只需 BGE（~1.3GB，一次性下载进 HF cache 后常驻）。
- 想本地也验 visual 路：把 `use_visual: true`，首次会下载 ColPali 进 HF cache（一次性，受 AGENTS.md 限制需用户主动触发 `make e2e-local`，脚本不自动跑）。

### 2.5 `Makefile`（改）

```
up / demo:    docker compose up -d --build          # 全栈起服务
db:           docker compose up -d db               # 只起 pgvector（本地 dev 用）
e2e-local:    pytest tests/e2e_local.py             # 本地端到端（需 db 容器 + 模型）
ingest-pdf:   python scripts/ingest_pdf.py --pdf <path>   # 本地/容器内触发 PDFIngestor
```

- 现有 `eval-vidore` 等评测 target 保留不动（不属于本切片，不碰）。
- 新增 `scripts/ingest_pdf.py` CLI 包装 `PDFIngestor`（与 `ingest_vidore.py` 并列）。

---

## 3. 数据流

```
[入库] PDF --(/ingest)--> 落盘 data/uploads/<doc_id>.pdf
        -> Parser(MinerU | SimplePDFParser) -> 逐页(markdown + page_image)
        -> TextChunker -> BGEEmbedder -> PgVectorStore (chunk + bge_vector)
        -> ColPaliEmbedder -> FaissColPaliStore.add_pages (增量)
        -> 返回 {doc_id, num_pages, num_chunks}

[问答] 问句 --(/ask)--> 三路检索(BM25+Dense+Visual) -> RRF -> rerank(Top-5)
        -> compress_context -> Generator(OpenAI SDK) -> 答案 + 引用{page_id,chunk_id,snippet}
```

---

## 4. 错误处理

| 场景 | 处理 |
|------|------|
| MinerU 解析失败 / 加密 PDF | `PDFIngestor` 抛 `IngestError`；API `/ingest` 回 422 + 清理该 `doc_id` 半成品 chunk(pgvector `DELETE WHERE doc_id=`) |
| FAISS 未加载 | `/ask` 前检查索引；未加载则回 503 提示先 `/ingest` |
| 检索无结果 / context 为空 | `Generator` 诚实拒答（复用现有拒答逻辑），`citations=[]` |
| LLM API 超时/限流 | `Generator` 捕获 `openai` 异常 → 回 502 + 错误明细；不污染索引 |
| 单文档限定检索但 doc_id 不存在 | 回 404 |

---

## 5. 测试与验证分层

| 层 | 命令 | 是否需要模型/PG | 内容 |
|----|------|----------------|------|
| 单元 | `make test` | 否（用 fixture + mock） | `test_pdf_ingestor`（SimplePDFParser + 桩 store）、`test_generator`（mock openai）、`test_api`（TestClient 测 422/503/正常） |
| **本地端到端** | `make e2e-local` | 是（pgvector 容器 + BGE；可选 ColPali） | `docker compose up db` → `PDFIngestor`(local-dev profile) 入库样例 PDF → `POST /ask` 拿答案+引用。首次需一次性下载 BGE 进 HF cache |

- `make test` 纯单元，本地随时可跑（遵循 AGENTS.md 本地禁重活，不触发模型下载）。
- `make e2e-local` 是"本地能跑"的硬验证：脚本会先检查 `DATABASE_URL` 可达（pgvector 容器），模型缺失时**明确提示用户手动触发下载**而非自动下载（遵循 AGENTS.md 不主动下大模型）。
- 本地 e2e 默认 `use_visual: false`，只用 BGE+OpenAI，避免 ColPali 下载；验证 visual 路为可选步骤。

---

## 6. 与路线图关系

- = 架构图 §1 ON 路径落地（真实 PDF → 入库 → 问答闭环）+ 容器化服务骨架。
- Repro Spine（`make eval-vidore` / CI / leaderboard）**本切片故意不做**，留待下一轮——本切片聚焦"服务能跑、本地能点"，评测纪律机制后续独立补。
- 下一轮加厚候选：Repro Spine、MinIO 接入、CI 跑消融、leaderboard、前端可点 demo、GraphRAG。

---

## 7. 验收点（切片完成标准）

1. **本地**：`docker compose up db` 起 pgvector → `make e2e-local` 上传样例 PDF 成功入库，`POST /ask` 返回答案 + 至少 1 条引用 `{page_id,chunk_id,snippet}`（无需云、无需 MinerU、默认免 ColPali）。
2. `docker compose up` 全栈起服务，`/health` 显示 page 数随入库增加。
3. `make test` + `make lint` 全绿（纯单元，无模型/PG 依赖）。
