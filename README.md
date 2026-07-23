# PrismRAG

多模态 **工业 PDF** RAG：BM25 + Dense + Visual 三路召回 → RRF 融合 → Cross-encoder 精排 → 生成（可选压缩 / Self-RAG Gate2）。  
在 **ViDoRe v3 Industrial**（英文 283 query）上做消融与分层评测；生产向 API 支持入库、检索、问答与全链路 Trace。

| | |
|--|--|
| **当前状态** | [handoff.md](handoff.md)（会话 / 云上操作 / 最新交付） |
| **模块架构** | [docs/architecture/](docs/architecture/)（实现级快照） |
| **评测协议** | [docs/eval-protocol.md](docs/eval-protocol.md)（NDCG 口径 v1 / Boot-A） |
| **Agent 规范** | [Agents.md](Agents.md)（本地禁全量、云上 GPU 纪律） |
| **下一主线** | [Content Pipeline Phase A/B](docs/superpowers/plans/2026-07-23-content-pipeline-phase-ab-roadmap.md)（入库语义 → 检索接缝） |

---

## 1. 系统概览

```text
PDF
  → 解析 (simple / MinerU) → 分块 (text | table) → 可选表摘要
  → BGE → pgvector          ─┐
  → BM25 倒排               ─┼→ RRF → Rerank (BGE | zerank-2) → 上下文过滤 → LLM 生成
  → Col* 页图 → FAISS MaxSim─┘         ↑
                                    可选 HyDE（默认关）
                                    可选 Visual 路由 / Self-RAG Gate2
```

| 路 | 技术 | 存储 | 粒度 |
|----|------|------|------|
| **BM25** | rank-bm25 | 进程内（冷启从 pg 重建） | chunk |
| **Dense** | BGE-large-en-v1.5 (1024d) | pgvector HNSW | chunk |
| **Visual** | ColPali / **ColQwen2**（默认评测） | FAISS multi-vec + MaxSim | **page** |

**Visual → 文本接缝：** 命中页后按 `page_id` 反查 pg chunk 做 grounding（见 [content-pipeline](docs/architecture/content-pipeline.md)）。

### 默认开关（`config/models.yaml`）

| 能力 | 默认 | 说明 |
|------|------|------|
| Visual 路由 | `enabled: false` | 可开 `heuristic \| always \| never` |
| 上下文过滤 | `context_filter.mode: bge` | `off \| bge \| llm \| …` |
| Self-RAG Gate2 | `enabled: false` | 推荐 `trigger: low_rerank` |
| HyDE | 消融可选，**正式默认关** | 本场景 Δ≈0 |
| 缓存 L3/L4 | `cache.enabled: true` | `index_version` 盐 |

---

## 2. 架构与文档

### 模块架构（推荐从这里读代码）

| 模块 | 文档 | 内容 |
|------|------|------|
| Trace | [trace.md](docs/architecture/trace.md) | 请求 Span、`X-Trace-Id`、排障二分 |
| Cache | [cache.md](docs/architecture/cache.md) | L3 检索 / L4 答案 + 版本盐 |
| Ingestion | [ingestion.md](docs/architecture/ingestion.md) | 三路写入、页 diff、删除一致 |
| Content Pipeline | [content-pipeline.md](docs/architecture/content-pipeline.md) | PDF/表/图解析与分块 |
| Evaluation | [evaluation.md](docs/architecture/evaluation.md) | 三层评测体系与口径 |
| Retrieval / Generation | — | ⏳ 待写 |

索引与约定：[docs/architecture/README.md](docs/architecture/README.md)

### 其它文档

| 类别 | 路径 |
|------|------|
| 全系统单页图 | [docs/prismrag-architecture.html](docs/prismrag-architecture.html) |
| 早期设计总览 | [docs/industrial-pdf-rag-architecture.md](docs/industrial-pdf-rag-architecture.md) |
| 增量验收 | [docs/incremental-verification-runbook.md](docs/incremental-verification-runbook.md) |
| Spec / 计划 | `docs/*-spec-*.md` · `docs/*-design-*.md` · `docs/superpowers/` |
| 实验复盘 | [docs/solutions/](docs/solutions/) |
| Run 归档 | [runs/](runs/) |

---

## 3. 评测结果（可辩护主结论）

**口径纪律：**

- **L1 主表 = 协议 v1**（`1/log2(i+1)` + page 去重）。勿与 2026-07-02 前旧公式 NDCG **绝对值**混比。  
- 样本数（283 / 150 / 100 / 50）必须写清。  
- 拒答 **不进** Faith / Rel 均值（`src/rejection.py`）。  
- 详情：[evaluation.md](docs/architecture/evaluation.md) · 数字以 `runs/` + handoff 为准。

### Layer 1 — 检索（Boot-A · 协议 v1 · 283q en）

来源：[`runs/20260720-bootA/`](runs/20260720-bootA/) · ColQwen2 · `GOLDEN_NO_HYDE` · `--skip-index`

| 配置 | NDCG@10 | 说明 |
|------|--------:|------|
| BM25_only | 0.4063 | |
| Dense_only | 0.3638 | |
| Visual_only | 0.1590 | 视觉单路仍弱 |
| Full_no_rerank | 0.4201 | 三路无精排 |
| Full_with_rerank (BGE) | 0.5161 | |
| **Full_zerank2** | **0.5318** | 主结论配置 |
| 同索引复跑 Full_zerank2 | **Δ = 0** | 漂移验收通过 |

**主结论：** Full_no_rerank → Full_zerank2 **+0.11 NDCG@10**（瓶颈在精排，不在再堆 Visual backbone）。  
HyDE：历史消融 Δ≈0 / 略负，**默认不跑**（见 [zerank2-hyde 复盘](docs/solutions/2026-07-02-zerank2-hyde-experiment.md)）。

可选：Visual 路由 150q（Boot-B）always 1244ms / NDCG@10 0.436 vs heuristic 1019ms / 0.401（延迟约 **−18%**）→ [`runs/20260720-bootB/`](runs/20260720-bootB/)。

### Layer 2 — 生成（RAGAS 自实现）

| 设置 | n | Faith | Rel | CtxRel | 来源 |
|------|--:|------:|----:|-------:|------|
| BGE 压缩默认管线 | **150** | **0.909** | 0.797 | **0.258** | [Boot-B](runs/20260720-bootB/) |
| 历史全量（旧压缩口径） | 283 | 0.772 | 0.810 | 0.076 | `20260706-ragas-full-283` |
| Gate2 OFF 重算（post-P0） | 100 | 0.919 | 0.816 | 0.255 | [Self-RAG 对照](runs/20260721-self-rag-on-only/) |
| Gate2 ON always（post-P0） | 100 | **0.928** | 0.814 | 0.261 | 同上 |

**注意：** 小样本 Faith 易偏高；Gate2 always 相对 OFF 仅 **边际 +0.9pt Faith**，延迟约 **×1.7**；默认 **关** Gate2。污染旧表（Faith 0.83→0.79）**作废**。

### Layer 3 — 端到端 QA（50 可答 + 20 应拒）

| 臂 | Correct | Reject | latency | 来源 |
|----|--------:|-------:|--------:|------|
| 历史 E2E（2026-07-05） | 0.64 | 0.95 | ~2.2s | `runs/20260705-e2e-qa/` |
| Gate2 OFF（post-P0 重算） | 0.60 | 0.90 | 2.24s | Self-RAG 对照 |
| Gate2 ON always | 0.62 | **0.95** | 3.81s | 同上 |

Combined 权重：Correct ×0.7 + RejectAcc ×0.3。E2E 主错多在 **错 chunk / 检索**，不在「Gate2 没开」。

### 关键 Run 索引（近期）

| Run | 内容 |
|-----|------|
| `runs/20260720-bootA/` | 协议 v1 黄金消融 + 漂移=0 |
| `runs/20260720-bootB/` | Visual 路由 + RAGAS 150 |
| `runs/20260721-self-rag-on-only/` | Gate2 干净对照（post-P0） |
| 更早 | 旧 NDCG 公式 / OOM 修复 / ColQwen2 等，见目录内 README |

---

## 4. 快速开始

### 环境

- Python **≥ 3.11**（推荐 [uv](https://github.com/astral-sh/uv)）
- PostgreSQL + **pgvector**
- macOS（M 系列）或 Linux GPU；**全量 283q / 视觉编码请上云**（见 Agents.md）
- 可选：Ollama（生成 / RAGAS Judge / 表摘要）

### 安装与冒烟

```bash
uv venv .venv --python 3.11 && source .venv/bin/activate
uv pip install -e ".[dev]"

# 本地 PG（示例）
make db          # docker pgvector，若已配置

# 最小入库 + 检索冒烟
python scripts/ingest_vidore.py --max-pages 10
python scripts/run_eval.py --max-queries 10 --skip-index --language en \
  --config-filter Full_zerank --visual-model colqwen2

make test
```

### 协议 v1 检索评测（云上）

```bash
# 黄金消融 283q，无 HyDE
python scripts/run_eval.py --skip-index --language en --expected-query-count 283 \
  --visual-model colqwen2 --no-hyde \
  --output-dir runs/YYYYMMDD-bootA/golden-ablation

# 或一键
bash scripts/cloud_boot_a.sh   # 需云环境 + source scripts/cloud_env.sh
```

### 生成层 / E2E

```bash
python scripts/run_ragas_metrics.py --max-queries 10
python scripts/run_e2e_qa.py --max-queries 10 --skip-index
# 云上 Boot-B / Self-RAG：cloud_boot_b.sh · cloud_self_rag_ab.sh
```

### API

```bash
python scripts/run_api.py
# GET  /health
# POST /search   {"query":"...", "k":5}
# POST /ask      检索 + 生成（± Gate2）
# POST /ingest   PDF 上传入库
# POST /cache/invalidate
# GET  /trace/{trace_id}
```

常用 Make：`make help` · `make test` · `make fetch-indexes` · `make ingest-pdf PDF=...`

---

## 5. 项目结构

```text
src/
  ingestion/     # parser · chunker · table_summarizer · encoders · pdf/vidore ingestor
  retrieval/     # bm25 · dense · visual · fusion · reranker · hyde · visual_router
  store/         # pgvector · faiss · snapshot
  generation/    # generator · context_filter · self_rag
  evaluation/    # ablation · ragas_metrics · e2e_qa · vidore_adapter
  cache/         # L3/L4 抽象
  observability/ # tracer · collectors · alerting · middleware
  api/           # FastAPI routes
  prompts/       # YAML 版本化 prompt
  rejection.py   # 统一拒答口径
scripts/         # ingest_* · run_eval · run_ragas* · run_e2e · cloud_* · pack_*
config/          # models.yaml · models.local-dev.yaml
docs/architecture/   # 模块架构（见 §2）
runs/            # 评测归档
tests/
```

---

## 6. 技术栈

| 层 | 选型 |
|----|------|
| 编码 | BGE-large-en-v1.5 · ColPali / ColQwen2 |
| 存储 | pgvector · FAISS (flat/HNSW) · BM25 内存 |
| 融合精排 | RRF · BGE-reranker-large / zerank-2 |
| 生成 / Judge | Ollama qwen2:7b 或 API（配置） |
| 服务 | FastAPI · PyTorch · Python 3.11 |
| 可观测 | structlog · Tracer/Collector · rich dashboard |

可观测架构见 [trace.md](docs/architecture/trace.md)；不必在 README 重复实现细节。

---

## 7. 云端与成本纪律

1. **有卡时段禁止大下载**；先查 HF / Ollama 缓存，缺包停并报告。  
2. `source scripts/cloud_env.sh` → 常设 `HF_HUB_OFFLINE=1`，索引挂数据盘。  
3. 一次开机串任务：`cloud_boot_a.sh` / `cloud_boot_b.sh` / `cloud_self_rag_ab.sh`。  
4. 发包：`pack_for_cloud.sh` + `cloud_apply_upload.sh`（避免盖掉云上索引）。  
5. 本地：代码与小样本；全量编码/283q/全量 RAGAS → 云上。

细节：handoff + Agents.md。

---

## 8. 关键设计决策

| 决策 | 取舍 |
|------|------|
| 三路混合 | 版式/表/关键词各有覆盖；消融看融合，**主增益在精排** |
| Visual 默认可关/可路由 | 存储与延迟贵，单路 NDCG 仍低 |
| HyDE 默认关 | 本场景 Δ≈0，保留开关与消融位 |
| 页 hash 增量 + 删除三路编排 | 防幽灵召回、省重编码 GPU |
| L3/L4 + index_version | 正确性优先于模糊命中率 |
| 三层评测 + 修尺子 | 防假 gap（旧 NDCG、拒答进 Faith） |
| Gate2 默认关 | 边际 Faith vs ×1.7 延迟；低置信再开 |

---

## 9. License / 状态

研究与工程演示项目（`pyproject.toml` version 0.1.0）。  
贡献前请读 [Agents.md](Agents.md) 与当前 [handoff.md](handoff.md)。
