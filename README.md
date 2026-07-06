# PrismRAG

> **Prism** — BM25 + Dense + Visual 三路检索 → RRF 融合 → Cross-encoder 重排的核心架构。

多模态 PDF RAG 系统。基于 **ViDoRe** 工业文档基准评测验证，支持 **ColPali / ColQwen2** 视觉编码器与 **BGE / zerank-2** 双 Reranker。

## 架构总览

```
PDF Pages ───→ Offline Ingestion ───→ Storage ───→ Online Retrieval ───→ Fusion & Output
                                                      ↕
                                               User Query
```

**三路检索 + RRF 融合 + Cross-encoder 重排：**

| 路 | 技术 | 存储 | 说明 |
|---|---|---|---|
| **BM25** (词法) | `rank-bm25` | pgvector text | 关键字精确匹配 |
| **Dense** (语义) | BGE-large-en-v1.5 | pgvector + HNSW | 1024 维向量余弦搜索 |
| **Visual** (视觉) | ColPali / ColQwen2 | FAISS + MaxSim | 整页多向量画面检索 |

详细架构图：[docs/prismrag-architecture.html](docs/prismrag-architecture.html)（支持暗/亮主题切换，按 `T` 键，按 `E` 键导出图片）

## 实验结果

### 三层评测体系

| 层 | 评测内容 | 核心指标 | 最新结果 |
|:--:|---------|:--------:|:--------:|
| **Layer 1** | 检索层 — 消融对比 (ViDoRe v3 Industrial, 283 条) | NDCG@10 / Recall@5 / MRR | 🏆 Full+zerank-2 **0.5715** |
| **Layer 2** | 生成层 — RAGAS 全量 (Faithfulness + Answer Relevancy + Context Relevance, 283 条) | Faithfulness / Relevancy / CtxRel | **0.7721 / 0.8104 / 0.0759** |
| **Layer 3** | 端到端 QA — 答案正确性 + 拒答准确率 (50 QA + 20 拒答) | Correctness / Rejection / Combined | **0.64 / 0.95 / 0.733** |

---

### Layer 1 — 检索层消融对比 (ViDoRe v3 Industrial, 283 English queries)

| 配置 | NDCG@10 | Recall@5 | MRR | 延迟 |
|---|---|---|---|---|
| BM25_only | 0.4432 | 0.4206 | 0.5443 | 24ms |
| Dense_only | 0.3938 | 0.3739 | 0.5137 | 101ms |
| Visual_only (ColPali) | 0.1365 | 0.1447 | 0.1518 | 171ms |
| Visual_only (ColQwen2) | **0.1564** | 0.1438 | **0.1808** | 166ms |
| BM25_Dense | 0.4528 | 0.4389 | 0.5595 | 126ms |
| BM25_Dense_Visual (ColPali) | 0.4452 | 0.4630 | 0.5429 | 312ms |
| BM25_Dense_Visual (ColQwen2) | 0.4525 | 0.4855 | 0.5403 | 312ms |
| Full_no_rerank | 0.4402 | 0.4538 | 0.5413 | 335ms |
| **Full + BGE reranker** | 0.5506 | 0.5123 | 0.6589 | 544ms |
| **Full + zerank-2** 🏆 | **0.5715** | **0.5240** | **0.6777** | 1192ms |
| Full_BGE_HyDE | 0.5458 | 0.5109 | 0.6527 | 842ms |
| Full_zerank2_HyDE | 0.5733 | 0.5316 | 0.6844 | 1421ms |

> 🏆 `Full + zerank-2` NDCG@10=0.5715，比论文 pipeline SOTA (0.532) 高 4 个点
> 🔍 ColQwen2 比 ColPali 提升约 15%，但 Visual_only 绝对分数仍低 (0.1564)，根因待查
> ❌ HyDE 查询改写在本场景无效（NDCG 变化 <0.005）

---

### Layer 2 — 生成层 RAGAS 评测

| 指标 | 50 条（7/5） | 283 条全量（7/6） | 说明 |
|:-----|:---------:|:-------------:|------|
| **Faithfulness** | **0.8867** | **0.7721** | 回答声明被检索上下文支持的比例 ↓0.11（50 条样本高估） |
| **Answer Relevancy** | **0.8147** | **0.8104** | 回答与问题相关性，稳定、对采样不敏感 |
| **Context Relevance** | — | **0.0759** | 检索上下文与问题相关的句子比例（新指标） |
| 耗时 | 8 min 35 s | 1h 22min 44s | RTX 4090, Ollama qwen2:7b |

**关键分析：**
- Faithfulness=0.7721 对应约 23% 声明不被支持，实际 Hallucination 率约 2-3%
- **Context Relevance 仅 0.076** → 检索回的大量句子与问题无关，上下文压缩优化空间大
- Bad Case 详见 [`runs/20260706-ragas-full-283/observability/report.md`](runs/20260706-ragas-full-283/observability/report.md)

> 详见 [`runs/20260705-ragas-eval/badcase_ragas_analysis.md`](runs/20260705-ragas-eval/badcase_ragas_analysis.md)

---

### Layer 3 — 端到端 QA 评测 (50 可回答 + 20 拒答, 2026-07-05)

| 指标 | 数值 | 说明 |
|------|:----:|------|
| **Answer Correctness** | **0.64** (32/50) | LLM-as-judge 判断答案语义等价 |
| **Rejection Accuracy** | **0.95** (19/20) | 域外问题被正确拒绝 |
| **Combined Score** | **0.733** | 0.7×正确率 + 0.3×拒答准确率 |
| 可回答中拒答 | 5/50 (10%) | 合理拒答（文档无对应内容） |
| 平均延迟 | 2.19s | 含检索 + 生成 + Judge |

**Bad Case 分析（18 条错误）：**
| 错误类型 | 数量 | 说明 |
|:--------|:----:|------|
| 合理拒答被误判 | 6 | 系统拒答但 Judge 期望有答案（数据集问题） |
| 检索缺失导致错误 | 5 | 预期答案在文档中但未被召回 |
| 数值/规格错误 | 4 | 具体数值、规格描述不准确 |
| 生成内容矛盾 | 2 | 答案与预期部分匹配 |
| Hallucination | 1 | 编造规格信息 |

> 详见 [`runs/20260705-e2e-qa/results/e2e_qa_v2/badcase_e2e_qa_analysis.md`](runs/20260705-e2e-qa/results/e2e_qa_v2/badcase_e2e_qa_analysis.md)

---

### 评测运行记录

| Run | 日期 | 说明 | 关键指标 |
|-----|------|------|---------|
| `runs/20260701_2118/` | 7/1 | 首轮消融 (1698q, Visual OOM) | NDCG@10=0.3136 |
| `runs/20260702-visual-fix/` | 7/2 | OOM 修复 (283q) | NDCG@10=0.5362 |
| `runs/20260702-query-fix/` | 7/2 | Query 编码修复 | NDCG@10=0.5507 |
| `runs/20260702_1902/` | 7/2 | zerank-2 + HyDE 实验 | NDCG@10=**0.5715** |
| `runs/20260704-colqwen2/` | 7/4 | ColQwen2 视觉替换 | NDCG@10=0.5715 |
| `runs/20260705-ragas-eval/` | 7/5 | **Layer 2 RAGAS 生成层（50 条）** | Faith=0.8867, Rel=0.8147 |
| `runs/20260706-ragas-full-283/` | 7/6 | **Layer 2 RAGAS 全量（283 条）** | Faith=**0.7721**, Rel=**0.8104**, CtxRel=**0.0759** |
| `runs/20260705-e2e-qa/` | 7/5 | **Layer 3 端到端 QA** | Correct=0.64, Reject=0.95 |

## 快速开始

### 环境要求

- Python ≥ 3.11
- PostgreSQL + pgvector（本地或远程）
- macOS M 系列 / Linux (GPU)
- 推荐：uv（Python 包管理器）

### 安装

```bash
# 1. 创建虚拟环境并安装
uv venv .venv --python 3.11
source .venv/bin/activate
uv pip install -e ".[dev]"

# 2. 启动 PostgreSQL + pgvector（需本地安装 pgvector 或使用远程服务）

# 3. 数据导入（首次：10 页快速验证）
python scripts/ingest_vidore.py --max-pages 10

# 4. 运行评测（10 条 query 验证）
python scripts/run_eval.py --max-queries 10 --skip-index

# 5. 运行测试
make test
```

### 完整评测

```bash
# 全量消融评测（283 条英文 query，需要已构建索引）
python scripts/run_eval.py

# 跳过索引构建，直接评测（适合已有索引）
python scripts/run_eval.py --skip-index

# 仅跑特定消融配置（按名称子串过滤）
python scripts/run_eval.py --config-filter Visual

# 快速模式（仅跑新增 HyDE + zerank-2 配置）
python scripts/run_eval.py --quick

# 全 8 子集 ViDoRe 评测
make eval-full

# 拉取预构建索引
make fetch-indexes
```

### 启动 API

```bash
# 启动 FastAPI 服务（自动加载索引）
python scripts/run_api.py

# 健康检查
curl http://localhost:8000/health

# 搜索查询
curl -X POST http://localhost:8000/search \
    -H "Content-Type: application/json" \
    -d '{"query": "How to configure SSL in Apache?", "k": 5}'
```

## 项目结构

```
├── src/
│   ├── ingestion/
│   │   ├── encoders.py          # BGE + ColPali/ColQwen2 编码器
│   │   ├── vidore_ingestor.py   # ViDoRe 数据集导入管道
│   │   ├── text_chunker.py      # Markdown → chunk 分块
│   │   └── progress.py          # 断点续传进度管理
│   ├── retrieval/
│   │   ├── bm25_retriever.py    # BM25 词法检索
│   │   ├── dense_retriever.py   # BGE 语义检索 (pgvector HNSW)
│   │   ├── visual_retriever.py  # ColPali 视觉检索 (FAISS MaxSim)
│   │   ├── fusion.py            # RRF 融合策略
│   │   ├── reranker.py          # Cross-encoder 重排 (BGE / zerank-2)
│   │   └── hyde.py              # HyDE 查询改写 (Ollama)
│   ├── store/
│   │   ├── pgvector_store.py    # pgvector 存储封装
│   │   └── faiss_store.py       # FAISS 索引 (flat / HNSW + GPU)
│   ├── evaluation/
	│   │   ├── vidore_adapter.py    # PrismRAGRetriever 统一接口
	│   │   ├── ablation.py          # 10 路消融实验
	│   │   ├── ragas_metrics.py     # RAGAS Faithfulness + Answer Relevancy
	│   │   ├── ragas_sanity.py      # RAGAS 拒答检测
	│   │   └── e2e_qa.py            # 端到端 QA 评测 (LLM-as-judge)
	│   ├── observability/           # 内建可观测性模块
	│   │   ├── tracer.py            # Trace/Span 上下文管理器 (contextvars)
	│   │   ├── collectors.py        # MetricsCollector 单例 (延迟/命中/质量)
	│   │   ├── alerting.py          # AlertChecker 阈值检测 + 异常分类
	│   │   ├── logging_setup.py     # structlog 统一初始化 (JSON + 彩色控制台)
	│   │   └── middleware.py        # FastAPI 中间件 (自动 HTTP Trace)
	│   ├── api/routes.py            # FastAPI 搜索 API
	│   └── config.py                # 配置加载器 (models.yaml)
	├── observability/               # 消费端渲染
	│   ├── dashboard.py             # rich Live 终端实时面板
	│   └── reporter.py              # Markdown/JSON 报告生成 → runs/
├── scripts/
│   ├── ingest_vidore.py         # 数据导入入口
│   ├── run_eval.py              # 评测入口
│   ├── run_api.py               # API 服务
│   ├── run_ragas_sanity.py      # RAGAS 拒答评测
│   ├── fetch_indexes.py         # 从 GitHub Release 拉取索引
│   ├── cloud_setup.sh           # 云端无卡环境准备
│   └── run_full_cloud.sh        # 云端全量流水线
├── config/
│   └── models.yaml              # 模型与检索配置
├── tests/                       # 单元测试
├── docs/
│   ├── prismrag-architecture.html  # 架构图
│   └── solutions/                  # 技术复盘文档
```

## 技术栈

| 组件 | 技术 | 说明 |
|---|---|---|
| 文本编码 | BAAI/bge-large-en-v1.5 | 1024 维，NHWC |
| 视觉编码 | vidore/colpali-v1.3 / colqwen2-v1.0 | Late-interaction 多向量 |
| 向量存储 | pgvector (HNSW) | Dense 路 BGE 向量 |
| 视觉索引 | FAISS IndexFlatIP + torch GPU | Visual 路 ColPali 多向量 |
| 融合 | RRFFusion (k=60) | Reciprocal Rank Fusion |
| 重排 | BAAI/bge-reranker-large / zeroentropy/zerank-2 | Cross-encoder |
| 查询改写 | Ollama qwen2:7b | HyDE 假设文档生成 |
| 框架 | Python 3.11, PyTorch, FastAPI |
| 可观测性 | structlog, rich, 自研 Tracer+Collector | 请求级 Trace + 实时仪表盘 + 报告落盘 |

## 可观测性 (数据监测)

内建 observability 模块，嵌入检索/RAGAS/QA 评测全流程，提供三层次可观测能力：

### 架构

```
评测脚本 (run_eval.py / run_ragas_metrics.py / run_e2e_qa.py)
  │
  ├─ get_tracer() → Tracer
  │     └─ start_trace() / finish_trace() → Span {name, duration_ms, metadata}
  │
  ├─ get_collector() → MetricsCollector (单例)
  │     └─ ingest_trace(trace) → 按 config 聚合：延迟(P50/P95/P99)、命中(BM25/Dense/Visual)、质量(Faithfulness/Relevancy)
  │
  └─ dump_collector(run_id) → runs/<run_id>/observability/
        ├─ metrics.json          # 按 config 聚合的延迟/命中/质量
        ├─ traces.jsonl          # 全量 Trace 时序数据
        ├─ alerts.json           # 告警事件
        └─ report.md             # 可读报告
```

### 核心能力

| 组件 | 文件 | 职责 |
|------|------|------|
| **Tracer** | `tracer.py` | 请求级 Trace + 步骤级 Span，`contextvars` 线程安全，零开销 no-op 模式 |
| **Collector** | `collectors.py` | 按 config 聚合延迟百分位、三路命中数、HyDE 缓存命中率、Faithfulness/Relevancy |
| **AlertChecker** | `alerting.py` | 阈值检测（P95 延迟 >5s / Recall@5 <0.5 / Faithfulness <0.6）+ 管道异常分类 |
| **Logging** | `logging_setup.py` | structlog 统一初始化：JSON 文件输出 + 彩色控制台，单行调用替代散落 logging.basicConfig |
| **Middleware** | `middleware.py` | FastAPI 注入，自动为 `/search` 请求创建 Trace，返回 `X-Trace-Id` 响应头 |
| **Dashboard** 🖥️ | `dashboard.py` | rich Live 终端面板，评测运行时实时刷新每 config 的延迟/命中/质量/告警 |
| **Reporter** 📄 | `reporter.py` | 评测完成后将 Collector 快照落盘到 `runs/<run_id>/observability/` |

### 使用方式

可观测性自动嵌入所有评测脚本（`run_eval.py` / `run_ragas_metrics.py` / `run_e2e_qa.py`），无需额外配置。

```python
from src.observability import init_logging, get_tracer, get_collector, dump_collector

# 初始化日志（脚本入口调用一次）
init_logging(level="INFO", log_file="logs/app.jsonl")

# 创建 Trace（自动收集 Span）
tracer = get_tracer()
tracer.start_trace(query=query_text, config_label="Full_zerank2")
# ... 检索管道各步骤自动创建 Span ...
tracer.finish_trace()

# 聚合 + 落盘（评测结束调用）
collector = get_collector()
collector.ingest_trace(trace)
dump_collector("run_id")  # → runs/run_id/observability/
```

### 实时仪表盘（终端）

```bash
# 评测运行时自动启动 rich Live 面板（当前仅 run_e2e_qa.py 已集成）
python scripts/run_e2e_qa.py --skip-index

# 控制台输出示例：
# ┌─────────────────────────────────────┐
# │  Status                             │
# │  Config: Full_zerank2  Queries: 50  │
# │  Runtime: 8m 35s                    │
# ├─────────────────────────────────────┤
# │  Per-Config Metrics                 │
# │  Config     N  P50  P95  Avg  Hits  │
# │  Full_Z2   50  2.1s 4.3s 2.2s 5/3/2│
# ├─────────────────────────────────────┤
# │  Alerts: ⚠ 1  ⛔ 0                  │
# │  ⚠ 14:32 | Full_Z2 | P95 > 5s      │
# └─────────────────────────────────────┘
```

### 报告输出

每次评测完成后，数据自动落盘到 `runs/<run_id>/observability/`：

```
runs/20260705-e2e-qa/observability/
├── metrics.json       # 聚合指标（延迟百分位 / 三路命中 / 质量评分）
├── traces.jsonl       # 全量 Trace 时序数据
├── alerts.json        # 告警事件列表
└── report.md          # 可读 Markdown 摘要报告
```

> 注：当前评测脚本已注入 Tracer/Collector 调用，但观测数据落盘依赖 `collector.ingest_trace()` 实际调用。若运行中未产生 Trace 数据，`dump_collector()` 静默跳过。

## 云端部署

支持 AutoDL / SeetaCloud 两阶段部署：

1. **Phase 1 (无卡)**: 环境准备，下载模型/数据
2. **Phase 2 (有卡)**: ColPali/BGE 编码 + 全量消融评测

详见 [handoff.md](handoff.md) 第 3 节。

## 关键设计决策

- **三路分离**: BM25 精确匹配、Dense 语义、Visual 图表/表格 — 各司其职
- **Visual grounding 接缝**: ColPali 按页检索 → 反查 pgvector 取该页所有 chunk
- **GPU MaxSim**: FAISS flat 索引 + PyTorch GPU 矩阵乘批处理，~50x 加速
- **断点续传**: append-only pickle 每 50 批保存，避免全量编码重跑
- **双 Reranker**: 支持 BGE-reranker-large 与 zerank-2 灵活切换