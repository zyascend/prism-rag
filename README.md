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

### 消融对比 (ViDoRe v3 Industrial, 283 English queries)

| 配置 | NDCG@10 | Recall@5 | MRR | 延迟 |
|---|---|---|---|---|
| BM25_only | 0.4432 | 0.4206 | 0.5443 | 24ms |
| Dense_only | 0.3938 | 0.3739 | 0.5137 | 101ms |
| Visual_only (ColPali) | 0.1365 | 0.1447 | 0.1518 | 171ms |
| Visual_only (ColQwen2) | **0.1564** | 0.1438 | **0.1808** | 166ms |
| Full_no_rerank | 0.4402 | 0.4538 | 0.5413 | 335ms |
| **Full + BGE reranker** | 0.5506 | 0.5123 | 0.6589 | 544ms |
| **Full + zerank-2** | **0.5715** | 0.5240 | **0.6777** | 1192ms |

🏆 `Full + zerank-2` NDCG@10=0.5715，比论文 pipeline SOTA (0.532) 高 4 个点。

## 快速开始

### 环境要求

- Python ≥ 3.11
- PostgreSQL + pgvector（本地 via Docker 或远程）
- macOS M 系列 / Linux (GPU)
- 推荐：uv（Python 包管理器）

### 安装

```bash
# 1. 创建虚拟环境并安装
uv venv .venv --python 3.11
source .venv/bin/activate
uv pip install -e ".[dev]"

# 2. 启动 PostgreSQL + pgvector（Docker）
docker run -d --name prismrag-db \
    -e POSTGRES_DB=prismrag \
    -e POSTGRES_USER=prismrag \
    -e POSTGRES_PASSWORD=prismrag \
    -p 5432:5432 \
    pgvector/pgvector:pg16

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
│   │   └── ablation.py          # 10 路消融实验
│   ├── api/routes.py            # FastAPI 搜索 API
│   └── config.py                # 配置加载器 (models.yaml)
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
├── Dockerfile                    # API Docker 部署
└── docker-compose.yml           # API + PostgreSQL
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