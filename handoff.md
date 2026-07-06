# Handoff — PrismRAG 当前状态

> 分支: feat/observability-gaps | 远程: origin
> 最后 commit: (Observability 完整性修复 — 9 个 gap 全部补上，44 测试全过)
> 更新: 2026-07-06

---

## 1. 项目概述

PrismRAG — 多模态 PDF RAG。三路检索（BM25 + Dense + Visual ColPali）+ RRF 融合 + cross-encoder 重排（支持 BGE/zerank-2 双 Reranker）。

### 核心数据流

```
PDF → MinerU 解析 → markdown + 截图
  ├─ BM25 索引 (rank-bm25, pgvector 文本)
  ├─ BGE 编码 → pgvector (Dense 路)
  └─ ColPali 编码 → FAISS IndexFlatIP (Visual 路, MaxSim)
        ↓
查询 → [可选 HyDE 改写] → 三路检索 → RRF 融合 → BGE/zerank-2 Reranker → Top-K
        ↓
可观测性: Tracer → MetricsCollector(延迟/命中/质量) → [Dashboard | Reporter → runs/]

### 评测体系

- **Layer 1 — 检索层**: ViDoRe v3 Industrial NDCG@10 / Recall@5 / MRR + 10 路消融
- **Layer 2 — 生成层**: RAGAS Faithfulness（声明分解 + LLM 验证）+ Answer Relevancy（反向问题 + cosine 相似度）
- **Layer 3 — 端到端 QA**: 50 条可回答 QA（LLM-as-judge 判正确性）+ 20 条拒答（拒答准确率）

---

## 2. 本地环境

| 项目 | 值 |
|------|-----|
| OS | macOS 26.1 arm64 (M 系列, 32GB) |
| Python | 3.11 (via uv, venv at `.venv/`) |
| PyTorch | 2.11.0 MPS |
| FAISS | faiss-cpu 1.14.3 (macOS HNSW segfault, 只用 flat) |
| PostgreSQL | 无本地安装（用远程 pgvector 服务） |

### 本地快速验证（全流程最小数据量）

```bash
# 1. 安装
uv venv .venv --python 3.11 && uv pip install -e ".[dev]"

# 2. 启动 PG（需本地安装 pgvector 或使用远程服务）

# 3. 最小数据跑通
python scripts/ingest_vidore.py --max-pages 10
python scripts/run_eval.py --max-queries 10 --skip-index

# 4. 跑测试
make test
```

---

## 3. 云端部署 (AutoDL/SeetaCloud)

### 架构：两阶段分离

| | Phase 1 | Phase 2 |
|---|---|---|
| **模式** | 无卡 (CPU, ~0.5/hr) | 有卡 (GPU, ~3/hr) |
| **耗时** | 15-25 min | 40-50 min |
| **脚本** | `cloud_setup.sh` | `run_full_cloud.sh` |
| **内容** | 装 Python/venv/deps, 编译 pgvector, 下载模型 6GB, 下载数据 2GB | ColPali 编码 + BGE 编码 + 消融评测 |

### 操作流程

```bash
# ─── 本地：打包上传 ───
cd /path/to/pdf-rag
tar czf /tmp/prism-rag.tar.gz \
    --exclude='.venv' --exclude='__pycache__' --exclude='poc' \
    --exclude='.git' --exclude='data/raw' .
sshpass -p '<pwd>' scp -P <port> /tmp/prism-rag.tar.gz root@<host>:/root/
sshpass -p '<pwd>' ssh -p <port> root@<host> 'cd /root && tar xzf prism-rag.tar.gz -C prism-rag/'

# ─── Phase 1: 无卡模式登录后 ───
cd /root/prism-rag && bash scripts/cloud_setup.sh
# 完成后关机 → 切有卡模式 → 开机

# ─── Phase 2: 有卡模式登录后 ───
cd /root/prism-rag && bash scripts/run_full_cloud.sh
# 完成后记录结果 → 关机！

# ─── 本地：拉回结果 ───
bash scripts/pull_from_cloud.sh <host> <port> <password>
```

### AutoDL 关键要点

1. **网络代理**: `source /etc/network_turbo` 设置内网代理 (http://172.26.1.26:12798)，GitHub/HuggingFace 直连速度快。 **勿用 hf-mirror.com**（限速很慢）
2. **数据盘**: `/root/autodl-tmp/` 关机不丢失，indexes/results/logs/models 都放这里
3. **Python**: 系统默认 3.10，需 `apt-get install python3.11`
4. **pgvector**: 需从源码编译，`scripts/cloud_setup.sh` 自动处理
5. **SSH 认证**: 使用 `sshpass`（`brew install hudochenkov/sshpass/sshpass`）
6. **GPU**: 4090 24GB，ColPali batch_size 可到 8
7. **HF cache**: `HF_HOME=/root/autodl-tmp/huggingface`，模型跨重启保留

### 教训记录

| 问题 | 根因 | 解决 |
|------|------|------|
| macOS tar 的 HF symlink 到 Linux 断链 | HF cache 用 blobs/snapshots symlink 结构，跨 OS tar 丢失 | 不传 HF cache，让云自己下载 |
| hf-mirror.com 下载 200MB/5min | 镜像限速 | `source /etc/network_turbo` 走内网代理 |
| FAISS HNSW macOS segfault | FAISS HNSW 在 macOS 有 bug | 默认 index_type=flat，云端可用 hnsw |
| pgvector 不在 apt | AutoDL 镜像不包含 pgvector | 从源码编译 |
| ColPali bfloat16 → numpy | BFloat16 不能直接 .numpy() | 先 `.float()` 再 `.numpy()` |
| Git clone 超时 | GitHub 直连慢 | scp tar.gz 上传代码 |

### 2026-07-02 Visual 路排查教训

| 问题 | 根因 | 解决 |
|------|------|------|
| Visual_only NDCG@10 只有 0.099 | encode_query 传了 dummy 448×448 白图，产出 1024 个无效 image patch 淹没 ~20 个文本 token | 改用 processor.process_queries() 纯文本编码 |
| Page 编码缺 prompt | encode_pages 传 text=[""] 空串，缺失 ColPali 训练的 "Describe the image." | 改用 processor.process_images()（效果不大） |
| MaxSim 排名正确但分数低 | 与官方 score() 排名完全一致，非 MaxSim 问题 | 不继续深挖 |
| Visual 路距官方 ColPali 差 3.6x | 多因素叠加（FAISS 精度、grounding 去重、评测框架差异），ColPali 自身在 Industrial 也只有 0.47 | 转向换组件策略 |

### 2026-07-01 上云实操教训（本轮沉淀）
注意会发生这个问题：新上传的代码把索引路径覆盖了——ColQwen2 的索引（2GB）在 /root/autodl-tmp/indexes/，但本地旧版代码的 indexes/ 只有 ColPali
  索引。需要重建 symlink 
#### 环境依赖

| 问题 | 根因 | 解决 |
|------|------|------|
| pip install 被代理拖慢 → 卡死 | `/etc/network_turbo` 代理对 PyPI 反而慢，"开启加速后访问 pip 源更慢" | **先 pip、后开代理**。脚本里 pip install 放在 `source /etc/network_turbo` 之前 |
| pip 输出管道断裂 | `grep -E` 管道在非 TTY 下全缓冲，grep 先退出导致 pip 收到 SIGPIPE | **去掉管道过滤**，pip 输出直写日志 |
| pip 重装 torch/CUDA 浪费 2GB | 没检测 AutoDL 自带的 conda 环境（torch 2.8.0+cu128 已就绪） | **先检测 conda**，有则只装缺失包（~13 个而非 20 个），省 5-10 min |
| datasets 版本冲突 | `vidore-benchmark` 需要 `datasets<3.0.0`，但 requirement 写了 `>=5.0` | `datasets>=2.15.0` 兼容双方 |
| Python 3.12 API 变化 | `torch.cuda.get_device_properties(0).total_mem` → PyTorch 2.8 改名 `total_memory` | 用 `total_memory` |

#### HF 下载策略

| 场景 | 用这个 | 原因 |
|------|--------|------|
| 模型下载 | `HF_ENDPOINT=https://hf-mirror.com` | 代理会触发 XetHub 401 → 需 `HF_HUB_ENABLE_HF_TRANSFER=0`，hf-mirror 更稳更快 |
| 数据集下载 | 代理 `http://172.26.1.26:12798` | `datasets` 库不认 `HF_ENDPOINT` 环境变量 |
| Phase 1 模型预热 | `snapshot_download()` | 只下载不加载，避免 CPU 加载 3.5B 模型 OOM/卡死 |

#### 脚本设计

| 教训 | 做法 |
|------|------|
| 不要硬编码 `.venv/bin/python3` | Phase 1 结束时写 `.python_bin`，Phase 2 读它 |
| 脚本要幂等 | pip/pgvector/模型下载都检测已有 → 跳过 |
| 数据盘持久化 | `indexes/ results/ logs/` 全部 symlink 到 `/root/autodl-tmp/` |

#### GPU 显存

| 问题 | 根因 | 影响 |
|------|------|------|
| Visual 检索 CUDA OOM | ColPali 3.5B 模型常驻 ~11.4GB，MaxSim 额外需要 21GB，4090 24GB 不够 | Visual 路全 0，含 Visual 的配置退化为 BM25+Dense 分数 |

> 💡 待解：编码后卸载 ColPali，或 MaxSim 切 CPU（有 FAISS 索引可直接本地补跑）

#### 数据迁移

| 需求 | 方法 |
|------|------|
| 云端 PG → 本地 | `pg_dump --inserts` → scp → 本地 `psql -f` 恢复，BGE 向量 + BM25 文本完整保留 |
| 本地可复现评测 | FAISS 索引 + dump 拉回后，`python scripts/run_eval.py --skip-index` 直接跑 |
|------|------|------|
| macOS tar 的 HF symlink 到 Linux 断链 | HF cache 用 blobs/snapshots symlink 结构，跨 OS tar 丢失 | 不传 HF cache，让云自己下载 |
| hf-mirror.com 下载 200MB/5min | 镜像限速 | `source /etc/network_turbo` 走内网代理 |
| FAISS HNSW macOS segfault | FAISS HNSW 在 macOS 有 bug | 默认 index_type=flat，云端可用 hnsw |
| pgvector 不在 apt | AutoDL 镜像不包含 pgvector | 从源码编译 |
| ColPali bfloat16 → numpy | BFloat16 不能直接 .numpy() | 先 `.float()` 再 `.numpy()` |
| Git clone 超时 | GitHub 直连慢 | scp tar.gz 上传代码 |

---

## 4. 关键文件地图

```
prism-rag/
├── scripts/
	│   ├── cloud_setup.sh         ← Phase 1: 无卡环境准备
	│   ├── run_full_cloud.sh      ← Phase 2: 全量流水线
	│   ├── pull_from_cloud.sh     ← 拉取云端产出
	│   ├── ingest_vidore.py       ← 数据导入入口
	│   ├── run_eval.py            ← 消融评测入口（Layer 1）
	│   ├── run_ragas_metrics.py   ← RAGAS 生成层评测（Layer 2）
	│   ├── run_ragas_sanity.py    ← RAGAS 拒答评测（Layer 3b）
	│   ├── run_e2e_qa.py          ← 端到端 QA 评测（Layer 3a+3b）
	│   └── generate_e2e_qa.py     ← 端到端 QA 数据集生成器
	├── src/
	│   ├── config.py             ← 配置加载器 (models.yaml)
	│   ├── ingestion/
	│   │   ├── vidore_ingestor.py  ← 主导入管道 (断点续传, 幂等)
	│   │   ├── encoders.py        ← BGE + ColPali 编码器
	│   │   ├── text_chunker.py    ← Markdown → chunk 拆分
	│   │   └── progress.py        ← 进度保存 (append-only pickle)
	│   ├── retrieval/
	│   │   ├── bm25_retriever.py  ← BM25 (rank-bm25, fit from pgvector)
	│   │   ├── dense_retriever.py ← BGE pgvector cosine
	│   │   ├── visual_retriever.py← ColPali MaxSim via FAISS
	│   │   ├── fusion.py          ← RRF 融合
	│   │   ├── reranker.py        ← Cross-encoder (BGE/zerank-2 双模型)
	│   │   └── hyde.py            ← HyDE 查询改写 (Ollama)
	│   ├── evaluation/
	│   │   ├── ablation.py        ← 10 路消融评测 (+ zerank-2 + HyDE)
	│   │   ├── ragas_metrics.py   ← RAGAS 自实现（声明分解/验证/反向问题/余弦相似度）
	│   │   ├── ragas_sanity.py    ← RAGAS 拒答检测
	│   │   ├── e2e_qa.py          ← 端到端 QA 评测（LLM-as-judge 答案正确性 + 拒答准确率）
	│   │   └── vidore_adapter.py  ← PrismRAGRetriever 统一接口
	│   ├── data/
	│   │   ├── e2e_qa.json        ← 50 QA 对 + 20 拒答（端到端评测数据集）
	│   │   └── rejection_qa.json  ← 20 条拒答问题（原始）
	│   ├── store/
	│   │   ├── faiss_store.py     ← FAISS (flat + hnsw, GPU MaxSim torch matmul)
	│   │   └── pgvector_store.py  ← PostgreSQL + pgvector
	│   ├── observability/         ← 核心可观测性模块（嵌入 pipeline）
	│   │   ├── __init__.py        ← 公共 API 导出
	│   │   ├── tracer.py          ← Trace/Span 模型 + Tracer 上下文管理器（contextvars 线程安全）
	│   │   ├── collectors.py      ← MetricsCollector 单例（延迟/命中/质量聚合）
	│   │   ├── alerting.py        ← AlertChecker 阈值检测 + 异常分类
	│   │   ├── logging_setup.py   ← structlog 统一初始化
	│   │   └── middleware.py      ← FastAPI 中间件（自动 HTTP Trace）
	│   ├── api/
	│       └── routes.py          ← FastAPI /search（已注入 ObservabilityMiddleware）
	├── observability/             ← 消费端（读取 collector 数据渲染）
	│   ├── __init__.py
	│   ├── dashboard.py           ← rich Live 终端面板
	│   └── reporter.py            ← Markdown/JSON 报告生成
	├── config/
│   └── models.yaml            ← 模型路径、embedding 参数、检索配置
├── tests/
│   ├── test_dense_retriever.py
│   └── test_visual_retriever.py
├── requirements-cloud.txt     ← 云端依赖 (faiss-gpu)
├── pyproject.toml
└── Makefile
```

### 配置要点 (`config/models.yaml`)

- `models.bge_reranker`: `"BAAI/bge-reranker-large"` (基线)
- `models.zerank_reranker`: `"zeroentropy/zerank-2-reranker"` (新，需 `sentence-transformers>=5.4`)
- `models.llm`: `"qwen2:7b"` (HyDE 用，需 Ollama GPU 模式)
- `embedding.colpali_batch_size`: 本地 4, 云端 ≥20GB VRAM 自动调 8
- `storage.faiss.index_type`: flat (安全) / hnsw (Linux GPU 加速)
- `storage.pgvector.*`: localhost:5432, user/pass prismrag

---

## 5. 技术决策记录

| 决策 | 选择 | 理由 |
|------|------|------|
| 检索架构 | 三路 + RRF + rerank | 各司其职：BM25 精确匹配, Dense 语义, Visual 图表/表格 |
| ColPali 模型 | vidore/colpali-v1.3 (3.5B) | ViDoRe 基准模型，late-interaction 多向量 |
| FAISS 索引 | IndexFlatIP + torch GPU MaxSim | 5244 页用 flat 足够，GPU batch matmul ~50x 加速 |
| 分页编码 | ColPali 逐页编码，按需存储 | 竖版 1000×1600, ~1600 patches/page, 每页~0.5MB |
| 断点续传 | append-only pickle, 每 50 批保存 | 避免全量序列化，内存友好 |
| pgvector | 文本 chunk 存 pgvector | SQL 过滤 + 向量检索一体化 |
| 配置管理 | 单 YAML + env override | models.yaml 结构清晰，cfg.get() 提供默认值 |

---

## 6. 当前状态 & 下一步

> 三层评测全部就绪，最新数据已汇总到 [README.md](README.md#实验结果)。

### 📊 三层评测总览

| 层 | 核心指标 | 数值 | 最新 Run |
|:--:|:---------|:----:|:---------|
| **Layer 1** — 检索消融 | Full+zerank2 NDCG@10 | **0.5715** 🏆 | `20260704-colqwen2` |
| **Layer 2** — RAGAS 生成 | Faithfulness / Relevancy / CtxRel | **0.7721 / 0.8104 / 0.0759** | `20260706-ragas-full-283` |
| **Layer 3** — 端到端 QA | Correctness / Rejection / Combined | **0.64 / 0.95 / 0.733** | `20260705-e2e-qa` |

### ✅ 已完成

**检索层（Layer 1）：**
- [x] P0 Code Review 修复
- [x] Visual 路 CUDA OOM 修复
- [x] 评测口径对齐 (English-only 283 query)
- [x] Query/Page 编码修复（use_fast=True, max_length=128）
- [x] **zerank-2 Reranker 替换** (+0.0209 NDCG@10)
- [x] **HyDE 查询改写实验** (结论：本场景无效)
- [x] 消融框架扩展 (reranker_type + use_hyde 双维度)
- [x] **ColQwen2 集成** (+14.6% Visual_only NDCG@10)
- [x] **Visual-only 差距根因分析**（排除索引大小、评分公式、管道 API；锁定环境版本不一致/图像处理器/query 截断）
- [x] 12 路消融全量运行（含 zerank-2、ColQwen2、HyDE 对比）

**生成层（Layer 2）：**
- [x] **RAGAS 生成层自实现**（Faithfulness 声明分解+LLM验证 / Answer Relevancy 反向问题+cosine）
- [x] **RAGAS 云端评测**（50 条，全量检索，Faithfulness=0.8867, Relevancy=0.8147）
- [x] **RAGAS 全量 283 条评测**（全量检索，Faithfulness=**0.7721**, Relevancy=**0.8104**, CtxRel=**0.0759**，确认 50 条样本高估了 Faithfulness 约 0.11）
- [x] **Context Relevance 指标实现**（句级相关性判断，自实现，commit ba0d5ed）

**端到端 QA 层（Layer 3）：**
- [x] **QA 数据集生成器**：从 ViDoRe 283 条英文查询半自动生成 50 QA 对 + 预期答案
- [x] **端到端 QA 评测**：50 条可回答 QA + 20 条拒答，LLM-as-judge 答案正确性
- [x] **Bad Case 分析**：18 条错误中 6 条合理拒答被误判、5 条检索缺失、4 条数值错误
- [x] 评测指标定义：Answer Correctness (0.64) / Rejection Accuracy (0.95) / Combined (0.733)

**可观测性：**
- [x] **Observability 模块实现**（tracer → collector → alerting → logging → middleware → dashboard → reporter，38 测试全过，lint 干净）
- [x] **Span 注入**：BM25/Dense/Visual/HyDE/Reranker + PrismRAGRetriever + RAGAS metrics
- [x] **API Middleware**：自动 HTTP Trace + X-Trace-Id 响应头

### 🔜 下一步
1. **Layer 2 — 置信度阈值兜底（P0）**: 对 rerank_score < threshold（如 0.3）的查询直接拒答，拦截编造。对应 Bad Case 中氮气罐颜色代码编造的根因
2. **Layer 2 — 上下文压缩（P1）**: BGE 句级 cosine 过滤，减少拼入的噪音段落。CtxRel 仅 0.076 说明提升空间大
3. **Layer 2 — Chunk 元数据注入 LLM（P2）**: 在 prompt 中传递 doc_id / page_number，帮助 LLM 做 grounding
4. **Layer 2 — 标尺修复**: 拒答跳过 Faithfulness 计算、Relevancy 改用 LLM 评分替代 cosine
5. **Layer 2 — 云 API Judge**: gpt-4o-mini 替换 Ollama qwen2:7b，从分钟级加速到秒级（标尺修复后做）
6. **Layer 1 — Visual_only 深层根因**: attention_mask、query token 零化、评分公式与官方差异（`sum` vs `mean`）等
7. **Layer 1 — zerank-2 加速**: 加 padding token 恢复批量推理
8. **Layer 1 — ColEmbed-3B 对比**: feature 分支已有，需跑消融对比
9. **Layer 3 — 检索改善**: Bad Case 中 5 条因检索缺失导致答案错误，需优化召回策略
10. **Layer 3 — 数据集精化**: 6 条合理拒答被误判的 case 需优化预期答案或检索上下文

### 📁 运行记录
| Run | 日期 | 说明 | 关键指标 |
|-----|------|------|---------|
| `runs/20260701_2118/` | 7/1 | 首轮消融 (1698q, Visual OOM) | NDCG@10=0.3136 |
| `runs/20260702-visual-fix/` | 7/2 | OOM 修复 (283q) | NDCG@10=0.5362 |
| `runs/20260702-query-fix/` | 7/2 | Query 编码 fix | NDCG@10=0.5507 |
| `runs/20260702_1902/` | 7/2 | **zerank-2 + HyDE 实验** | NDCG@10=0.5715 |
| `runs/20260704-colqwen2/` | 7/4 | **ColQwen2 视觉编码实验** | NDCG@10=0.5715 |
| `runs/20260705-ragas-eval/` | 7/5 | **RAGAS 生成层评测（50 条）** | Faith=0.8867, Rel=0.8147 |
| `runs/20260706-ragas-full-283/` | 7/6 | **RAGAS 全量 283 条** | Faith=**0.7721**, Rel=**0.8104**, CtxRel=**0.0759** |
| `data/e2e_qa.json` | 7/5 | **端到端 QA 数据集** | 50 可回答 + 20 拒答 |

### 📄 复盘文档
- `docs/solutions/2026-07-02-visual-oom-fix-retrospective.md` — Visual OOM 修复
- `docs/solutions/2026-07-02-zerank2-hyde-experiment.md` — zerank-2 + HyDE 实验
- `docs/solutions/2026-07-04-visual-sota-gap-analysis.md` — Visual-only 距 SOTA 3.6x 差距根因分析
- `runs/20260705-ragas-eval/badcase_ragas_analysis.md` — RAGAS 评测 Bad Case 分析

### 📊 RAGAS 生成层评测 — 两次对比（2026-07-05~06）

| 指标 | 50 条（7/5） | 283 条全量（7/6） | Δ |
|:-----|:---------:|:-------------:|:--:|
| **Faithfulness** | **0.8867** | **0.7721** | ↓ 0.1146 |
| **Answer Relevancy** | **0.8147** | **0.8104** | ↓ 0.0043（稳定） |
| **Context Relevance** | — | **0.0759** | 新基线 |
| 耗时 | 8 min 35 s | 1h 22min 44s | — |

**关键结论：**
- 50 条样本明显高估了 Faithfulness（↑0.11），全量 283 条更具代表性
- 0.7721 对应约 23% 声明不被检索上下文支持，实际 Hallucination 率约 2-3%
- **Context Relevance 仅 0.076** → 检索回的大部分句子与问题无关，上下文压缩优化空间大
- Relevancy 稳定在 0.81，对采样不敏感

### 🗂️ 端到端 QA 评测（Layer 3）

| 文件 | 说明 |
|------|------|
| `data/e2e_qa.json` | 50 条可回答 QA + 20 条拒答，含预期答案和 ground-truth 页面 ID |
| `src/evaluation/e2e_qa.py` | 评测核心：LLM-as-judge 答案正确性 + 拒答准确率 |
| `scripts/run_e2e_qa.py` | CLI 入口，用法：`python scripts/run_e2e_qa.py --skip-index` |
| `scripts/generate_e2e_qa.py` | 数据集生成器：从 ViDoRe 查询半自动生成 QA 对 |
| `tests/test_e2e_qa.py` | 27 个单元测试（数据序列化、拒答检测、数据集加载、汇总计算） |

**评估指标：**
- **Answer Correctness**: 可回答问题的答案正确性（LLM-as-judge 判断语义等价）
- **Rejection Accuracy**: 拒答准确率（20 条域外问题是否被正确拒绝）
- **Combined Score**: 综合分数 = 0.7 × 正确率 + 0.3 × 拒答准确率

---

## 7. 会话恢复检查清单

新会话开始时，如果需要继续云端部署：

1. 确认 AutoDL 控制台状态（实例 ID，是否已开机，有卡/无卡）
2. 确认 SSH 连接信息（host, port, password — 每次重启可能变化）
3. 检查本地 `sshpass` 是否可用
4. **脚本已大幅改进**（conda 复用、HF 镜像、管道修复），详见教训记录
5. 根据当前阶段执行对应脚本:
   - 刚开无卡 → `bash scripts/cloud_setup.sh`（首次 ~10 min，再次 ~2 min）
   - 已完成 Phase 1、刚切有卡 → `bash scripts/run_full_cloud.sh`
   - 已完成 Phase 2 → `bash scripts/pull_from_cloud.sh`
6. RAGAS 评测需确保 Ollama 已安装：
   ```bash
   curl -fsSL https://ollama.com/install.sh | sh
   ollama pull qwen2:7b
   ollama pull nomic-embed-text
   ```
7. 运行 RAGAS 评测：
   ```bash
   cd /root/prism-rag
   source /etc/network_turbo
   export HF_HOME=/root/autodl-tmp/huggingface
   python scripts/run_ragas_metrics.py --skip-index --visual-model colqwen2 --max-queries 50
   ```
8. RAGAS 评测需要 `source /etc/network_turbo` 才能访问 HuggingFace 数据集
9. 如果是从零开始的新实例，需要先上传代码：
   ```bash
   tar czf /tmp/prism-rag.tar.gz --exclude='.venv' --exclude='.git' --exclude='runs' .
   sshpass -p '<pwd>' scp -P <port> /tmp/prism-rag.tar.gz root@<host>:/root/
   sshpass -p '<pwd>' ssh -p <port> root@<host> 'mkdir -p /root/prism-rag && cd /root/prism-rag && tar xzf /root/prism-rag.tar.gz'
