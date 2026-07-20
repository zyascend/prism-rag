# Handoff — PrismRAG 当前状态

> 分支: feat/bullet-strengthening（进行中）| 远程: origin
> 更新: 2026-07-20 — Self-RAG 设计文档修订为 v2 MVP（仅 Gate2）；Boot-A/B 仍待云上

### Bullet 强化进度（Cloud Boot Packing）

| Boot | 状态 | 说明 |
|------|------|------|
| **本地准备 Boot-A** | ✅ | eval protocol v1；`cloud_boot_a.sh`；`--no-hyde`；NDCG 单测 |
| **本地准备 Boot-B** | ✅ | `VisualRouter` + 配置默认关闭；`context_filter`；`cloud_boot_b.sh` |
| **Boot-A**（黄金消融 + 漂移） | ✅ **2026-07-20** | 见 `runs/20260720-bootA/`：Full_zerank2 **0.5318**，no_rerank **0.4201**（Δ+0.11）；漂移 **Δ=0** |
| **Boot-B**（路由 + RAGAS100） | ⏳ 待有卡 | **已定**：always vs heuristic 双跑 + RAGAS **100q 默认 BGE 压缩**；**不跑** LLM 句过滤。命令：`bash scripts/cloud_boot_b.sh`（默认即该组合） |
| Boot-C（RAGAS283） | 默认可跳过 | — |

计划全文：`docs/superpowers/plans/2026-07-20-bullet-strengthening-roadmap.md`  
配置：`retrieval.visual_routing.enabled` 默认 **false**；`context_filter.mode` 默认 **bge**。

### Self-RAG 设计（文档 only，未实现）

| 项 | 状态 |
|----|------|
| 设计文档 | ✅ v2：`docs/self-rag-closed-loop-design-2026-07-09.md` |
| MVP 范围 | **仅 Gate2**（生成后忠实性门 → regenerate/abstain）；Gate1 充分性 = Phase 2 |
| 相对 Boot | 实现排在 Boot-A/B 数字钉死之后（可选 A 级 bullet） |
| 关键废止 | 不用 CtxRel 当「够不够答」；不默认 HyDE；不无 cap 直接上线 claim 级 `compute_faithfulness` |

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
| **Layer 1** — 检索消融 | Full_zerank2 NDCG@10 (全量 283q, 表格摘要分块+ColQwen2) | **0.5357** | `20260709-table-summary-ndcg` |
| **Layer 2** — RAGAS 生成 | Faithfulness / Relevancy / CtxRel (表格摘要 ON, 100q) | **0.901 / 0.804 / 0.263** | `20260709-table-summary-ndcg` |

> 历史对照（供追溯，非直接可比）：NDCG 修复后 50q 全配置 `20260707-bottleneck-analysis` Full+zerank2 = 0.402；RAGAS 修复后 100q `20260708-ctxrel-fix` = 0.882/0.791/0.294（旧分块）。⚠️ 当前 run 与历史 run 存在**混淆变量**（视觉编码器 ColQwen2 vs ColPali-v1.3 + 表格摘要分块 vs 旧切分），干净归因见 §9。

> ⚠️ **NDCG 公式已修复**: 从自实现 `1/(i+1)` 改为国际标准 `1/log2(i+1)`（对齐 pytrec_eval）。
> 旧公式下 Full+zerank2 为 0.5715，新公式同等分数约 0.65+（50q 采样）。历史 run 数据不可直接对比。

### 📈 CtxRel 改善历程（100-query）

| 阶段 | PR | CtxRel | Faithfulness | 改动 |
|:--|:--:|:--:|:--:|------|
| 基线 | — | 0.076 | 0.772 | 283-query 全量（旧） |
| +上下文压缩 | #19 | 0.087 | 0.894 | BGE 句级 cosine 过滤 0.4 |
| +TO 清洗 | #20 | 0.117 | 0.886 | 6 步正则去噪音行 |
| +doc_ref | #21 | 0.116 | 0.889 | TO 编号存 metadata |
| +metric 修复 | 本分支 | **0.294** | 0.882 | CtxRel 改评压缩后 context（见下） |

> ⚠️ **CtxRel 根因修复（2026-07-08，分支 `fix/ctxrel-compressed-context`）**：
> `compute_context_relevancy` 原本传入**原始检索 chunk**（`evaluate_generation` 中重新构造 `context_chunks`），
> **绕过了 `compress_context` 的 0.4 句级过滤**。但 RAGAS Context Relevance 定义为
> "喂给 LLM 的上下文"中相关句占比——应与 `generate_answer` / `compute_faithfulness` 使用同一份 `context`。
> 这导致 CtxRel 系统性低估：实测 100q 平均 75 句/查询、仅 ~8 句相关 = 0.116；
> 压缩后 ~30 句中保留大部分相关句，预计 CtxRel 升至 ~0.25–0.30（待云端重跑确认）。
> 修复：在 doc_ref 前缀注入前捕获 `ctx_for_eval = context`，传 `[ctx_for_eval]` 给 CtxRel。
> 回归测试 `TestCtxRelUsesCompressedContext` 已加（47 ragas 测试全过）。
> **云端复测确认（2026-07-08, `runs/20260708-ctxrel-fix`）**：CtxRel 0.116 → **0.294 (+154%)**，
> num_sentences 75.3→29.8（压缩到 40%）、num_relevant 7.8→8.2（相关句保留），Faithfulness/Relevancy 无回归。

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
- [x] 12 路消融全量运行（含 zerank-2、ColQwen2、HyDE 对比）
- [x] **NDCG 公式修复** (`1/(i+1)` → `1/log2(i+1)`，对齐 pytrec_eval 国际标准)
- [x] **NDCG/Recall page_id 去重**（修复 VisualRetriever chunk 级重复导致 NDCG 被低估 ~40%）
- [x] **Visual gap 终局定性** — 详见 §7 突破

**§7 Visual-only SOTA gap 终局定性（2026-07-07 cloud 实验）：**
- [x] Root Cause A/B/C **全部证伪**（同进程重建分不变、Qwen2VL 默认 fast、max_length 已修）
- [x] MaxSim 与官方 `score_multi_vector` **完全一致**（NDCG 0.344 vs 0.340, 99% match）
- [x] 官方 SOTA 确认：ColPali-v1.3 Industrial = **0.470**, ColQwen2 Industrial = **0.498**
- [x] "3x 缺口" = NDCG 公式差（~2x）+ page_id 去重缺失（~0.4x），修复后 Visual_only ≈ **0.34** vs 官方 0.50
- [x] 剩余 1.45x 差距来自编码/图像质量，非管线 bug

**生成层（Layer 2）：**
- [x] **RAGAS 生成层自实现**（Faithfulness 声明分解+LLM验证 / Answer Relevancy 反向问题+cosine）
- [x] **RAGAS 云端评测**（50 条，全量检索，Faithfulness=0.8867, Relevancy=0.8147）
- [x] **RAGAS 全量 283 条评测**（全量检索，Faithfulness=**0.7721**, Relevancy=**0.8104**, CtxRel=**0.0759**，确认 50 条样本高估了 Faithfulness 约 0.11）
- [x] **Context Relevance 指标实现**（句级相关性判断，自实现，commit ba0d5ed）
- [x] **CtxRel 口径修复**（`compute_context_relevancy` 改评压缩后 context，与 Faithfulness/生成口径一致；分支 `fix/ctxrel-compressed-context`，待云端重跑确认数值）

**端到端 QA 层（Layer 3）：**
- [x] **QA 数据集生成器**：从 ViDoRe 283 条英文查询半自动生成 50 QA 对 + 预期答案
- [x] **端到端 QA 评测**：50 条可回答 QA + 20 条拒答，LLM-as-judge 答案正确性
- [x] **Bad Case 分析**：18 条错误中 6 条合理拒答被误判、5 条检索缺失、4 条数值错误
- [x] 评测指标定义：Answer Correctness (0.64) / Rejection Accuracy (0.95) / Combined (0.733)

**可观测性：**
- [x] **Observability 模块实现**（tracer → collector → alerting → logging → middleware → dashboard → reporter，38 测试全过，lint 干净）
- [x] **Span 注入**：BM25/Dense/Visual/HyDE/Reranker + PrismRAGRetriever + RAGAS metrics
- [x] **API Middleware**：自动 HTTP Trace + X-Trace-Id 响应头
- [x] **线上单条答案排查闭环（2026-07-18, `feat/observability-trace-gaps`）**：补齐此前三块缺口 —
  - ① 生成层埋点：`Generator.answer` 包 `tracer.start_span("generation")`，metadata 含 model/k_context/num_retrieved/num_citations/citations/完整 context
  - ② `/ask` 返回 `retrieval_trace`（bm25/dense/visual top5，复用 `search_with_trace`）；新增 `GET /trace/{trace_id}` 反查端点（404 if not found）
  - ③ `MetricsCollector` 增 `_trace_by_id` 内存索引（FIFO cap=2000）+ 磁盘 JSONL 持久化（`logs/api_traces.jsonl`，由 `config observability.trace_persist_path` 控制，空串关闭）；`get_trace(id)` 内存优先、未命中回退扫描磁盘（覆盖进程重启）
  - **效果**：拿响应 `X-Trace-Id` → `GET /trace/{id}` 即可看 retrieval_trace + generation.context，二分定位"检索层 vs 生成层"错误，端到端可落地（聚焦单测已验证）

**检索缓存（2026-07-18, `feat/cache-layers`）— L3 结果缓存 + L4 Answer 缓存 + 全局开关 + 可观测命中率：**
- [x] **CacheStore 抽象 + InMemoryLRUCache**（`src/cache/store.py`）：进程内 LRU 淘汰 + 可选 TTL 兜底；RedisCache 预留接口（多 worker 场景）
- [x] **L3 检索结果缓存**（`PrismRAGRetriever.search_with_trace` 包裹）：key = 归一化 query + k + 各路开关 + reranker_type + index_version 盐；命中跳过三路检索+融合+重排
- [x] **L4 Answer 缓存**（`/ask` 包裹 + `PrismRAGRetriever._answer_cache`/`answer_cache_key`）：命中跳过整次 LLM 生成；key 含 归一化 query + model + k_context + doc_id + index_version 盐
- [x] **Generator.cacheable 确定性守卫**（`generator.py`：`temperature==0` 才可缓存，非确定性生成不读不写 L4）
- [x] **index_version 版本盐失效（`invalidate_cache` 同时清 L3 + L4）**：`delete_document` 调 `invalidate_cache()`（版本+1，旧 key 天然失效，零脏读）；新增 `POST /cache/invalidate` 端点供重索引后失效服务侧缓存
- [x] **全局开关**（`cache.enabled`，`src/config.py` CacheConfig + `models.yaml` cache 段）：门控所有缓存层（L3/L4），运行时每请求读取，关闭即穿透
- [x] **可观测 cache 命中率**（`MetricsCollector.record_cache_event` + `ConfigMetrics.retrieval_cache_hit_rate` + `answer_cache_hit_rate`）：命中写 `retrieval`/`answer` 事件（供 `GET /trace/{id}` 可见）+ 聚合两层命中率进 report
- [x] **K1 修复**：`vidore_adapter.py:349` fused 末路径 cache miss 补 `config_label=config_label`（原漏传默认 `"api"`，仅指标归类偏差，非正确性 bug）
- [x] 正确性约束：L3 key 含全部检索开关、`doc_id` 在路由层后置过滤不入 key（C1）；L4 key 含 doc_id/model/k_context（C6）；`visual_query_embedding` 非 None 按 tensor hash 编 key；TTL 仅兜底不依赖正确性
- 聚焦单测 `tests/test_retrieval_cache.py` 全过（11 个：LRU/TTL、retrieval+answer 命中率聚合、cache_key 归一化+版本盐、L4 answer_cache_key/doc_id/版本盐、命中/未命中、全局开关关闭、K1 回归、`/ask` L4 集成）
- 规格文档：`docs/cache-retrieval-spec-2026-07-18.md`（v1.1，含 L4 设计/C6/K1 已修复）

**表格摘要 + 大表保护（2026-07-09, `runs/20260709-table-summary-ndcg`）— 实现后首次云端全量验证：**
- [x] 重新入库（`ingest_vidore.py --skip-faiss`, `table_summary_enabled=True`；chunks 8835 = text 6530 + table 2305，table_summary 100% 非空）
- [x] 复用 ColQwen2 视觉 FAISS 索引（未重编码，省 GPU 时间）
- [x] 全量 283q NDCG 10 路消融（Full_zerank2 = **0.5357** ⭐, MRR 0.6658）
- [x] RAGAS 100q 生成端评测（Faith=**0.901** / Rel=0.804 / CtxRel=**0.263**，82 生成 / 18 拒答）
- [x] 干净归因：表格摘要分块削弱文本路 NDCG（Dense -7.6% / BM25 -4.1%），视觉路持平；生成端 ContextRelevancy = **0.2626**（⚠️ 见 §9 footnote ①：此数非本特性增益，且非全 run 最高）
- [x] 结论：特性主目标（生成端收益）达成，方向正确，保留；详见 §9

**增量更新/删除优化（已合并 `main` via PR #26，commit `038e6dd`；Spec `docs/incremental-update-optimization-spec-2026-07-16.md`）：**
- [x] **P0（D2 正确性）**：删文档后已删内容仍被 BM25 召回的 bug 修复（`delete_document` 三路统一编排：pg→bm25→faiss；pg 删除前先取受影响 id）。
- [x] **P1（D1 FAISS orphan + U2 副本幂等）**：FAISS 墓碑 + 异步 compact；`documents` 表内容哈希幂等（同 PDF 重入库复用 doc_id，不产副本）；ingest 哈希覆盖。
- [x] **P2（效率 + 规模）**：① BM25 弃用 `rank_bm25.BM25Okapi` 全量重建，改为自维护统计 + `fit_incremental`/`remove_chunks`（O(vocab) 重算 idf，消 U1）；打分公式与本项目所装 rank_bm25 逐位一致（NDCG 不漂移）。② page 级 `page_hash` diff：同 doc_id 修改版仅重编码变化页，未变页跳过 ColQwen2（省 GPU）；`routes.py` 跳过每次 ingest 后的全量 `fit_from_pgvector`。③ 原子快照切换：FAISS `os.replace` + BM25 临时文件 `os.replace` + pg `chunks_staging` 事务内 RENAME swap（零停机大批量刷新）。
- [x] **验证（本地）**：`test_p2_incremental` + `test_lifecycle` + `test_faiss_lifecycle` + `test_pdf_ingestor` 共 21 项全过；ruff 全绿。评测公式对照 `test_bm25_scores_match_rank_bm25` 通过。
- [x] **收尾**：分支 `feat/incremental-update-delete` 已 squash 合并入 `main`（PR #26，commit `038e6dd`），远程/本地分支已清理。✅
- [ ] **待办（验证）**：上云用已部署 ColQwen2 跑全量评测，确认 NDCG 不漂移、page diff 省 GPU 生效。⚠️ 本地禁全量评测（依赖远程 PG / ColQwen2）。

### 🔜 下一步

**P0 — NDCG 修复后重跑全量评测：**
- [x] 全量 283q 10 路消融已跑（`runs/20260709-table-summary-ndcg`，标准 NDCG + 去重）→ 含表格摘要分块的新基线（Full_zerank2 NDCG@10 = 0.5357）
- [ ] RAGAS 全量 283 条重跑（确认 Faithfulness 是否受检索质量影响；当前仅 100q，见 §9）

**P1 — 质量改进：**
3. Chunk metadata 注入 LLM prompt（page_number/section_title）
4. CtxRel 句级 LLM 预过滤替代 BGE cosine
   - ⚠️ **ratio 调参实验已做（2026-07-08, `runs/20260708-compress-ratio-025`）**：0.4→0.25 时 CtxRel 0.294→0.410(+39%) **但** Faithfulness 0.882→0.855(-3%)、拒答 16→19、num_relevant 8.2→7.4（BGE 开始砍相关句）。结论：**0.25 过激，0.4 保留为安全默认；甜区估计在 0.3（待测）**。调参是零风险但边际有限，LLM 预过滤才是质变路径（注意避免与 CtxRel 评分 LLM 自循环）。
5. 换 Unstructured.io 重解析 PDF（根治 TO 手册噪音）

**P2 — 效率/工程：**
6. Visual 路按需路由（含表格/图表 query 启用，纯文本跳过）
7. `run_eval.py` 加 `--no-hyde` flag（省 ~3.5 min/run）
8. gpt-4o-mini 替换 Ollama qwen2:7b（分钟级→秒级）

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
| ☁️ Option A | 7/7 | 复用 7/4 cache, Visual_only 50q | NDCG@10=0.1676（旧公式） |
| ☁️ Option B | 7/7 | 同进程重建, Visual_only 50q | NDCG@10=0.1676 → Root Cause A **证伪** |
| ☁️ MaxSim 对比 | 7/7 | 官方 `score_multi_vector` vs `_maxsim_torch` 同嵌入 | NDCG 0.344 vs 0.340, **99% 一致** |
| ☁️ NDCG 修复后 | 7/7 | 标准 `1/log2` 公式, 50q 全配置消融 | Visual=0.202, Full+zerank2=**0.402** (旧公式 0.167→0.202) |
| `runs/20260708-ctxrel-fix/` | 7/8 | **CtxRel 口径修复复测（100q）** | CtxRel=**0.294**(+154%), Faith=0.882, Rel=0.791 |
| `runs/20260709-table-summary-ndcg/` | 7/9 | **表格摘要+大表保护 首次云端全量验证** | NDCG Full_zerank2=**0.5357**(283q); RAGAS Faith=**0.901**/Rel=0.804/CtxRel=**0.263**(100q) |

### 📄 复盘文档
- `docs/solutions/2026-07-02-visual-oom-fix-retrospective.md` — Visual OOM 修复
- `docs/solutions/2026-07-02-zerank2-hyde-experiment.md` — zerank-2 + HyDE 实验
- `docs/solutions/2026-07-04-visual-sota-gap-analysis.md` — Visual-only 距 SOTA 差距根因分析 + **终局定性**（§7）
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
  - ⚠️ 注：该值含**口径 bug**——CtxRel 原评原始 chunk 而非压缩后 context（已修复，见 §6 CtxRel 表），真实值约为其 ~2x
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

## 8. 生产服务骨架（2026-07-08, `feat/production-spine`）

> 方向：从"刷 benchmark"转向"贴近生产"。薄垂直切片 = 真实 PDF 入库 + `/ask` 问答闭环 + 本地可跑 + 容器化。
> 设计：`docs/superpowers/specs/2026-07-08-production-service-spine-design.md`
> 计划：`docs/superpowers/plans/2026-07-08-production-service-spine.md`（9 task，全过 + 终审 Ready to merge）

### 交付内容
- `src/ingestion/parser.py`：`Parser` 抽象 — `SimplePDFParser`(PyMuPDF, 本地零依赖兜底) + `MinerUParser`(MinerU CLI, 生产)
- `src/ingestion/pdf_ingestor.py`：`PDFIngestor.ingest(pdf)` → 解析→分块→BGE→pgvector + ColPali→FAISS 增量(`add_pages`)
- `src/store/faiss_store.py`：新增 `add_pages` 增量写入（flat + hnsw 同步）；`src/store/pgvector_store.py`：新增 `delete_by_doc_id`
- `src/generation/generator.py`：`Generator`(OpenAI SDK) + `GenerationError`，引用以检索 chunk 为准，空检索诚实拒答
- `src/api/routes.py`：新增 `POST /ingest`（上传 PDF 入库）、`POST /ask`（检索→生成→答案+引用回链）；失败清理 + 502/500 错误处理
- `src/config.py`：`CONFIG_PROFILE` 合并机制；`config/models.local-dev.yaml`（parser=simple, use_visual=false, 免 MinerU/ColPali）
- `Dockerfile` + `docker-compose.yml`（pgvector + api，OPENAI_API_KEY 映射）+ `Makefile`(`db`/`up`/`e2e-local`/`ingest-pdf`) + `scripts/ingest_pdf.py`
- `tests/e2e_local.py`：真·端到端，无 PG/OPENAI 时自动 skip

### 本地可跑验证
- `docker compose up db` 起 pgvector → `python scripts/ingest_pdf.py --pdf x.pdf` 入库 → `POST /ask` 拿答案+引用
- `use_visual:false` 本地 smoke 只需 BGE(~1.3GB, 一次性下载进 HF cache) + OpenAI API，免 ColPali
- 单测：`make test`（9 个 spine 单测全过，纯单元无 PG/模型）；全量 174 单测 + ruff 全绿（Task 9 确认）

### 已知 follow-up（非阻塞）
- `add_pages` 在 `index_type=hnsw` 现已同步 HNSW；默认 flat 不受影响
- 上传文件失败时已清理 `data/uploads/<doc_id>.pdf`（mineru 中间产物清理为 best-effort）
- 本切片**不含** `make eval-vidore`/Repro Spine/GraphRAG/MinIO/CI（已与用户确认拿掉，留待下一轮）

---

## 9. 表格摘要+大表保护 首次云端全量验证 (2026-07-09)

> 背景：特性 `fe9ceba`（表格摘要+大表保护）已合并 main（PR #24）。本 run 是**实现后首次云端全量验证**——重新入库（表格摘要默认开 + 大表按行切）后跑 NDCG 消融 + RAGAS，确认特性是否达成设计目标（设计文档 `docs/table-summary-large-table-design-2026-07-09.md`）。
> 数据归档：`runs/20260709-table-summary-ndcg/`（results/ 含 ablation_results.json + ragas_metrics_default.json；logs/ 含 eval_ndcg.log + eval_ragas.log）。⚠️ FAISS 索引（~8GB，含 ColQwen2 视觉索引 + page_embeddings_cache）**未拉回**，仍留云端 `/root/autodl-tmp/indexes`，如需本地 `--skip-index` 复跑需先 rsync。
> ⚠️ **混淆变量**：本 run 用 **ColQwen2** 视觉 + 表格摘要分块；历史最好 run（20260702）用 **ColPali-v1.3** + 旧切分。两者不只差"表格摘要"，还差"视觉编码器"，直接比会混淆。

### 实验设置
- 环境：seetacloud RTX 4090 24G，ColQwen2 视觉 FAISS 复用（未重编码）
- 入库：`ingest_vidore.py --skip-faiss`，`table_summary_enabled=True`
- 数据集：vidore/vidore_v3_industrial（English，283 queries 全量）
- chunks：8835（text 6530 + table 2305），table_summary 100% 非空

### NDCG@10 消融（全量 283q，10 config）

| Config | NDCG@10 | MRR | 备注 |
|--------|---------|-----|------|
| BM25_only | 0.4248 | 0.5302 | |
| Dense_only | 0.3638 | 0.4718 | |
| Visual_only | 0.1590 | 0.1727 | 视觉路，分块无关 |
| BM25_Dense | 0.4296 | 0.5376 | |
| BM25_Dense_Visual | 0.4334 | 0.5071 | |
| Full_no_rerank | 0.4334 | 0.5071 | |
| Full_with_rerank (bge) | 0.5162 | 0.6356 | |
| Full_BGE_HyDE | 0.5054 | 0.6150 | |
| **Full_zerank2** ⭐ | **0.5357** | **0.6658** | 当前最优 |
| Full_zerank2_HyDE | 0.5273 | 0.6518 | |

### RAGAS 100q（生成端 verdict，表格摘要 ON）

| 指标 | 当前 | 历史干净 run (20260707-ragas-100-clean, 旧分块) | Δ |
|------|------|------|---|
| Faithfulness | **0.901** | 0.8862 | +0.015 (+1.7%) |
| AnswerRelevancy | **0.804** | 0.7984 | +0.006 (+0.7%) |
| ContextRelevancy | **0.2626** | 0.1175¹ | +0.145 (+14.5pp) — 非本特性增益（见 ①） |
| 生成/拒答 | 82 / 18 | 85 / 15 | — |

> **① ContextRelevancy 口径警示（重要，纠正前版"翻倍"误述）**：本指标 = `num_relevant / num_sentences`，即**检索上下文的精确度（precision）**，**对上下文体积高度敏感**。横向核对全部 run（同一 100q 集）实测：
> | run | CtxRel | 平均句数 |
> |---|---|---|
> | 20260708-compress-ratio-025 | **0.4102（全 run 最高）** | 18.5 |
> | 20260708-ctxrel-fix（metric 修复） | 0.2943 | — |
> | **20260709-table-summary-ndcg（本 run）** | **0.2626** | 32.1 |
> | 20260707-ragas-100-clean（旧分块） | 0.1175 | — |
> - 0.087→0.294 的跃升来自 **7/8 的 `compute_context_relevancy` metric 修复**（改评压缩后 context），**不是任何特性**；本 run 0.2626 甚至**低于** ctxrel-fix 的 0.2943。
> - 表格摘要会**向上下文注入摘要文本**，使平均句数从 ~18 涨到 32 → 分母变大 → precision **被动下降**。最高值 0.4102 来自 `compress-ratio-025`（把上下文压到 0.25），属"砍掉上下文换精度"的假象，**不可作为质量增益**。
> - 结论：CtxRel 在本项目里**不是特性收益信号**；生成端主目标看 **Faithfulness 0.901**。CtxRel 仅用于监控"上下文是否被稀释"。

### 干净归因（消掉视觉编码器变量）

| 路 | 旧分块 | 表格摘要分块 | Δ | 解读 |
|----|--------|--------------|---|------|
| Dense_only (BGE文本) | 0.3938 | 0.3638 | **-0.030 (-7.6%)** | 纯"摘要替代整表编码"效应 |
| BM25_only (词法) | 0.4432 | 0.4248 | **-0.018 (-4.1%)** | 摘要关键词比整表少 |
| Visual_only (ColQwen2) | 0.1564 | 0.1590 | +0.003 ≈ 持平 | 分块不影响视觉检索 ✅ |

公平对照（各自最优 reranker）：历史最好 20260702(bge) NDCG 0.5507/MRR 0.6595 vs 当前 Full_zerank2 NDCG 0.5357/MRR **0.6658** → NDCG -2.7%、MRR **+1.0%**。

### 结论
1. **NDCG 非本特性目标指标**（设计目标要改善的是生成端 Faithfulness/AnswerRelevancy，非检索排序）。文本路 NDCG 小幅下降是**预期权衡**。
2. **RAGAS 印证生成端收益（主目标达成）**：Faithfulness 0.901(>0.90)。⚠️ CtxRel **0.263 既非"翻倍"也非本特性增益**（见 ①）；答案忠实度提升才是主证据。
3. 视觉路 NDCG 完全不受影响（持平），ColQwen2 升级是净加分。
4. 判定：特性方向正确，保留。若要压低文本路 NDCG 损失，可试点"Dense 同时编码摘要+关键单元格"混合 embed。

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
