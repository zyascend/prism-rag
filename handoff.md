# Handoff — PrismRAG 当前状态

> 分支: feat/ragas-faithfulness | 远程: origin
> 最后 commit: (RAGAS 生成层评测 — Faithfulness + Answer Relevancy 自实现)
> 更新: 2026-07-05（新增 RAGAS 生成层评测体系，Feithfulness=0.8867, Relevancy=0.8147）

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
```

### 评测体系

- **ViDoRe v3 Industrial**: 27 份工业 PDF, 5244 页, 1698 条 query
- **10 路消融**: 7 基线 + zerank-2 + 2×HyDE 变体
- **RAGAS 拒答**: 20 条无答案 query，验证拒答率
- **RAGAS 生成层评测**: Faithfulness（声明分解 + LLM 验证）+ Answer Relevancy（反向问题 + cosine 相似度）

---

## 2. 本地环境

| 项目 | 值 |
|------|-----|
| OS | macOS 26.1 arm64 (M 系列, 32GB) |
| Python | 3.11 (via uv, venv at `.venv/`) |
| PyTorch | 2.11.0 MPS |
| FAISS | faiss-cpu 1.14.3 (macOS HNSW segfault, 只用 flat) |
| PostgreSQL | 无本地安装（用 Docker `pgvector/pgvector:pg16` 或 remote） |

### 本地快速验证（全流程最小数据量）

```bash
# 1. 安装
uv venv .venv --python 3.11 && uv pip install -e ".[dev]"

# 2. 启动 PG (Docker)
docker run -d --name prismrag-db \
    -e POSTGRES_DB=prismrag -e POSTGRES_USER=prismrag -e POSTGRES_PASSWORD=prismrag \
    -p 5432:5432 pgvector/pgvector:pg16

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
│   ├── cloud_setup.sh        ← Phase 1: 无卡环境准备
│   ├── run_full_cloud.sh     ← Phase 2: 全量流水线
│   ├── pull_from_cloud.sh    ← 拉取云端产出
│   ├── ingest_vidore.py      ← 数据导入入口
│   ├── run_eval.py           ← 消融评测入口
│   ├── run_ragas_metrics.py  ← RAGAS 生成层评测（Faithfulness + Relevancy）
│   └── run_ragas_sanity.py   ← RAGAS 拒答评测
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
│   │   └── vidore_adapter.py  ← PrismRAGRetriever 统一接口
│   ├── store/
│   │   ├── faiss_store.py     ← FAISS (flat + hnsw, GPU MaxSim torch matmul)
│   │   └── pgvector_store.py  ← PostgreSQL + pgvector
│   └── api/
│       └── routes.py          ← FastAPI /search (含 retrieval_trace)
├── config/
│   └── models.yaml            ← 模型路径、embedding 参数、检索配置
├── tests/
│   ├── test_dense_retriever.py
│   └── test_visual_retriever.py
├── Dockerfile
├── docker-compose.yml
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

### 📊 消融结果 (ViDoRe V3 Industrial, English 283 queries, 2026-07-04)

| 配置 | NDCG@10 | Recall@5 | MRR | Latency | 备注 |
|---|---|---|---|---|---|
| BM25_only | 0.4432 | 0.4206 | 0.5443 | 24ms | |
| Dense_only | 0.3938 | 0.3739 | 0.5137 | 101ms | |
| Visual_only (ColPali) | 0.1365 | 0.1447 | 0.1518 | 171ms | |
| **Visual_only (ColQwen2)** 🆕 | **0.1564** | 0.1438 | **0.1808** | 166ms | +14.6% vs ColPali |
| BM25_Dense | 0.4528 | 0.4389 | 0.5595 | 126ms | |
| BM25_Dense_Visual (ColPali) | 0.4452 | 0.4630 | 0.5429 | 312ms | |
| BM25_Dense_Visual (ColQwen2) 🆕 | 0.4525 | 0.4855 | 0.5403 | 312ms | |
| Full_no_rerank | 0.4402 | 0.4538 | 0.5413 | 335ms | |
| Full_with_rerank (BGE) | 0.5506 | 0.5123 | 0.6589 | 544ms | 基线 |
| **Full_zerank2** 🆕 | **0.5715** | 0.5240 | 0.6777 | 1192ms | zerank-2 +0.0209 |
| Full_BGE_HyDE 🆕 | 0.5458 | 0.5109 | 0.6527 | 842ms | HyDE 无效 |
| Full_zerank2_HyDE 🆕 | 0.5733 | 0.5316 | 0.6844 | 1421ms | HyDE≈无效 |

> 🏆 Full+zerank2 NDCG@10=0.5715，比 BGE 基线 +0.0209，比论文 pipeline SOTA (0.532) 高 0.04
> 🆕 ColQwen2 提升约 15%，但 Visual_only 绝对分数仍很低（0.1564），根因待查

### ✅ 已完成
- [x] P0 Code Review 修复
- [x] Visual 路 CUDA OOM 修复
- [x] 评测口径对齐 (English-only 283 query)
- [x] Query/Page 编码修复（use_fast=True, max_length=128）
- [x] **zerank-2 Reranker 替换** (+0.0209 NDCG@10)
- [x] **HyDE 查询改写实验** (结论：本场景无效)
- [x] 消融框架扩展 (reranker_type + use_hyde 双维度)
- [x] `--quick` / `--config-filter` 按名过滤消融配置
- [x] **ColQwen2 集成** (+14.6% Visual_only NDCG@10)
- [x] **Visual-only 差距根因分析**（排除索引大小、评分公式、管道 API；锁定环境版本不一致/图像处理器/query 截断）
- [x] **RAGAS 生成层自实现**（Faithfulness 声明分解+LLM验证 / Answer Relevancy 反向问题+cosine）
- [x] **RAGAS 云端评测**（50 条，全量检索，Faithfulness=0.8867, Relevancy=0.8147）
- [x] **`run_ragas_metrics.py` `--skip-index` 修复**（正确加载 FAISS + BM25）
- [x] **RAGAS Bad Case 分析**（含标尺缺陷分析）

### 🔜 下一步
1. **排查 Visual_only 深层根因**: attention_mask、query token 零化、评分公式与官方差异（`sum` vs `mean`）等
2. **zerank-2 加速**: 研究加 padding token 恢复批量推理（目前逐条，2x 慢）
3. **换视觉模型**: ColEmbed-3B（feature 分支已有，需跑消融对比）
4. **RAGAS 全量 283 条评测**: 看结果是否稳定
5. **RAGAS 云 API Judge**: 用 gpt-4o-mini 替换 Ollama qwen2:7b，从分钟级加速到秒级
6. **RAGAS 标尺修复**: 拒答不计入 Faithfulness、Relevancy 改用 LLM 评分替代 cosine

### 📁 运行记录
| Run | 日期 | 说明 | 关键指标 |
|-----|------|------|---------|
| `runs/20260701_2118/` | 7/1 | 首轮消融 (1698q, Visual OOM) | NDCG@10=0.3136 |
| `runs/20260702-visual-fix/` | 7/2 | OOM 修复 (283q) | NDCG@10=0.5362 |
| `runs/20260702-query-fix/` | 7/2 | Query 编码 fix | NDCG@10=0.5507 |
| `runs/20260702_1902/` | 7/2 | **zerank-2 + HyDE 实验** | NDCG@10=0.5715 |
| `runs/20260704-colqwen2/` | 7/4 | **ColQwen2 视觉编码实验** | NDCG@10=0.5715 |
| `runs/20260705-ragas-eval/` | 7/5 | **RAGAS 生成层评测** | Faith=0.8867, Rel=0.8147 |

### 📄 复盘文档
- `docs/solutions/2026-07-02-visual-oom-fix-retrospective.md` — Visual OOM 修复
- `docs/solutions/2026-07-02-zerank2-hyde-experiment.md` — zerank-2 + HyDE 实验
- `docs/solutions/2026-07-04-visual-sota-gap-analysis.md` — Visual-only 距 SOTA 3.6x 差距根因分析
- `runs/20260705-ragas-eval/badcase_ragas_analysis.md` — RAGAS 评测 Bad Case 分析

### 📊 RAGAS 生成层评测（50 条，全量检索，2026-07-05）

| 指标 | 数值 | 说明 |
|------|:----:|------|
| Faithfulness | **0.8867** | 回答 88.7% 的声明被检索上下文支持 |
| Relevancy | **0.8147** | 回答与问题高度相关 |
| 生成回答 | 45/50 (90%) | |
| 拒答 | 5/50 (10%) | 全部合理 |
| 耗时 | 8 min 35 s | RTX 4090, Ollama qwen2:7b |

**Bad Case 总结：**
- 🟡 合理拒答 4 条（Faithfulness=0 误伤）
- 🟠 真正 Hallucination 1 条（氮气罐颜色代码编造，检索未召回相关文档）
- 🔵 Relevancy 标尺偏差 5 条（cosine similarity 对词面不同不敏感）
- 标尺缺陷：拒答应跳过 Faithfulness 计算、Relevancy 应改用 LLM 评分

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
