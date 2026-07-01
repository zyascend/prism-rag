# Handoff — PrismRAG 当前状态

> 分支: main | 远程: origin/main
> 最后 commit: e520557 perf: MaxSim 切 torch GPU 加速
> 更新: 2026-07-01（上云完成，首轮消融结果到手）

---

## 1. 项目概述

PrismRAG — 多模态 PDF RAG。三路检索（BM25 + Dense + Visual ColPali）+ RRF 融合 + cross-encoder 重排。

### 核心数据流

```
PDF → MinerU 解析 → markdown + 截图
  ├─ BM25 索引 (rank-bm25, pgvector 文本)
  ├─ BGE 编码 → pgvector (Dense 路)
  └─ ColPali 编码 → FAISS IndexFlatIP (Visual 路, MaxSim)
        ↓
查询 → 三路检索 → RRF 融合 → BGE-Reranker 重排 → Top-K
```

### 评测体系

- **ViDoRe v3 Industrial**: 27 份工业 PDF, 5244 页, 1698 条 query
- **7 路消融**: BM25 → Dense → Visual → BM25+Dense → 三路 → Full → Full+Rerank
- **RAGAS 拒答**: 20 条无答案 query，验证拒答率

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
│   ├── run_eval.py           ← 评测入口
│   └── run_ragas_sanity.py   ← RAGAS 拒答
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
│   │   └── reranker.py        ← BGE Cross-encoder
│   ├── evaluation/
│   │   ├── ablation.py        ← 7 路消融评测
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

### 📊 首轮云上消融结果 (2026-07-01, RTX 4090)

| 配置 | NDCG@5 | Recall@5 | MRR | 延迟 |
|------|--------|----------|-----|------|
| BM25_only | 0.1371 | 0.1485 | 0.1701 | 29ms |
| Dense_only | 0.1299 | 0.1342 | 0.1916 | 105ms |
| Visual_only | ⚠️ 0.0000 | 0.0000 | 0.0000 | 95ms |
| BM25_Dense | 0.1590 | 0.1767 | 0.2120 | 129ms |
| BM25_Dense_Visual | 0.1590* | 0.1767* | 0.2120* | 221ms |
| Full_no_rerank | 0.1590* | 0.1767* | 0.2120* | 224ms |
| **Full_with_rerank** 🏆 | **0.2468** | **0.2331** | **0.3136** | 438ms |

> \* Visual 路 CUDA OOM 被跳过，分数等同于 BM25_Dense  
> 🏆 Reranker 提分显著：NDCG +55% vs BM25 only  
> 产出保存在 `runs/20260701_2118/`

### ✅ 已完成
- [x] P0 Code Review 修复 (FAISS HNSW, API trace, CI, Docker/index 版本化)
- [x] GPU 优化 (torch MaxSim, batch_size 自适应)
- [x] 本地最小数据量全流程验证通过
- [x] AutoDL 云部署脚本 (两阶段)
- [x] Docker Compose (API + pgvector + ollama)
- [x] CI (lint+test / ablation / full weekly)
- [x] ~~首轮云上 ViDoRe 消融完成~~（Visual 路 CUDA OOM，见教训）
- [x] 云上产物拉回本地（FAISS 索引 + PG dump）

### 🔜 下一步
1. **立即可做**: 本地 Docker 环境跑 `--skip-index` 验证评测可复现
2. **Visual 路修复**: 编码后卸载 ColPali 模型释放显存，让 MaxSim 能跑
3. **优化**: 本地补跑 Visual 路（已有 FAISS 索引，CPU MaxSim 可行）
4. **后续规划**: 参考 `docs/` 下 roadmap 文档

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
6. 如果是从零开始的新实例，需要先上传代码：
   ```bash
   tar czf /tmp/prism-rag.tar.gz --exclude='.venv' --exclude='.git' --exclude='runs' .
   sshpass -p '<pwd>' scp -P <port> /tmp/prism-rag.tar.gz root@<host>:/root/
   sshpass -p '<pwd>' ssh -p <port> root@<host> 'mkdir -p /root/prism-rag && cd /root/prism-rag && tar xzf /root/prism-rag.tar.gz'
