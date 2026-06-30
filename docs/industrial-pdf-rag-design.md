# 工业 PDF 多模态 RAG 系统 — 设计文档

> 独立项目，参考 WeKnora 架构思想 + ViDoRe 评测验证，面向 AI Agent 开发工程师面试。

---

## 1. 项目定位

**一句话**：借鉴腾讯 WeKnora 的多模态 RAG 架构，自研工业 PDF 智能检索问答系统，用 ViDoRe 国际基准量化验证，两阶段交付。

**目标岗位**：AI Agent 开发工程师

**独立仓库**：与 `finqa-rag-agent` 分属不同项目

---

## 2. 开发阶段

| | 第一阶段 | 第二阶段 |
|------|---------|---------|
| **核心交付** | 多模态解析 + 三路检索 + ViDoRe 评测 | + GraphRAG + ReACT Agent |
| **代码模块** | `ingestion/` `retrieval/` `evaluation/` `api/` `ui/` | + `graphrag/` `agent/` |
| **评测产物** | ViDoRe NDCG@10 + 消融表 | + Multi-hop 消融 + Agent 拆解正确率 |
| **面试 Demo** | 检索效果实时演示 + Leaderboard 对比 | + Agent 思考链可视化 + 知识图谱浏览 |
| **工期估计** | 2-3 周 | + 2-3 周 |

---

## 3. 架构设计

### 3.1 架构风格：逻辑分层单体

参考 WeKnora 五层解耦思想，但在同一 Python 进程内以模块边界实现分层——架构清晰可讲，无跨进程调试负担。

```
┌────────────────────────────────────────────────────────┐
│                    React 前端                           │
│         查询输入 / 结果展示 / 引用高亮 / 评测面板          │
└────────────────────┬───────────────────────────────────┘
                     │ REST API
┌────────────────────┴───────────────────────────────────┐
│                 FastAPI 路由层                          │
│       POST /search   POST /ingest   GET /eval           │
└────────────────────┬───────────────────────────────────┘
                     │ 同进程调用
┌────────────────────┴───────────────────────────────────┐
│                  Python 核心引擎                         │
│                                                         │
│  ┌──────────┐  ┌───────────┐  ┌──────────────────┐    │
│  │ingestion/│  │retrieval/ │  │  evaluation/      │    │
│  │          │  │           │  │  ViDoRe + RAGAS   │    │
│  │ PDF解析  │→│ BM25      │  │  评测脚本          │    │
│  │ 布局分析  │  │ Dense     │  └──────────────────┘    │
│  │ OCR      │  │ Visual    │                            │
│  │ 分块     │  │ 融合排序   │                            │
│  └──────────┘  └───────────┘                            │
│                                                         │
└──────────────┬──────────────────┬──────────────────────┘
               │                  │
     ┌─────────┴──┐     ┌────────┴─────┐
     │ pgvector    │     │   MinIO      │
     │ + BM25 索引  │     │  文档+图片    │
     └────────────┘     └──────────────┘
```

### 3.2 技术栈

| 层 | 技术 | 备注 |
|----|------|------|
| 前端 | React + TypeScript | UI 库待定 |
| 后端 | Python FastAPI | |
| 文档解析 | **MinerU**（主选）/ Surya（备选，单页轻量） | 2026 SOTA 工业级 Pipeline；淘汰旧版 LayoutLMv3 + PyMuPDF + PaddleOCR 三件套 |
| OCR 补充 | PaddleOCR | 仅对扫描页兜底 |
| 文本 Embedding | BGE-large-en-v1.5 | 英文，768 维 |
| 视觉 Embedding | **ColPali**（整页多向量 late interaction） | **不参与分块，按页建索引**；CLIP 仅用于 ingestion 期对 figure 块打辅助标签，不进检索路 |
| 文本/单向量存储 | pgvector | HNSW 索引，存 BGE 768 维向量 |
| ColPali 多向量存储 | **FAISS（IndexFlat / HNSW）** | pgvector 无原生 MaxSim 算子，多向量沿用 ColBERT 范式落 FAISS，进程内 MaxSim 后处理。生产用 HNSW32 + PQ 量化，Demo 7k 页 MaxSim 实测 <3s；POC 验证每页 ~0.5 MB / ~1031 patches × 128 维 |
| BM25 | rank_bm25 或 Elasticsearch | 第一版用 rank_bm25 |
| 对象存储 | MinIO | 文档 + 页面截图 + 图片块 |
| 重排 | cross-encoder（bge-reranker-large） | RRF 之后做 top-20 → top-5 |
| LLM (推理) | Ollama qwen2:7b（本地）/ DeepSeek API（可选） | Ollama 零成本 |
| LLM (抽取) | Ollama qwen2:7b → DeepSeek API 升级 | 第二阶段 GraphRAG 用 |
| 评测 | vidore-benchmark + RAGAS | |

### 3.3 硬件环境

- MacBook M 系列，Ollama Metal 推理
- 必要时切 DeepSeek API

### 3.4 容量与延迟预算

> 单机 M 系列 32GB 内存下的可行性评估；Demo 期不可回避的硬约束。以下数据基于 POC 实测（vidore/colpali-v1.3, MPS bfloat16, batch=4, 10 页合成 PDF）。

| 维度 | 规模估算 | 说明 |
|------|---------|------|
| ViDoRe 文本索引 | ~24,000 页 × 768 维 × 4B ≈ 70 MB | pgvector 单库轻量 |
| ViDoRe ColPali 索引 | ~24,000 页 × 1031 patch × 128 维 × 4B ≈ **12 GB** | POC 验证 ~0.5 MB/页；FAISS IndexFlat 单进程内存常驻 |
| Demo 知识库 ColPali 增量 | ~350 份 × 平均 20 页 ≈ **3.5 GB** | 与 ViDoRe 共进程时选载其一；在线只载 Demo |
| 在线查询延迟（含 MaxSim） | **naïve: ~205s**（24k pg）/ **~60s**（7k pg）<br/>**HNSW: <3s**（7k pg） | Ollama generate 不计；MaxSim 全表扫是瓶颈。naïve 矩阵乘 86ms/10pg 实测，HNSW 可降至秒级 |
| 冷启动编译 | 首次 query ~1s，后续 ~80ms | torch.mps 编译开销，预热一次即可消除 |
| Ollama qwen2:7b 常驻显存 | ~5 GB（Metal 统一内存） | 与 FAISS 共享内存压力 |
| ColPali 编码吞吐 | ~1.5 pg/s（MPS, bfloat16, batch=4, 1024×768） | 全量 24k 页约 4.5h 离线编码；7k 页 Demo 约 1.3h |

**优化备选**：
- ColPali 索引换 FAISS HNSW32 + PQ 量化，内存压到 1–2 GB（精度略降）；POC 验证 HNSW 近似搜索可代替精确全表扫，延迟从分钟级降至秒级。
- 全量 ViDoRe 评测离线跑、在线服务只载 Demo 350 份索引。POC 验证 Demo 7k 页索引仅 ~3.5 GB，FAISS 配置得当的情况下在线可用。
- 首次 query 冷启动 ~1s（torch.mps 编译开销），生产需另做一次 dummy query 预热，后续稳定。
- 复杂生成切 DeepSeek API，本地 Ollama 仅做抽取/judge。

---

## 4. 核心数据流

### 4.1 文档摄入 (ingest)

```
PDF 文件
  │
  ├─→ MinerU 解析 → 文本 / 表格 / 图片区域 + 页面截图（150 DPI）
  ├─→ PaddleOCR 兜底扫描页
  ├─→ [文本路] 按区域分块 → 每块产 BGE 向量（768维）
  │        ├─ 文本块   → BGE 向量
  │        └─ 表格块   → 文本化(行列+单元格) + BGE 向量
  │
  └─→ [视觉路] 整页 ColPali 编码（#patch 个 128 维向量，不切块）
          └─ image_url = 页面截图 URL（供前端渲染）

入库：
  ├─ pgvector:        chunk_id, chunk_text, chunk_type, bge_vector, page_id
  └─ FAISS(进程内):   page_id → ColPali multivector list
                      query 时 MaxSim 全表扫 → 取 Top-N 页
                      命中页 → 反查该页所有 BGE chunk 拼上下文
```

**设计要点**：
- ColCali 整页检索 ≠ 分块检索；切图块会破坏 patch 间 late interaction，因此 Visual 路**按页编码、按页建索引**。
- 命中视觉页后，回到 pgvector 反查该页的文本/表格 chunk，拼出 grounding 上下文——这是 Visual 路给 RAG 提供 grounding 的工程接缝。

**性能实测（POC）**：
- ColPali 编码吞吐约 **1.5 pg/s**（MPS bfloat16, batch=4, 1024×768 页面）。全量 ViDoRe 24k 页约 4.5h 离线完成，Demo 7k 页约 1.3h。建议用 tmux 后台批量运行，避免中断。
- 首次编码有 torch.mps 编译开销（~30s），后续 batch 稳定在 ~0.7s/pg。

### 4.2 在线检索 (search)

```
用户查询 "What is the load capacity of the conveyor belt?"

  ├─→ 第一路 BM25：查询词 → 关键词匹配 → Top-20（chunk 级）
  ├─→ 第二路 Dense：BGE encode 查询 → pgvector HNSW → Top-20（chunk 级）
  └─→ 第三路 Visual：ColPali encode 查询 → FAISS MaxSim → Top-20（页级）
        └─ Visual 命中的页 → 反查 pgvector 该页所有 BGE chunk 纳入候选集

  └─→ RRF 融合三路 Top-20（chunk 评分对齐） → Top-20
      └─→ cross-encoder rerank → Top-5 返回
```

**融合策略阶段性切换**：
- 第一阶段：**RRF**（`score = Σ 1/(k+rank)`，非参数，无可调权重，工程上稳）。
- 第二阶段：GraphRAG 路加入后切换为**凸加权** `Σ w_i · s_i`，权重 `{bm25, dense, visual, graph}` 可调，由消融实验定标。

### 4.3 检索返回格式

```json
{
  "query": "...",
  "results": [
    {"chunk_id": "page_042_text_03", "score": 0.95, "type": "text", "content": "..."},
    {"chunk_id": "page_042_figure_01", "score": 0.88, "type": "figure", "image_url": "..."}
  ],
  "retrieval_trace": {
    "bm25_top5": ["page_042", "page_015", ...],
    "dense_top5": ["page_042", "page_108", ...],
    "visual_top5": ["page_042", "page_067", ...]
  }
}
```

**设计要点**：返回 `type` 标记（text/table/figure）前端差异化渲染；返回 `retrieval_trace` 展示每条通路贡献。

---

## 5. 评测体系

### 5.1 三层评测

| 层 | 测什么 | 用什么 | 时机 |
|----|--------|--------|------|
| 检索层 | 文档找没找对 | ViDoRe（NDCG@10, Recall@5） | 第一阶段 |
| 生成层 | 答案有没有瞎编 | RAGAS faithfulness（Ollama qwen2:7b 当 judge） | **第一阶段轻量版**（20 条拒答抽样） → 第二阶段全量 |
| 端到端 | 答案对不对、该不该回答 | 自建 50 QA 对 + 20 条拒答 | 第二阶段全量（第一阶段先吃 20 条拒答做 sanity） |

> 写入轻量版生成层评测的目的：让"评测驱动迭代"叙事在第一阶段就自洽——光有 ViDoRe NDCG 不等于系统好用，至少要有"答案没瞎编"这一层兜底。Judge 用 Ollama 同模型自评存在已知偏好风险，第二阶段会切到 DeepSeek-V3 / GPT-4o-mini 异家族做复评。

### 5.2 ViDoRe 评测流程

```
python -m evaluation.vidore_eval \
    --dataset vidore_v3_industrial \
    --output results/industrial.json

数据流：
  ViDoRe 数据集 (HuggingFace)
    ├── corpus: 3000+ 页 PDF 图片 + OCR 文本
    ├── queries: 300+ 条人工标注查询
    └── qrels: 标准答案（哪些页相关）
        │
        ▼
  ingestion pipeline → 建索引到 pgvector + FAISS
        │
        ▼
  retrieval pipeline → 每个查询返回 Top-10
        │
        ▼
  vidore-benchmark → 自动算分（NDCG@5/10, Recall@5/10, MRR）
```

**与 vidore-benchmark 的对接方式**：
- 适配 `BaseBeIRRetriever` 子类 → 实现 `search(queries, k)` 接口，把 BM25/Dense/Visual + RRF + rerank 整条 pipeline 包成统一检索器。
- corpus 端按 ViDoRe 提供的 page 图片喂 ColPali，按附带 OCR 文本喂 BGE；不私自重切 corpus，保证结果可对比 leaderboard。
- 评测只在离线 pipeline 跑，与在线服务共用 `retrieval/` 模块、**不复用在线服务的索引实例**，避免评测和 Demo 互相干扰。

### 5.3 消融实验

| 配置 | 维度 | 预期 | 面试话术 |
|------|------|------|---------|
| 纯 BM25 | 路由增量 | NDCG@10 最低 | "关键词盲区：语义相近但用词不同的查询全漏" |
| 纯 Dense | 路由增量 | 中 | "语义盲区：设备编号 XJ-203 这种硬匹配不够" |
| 纯 Visual | 路由增量 | 中 | "能看图，但文字密集页面区分度不够" |
| BM25 + Dense | 路由增量 | 中高 | "文本双路已基本覆盖，但图表查询仍有差距" |
| **BM25 + Dense + Visual** | 路由增量 | **最高** | "三路互补，每路都有不可替代的增量" |
| 三路 RRF（无 rerank） | 重排增量 | 中高 | "RRF 已能融合，但分数粒度粗" |
| **三路 RRF + cross-encoder rerank** | 重排增量 | **更高** | "重排把粗排错位的边界 case 拉回 Top-5" |

**消融读法**：上半表证明三路检索各自不可或缺；下半表证明 rerank 的工程增量。两组分开报，避免被面试官追问"你的提升到底是检索的功劳还是重排的功劳"。

### 5.4 面试评测产物

1. **Leaderboard 对比**：你的系统 vs ColPali / ColQwen / nemo-colembed
2. **消融表**：三路各自贡献
3. **案例卡**：5 个典型查询（BM25 立功 / Dense 立功 / Visual 立功 / 三者共同 / 全失败）

---

## 6. 数据策略

### 6.1 数据全景

| 用途 | 数据 | 量 | 来源 |
|------|------|-----|------|
| **检索评测** | ViDoRe v3 (8 个 domain) | ~24,000 页 + 3,099 查询 | HuggingFace 下载 |
| **Demo 知识库** | RealKIE 合同/发票/NDA | ~200 份 | GitHub 开源下载 |
| **Demo 知识库** | CHIC 发票/采购单 | ~100 份 | GitHub 开源下载 |
| **Demo 知识库** | Siemens 设备手册 | ~50 份 | 公开采集 |
| **自建评测** | 拒答测试集 | 20 条 out-of-scope | 自己写 + LLM 辅助 |
| **自建评测** | 端到端 QA 对 | 50 条 | 人工标注 |

### 6.2 数据就位顺序

1. 下载 ViDoRe Industrial 单子集 → 跑通检索评测闭环
2. 下载 RealKIE + CHIC → 搭建 Demo 知识库
3. 下载 Siemens 手册 → 丰富工业场景
4. 自建 20 条拒答 + 50 条 QA → 端到端评测

### 6.3 开发中快速验证

| 阶段 | 测什么 | 用什么数据 | 反馈速度 |
|------|--------|-----------|---------|
| 解析管道 | 文本提取质量 | 1 个复杂 PDF + 肉眼 | 秒级 |
| 分块策略 | 语义完整性 | 10 个 PDF + 人工检查 | 分钟级 |
| Embedding | 语义匹配 | 自建 50 对验证集 | 1 分钟 |
| 检索效果 | 单路/融合 | ViDoRe Industrial 单子集 | 5 分钟 |
| 里程碑 | 全量 | ViDoRe 全部 8 子集 | 1 小时 |

---

## 7. 第二阶段预留

### 7.1 GraphRAG

第一阶段代码预留：

- `chunk` 表加 `post_processed` 字段（默认 false），第二阶段实体抽取后置 true
- `retrieval/` 融合函数抽象成 `FusionStrategy` 接口，第一阶段注册 `RRFFusion`，第二阶段注册 `ConvexFusion`，**不在调用点 if/else 硬切**：
  ```python
  # 第一阶段
  fusion = RRFFusion(k=60)
  # 第二阶段切换
  fusion = ConvexFusion(weights={"bm25":0.3,"dense":0.45,"visual":0.2,"graph":0.15})
  # graph 权重从 0 → 0.15，由消融实验定标
  ```

第二阶段实现：

- Asynq/Redis 异步任务 → Ollama qwen2:7b 实体提取 + 关系抽取
- Neo4j 存储三元组 → retrieval/ 新增 `graph_search()` → 融合公式加第四项
- 抽取质量不够时切 DeepSeek API

### 7.2 ReACT Agent

第一阶段预留：

- `/search` 和未来的 `/agent/search` 共用同一个 `retrieval/` 模块
- Agent 只新增编排层，不修改检索核心

第二阶段实现：

- ReACT 循环：Thought → Action (调用 search/graph_search) → Observation → 反思迭代
- 工具集（刻意收窄到私有知识域）：
  - `knowledge_search`：调用第一阶段 retrieval pipeline
  - `query_knowledge_graph`：对 Neo4j 发 Cypher 查询
  - `query_refiner`：多跳查询改写（把"上述比率"这种指代在多轮内显式解析）
- **不引入 `web_search`**：项目核心是私有知识库 RAG，外网接入会带来 prompt injection、内容去重、来源可信度等次生问题，超出 demo 工程边界；如面试问及"能否扩展 web 检索"，作为 roadmap 而非已实现点回答。
- 前端可视化：Agent 思考链展示

---

## 8. 成本和部署

### 8.1 零成本方案

| 组件 | 免费替代 |
|------|---------|
| LLM | Ollama qwen2:7b |
| Embedding | Ollama BGE-large-en-v1.5 |
| 向量存储 | pgvector |
| 图存储 | Neo4j Community |
| 对象存储 | MinIO |
| 文档解析 | PyMuPDF + PaddleOCR（均开源） |

月费：¥0（电费除外）。

### 8.2 低成本升级

日常用 Ollama 本地模型，复杂查询投递 DeepSeek API：日 100 次查询 ≈ ¥3/月。

### 8.3 部署

单机 Docker Compose：
```yaml
services:
  api:        # FastAPI
  frontend:   # React (nginx 静态服务)
  db:         # PostgreSQL + pgvector
  minio:      # 对象存储
  ollama:     # 本地 LLM
  neo4j:      # 第二阶段
```

---

## 9. 项目叙事

### 对外一句话

> 借鉴腾讯 WeKnora 的多模态混合检索架构，在 ViDoRe 国际基准上验证效果，多模态解析管道全自研，分层评测驱动迭代。

### 面试话术

**架构**：
> 「架构设计参考腾讯 WeKnora 的多模态解析管道和 BM25+Dense+Visual 三路检索思想。采用逻辑分层单体——模块独立有清晰边界，但不引入 gRPC 跨进程通信开销。第一阶段聚焦检索层，第二阶段加 GraphRAG 和 ReACT Agent。」

**评测**：
> 「用 ViDoRe Industrial 基准做量化验证。NDCG@10 达到 xx，消融实验证明三路检索各自不可或缺——纯 BM25 只有 0.xx，加语义检索到 0.xx，加视觉检索才突破 0.xx；再叠 cross-encoder 重排还能再涨 xx 个点。我不只报系统总分，而是把检索路增量与重排增量分开报，避免提升来源说不清。第一阶段除 ViDoRe 外还跑 20 条拒答的 RAGAS faithfulness sanity——只看 NDCG 高不代表系统好用。」

**数据**：
> 「数据分三块。评测用 ViDoRe——有公开 Leaderboard，谁都可以复现。知识库用 RealKIE 合同/发票和 Siemens 设备手册——真实业务文档，不是玩具数据。自建 50 条 QA 和 20 条拒答用例做端到端验证。」

**成本**：
> 「整个系统零成本搭建。LLM 用 Ollama 本地 Qwen2-7B，Embedding 用 BGE，文本向量存 pgvector，ColPali 多向量存 FAISS 进程内 MaxSim，对象存储用 MinIO。生产化只需把 LLM 切到 DeepSeek API，月成本控制在 ¥50 以内。我做了一次容量预算：ViDoRe 全量 ColPali 索引约 12GB，单机 32GB 内存跑得动但在线 QPS 不高，Demo 期可接受。」

---

## 10. 已确认的设计决策清单

| # | 决策 | 结论 |
|---|------|------|
| 1 | 与 finqa-rag-agent 关系 | 独立仓库 |
| 2 | 目标岗位 | AI Agent 开发工程师 |
| 3 | 开发节奏 | 先检索层，后 GraphRAG + Agent |
| 4 | 多模态解析 | MinerU 为主选（淘汰 LayoutLMv3 + PyMuPDF + PaddleOCR 三件套） |
| 5 | 文档语言 | 纯英文 |
| 6 | Agent 方案 | ReACT 循环 |
| 7 | GraphRAG 抽取 | 先 Ollama 本地，后 DeepSeek API |
| 8 | 硬件 | MacBook M 系列 + 可选 DeepSeek API |
| 9 | 前端 | React |
| 10 | 第一阶段评测 | 检索层 ViDoRe **+ 生成层 20 条拒答 RAGAS sanity** |
| 11 | 架构风格 | 逻辑分层单体（方案 C） |
| 12 | Visual 检索范式 | **ColPali 整页多向量**（不用图块 CLIP 检索） |
| 13 | 多向量存储 | FAISS 进程内 MaxSim（pgvector 无原生算子） |
| 14 | 文本向量融合 | 第一阶段 RRF，第二阶段 GraphRAG 加入后切凸加权 |
| 15 | 重排 | cross-encoder（bge-reranker-large），消融单独报增量 |
| 16 | Agent 工具集 | 收窄到私有知识域，**不引入 web_search** |
| 17 | 容量预算 | ViDoRe 全量 ColPali ~12GB，单机 32GB 可跑 |
