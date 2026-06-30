# 工业 PDF 多模态 RAG 系统 — 架构总览

> 单图版架构总览。配套根设计 `industrial-pdf-rag-design.md`（详细条款、决策清单、面试话术）阅读。
> 本文只画图与速读表，所有"为什么这么选"的论证一律回查设计文档对应章节。

---

## 1. 总览图

```mermaid
flowchart TB
    classDef data fill:#e8f5e9,stroke:#388e3c,color:#1b5e20
    classDef offline fill:#fff3e0,stroke:#f57c00,color:#e65100
    classDef online fill:#e3f2fd,stroke:#1976d2,color:#0d47a1
    classDef visual fill:#f3e5f5,stroke:#7b1fa2,color:#4a148c
    classDef judge fill:#fce4ec,stroke:#ad1457,color:#880e4f
    classDef store fill:#f5f5f5,stroke:#616161,color:#212121
    classDef phase2 fill:#e0f7fa,stroke:#00838f,color:#006064
    classDef out fill:#fffde7,stroke:#f9a825,color:#f57f17

    subgraph OFF["离线一次性 / 离线批量 (§4.1)"]
        direction TB

        subgraph RAW["原始文档 (§6.1)"]
            VID[ViDoRe v3 industrial<br/>~24,000 页]:::data
            RK[RealKIE 200 份<br/>合同/发票]:::data
            CHIC[CHIC 100 份]:::data
            SIE[Siemens 设备手册 50 份]:::data
        end

        subgraph INGEST["文档摄取 ingestion"]
            direction TB
            MINER["MinerU 解析<br/>文本/表格/图片区域 + 页面截图 150 DPI"]:::offline
            OCR["PaddleOCR 兜底扫描页"]:::offline
            TXT["文本路: 按区域分块<br/>文本块 / 表格块(文本化)"]:::offline
            BGE["BGE-large-en-v1.5 encode<br/>768 维"]:::offline
            COLP["视觉路: ColPali 整页 encode<br/>#patch × 128 维 (不切块)"]:::visual
        end

        subgraph STORE["存储 (§3.2 / §3.4)"]
            direction TB
            PG[(pgvector<br/>chunk + bge_vector<br/>HNSW)]:::store
            FAISS[(FAISS 进程内<br/>page_id → ColPali multivector<br/>MaxSim 后处理)]:::visual
            MINIO[(MinIO<br/>PDF + 页面截图 + 图片块)]:::store
        end
    end

    subgraph ON["在线每次问句 (§4.2)"]
        direction TB

        U[用户问句]:::out

        subgraph RET["三路检索 + 融合 + 重排"]
            direction TB
            BM25["BM25 路<br/>rank_bm25 → Top-20 chunk"]:::online
            DENSE["Dense 路<br/>BGE encode → pgvector HNSW<br/>→ Top-20 chunk"]:::online
            VIS["Visual 路<br/>ColPali encode → FAISS MaxSim<br/>→ Top-20 页"]:::visual
            GROUND["Visual grounding 接缝<br/>命中页 → 反查 pgvector 该页<br/>所有 BGE chunk 纳入候选"]:::visual
            FUSION["RRF 融合 (第一阶段)<br/>score = Σ 1/(k+rank)"]:::online
            RERANK["cross-encoder rerank<br/>bge-reranker-large<br/>Top-20 → Top-5"]:::online
        end

        CTX["拼 grounding 上下文<br/>Top-5 chunk + 对应页面截图 URL"]:::out
        GEN["LLM 生成<br/>Ollama qwen2:7b / DeepSeek API"]:::online
        ANSWER[回答 + 引用 page_id+chunk_id<br/>+ 页面截图回链]:::out
    end

    subgraph EVAL["评测 (§5)"]
        direction TB
        VDR["ViDoRe leaderboard<br/>NDCG@5/10, Recall@5/10, MRR<br/>via BaseBeIRRetriever 适配"]:::judge
        RAGAS["RAGAS faithfulness<br/>Ollama qwen2:7b 当 judge<br/>第二阶段切异家族复评"]:::judge
        E2E["端到端 QA<br/>50 QA + 20 拒答"]:::judge
        ABL["消融实验<br/>路由增量 + 重排增量分开报"]:::judge
    end

    subgraph P2["第二阶段预留 (§7)"]
        direction TB
        GRAPH["GraphRAG<br/>Neo4j 实体/关系图谱"]:::phase2
        AGENT["ReACT Agent<br/>knowledge_search /<br/>query_knowledge_graph /<br/>query_refiner"]:::phase2
        CONVEX["凸加权 fusion<br/>weights bm25/dense/visual/graph<br/>消融定标"]:::phase2
    end

    %% 离线连接
    VID --> MINER
    RK --> MINER
    CHIC --> MINER
    SIE --> MINER
    MINER --> OCR
    MINER --> TXT
    MINER --> COLP
    TXT --> BGE
    BGE --> PG
    COLP --> FAISS
    MINER --> MINIO

    %% 在线连接
    U --> BM25
    U --> DENSE
    U --> VIS
    VIS --> GROUND
    GROUND --> FUSION
    BM25 --> FUSION
    DENSE --> FUSION
    PG -.反查该页 chunk.-> GROUND
    FUSION --> RERANK
    RERANK --> CTX
    CTX --> GEN
    GEN --> ANSWER

    %% 评测连接
    ANSWER -.cases.-> RAGAS
    ANSWER -.cases.-> E2E
    RET -.retrieval.-> VDR
    RET -.ablation.-> ABL

    %% 第二阶段扩展
    GRAPH --> CONVEX
    AGENT --> BM25
    AGENT --> DENSE
    AGENT --> VIS
    AGENT --> GRAPH
    CONVEX -.替换.-> FUSION
```

---

## 2. 关键节点速读

### 离线：原始文档 → 双路索引

| 节点 | 干啥 | 锚点 |
|------|------|------|
| MinerU 解析 | 2026 SOTA 工业级 Pipeline，替代旧版 LayoutLMv3+PyMuPDF+PaddleOCR 三件套 | §3.2 / §4.1 |
| 文本路分块 | 文本块 + 表格块(行列+单元格 文本化)，喂 BGE | §4.1 |
| 视觉路 ColPali | **按页** encode，产出多向量；不切块——切图块会破坏 patch 间 late interaction | §3.2 / §4.1 |
| pgvector | 存 BGE 768 维单向量，HNSW 索引 | §3.2 |
| FAISS | 存 ColPali 多向量，进程内 MaxSim；pgvector 无原生 MaxSim 算子 | §3.2 |

### 在线：问句 → 三路 → 融合 → 重排 → 生成

| 节点 | 干啥 | 锚点 |
|------|------|------|
| BM25 / Dense | chunk 级 Top-20，硬匹配 + 语义两条 | §4.2 |
| Visual 路 | **页级** Top-20，命中后回 pgvector 反查该页所有 BGE chunk | §4.1 接缝 |
| RRF 融合 | 第一阶段非参数 `Σ 1/(k+rank)`，无可调权重，工程上稳 | §4.2 |
| cross-encoder rerank | bge-reranker-large，Top-20 → Top-5 | §3.2 / §4.2 |
| 生成 | Ollama qwen2:7b 本地，复杂场景切 DeepSeek API | §3.2 |

### 评测：三层 + 消融

| 节点 | 干啥 | 锚点 |
|------|------|------|
| ViDoRe | 检索层 NDCG/Recall/MRR，适配 `BaseBeIRRetriever`，不私自重切 corpus | §5.2 |
| RAGAS faithfulness | 生成层"答案没瞎编"，第一阶段先吃 20 条拒答 sanity | §5.1 |
| 端到端 QA | 50 QA + 20 拒答，第二阶段全量 | §5.1 |
| 消融 | 路由增量 + 重排增量两组分开报 | §5.3 |

---

## 3. Visual 路 grounding 接缝（项目最关键的工程接缝）

```mermaid
flowchart LR
    classDef q fill:#e3f2fd,stroke:#1976d2,color:#0d47a1
    classDef v fill:#f3e5f5,stroke:#7b1fa2,color:#4a148c
    classDef t fill:#e8f5e9,stroke:#388e3c,color:#1b5e20
    classDef out fill:#fffde7,stroke:#f9a825,color:#f57f17

    Q[用户问句]:::q --> CE[ColPali encode 查询]:::v
    CE --> MS[FAISS MaxSim 全表扫]:::v
    MS --> TOP[Top-20 页]:::v
    TOP --> REV[反查 pgvector<br/>WHERE page_id IN (命中页)]:::t
    REV --> CH[该页所有 BGE chunk]:::t
    CH --> CAND[纳入 RRF 候选集<br/>与 BM25/Dense 路对齐评分]:::out

    %% 接缝的工程价值
    style REV stroke-width:3px
```

**为什么这条是接缝**：ColPali 只能告诉你"哪一页相关"，不能直接喂生成；必须回到 pgvector 把那一页的文字/表格 chunk 捞出来拼上下文。这一步决定了 Visual 路"看得见图"能不能落地为"答得出字"。面试常在此处追问。

---

## 4. 融合策略阶段性切换（§4.2 / §7.1）

```mermaid
flowchart LR
    classDef p1 fill:#e3f2fd,stroke:#1976d2,color:#0d47a1
    classDef p2 fill:#e0f7fa,stroke:#00838f,color:#006064

    subgraph P1["第一阶段"]
        RRF["RRFFusion<br/>score = Σ 1/(k+rank)<br/>非参数 / 无可调权重"]:::p1
    end
    subgraph P2["第二阶段 (GraphRAG 加入后)"]
        CONV["ConvexFusion<br/>Σ w_i · s_i<br/>weights 可调"]:::p2
    end
    P1 -->|graph 路就绪| P2
```

- 用 `FusionStrategy` 接口抽象，调用点不 if/else 硬切。
- 第二阶段权重 `{bm25:0.3, dense:0.45, visual:0.2, graph:0.15}` 由消融定标，不接受拍脑袋。
- RRF → 凸加权不是因为 RRF 错，而是 graph 路得分尺度与前三路不同，需要可调权重做对齐。

---

## 5. 评测分层（§5.1）

```mermaid
flowchart TB
    classDef lay1 fill:#e8f5e9,stroke:#388e3c,color:#1b5e20
    classDef lay2 fill:#fff3e0,stroke:#f57c00,color:#e65100
    classDef lay3 fill:#fce4ec,stroke:#ad1457,color:#880e4f

    L1["检索层<br/>ViDoRe NDCG@10 / Recall@5<br/>第一阶段"]:::lay1
    L2["生成层<br/>RAGAS faithfulness<br/>第一阶段 20 条拒答 sanity → 第二阶段全量<br/>第二阶段切异家族 judge 复评"]:::lay2
    L3["端到端<br/>50 QA + 20 拒答<br/>第二阶段全量"]:::lay3

    L1 --> L2 --> L3
```

**叙事自洽点**：第一阶段不能只看 ViDoRe NDCG——NDCG 高不等于系统好用。20 条拒答 RAGAS 是让"评测驱动迭代"站得住脚的最小必要条件，judge 自评偏好风险已在文档显式承认。

---

## 6. 消融实验读法（§5.3）

```mermaid
flowchart TB
    classDef route fill:#e3f2fd,stroke:#1976d2,color:#0d47a1
    classDef rerank fill:#fff3e0,stroke:#f57c00,color:#e65100
    classDef best fill:#e8f5e9,stroke:#388e3c,color:#1b5e20

    subgraph ROUTE["路由增量组 — 证明三路各自不可或缺"]
        direction TB
        R1[纯 BM25<br/>关键词盲区]:::route
        R2[纯 Dense<br/>设备编号硬匹配弱]:::route
        R3[纯 Visual<br/>文字密集页区分度低]:::route
        R4[BM25 + Dense<br/>图表查询有差距]:::route
        R5[BM25 + Dense + Visual<br/>三路互补最高]:::best
    end

    subgraph RER["重排增量组 — 证明 rerank 的工程增量"]
        direction TB
        RR1[三路 RRF 无 rerank]:::rerank
        RR2[三路 RRF + cross-encoder rerank<br/>粗排错位的边界 case 拉回 Top-5]:::best
    end

    R5 -.同系统两种配置.-> RR1
```

**两组分开报**：避免被面试官追问"你的提升到底是检索的功劳还是重排的功劳"。这是评测纪律，不是图表装饰。

---

## 7. 容量与延迟预算速读（§3.4）

| 维度 | 规模 | 单机 32GB 可行性 |
|------|------|----------------|
| ViDoRe 文本索引 | ~70 MB | ✅ 轻量 |
| ViDoRe ColPali 索引 | ~12 GB | ⚠️ 主要内存压力源 |
| Demo 知识库 ColPali 增量 | ~7 GB | 与 ViDoRe 共进程 |
| 单 query 延迟 | 2–5 s | ⚠️ MaxSim 全表扫是瓶颈 |
| Ollama qwen2:7b 常驻 | ~5 GB | 与 FAISS 共享内存 |

**降内存备选**：FAISS HNSW / PQ 量化 → 1–2 GB（精度略降）；在线服务只载 Demo 350 份索引，ViDoRe 离线跑。

---

## 8. 阶段路线图（§2）

```mermaid
flowchart LR
    classDef p1 fill:#e3f2fd,stroke:#1976d2,color:#0d47a1
    classDef p2 fill:#e0f7fa,stroke:#00838f,color:#006064
    classDef p3 fill:#fff3e0,stroke:#f57c00,color:#e65100

    P1["第一阶段<br/>检索层 MVP<br/>ViDoRe + 20 条拒答 sanity"]:::p1
    P2["第二阶段<br/>GraphRAG + ReACT Agent<br/>凸加权切换"]:::p2
    P3["第三阶段<br/>端到端评测全量<br/>50 QA + 20 拒答"]:::p3
    P1 --> P2 --> P3
```

---

## 9. 与设计文档章节对照

| 总览图区块 | 设计文档锚点 |
|------------|--------------|
| §1 总览图 离线 ingest | §4.1 文档摄入 |
| §1 总览图 在线 search | §4.2 在线检索 |
| §1 总览图 评测 | §5 评测体系 |
| §1 总览图 第二阶段 | §7 第二阶段预留 |
| §2 速读表 | §3.2 / §4.1 / §4.2 / §5 |
| §3 Visual grounding 接缝 | §4.1 设计要点 |
| §4 融合策略切换 | §4.2 / §7.1 |
| §5 评测分层 | §5.1 |
| §6 消融读法 | §5.3 |
| §7 容量预算 | §3.4 |
| §8 阶段路线图 | §2 开发阶段 |

---

## 10. 与隔壁 `architecture-overview.md` 的关系

`architecture-overview.md` 是 **finqa-rag-agent（财报问答）** 项目的总览图，配套根 spec `2026-06-29-finqa-rag-agent-design.md`，与本文件**不是同一项目**。两份文档刻意分开维护，避免工业 PDF 项目与财报项目的技术选型互相污染。两份的共同点仅在于：都强调"评测驱动迭代"、都用 RRF+cross-encoder rerank、都把 Agent 工具集刻意收窄——这些是同一作者的工程纪律延续，不是项目耦合。