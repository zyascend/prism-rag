# MRAG 综述优化思路对照分析 — 全量 ROI 清单

> **日期：** 2026-07-25  
> **来源论文：** Lang Mei et al., *A Survey on Multimodal Retrieval-Augmented Generation*, arXiv:2504.08748v1（华为云，80 页）  
> **本地 PDF：** `/Users/theyang/Documents/pdfs/2504.08748v1.pdf`  
> **性质：** 对照 PrismRAG 的**借鉴价值分析**（含高/中/低 ROI 与明确不借）；非单一功能 Spec。  
> **关联：**  
> - 瓶颈诊断：[`docs/bottleneck-analysis-2026-07-07.md`](../bottleneck-analysis-2026-07-07.md)  
> - 入库主线：[`docs/superpowers/plans/2026-07-23-content-pipeline-phase-ab-roadmap.md`](../superpowers/plans/2026-07-23-content-pipeline-phase-ab-roadmap.md)  
> - P0/P1 实现方案：[`docs/superpowers/plans/2026-07-25-p0-p1-refiner-planning-impl.md`](../superpowers/plans/2026-07-25-p0-p1-refiner-planning-impl.md)  
> - 云验收脚本：`scripts/cloud_boot_r.sh`  
> - 阴性实验：`runs/20260722-crag-on/` · `runs/20260721-self-rag-on-only/`

---

## 0. TL;DR

这篇综述对 PrismRAG **最有用的不是「再上一个 multimodal 模型」**，而是把系统语言统一为：

| 综述概念 | 含义 | PrismRAG 映射 |
|----------|------|----------------|
| **Search Planning** | 是否检索、走哪条模态 | `search_planner` / VisualRouter |
| **Retriever → Reranker → Refiner** | 召回 / 精排 / **入模前加工** | 三路+RRF+zerank2 / **refiner** |
| **Extraction ∥ Representation** | 文本结构 + 页截图双轨 | BM25+Dense + ColQwen2 |
| **Adaptive vs Fixed pipeline** | 固定三路会伤正确率/成本 | 强制 Visual / 默认 CRAG 已踩坑 |

**一句话：** 从「固定三路 + 硬门控」推进到「结构更完整的入库 + 自适应选路 + 检索后 Refiner」；多模态截图路**按需**启用。

**与现有实验一致：**

- NDCG 不差但 CtxRel/Correct 吃亏 → 瓶颈在 **chunk/上下文**，不是再叠 LLM 门。  
- CRAG 硬滤：CtxRel↑、Correct↓、延迟×3 → **机制有效 ≠ 有用**。  
- Gate2：Faith 仅边际，Correct 主错仍在检索/错 chunk。

---

## 1. 论文在讲什么（压缩版）

### 1.1 MRAG 三代演进

| 代 | 核心 | 局限（论文观点） | PrismRAG 位置 |
|----|------|------------------|---------------|
| **MRAG1.0**「伪多模态」 | 多模态→caption 再走纯文本 RAG | 结构/细节丢失；解析重 | 早期表/图变文字思路；**已超越** |
| **MRAG2.0** | 保留原模态 + 跨模态检索 + MLLM 生成 | 跨模态检索仍弱；多模态如何组织进 prompt 难 | **接近现状**：text/table + Col* 页图 + 表摘要 |
| **MRAG3.0** | 页截图索引 + **Search Planning** + 可选多模态输出 | 规划与评测仍不成熟 | 截图索引有；**规划曾弱**；多模态输出暂不需要 |

### 1.2 五个技术组件（§3）

1. **Multimodal Document Parsing & Indexing** — Extraction-based vs Representation-based  
2. **Multimodal Search Planning** — Fixed vs Adaptive；`a_none / a_text / a_image`  
3. **Multimodal Retrieval** — Retriever / Reranker / **Refiner**  
4. **Multimodal Generation** — 文本答案 vs 原生/增强多模态输出  
5. **Datasets & Evaluation** — VQA / 文档 / 工业等（偏学术基准）

### 1.3 论文点名的关键教训（与我们数据同构）

1. **强制 image-to-image 检索可引入误导图**，反而伤 MLLM（综述引 [125] 等）。  
2. **并非所有 query 都需要检索**（mR2AG 等）。  
3. **LLM 对输入质量极敏感**；解析+检索丢信息 → 噪声进生成。  
4. **Refiner（检索后压缩/蒸馏）被低估**，却是长上下文落地关键。  
5. **改写/规划需自适应**；固定 rewrite 易漂。  
6. 评测要分层、要成本；客观题堆砌 ≠ 工业文档 QA。

---

## 2. PrismRAG 现状锚点（分析前提）

### 2.1 架构快照

```text
PDF → Parse (simple/MinerU) → Chunk (text|table|image)
    → TableSummarizer(可选 context) → BGE + BM25 + Col*(page)
Query → [SearchPlan?] → BM25 ∥ Dense ∥ Visual?
      → RRF → [modality_boost?] → [neighbor/crossref expand?]
      → Rerank (BGE/zerank2)
      → [CRAG? 默认关]
      → Refiner (bge | soft_rank | llm…) → LLM
      → [Gate2? 默认关]
```

### 2.2 关键实测结论（决策约束）

| 结论 | 证据指针 |
|------|----------|
| 主矛盾：错 page/错 chunk / 上下文噪声 | handoff；bottleneck 报告；E2E badcase |
| Rerank 是 L1 主增益 | Boot-A：no_rerank → zerank2 Δ NDCG@10 ≈ +0.11 |
| Visual_only 弱 | 历史 NDCG@10 ≈ 0.16；强制开有噪声风险 |
| CRAG 默认禁开 | Correct −12pt；latency ×3.1；误拒↑ |
| Gate2 默认关 / 可 low_rerank 试 | Faith +0.9pt 边际；Correct +0.02 |
| Content Pipeline A/B 代码齐 | Boot-CP page 级 NDCG 三臂同 0.3575；默认 expand/boost 关 |
| Goal-A（ON 索引） | 283q NDCG@10 0.5337；E2E Correct 0.66 / Reject 0.95 |

### 2.3 分析时的产品边界

- **域：** 工业 PDF 手册 QA（表、参数、拒答），非通用 VQA/创意多模态输出。  
- **成本：** 云 GPU 按小时计；禁止默认叠贵链路。  
- **评测否决项：** E2E Correct、Faith、误拒、latency；**CtxRel 不作上线否决**（P09）。

---

## 3. 全量优化思路 ROI 总表

### 图例

| 状态 | 含义 |
|------|------|
| ✅ 已落地 | 代码在仓库；默认见备注 |
| 🔶 部分 | 有相近能力，未完全按综述形态 |
| ⏳ 待验 | 已实现，等 Boot-R / 云数字 |
| 📋 规划 | 仅分析/路线图，未做或低优先 |
| 🚫 不做 | 明确排除 |

### 3.1 高 ROI（应做 / 已在做）

| ID | 思路 | 综述锚点 | 对 PrismRAG 的价值 | 状态 | 默认 |
|----|------|----------|-------------------|------|------|
| **H1** | **Refiner 正式化**（query-focused 句级 soft_rank，表保护） | §3.3.3 RECOMP / FILCO / LongLLMLingua / CPC / AdaComp | 直接打「找对页、喂错句」；比硬 CRAG 安全 | ✅ + ⏳ Boot-R | `mode=bge`；`soft_rank` 待阳性 |
| **H2** | **Adaptive Search Planning**（规则 `text/visual/none`） | §3.2 OmniSearch / mR2AG；强制 image 有害 | 降 Visual 噪声与延迟 | ✅ + ⏳ | `enabled=false` |
| **H3** | **Extraction ∥ Representation 双轨 + 融合** | §3.1.2 并行 OCR+截图再 fuse | 已是主架构；继续而非推倒 | 🔶 已有三路 | 三路默认开（planning 可关 visual） |
| **H4** | **层级元数据 / 邻居 expand**（section、prev/next、page expand） | §7.1 Contextual Relationship Preservation | 跨段/半截表 | ✅ Phase A3+B1 | expand 默认关 |
| **H5** | **上下文感知表摘要 + 类型化 content_list** | §7.1 Advanced Captioning；结构保留 | 表题 miss / 错表 | ✅ Phase A1–A2 | context 默认关 |
| **H6** | **Cross-ref expand**（见表/见图） | 结构关系 / 交叉引用（future + 工程补全） | 引用句 hit 但表体未进 top-k | ✅ + ⏳ | `enabled=false` |
| **H7** | **强精排（zerank2）保持主增益** | §3.3.2 Reranker 族 | L1 最大单点增益之一 | ✅ | 黄金臂 Full_zerank2 |
| **H8** | **三层评测 + 成本指标 + 默认关新开关** | §6.5 / 工业评测纪律 | 避免 CtxRel 单飞上线 | ✅ | 纪律性 |

### 3.2 中 ROI（有条件、子集或二期）

| ID | 思路 | 综述锚点 | 为何中 ROI | 建议触发条件 | 状态 |
|----|------|----------|------------|--------------|------|
| **M1** | **动态压缩比**（低 rerank 置信 → 保留更多句） | AdaComp | 实现便宜，但需校准阈值；与 soft_rank 叠乘 | Boot-R R1 阳性后开 `adaptive_ratio` | 📋 配置位已留，`enabled=false` |
| **M2** | **soft_rank + 轻量 prune**（`prune_below`） | 介于 compress 与 CRAG 之间 | 有硬删风险，易重演 CRAG | 仅实验臂；Correct 不掉才考虑 | 📋 配置有，**禁止默认** |
| **M3** | **Listwise LLM rerank**（RankGPT / TourRank / FIRST） | §3.3.2 | 延迟高；zerank2 已强 | 仅 `max(rerank)<θ` 对 top-20 | 📋 未做 |
| **M4** | **低置信 Gate2 / claim 级 Gate2** | 生成后忠实性（非综述核心，工程延伸） | 全量 always 边际；claim 更准但贵 | `trigger=low_rerank` 或 claim 子集 | 🔶 Gate2 有；claim 未做 |
| **M5** | **多跳 sub-query 分解** | OmniSearch / CogPlanner | 工业题部分多跳；全量 rewrite 已阴性 | Failure Clinic multi-hop 子集 only | 📋 未做 |
| **M6** | **CREAM 式 coarse-to-fine 多页** | ColPali 系多页 | neighbor expand 可近似；真 coarse-to-fine 更复杂 | 跨页流程题 miss 仍高时 | 📋 用 B1 近似 |
| **M7** | **空 caption 才 VLM 描述图块** | Advanced captioning；MRAG2 统一 MLLM caption | 全库 VLM 贵；空 caption 子集有价值 | Phase C；与 content_list image 块 | 📋 roadmap 已写「不做默认」 |
| **M8** | **HyDE 条件触发** | Query reformulation 族 | 历史 HyDE 消融不稳；全局开易漂 | 仅短/模糊 query 或检索分极低 | 🔶 代码有，默认关 |
| **M9** | **CRAG 软化重试**（只 grade 不 rewrite；软降权不丢 chunk） | Corrective RAG 思想 | 硬滤已阴性；软化可能回血但 ROI 低于 H1 | 预算富余且 Refiner 不够时 | 🔶 实现在，默认关；协议须改 |
| **M10** | **Modality boost 调参/默认** | 模态融合权重 | Boot-CP 未显示 page 级 NDCG 增益 | 表子集 E2E 再验 | ✅ 代码有，默认关 |
| **M11** | **解析后表质检（offline MLLM/HITL）** | §7.1 Error detection；HITL | 不进在线路径；提高库质量 | 关键手册入库流水线 | 📋 未做 |
| **M12** | **Prompt/压缩信息论方法**（LLMLingua 系 token PPL） | LongLLMLingua / Selective Context | 与 soft_rank 同族更细；依赖小 LM PPL | soft_rank 阳性后的增强版 | 📋 未做 |
| **M13** | **Failure Clinic 标签扩展**（规划错关 visual、crossref 错表） | 工程可观测 | 不直接涨分，但加速决策 | 与 Boot-R 并行 | 🔶 Clinic 有；标签可补 |

### 3.3 低 ROI / 现阶段不借

| ID | 思路 | 综述锚点 | 为何低 / 不做 | 状态 |
|----|------|----------|---------------|------|
| **L1** | 多模态**输出**（答案插图/视频） | MRAG3.0 Augmented Multimodal Output | 产品非刚需；评测无覆盖 | 🚫 |
| **L2** | 默认统一 **MLLM 端到端生成**（页图+文进大 VLM） | MRAG2/3 Generation | 延迟/成本；7B Ollama 不稳；工业答案以文本为准 | 🚫 默认 |
| **L3** | **Generative Retrieval**（DSI / DocID / SEAL） | §3.3.1 Generative Structure | 替换 FAISS+PG 代价巨大；研究向 | 🚫 |
| **L4** | **Multi-agent 规划**（多代理并行检索计划） | §6.2 multi-agent coordination | 过度工程；先规则 Planning | 🚫 |
| **L5** | **全库多模态知识图谱 / LightRAG 全迁** | 结构增强的重型形态 | 已路线图排除；维护成本高 | 🚫 |
| **L6** | **默认全局 CRAG + reformulate** | Corrective 流水线 | 实测 Correct −12pt | 🚫 默认 |
| **L7** | **默认 Gate2 always** | 生成后全量过门 | 边际 Faith；延迟 ×1.7 | 🚫 默认 |
| **L8** | **默认 HyDE / 默认 VLM query 增强** | Query reformulation / visual query | 漂查询；成本 | 🚫 默认 |
| **L9** | **音频 / 视频 MRAG** | 综述模态覆盖 | 域外 | 🚫 |
| **L10** | **统一能力 taxonomy 的学术基准堆砌** | §4–5 Datasets | 与工业手册 E2E 不对齐；可作调研 | 📋 参考 only |
| **L11** | **Native 多模态 DocID 学习型标识** | Learnable DocID | 同 L3 | 🚫 |
| **L12** | **强制 image-centric 固定规划**（凡图必检索） | Fixed image-centric planning | 论文与我们 Visual 数据均反对 | 🚫 |
| **L13** | **Office 全家桶 / 公式专用管线** | 解析扩展 | 当前 PDF 工业手册优先 | 📋 远期 |
| **L14** | **图块级 Col* 全库索引**（非页级） | 细粒度 visual index | 存储/编码成本高；页级已在 | 📋 Phase C 研究 |

---

## 4. 分组件深挖（高+中+低）

### 4.1 Document Parsing & Indexing

| 方向 | ROI | 说明 | 动作 |
|------|-----|------|------|
| 结构保留（heading / table type / caption） | **高** | 综述 extraction 丢结构；我们 A2/A3 对症 | 已做；re-ingest 后价值最大 |
| 表摘要 + 邻段 context | **高** | domain caption | 已做；默认 context 关，Boot 阳性再开 |
| 页截图 Col* 索引 | **中高（架构）** | Representation-based；单独 Visual_only 弱但可补版式 | 保持；靠 Planning 按需 |
| 页内图块单独 embedding | **低–中** | 贵；空 caption VLM 更划算 | 不做默认 |
| 解析后 LLM 纠错 | **中** | offline | 关键库可加 |
| 纯 caption 伪多模态（1.0） | **低** | 倒退 | 禁止回退 |

### 4.2 Search Planning

| 方向 | ROI | 说明 | 动作 |
|------|-----|------|------|
| 规则 heuristic 选 visual | **高** | 对齐「强制 image 有害」 | ✅ `search_planner` |
| `a_none` 跳过检索 | **中** | 闲聊省成本；工业误伤风险 | 配置 `allow_skip_retrieval` 默认 false |
| LLM Retrieval Classification | **低–中** | 不稳+延迟 | 不做默认 |
| 多跳 sub-query | **中** | 子集 | Failure Clinic 后再开 |
| Multi-agent 规划 | **低** | 过度 | 🚫 |

### 4.3 Retrieval（Retriever / Reranker / Refiner）

| 方向 | ROI | 说明 | 动作 |
|------|-----|------|------|
| 三路 + RRF + zerank2 | **高（已有）** | 保持 | 黄金臂 |
| Refiner soft_rank | **高** | 综述最该补的环 | ✅ 待 Boot-R |
| neighbor / crossref expand | **高–中** | 候选集补全 | ✅ 默认关 |
| modality boost | **中** | 轻推 table | ✅ 默认关 |
| Listwise LLM rerank | **中** | 低置信 | 📋 |
| Generative retrieval | **低** | 架构替换 | 🚫 |
| CRAG 硬滤 | **实测负** | 勿当高 ROI | 默认关 |

### 4.4 Generation & 后处理

| 方向 | ROI | 说明 | 动作 |
|------|-----|------|------|
| 表 chunk 不句压 | **高（已有）** | 防拆表 | refiner 保留 |
| Gate2 low_rerank | **中** | 边际 Faith | 可试 |
| claim 级忠实性 | **中** | 更准更贵 | 未做 |
| 答案内插图 | **低** | 产品 | 🚫 |
| 默认 MLLM 生成 | **低** | 成本 | 🚫 |

### 4.5 评测与工程纪律

| 方向 | ROI | 说明 |
|------|-----|------|
| L1/L2/L3 + latency | **高** | 已有；坚持 Correct 否决 |
| 默认关新开关 + cache 盐 | **高** | L3/L4 已扩 refiner/plan/crossref 盐 |
| 学术 VQA 大盘替换工业 E2E | **低** | 不对齐 |

---

## 5. 与「已踩坑」对照（避免重复投资）

| 已做实验 | 结果 | 综述如何解释 | 后续策略 |
|----------|------|--------------|----------|
| CRAG ON 100q | CtxRel +11pt；Correct −12pt；latency ×3.1 | 硬滤/改写伤害证据充分性；非「相关句」即「可答」 | 默认关；优先 Refiner 软策略 |
| Gate2 always | Faith +0.9pt；Correct +0.02 | 生成后补救改不动检索错 chunk | 默认关；low_rerank 可选 |
| Visual 弱 + 强制三路 | Visual_only 低；融合增量有限 | compulsive image retrieval | Planning 按需 visual |
| HyDE 消融 | 不稳定/有害场景 | 固定 reformulation 漂 | 保持关；条件触发属中 ROI |
| Boot-CP B1/B2 | page NDCG 三臂同 | expand 增益在 chunk/E2E 层 | 看 E2E/子集，不单看 NDCG@10 |

---

## 6. 落地状态与流水线位置（2026-07-25）

### 6.1 代码交付（`feat/p0-p1-retrieval-context`）

| 包 | 路径 | 配置键 | 默认 |
|----|------|--------|------|
| Refiner | `src/generation/refiner.py` | `refiner.mode` | **bge** |
| Planning | `src/retrieval/search_planner.py` | `retrieval.search_planning.enabled` | **false** |
| Crossref | `src/retrieval/expand.py` + `find_chunks_by_ref` | `retrieval.crossref_expand.enabled` | **false** |
| 云脚本 | `scripts/cloud_boot_r.sh` | 多臂 Base/R1/Pl/Cr | — |
| 实现方案 | `docs/superpowers/plans/2026-07-25-p0-p1-refiner-planning-impl.md` | — | — |

### 6.2 目标流水线（含中 ROI 预留位）

```text
Query
  → SearchPlan (H2) ………………… 默认关 → 透传三路
  → BM25 ∥ Dense ∥ Visual?
  → RRF → modality_boost (M10) …… 默认关
  → neighbor_expand (H4) ………… 默认关
  → Rerank (H7)
  → crossref_expand (H6) ………… 默认关
  → [CRAG (M9) 禁止默认]
  → Refiner soft_rank/bge (H1) … 默认 bge
      └ adaptive_ratio (M1) ……… 默认关
  → LLM
  → [Gate2 (M4) 默认关]
```

### 6.3 Boot-R 验证（进行中 / 待归档）

| 臂 | 变量 | 成功标准（相对 base） |
|----|------|----------------------|
| base | bge · 全实验关 | 基线 |
| r1 | soft_rank | Correct ≥ −2pt；Faith ≥ −2pt；latency ≤ ×1.15 |
| pl | planning on | Correct ≥ −2pt；latency↓ 或 Visual 调用率↓ |
| cr | crossref on | Correct ≥ −2pt；latency ≤ ×1.10 |
| CtxRel | 任意 | **仅观察，否决禁用** |

产物目录约定：`runs/YYYYMMDD-boot-r/`（见 `cloud_boot_r.sh`）。

---

## 7. 优先级路线图（含中低 ROI 排序）

### 近端（本季度）

1. **Boot-R 出数** → 决定 soft_rank / planning / crossref 是否试开默认  
2. **Content Pipeline 默认开关**（表 context / expand）按既有 decide 协议，不与 Boot-R 混变量  
3. **Failure Clinic 补标签**（M13）：planning 错关 visual、crossref 错表、refiner 过剪  

### 中端（Boot-R 后、有预算）

| 序 | 项 | 依赖 |
|----|----|------|
| 1 | M1 adaptive_ratio | R1 阳性 |
| 2 | M4 low_rerank Gate2 再验 | 稳定 base |
| 3 | M5 multi-hop 子集 | Clinic 样本 |
| 4 | M3 低置信 listwise | 延迟预算 |
| 5 | M9 CRAG 软化（可选） | 仅当 H1 不够 |

### 远端 / 研究

- M7 空 caption VLM  
- M12 LLMLingua 级 token 压缩  
- M6 真 coarse-to-fine 多页  
- L* 全表保持 🚫 除非产品边界变化  

---

## 8. 决策纪律（写进配置文化）

1. **新能力默认关**；阳性写 handoff 再改默认。  
2. **Correct / Faith / 误拒 / latency** 否决；CtxRel、中间 grade 分不否决上线。  
3. **禁止**「再叠一个 LLM 门」作为检索坏 chunk 的第一反应。  
4. **改写类**（HyDE / CRAG reformulate / LLM plan）必须条件触发 + 对照臂。  
5. **Visual** 按需，不 compulsive。  
6. **表结构** 任何 compress/refiner 必须 protect。  
7. **一次开机多臂**；禁止半成品连开。  
8. 简历/对外：可写「实现 + 对照」；**禁止**写未达标的涨分话术。  

---

## 9. 附录

### A. 论文元数据

| 项 | 值 |
|----|-----|
| 标题 | A Survey on Multimodal Retrieval-Augmented Generation |
| 作者 | Lang Mei, Siyu Mo, Zhihan Yang, Chong Chen（华为云） |
| 标识 | arXiv:2504.08748v1 |
| 篇幅 | 80 pages |
| 关键词 | MRAG, MLLM, Document Parsing, Multimodal Search Planning |

### B. 推荐精读章节（再读论文时）

| 章节 | 内容 | 对我们 |
|------|------|--------|
| §2.1–2.3 | MRAG 1/2/3 | 定位 |
| §3.1 | Parsing extraction vs representation | 入库 |
| §3.2 | Planning fixed vs adaptive | H2 |
| §3.3.3 | Refiner | H1 |
| §6 | Limitations | 避坑 |
| §7.1 | Future parsing | A/B 与 crossref |

### C. 文档关系

```text
bottleneck-analysis (问题是什么)
        ↓
本分析 (综述对照 → 做什么/不做什么)
        ↓
content-pipeline roadmap (入库 H4/H5)
        ↓
p0-p1-refiner-planning-impl (H1/H2/H6 怎么实现)
        ↓
cloud_boot_r.sh + runs/ (是否上线)
```

### D. 变更记录

| 日期 | 说明 |
|------|------|
| 2026-07-25 | 首版：全量高/中/低 ROI + 落地状态 + Boot-R 纪律 |

---

## 10. 结语

综述提供的是 **MRAG 能力地图与反模式**，不是可照抄的 SOTA 配置。  
对 PrismRAG：

- **高 ROI** 集中在 **入库结构、按需规划、检索后 Refiner、引用补全、强精排与评测纪律**。  
- **中 ROI** 是「子集 / 低置信 / 二期增强」，服务高 ROI 阳性之后。  
- **低 ROI** 多为模态炫技、架构重写或已实测阴性的硬门控——**明确不借，省 GPU 与注意力**。

以 Correct 为北，以默认关为闸，以一次 Boot 多臂为尺。
