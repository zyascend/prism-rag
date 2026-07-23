# Content Pipeline Phase A/B — 检索质量主线路线图

> **For agentic workers:** 实现时用 superpowers:subagent-driven-development（推荐）或 executing-plans，按 Task checkbox 推进。  
> **灵感来源：** [RAG-Anything](https://github.com/HKUDS/RAG-Anything)（ContextExtractor / typed content_list / 轻量层级）— **只借鉴入库语义，不迁 LightRAG / 不默认上全量 KG**。  
> **关联：** handoff「检索 badcase P2 主矛盾」；CRAG 默认关；Gate2 默认关。  
> **日期：** 2026-07-23  
> **分支建议：** 优先单分支 `feat/content-pipeline-phase-ab`（A+B 代码齐后再开机）；拆 PR 亦可，**禁止为半成品单独开机**  
> **云策略（定稿）：** 默认 **1 次 Boot-CP** 验收 A+B（见 [Cloud Boot Packing](#cloud-boot-packing减小开机次数--主策略)）

---

## Goal

在 **不叠生成前/后 LLM 门** 的前提下，把入库与检索接缝做厚，直接打「错 page / 错 chunk / 表语境丢失」：

| Phase | 一句话 | 在线延迟影响 |
|-------|--------|-------------|
| **A** | 入库侧：上下文表摘要 + MinerU 类型化块 + 轻量元数据 | **0**（ingest 时付费） |
| **B** | 检索侧：parent–child expand + 查询模态轻 boost | **低～中**（可开关，默认关） |

**成功定义（必须同时满足）：**

1. 新能力 **默认关或可灰度**，黄金消融（Full_zerank2）不回退超协议噪声。  
2. 表/结构相关子集有 **可辩护 Δ**（NDCG 或 E2E Correct），不是只有 CtxRel。  
3. 全量 re-index **只上云**；本地 ≤10q 冒烟 + 单测。  
4. **云验收默认 1 次开机**（A 与 B 同机多臂）；禁止 A1 mini-boot 后再 Full A / 再 B 连开三次。

---

## Architecture（目标态）

```text
PDF
  ├─ simple: markdown 页（启发式 table 兜底，保持兼容）
  └─ mineru: content_list 优先
        type: text | table | image
        + caption / table_body / page_idx / section hints
              │
              ▼
        Chunk(
          text, chunk_type,
          table_summary,          # 已有
          caption?,               # A2/A3 新增
          section_path?,          # A3
          prev/next_chunk_id?,    # A3
        )
              │
     TableSummarizer(context=邻页/标题/caption)   # A1
              │
     BGE(embed summary|text) + BM25(text) + Col*(page image)
              │
              ▼
     search → [可选 B1 expand] → [可选 B2 modality boost] → RRF → Rerank
```

**明确不做（本路线图范围外 / Phase C 研究项）：**

- 全库多模态知识图谱 / LightRAG 迁移  
- 默认 VLM-enhanced query / 默认 CRAG / 默认 Gate2  
- 公式专用管线、Office 全家桶  
- 图块级 Col* 索引（Phase C：空 caption 才 VLM 描述进 Dense）  
- 本地全量 5244 页 re-index  

---

## 现状锚点（改之前先认清）

| 组件 | 路径 | 现状 |
|------|------|------|
| 解析 | `src/ingestion/parser.py` | `Page(markdown, image)`；MinerU 只拼 md+截图，**未消费 content_list** |
| 分块 | `src/ingestion/text_chunker.py` | `chunk_type: text\|table`；表靠 `_looks_like_table` 启发式 |
| 表摘要 | `src/ingestion/table_summarizer.py` + `prompts/table_summary.yaml` | **只喂 table_md**，无邻页/章节 |
| 入库 | `pdf_ingestor.py` / `vidore_ingestor.py` | dual：embed=summary，存=全文 |
| 存储 | `pgvector_store.py` | 已有 `table_summary`；**无 section/caption/neighbor 列** |
| 检索 | `vidore_adapter.PrismRAGRetriever` | 三路 + RRF + rerank；VisualRouter 已有 |
| 诊断 | `src/diagnostics/` | Failure Clinic P01–P12；可补解析/表语境标签 |

---

## 原则

1. **先 A 后 B**：B 依赖 A 的元数据字段；A 未验收禁止开 B 默认。  
2. **兼容 simple 解析器**：无 content_list 时退回现状路径，评测与本地 dev 不炸。  
3. **index_version 盐**：schema / 摘要逻辑变更必须 bump，避免 L3/L4 脏缓存。  
4. **Agents.md**：本机禁止全量 ingest / 全量 RAGAS；云上先查 HF 缓存。  
5. **默认关新开关**：配置进 `config/models.yaml`，生产/黄金消融保持旧行为直到 Boot 阳性。

---

## Phase A — 入库语义（主矛盾直接解）

**目标：** 表找得到、块类型可信、chunk 自带「在哪一节 / 邻居是谁」。  
**云开机：** **不单独为 A 开机**。A1–A3（及 B 代码）本地全绿后，进入唯一 **Boot-CP**；A 的 Go/No-Go 看同机 **Arm-A**（expand/boost 全关）。

### A0 — 基线与探针（0.5d，可本地）

- [ ] **A0.1** 从已有 badcase / E2E 抽出 **表相关 query 子集**（建议 20–40 条；无金标则用 Failure Clinic 手工标签）
  - 产物：`data/table_subset_queries.json` 或 `runs/.../table_subset.md`（路径自定，README 写清）
- [ ] **A0.2** 冻结对照数字指针：Boot-A Full_zerank2、post-P0 E2E Correct、CRAG 阴性结论（只引用 `runs/`，不重跑）
- [ ] **A0.3** Failure Clinic 增补标签草案（可先文档不写代码）
  - 解析/OCR 脏 → 建议 **P10**
  - 表结构丢失 / 错表 → **P02 细化** 或 **P11**
  - 表语境丢失（对表错章）→ **P12**
  - 与 `docs` 中 RAG-Anything failure modes 对齐一句说明即可

**验收：** 子集列表 + 基线指针写进本文件或 run README；无需 GPU。

---

### A1 — 上下文感知 TableSummarizer ⭐ 最高 ROI

**问题：** 摘要只看表身 → Dense 丢章节/单位/工况语境 → 错表 / miss。  
**做法：** 摘要输入 = `section_hint + caption + neighbor_text + table_md`；embed 仍用 summary，生成仍用全文。

#### Tasks

- [ ] **A1.1** 扩展 API（向后兼容）
  - 文件：`src/ingestion/table_summarizer.py`
  - `summarize(table_md, *, context: str = "") -> str`
  - 空 context 行为 = 今日行为
- [ ] **A1.2** Prompt 新版本（registry，勿破坏 active 默认直到开关打开）
  - 文件：`src/prompts/prompts/table_summary.yaml`
  - 新 version：要求输出仍 1–3 句；上下文仅作消歧，禁止编造表中不存在的数字
- [ ] **A1.3** 上下文装配
  - 文件：`text_chunker.py` 和/或 `pdf_ingestor` / `vidore_ingestor` 调用点
  - 最小实现：同页非表段落截断拼接（`max_context_chars` 可配置，默认 ~1500）
  - 有 caption / 标题行时优先拼前缀（chunker 已部分保留 caption 前缀，需显式传入 summarizer）
- [ ] **A1.4** 配置
  - `config/models.yaml` 示例：
    ```yaml
    ingestion:
      table_summary_enabled: true          # 已有
      table_summary_context_enabled: false # 新，默认关
      table_summary_context_max_chars: 1500
    ```
- [ ] **A1.5** 单测
  - `tests/test_*`：同表 + 不同 context → 摘要 prompt 含 context；开关关 → 与旧行为一致
  - mock `call_llm`，不真调模型
- [ ] **A1.6** 文档
  - 更新 `docs/architecture/content-pipeline.md` Enrich 段
  - 与 `docs/table-summary-large-table-design-2026-07-09.md` 加「上下文摘要」一小节（链接本 roadmap）

**验收：**

| 项 | 标准 |
|----|------|
| 单测 | 绿 |
| 本地 | 无需 GPU；可选 1 页 fixture PDF |
| 云（Boot-CP · Arm-A） | 子集 NDCG 或 E2E 表题 **不显著变差**；理想 **Correct↑ 或 表 miss↓** |
| 默认 | `table_summary_context_enabled: false` 直到阳性 |

**风险：** 上下文引入噪声导致摘要跑题 → prompt 强调「以表为准」+ 可关。

---

### A2 — MinerU content_list → typed chunks

**问题：** 仅 markdown + `_looks_like_table` → 复杂表/OCR 脏 md 类型错。  
**做法：** MinerU 路径优先 `content_list`；simple 路径保持启发式。

#### Tasks

- [ ] **A2.1** 解析输出扩展
  - 文件：`src/ingestion/parser.py`
  - 方案二选一（实现时定，推荐 B）：
    - **A：** `Page` 增加 `blocks: list[ContentBlock] | None`
    - **B：** 新方法 `parse_structured(pdf) -> list[PageStructured]`，旧 `parse` 保留
  - `ContentBlock`: `type`, `text`/`table_body`, `caption`, `page_idx`, 可选 `bbox`
- [ ] **A2.2** MinerU 读取 content_list
  - 定位 MinerU 输出 `*_content_list.json`（或等价）；缺失时 **降级** 现网 md 路径并打 log
- [ ] **A2.3** Chunker 入口
  - `chunk_page` 保持；新增 `chunk_blocks(blocks, page_meta) -> list[Chunk]`
  - `chunk_type` 由 block.type 映射：`table` / `text`；（`image` 见 A2.4）
- [ ] **A2.4** image 块最小处理（不做 VLM）
  - 若有 caption：生成 `chunk_type=text` 或 `image` 的 **caption 锚点 chunk**（Dense/BM25 可命中）
  - 无 caption：可跳过或仅记 metadata（配置 `ingestion.image_caption_chunks: true`）
- [ ] **A2.5** Ingestor 接线
  - `PDFIngestor`：parser=mineru 时走 structured；simple 走旧路径
  - `ViDoReIngestor`：若语料无 content_list，**不强制**（避免评测语料路径大改）；仅生产 PDF 路径先落地亦可
- [ ] **A2.6** 单测
  - fixture：迷你 content_list JSON + 期望 chunk_type 分布
  - 降级：无 JSON → 旧路径
- [ ] **A2.7** 文档
  - `content-pipeline.md` Parse 段补 content_list 分支 Mermaid

**验收：**

| 项 | 标准 |
|----|------|
| 类型准确 | fixture 上 table/text 与标注一致 |
| 兼容 | simple + 无 list 不回归 |
| 云 | 与 A1 同 Boot 测；看表子集 + 全量 100q NDCG 漂移 |

**风险：** MinerU 版本/输出路径漂移 → 强降级 + 集成测试锁文件布局。

---

### A3 — 轻量层级元数据（不上全图）

**问题：** hit 对页错段、无法 expand 邻居、rerank/生成缺「章节路径」。  
**做法：** chunk 元数据 + DB 列；**不建 entity KG**。

#### Tasks

- [ ] **A3.1** 数据模型
  - `Chunk` 增字段（可选默认空）：
    - `section_path: str = ""`
    - `caption: str = ""`
    - `prev_chunk_id: str = ""`
    - `next_chunk_id: str = ""`
  - 入库后二次 pass 填 prev/next（同 doc 顺序）
- [ ] **A3.2** section_path 提取（最小）
  - 从 md/heading/`text_level` 维护 running heading stack
  - 无 heading 时：`section_path=""` 合法
- [ ] **A3.3** pg  schema
  - `pgvector_store.py`：`ALTER ... ADD COLUMN IF NOT EXISTS` 模式（与 `table_summary` 一致）
  - insert/select/search 投影带上新列
  - BM25 rebuild 是否带元数据：至少 search 返回 dict 含字段
- [ ] **A3.4** 检索结果透出
  - `PrismRAGRetriever` 结果 dict / API `Citation` 可选带 `section_path`（便于 Clinic 与前端）
- [ ] **A3.5** `index_version` / 文档
  - 语料重灌规则写入 `docs/architecture/ingestion.md`
  - cache 盐：摘要逻辑或 schema 大变时 bump `index_version` 语义说明
- [ ] **A3.6** 单测
  - 三块连续文本 → prev/next 链正确
  - heading 栈 → section_path 拼接

**验收：**

| 项 | 标准 |
|----|------|
| 写读一致 | insert 后 get 回 section/caption/neighbors |
| 旧索引 | 缺列迁移不炸（空串默认） |
| Phase B | B1 可依赖 `prev/next` 或同 `page_id` 查询 |

---

### Phase A 退出标准（Go / No-Go）

> 在 **Boot-CP 的 Arm-A** 上判（与 B 同机）；逻辑上仍可「A No-Go 则忽略 B 臂数字、默认保持关」。

| # | 标准 | Go |
|---|------|----|
| 1 | A1–A3 单测全绿；simple 路径 e2e 本地冒烟 | 必须（开机前） |
| 2 | Boot-CP **Arm-A**：Full_zerank2 **100q**（决策）相对历史基线 Δ NDCG@10 ≥ **−0.01**（不崩） | 必须 |
| 3 | 表子集：NDCG 或 E2E 表相关 Correct **↑ 或 miss↓** 且可写清机制 | 理想；若持平但 Clinic 表语境错减少也可条件 Go |
| 4 | 默认配置仍可复现旧行为（context 关 / 无 list 降级） | 必须 |
| 5 | `content-pipeline.md` + handoff 更新 | 必须 |

**No-Go：** Arm-A 全量 NDCG 掉 >1pt 且表子集无增益 → 修 A1 prompt / 关 context；**B 臂结果不采纳、开关保持 false**（同机已跑的 B 数据可归档但不写「可上线」）。

---

## Phase B — 检索接缝（与 A 同机验收）

**目标：** 命中后补邻居；查询意图轻推 table/visual。  
**前提（实现）：** A3 元数据可查（`prev/next` 或同 `page_id`）。  
**前提（云）：** B 代码与 A **同一 commit 上机**；看 Boot-CP **Arm-B\*** vs **Arm-A**。

### B1 — Parent–child / 邻居 expand

**问题：** top-k 命中半截表或答句在邻块。  
**做法：** rerank 前后对 top 命中 **按 page 或 prev/next 扩 1 跳**，去重后截断到 `rerank_k` 或更高 cap。

#### Tasks

- [ ] **B1.1** 设计参数（写入 models.yaml，默认关）
  ```yaml
  retrieval:
    neighbor_expand:
      enabled: false
      mode: page          # page | prev_next
      max_extra: 2        # 每个 hit 最多扩几块
      stage: post_rerank  # pre_rerank | post_rerank（先实现 post，省 rerank 成本）
  ```
- [ ] **B1.2** 实现
  - 位置：`vidore_adapter.search_with_trace` 或独立 `src/retrieval/expand.py`
  - 数据：pg `get_chunks_by_page` / id 批量取（避免 N+1）
  - Trace：`retrieval_trace.expand` 记录扩了哪些 id
- [ ] **B1.3** 与 context_filter / 生成协调
  - 扩进来的 table 仍走 table 保护（不压缩）
  - token 预算：expand 后仍受 compression_ratio / rerank_k 约束
- [ ] **B1.4** 单测 + 10q 冒烟
- [ ] **B1.5** 文档：`docs/architecture/` 可先在 evaluation 或新建 retrieval 片段链到本 roadmap

**验收：** E2E 可答题 Correct 不降；表/跨段题 miss 下降；延迟增量可测（目标 **< +15%** p50）。

---

### B2 — 查询侧模态轻 boost（扩展 VisualRouter 思路）

**问题：** 表/图题被文本路淹没（RAG-Anything failure mode #4）。  
**做法：** 启发式意图 → 融合前权重或候选池，**不是**再调 LLM。

#### Tasks

- [ ] **B2.1** 意图检测（规则，可测）
  - table-ish：table / 表 / spec / range / limit / parameter / 阈值 / 阈值 …
  - visual-ish：figure / diagram / schematic / wiring / layout / 图 …
  - 文件建议：`src/retrieval/query_intent.py` 或并入 `visual_router.py`
- [ ] **B2.2** 作用点（选一，先做简单）
  - **Boost：** RRF 前对 `chunk_type==table` 或 visual 路 rank 加权
  - **Gate：** table-ish 时提高 dense_k 对 table 的保留；visual-ish 强制 `use_visual=True`
- [ ] **B2.3** 配置默认关
  ```yaml
  retrieval:
    modality_boost:
      enabled: false
      table_rrf_bonus: 0.0    # 或 rank 偏移，实现时定可解释量纲
      force_visual_on_visual_intent: false
  ```
- [ ] **B2.4** 消融
  - Boot-CP 内 **Arm-B2**（boost on）vs **Arm-A**；Full_zerank2 100q + 表子集
- [ ] **B2.5** 单测：规则命中表；无关 query 不改变排序（bonus=0 或 enabled=false）

**验收：** 表子集升；全量不掉 >1pt；延迟可忽略（纯规则）。

---

### Phase B 退出标准

| # | 标准 |
|---|------|
| 1 | B1/B2 默认 **false**；单测绿（开机前） |
| 2 | Boot-CP：**Arm-B1 或 Arm-B2** 相对 **Arm-A** 有 **E2E Correct 或 表子集** 增益 |
| 3 | 延迟：B1 p50 增幅写进 run README；不可接受则改 `max_extra`/`stage` |
| 4 | handoff 更新默认建议（哪个开关可生产试开） |

---

## 任务依赖图

```text
本地（0 云费）
  A0 基线/子集
   ├─► A1 上下文表摘要  ─┐
   ├─► A2 content_list  ─┼─► A3 元数据
   └─► B1 expand ∥ B2 boost（与 A 并行写完）
                         │
                         ▼
云 ×1  Boot-CP（唯一开机）
  re-index 一次
   ├─ Arm-A   → 判 Phase A
   ├─ Arm-B1 / Arm-B2 → 判 Phase B
   └─ 归档 → 关机
```

**并行（本地）：** A1 ∥ A2；B1 ∥ B2；**B 不必等云上 A Go**（等的是 A 的代码与 schema，不是云数字）。  
**禁止并行（云）：** 为 A1、A2、B 各开一次机。

### 阶段合并原则（减成本）

| 合并 | 做法 | 省什么 |
|------|------|--------|
| A1+A2+A3 | 同一 PR/迭代，**一次 re-index** | 第二次灌库 |
| A 验收 + B 验收 | **同 Boot 多臂** | 第二次开机 |
| B1+B2 | 同索引 2～3 臂 | 为 boost 再开机 |
| NDCG+E2E+表子集 | 同机串行 | 重复加载环境 |
| 决策 100q / 定稿 283 | 仅最优臂可选 283 | 每臂 283 |

**禁止假省钱：** A1-only mini-boot → 再 Full A → 再 B（最多 3 次灌/开）。

---

## 建议日程（弹性 · 对齐 1 次开机）

| 周 | 内容 | 环境 |
|----|------|------|
| W1 | A0 + A1 + 文档；启动 A2 | 本地 |
| W2 | A2 + A3 + **B1 + B2** 代码与单测全绿 | 本地 |
| W3 | **Boot-CP 一次**（re-index + 多臂）+ handoff 定稿 | **云 ×1** |

预算紧可砍到 **Minimal**（见下）：弱化 A2、可砍 Arm-B2 / E2E。

---

## Cloud Boot Packing（减小开机次数 · 主策略）

> **计费单位是「开机窗口」不是「Phase」。**  
> 原则：① **A+B 代码齐再开机**；② **整段 A 只 re-index 一次**；③ B 只 `--skip-index` 多臂；④ 决策用 100q，283/RAGAS 仅最优臂可选。

### 三档预算（选一执行）

| 档位 | GPU 开机 | 覆盖 | 适用 |
|------|:--------:|------|------|
| **Minimal** | **1** | A1+A3（A2 可仍走启发式）；B1 优先；Text-only 灌库；100q + 表子集；E2E 可砍 | 钱紧、先要信号 |
| **Standard（推荐 · 默认）** | **1** | **A0–A3 + B1+B2**；一次 re-index；Arm-A / B1 / B2；全臂 100q + 最优臂 E2E | 默认执行 |
| **Full** | **1**（尽量） | Standard + 最优臂 283q 和/或 RAGAS 100；**仅当同机余量不够才** 第二次短开机 skip-index 补定稿 | 简历要厚数字 |

**默认按 Standard：整条主线 1 次开机。** 旧写法「Boot-CP-A + Boot-CP-B = 2 次」**作废**。

### Boot-CP — 唯一验收窗口（Standard）

**本地先完成（0 云费）：**

- [ ] A1–A3 + B1–B2 单测绿  
- [ ] 配置键齐全，**默认 false** 时行为 = 现网  
- [ ] A0 表子集与历史基线指针写好  

```text
Boot-CP（同一进程环境 / 同一索引 · 中途不关机）
│
├─ Job0  环境：HF 缓存检查；拉代码；pg + faiss 就绪
├─ Job1  re-index 一次（策略见下表）
│
├─ Arm-A    expand=off, boost=off     ← 判 Phase A（vs 历史 Boot-A / post-P0）
├─ Arm-B1   expand=on,  boost=off     ← 判 B1（vs Arm-A）
├─ Arm-B2   expand=off, boost=on      ← 判 B2（vs Arm-A）
├─ Arm-B12  expand=on,  boost=on      ← 可选；预算紧可砍
│
├─ 各臂：Full_zerank2 @100q + 表子集
├─ 主臂（Arm-A）+ 最优 B 臂：E2E（50 可答 + 20 拒答）
├─ （Full 档）最优臂 +283q 或 RAGAS 100
└─ 归档 runs/YYYYMMDD-content-pipeline/ + README（Δ 表）→ 立刻关机
```

**如何读数（避免混淆）：**

| 对比 | 含义 |
|------|------|
| Arm-A vs `runs/20260720-bootA` Full_zerank2 | A 是否不崩 / 是否增益 |
| Arm-B\* vs **同机 Arm-A** | B 是否增益（同索引，干净） |
| Arm-A No-Go | B 数字仅存档；开关全 false 合入 |

**索引策略（Job1 选一，写进 run README）：**

| 策略 | 何时 | 成本 |
|------|------|------|
| **Text-only re-ingest**（默认） | 只动 summary/chunk 文本与 pg/bm25，**FAISS 页向量复用** | 省 |
| **Full re-ingest** | A2 改变页切分 / page_id 映射 | 贵 |

**第二开机（仅 Full 档例外）：** 第一次已阳性但没跑 283 → `skip-index` 只补定稿；**禁止**第二次再改代码灌库。

---

## 配置总表（目标写入 `config/models.yaml`）

| 键 | 默认 | Phase |
|----|------|-------|
| `ingestion.table_summary_context_enabled` | `false` | A1 |
| `ingestion.table_summary_context_max_chars` | `1500` | A1 |
| `ingestion.image_caption_chunks` | `false` 或 `true`（实现时定） | A2 |
| `retrieval.neighbor_expand.enabled` | `false` | B1 |
| `retrieval.neighbor_expand.mode` | `page` | B1 |
| `retrieval.neighbor_expand.max_extra` | `2` | B1 |
| `retrieval.neighbor_expand.stage` | `post_rerank` | B1 |
| `retrieval.modality_boost.enabled` | `false` | B2 |

---

## 关键文件清单（实施索引）

| 区域 | 文件 |
|------|------|
| 解析 | `src/ingestion/parser.py` |
| 分块 | `src/ingestion/text_chunker.py` |
| 摘要 | `src/ingestion/table_summarizer.py` |
| Prompt | `src/prompts/prompts/table_summary.yaml` |
| 入库 | `src/ingestion/pdf_ingestor.py`, `vidore_ingestor.py` |
| 存储 | `src/store/pgvector_store.py` |
| 检索 | `src/evaluation/vidore_adapter.py`, `src/retrieval/*` |
| API | `src/api/routes.py`（Citation 可选字段） |
| 配置 | `config/models.yaml`, `src/config.py` |
| 测试 | `tests/test_text_chunker.py`, `test_pdf_ingestor.py`, 新增 `test_table_summary_context.py` 等 |
| 文档 | `docs/architecture/content-pipeline.md`, `ingestion.md`, 本文件 |

---

## 与既有工作的边界

| 已有 | 本路线图关系 |
|------|----------------|
| 表摘要 + 大表保护（2026-07-09） | **A1 增强**，不推翻 dual embed |
| VisualRouter / Boot-B | **B2 互补**；不重复 always/heuristic 消融除非对照 |
| CRAG / Gate2 | **不纳入**；A/B 阳性前禁止用门控掩盖检索问题 |
| Failure Clinic | A0/A 后用标签验证「表语境/类型」类错误是否下降 |
| Bullet 强化路线图 | 本线补 bullet ① 检索质量与 ③ 上下文「入库侧」；不替代 Boot-A 黄金表 |

---

## 简历 / 对外话术（完成后）

**可写：**

- 工业 PDF 内容管道：MinerU 类型化块 + 上下文感知表摘要 + chunk 层级元数据  
- 检索接缝：邻居 expand / 模态意图 boost，且 **云上对照、默认关**  

**禁止写：**

- 未跑 Boot 的「大幅提升」  
- 把 CtxRel 当上线依据（CRAG 教训）  
- 「引入 RAG-Anything / 知识图谱」若实际未建图  

---

## 进度追踪

| ID | 状态 | 备注 |
|----|------|------|
| A0 | done | `data/table_subset_queries.json`（30 条 spec/numeric 探针） |
| A1 | done | context 摘要 + 默认关；单测 `test_table_summary_context.py` |
| A2 | pending | 本地 |
| A3 | pending | 本地 |
| B1 | pending | 本地；与 A 同迭代 |
| B2 | pending | 本地；与 A 同迭代 |
| Boot-CP | pending | **云 ×1**（Arm-A + Arm-B\*） |

状态枚举：`pending` → `in_progress` → `done` / `blocked` / `wontfix`。

---

## 变更记录

| 日期 | 变更 |
|------|------|
| 2026-07-23 | 初版：RAG-Anything 对照后 Phase A/B 落档 |
| 2026-07-23 | **云策略定稿：默认 1× Boot-CP**；废止 Boot-CP-A/B 两次开机；日程压成 W1–W3 |
