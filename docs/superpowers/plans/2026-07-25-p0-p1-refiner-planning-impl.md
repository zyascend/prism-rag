# P0 / P1 实现方案 — Refiner · Search Planning · Cross-ref Expand

> **For agentic workers:** 实现时用 subagent-driven-development 或按 Task checkbox 推进；**每 Task 先单测再接线**。  
> **灵感：** MRAG 综述 arXiv:2504.08748（Planning / Refiner / dual-stream）；CRAG/Gate2 阴性结论。  
> **关联：** [Content Pipeline Phase A/B](./2026-07-23-content-pipeline-phase-ab-roadmap.md) · handoff 检索 badcase P2。  
> **日期：** 2026-07-25  
> **分支建议：** `feat/refiner-soft-rank` · `feat/search-planning` · `feat/crossref-expand`（可单分支串行，禁止默认全开）  
> **云策略：** 本地单测 + ≤10q 冒烟；**云上最多 1 次 Boot-R** 验收 P0 Refiner + P1（多臂），禁止为半成品连开。

---

## 0. TL;DR

| 优先级 | 工作包 | 一句话 | 代码现状 | 默认 |
|--------|--------|--------|----------|------|
| **P0-A** | Content Pipeline A/B 收口 | 确认代码齐、默认关、缺什么补齐 | **大体已完成** | 仍 false |
| **P0-B** | **Refiner 正式化** | 检索后 query-focused 软排序 / 可选硬裁；trace 独立 | 仅有 `context_filter` bge/llm | 新 mode 默认关 |
| **P1-A** | **Search Planning** | 规则选路 `none/text/visual`，非强制三路 | `VisualRouter` + intent 碎片 | 默认关 |
| **P1-B** | **Cross-ref Expand** | 见表/见图 → 扩引用目标 chunk | 无 | 默认关 |

**成功标准（否决项）：** E2E Correct 不掉；Faith 不显著掉；latency 可解释。  
**禁止：** 默认开 CRAG / 默认全局 rewrite / 用 CtxRel 单指标上线 / 全库 VLM 描述。

---

## 1. 目标与非目标

### 1.1 目标

1. **上下文更干净**（Refiner）：入模句子对齐问题，**表结构不破坏**；软策略优先于硬丢 chunk。  
2. **检索更省、更准**（Planning）：非视觉题跳过 Visual；闲聊/可参数内答可 `a_none`（可选）；意图可测。  
3. **跨块引用可补**（Cross-ref）：命中「见表 3-2」时能扩到表体 chunk。  
4. **可观测**：`retrieval_trace` / generation span 有独立字段，Failure Clinic 可挂标签。

### 1.2 非目标（本方案范围外）

| 不做 | 原因 |
|------|------|
| 默认 CRAG / reformulate | 100q 阴性；Correct −12pt |
| Multi-agent / LLM 规划 | 延迟与不稳；先规则 |
| 默认 listwise LLM rerank | 贵；zerank2 已是主增益 |
| 多模态答案插图 | 产品非刚需 |
| 全量 KG / LightRAG | 已排除 |
| 本机全量 ingest / 283 RAGAS | Agents.md |

### 1.3 与现网流水线的位置

```text
Query
  │
  ▼
┌─────────────────────┐
│ P1-A SearchPlanner  │  → routes: bm25/dense/visual on/off
└──────────┬──────────┘
           ▼
   BM25 ∥ Dense ∥ Visual?
           ▼
        RRF fuse
           ▼
   modality_boost? (B2, 已有)
           ▼
   expand (B1 neighbor + P1-B crossref)
           ▼
        Rerank
           ▼
   expand post_rerank?
           ▼
┌─────────────────────┐
│ P0-B Refiner        │  → soft_rank | extractive | 兼容 bge/llm
│ (generation 前)     │
└──────────┬──────────┘
           ▼
      LLM generate  (± Gate2 默认关)
```

**原则：** Planning / Expand 改的是 **候选集**；Refiner 改的是 **入模文本**。  
CRAG 仍是可选旁路，**本方案不接线为默认**。

---

## 2. 现状锚点（改之前认清）

| 组件 | 路径 | 状态 |
|------|------|------|
| 表摘要 context | `table_summarizer` + yaml | A1 已有；默认 `table_summary_context_enabled: false` |
| content_list 分块 | `parser` / `text_chunker` | A2 已有 |
| section / prev-next | `pgvector_store` 列 | A3 已有 |
| neighbor expand | `src/retrieval/expand.py` | B1 已有；默认关 |
| modality boost | `query_intent.py` | B2 已有；默认关 |
| Visual 选路 | `visual_router.py` | 仅 visual on/off；默认关 |
| context_filter | `context_filter.py` + `generator.py` | `off/bge/llm/bge_then_llm`；**无 soft_rank / 无独立 trace 契约** |
| CRAG | `crag.py` | 默认关，保持 |

**P0-A 收口任务：** 不重写 A/B，只做缺口清单（见 §3）。  
**P0-B / P1 是新功能。**

---

## 3. P0-A — Content Pipeline A/B 收口（0.5–1d）

> 代码大体齐；本包保证「可开可关、可评、可归档」，**不强制改默认**。

### Tasks

- [ ] **P0-A.1** 对照 [Phase A/B roadmap](./2026-07-23-content-pipeline-phase-ab-roadmap.md) 勾选实现 vs 文档漂移  
  - 输出：本文件 §3 附录表或 `runs/.../gap.md` 一行清单  
  - 已知：Boot-CP 三臂 NDCG 同 0.3575（page 级看不到 B1）；Goal-A ON 索引 283q NDCG 0.5337 — **默认 expand/boost 仍关合理**
- [ ] **P0-A.2** 单测全绿（本地）  
  ```bash
  .venv/bin/python -m pytest tests/test_table_summary_context.py \
    tests/test_content_list_chunker.py tests/test_neighbor_expand.py \
    tests/test_modality_boost.py tests/test_chunk_metadata.py -q
  ```
  （文件名以仓库实际为准；缺则补测而非砍功能）
- [ ] **P0-A.3** 配置纪律再确认（`config/models.yaml`）  
  - `ingestion.table_summary_context_enabled: false`  
  - `retrieval.neighbor_expand.enabled: false`  
  - `retrieval.modality_boost.enabled: false`  
  - `retrieval.visual_routing.enabled: false`  
  - 阳性后再写 handoff「建议试开」——**不在本方案默认改 true**
- [ ] **P0-A.4** handoff 链到本方案；A/B 不再作为阻塞 P0-B 的前置（**schema/字段已存在即可**）

**退出：** 单测绿 + 默认关 + gap 无阻塞项 → 进入 P0-B。

---

## 4. P0-B — Refiner 正式化 ⭐ 主交付

### 4.1 问题

| 现象 | 根因 |
|------|------|
| 历史 CtxRel 极低 / 噪声入模 | chunk 粗 + 句级过滤弱 |
| CRAG 硬滤 chunk → Correct↓ | 证据被扔掉；误拒↑ |
| 现 `bge` 压缩 | 固定 ratio top-k 句；**无 soft score、无 chunk 级权重、trace 散在 compression span** |
| `llm` 过滤 | 硬 keep 列表；7B JSON 不稳（CRAG 同类风险） |

**设计决策：** Refiner = **生成前对已排序 hits 的上下文加工**；默认 **soft**（降权/重排句，不删 chunk 出候选集）。

### 4.2 模式定义

扩展 `context_filter.mode`（兼容旧值）或新命名空间 `refiner.*`（**推荐双写配置，内部统一入口**）：

```yaml
# 推荐：新块；旧 context_filter.mode 仍可读作兼容别名
refiner:
  enabled: true                    # false 时 = 今日 prepare_context 行为（看 context_filter.mode）
  mode: bge                        # 见下表；生产默认保持 bge 直至 Boot 阳性
  # soft_rank 专用
  soft_rank:
    min_sim: 0.25                  # 低于此句 soft 权重衰减，但不硬删（除非 prune）
    prune_below: null              # null=不硬删；设 0.15 则低于此丢弃（实验臂）
    temperature: 1.0               # 权重 = softmax(sim/T) 可选；MVP 用线性 map
    keep_ratio: 0.5                # 与旧 compression_ratio 对齐语义时可复用
  # 表保护（硬约束）
  protect_table_chunks: true       # 与 generator 现状一致：table 全文不进句压缩
  # 动态压缩（可选，P0-B 二期）
  adaptive_ratio:
    enabled: false
    low_rerank_threshold: 0.35     # max(rerank) 低 → 保留更多句（避免证据不足）
    ratio_high_conf: 0.4
    ratio_low_conf: 0.7
  # 可观测
  emit_trace: true

# 兼容：context_filter.mode 仍支持 off|bge|llm|bge_then_llm
# 新增合法值：soft_rank | soft_rank_then_llm
context_filter:
  mode: bge
```

| mode | 行为 | 延迟 | 风险 |
|------|------|------|------|
| `off` | 拼接 | 0 | 噪声 |
| `bge` | **现状** hard top-ratio | 低 | 可能砍关键句 |
| `soft_rank` | **P0 主推**：句级 sim 赋权，按序拼接；低分句可降权标注或尾部放置；**默认不删 chunk** | 低（同 bge encode） | 上下文略长 |
| `llm` / `bge_then_llm` | 现状硬 keep | 高 | JSON 失败 |
| `soft_rank_then_llm` | soft 后再 LLM（实验） | 更高 | 默认禁止开 |

### 4.3 算法（MVP `soft_rank`）

**输入：** `query`, `hits: List[dict]`（含 `text`, `chunk_type`, `rerank_score`…）  
**输出：** `context: str`, `refiner_trace: dict`

```text
1. 拆分 hits:
   - table_chunks: 保持全文，按原序，权重=1.0
   - text_chunks: 进入句级处理

2. 对 text 部分 split_context_to_sentences → sentences[i]
   BGE encode(query), encode(sentences) → sim[i] ∈ [-1,1] 或 [0,1]

3. Soft map（MVP，可测）:
   weight[i] = max(0, (sim[i] - min_sim) / (1 - min_sim))  # clamp
   if prune_below is not None and sim[i] < prune_below:
       drop sentence   # 仅实验臂
   else:
       keep all sentences with weight  # 或 keep_ratio 截断低权尾部

4. 排序策略（二选一，默认 A）:
   A. 保原文顺序拼接（与 bge 一致）—— 优先实现
   B. 高权句前移 —— 仅实验，防破坏叙述

5. 与 table 块按 hit 原序 merge（沿用 generator 逻辑）

6. refiner_trace:
   {
     mode, num_sentences_in, num_sentences_out,
     mean_sim, min_sim_kept, pruned: 0,
     table_chunks: N, text_chunks: M,
     ratio_effective, adaptive: bool
   }
```

**与 `compress_context` 关系：**

- 抽取 `compress_context` 的 encode+score 为共享函数 `score_sentences(query, sentences, embedder) -> List[float]`  
- `bge` mode = hard top-k on scores  
- `soft_rank` = soft weights + optional prune  
- **禁止**复制三份 encode 逻辑

### 4.4 代码改动清单

| # | 文件 | 变更 |
|---|------|------|
| 1 | `src/generation/refiner.py` **新建** | `refine_context(query, hits, embedder, **cfg) -> RefineResult` |
| 2 | `src/generation/context_filter.py` | `prepare_context` 委托 refiner；保留旧 API |
| 3 | `src/generation/generator.py` | 用 `refine_context` 替代手写 table/text 分叉；span metadata 写 `refiner` |
| 4 | `src/evaluation/ragas_metrics.py` | `eval_via_generator` 路径已走 Generator 则自动对齐；legacy compress 路径可选接同一入口 |
| 5 | `src/generation/self_rag.py` | Gate2 重生 `precomputed_context` 不变；首次生成走 refiner |
| 6 | `config/models.yaml` | 增加 `refiner:` 块 |
| 7 | `src/api/routes.py` | `/ask` 响应可选 `refiner` 摘要字段（debug 或 trace 内） |
| 8 | `tests/test_refiner.py` **新建** | 见 §4.6 |

**不做：** 改检索排序；不在 refiner 内二次检索。

### 4.5 Trace / 缓存盐

| 层 | 规则 |
|----|------|
| Generation span | `metadata.refiner = refiner_trace` |
| L4 Answer cache | key 必须含 `refiner.mode` + 关键阈值（与 self_rag / crag 盐同级） |
| L3 Retrieval cache | **不**含 refiner（检索结果不变） |
| RAGAS | 入模 context 必须是 refine 后文本（`eval_via_generator=true` 时天然满足） |

### 4.6 单测（TDD）

| 用例 | 期望 |
|------|------|
| `mode=bge` 与旧 `compress_context` 同输入同输出（golden） | 回归 |
| `mode=soft_rank` + 高 sim / 低 sim 句 | 低 sim 在 trace 中 weight 低；默认仍可出现在 context（不 prune） |
| `prune_below=0.9` 极端 | 低 sim 句消失；空 keep 时 fallback 原文或 top-3 |
| `chunk_type=table` | 表 markdown 完整保留，无行被拆丢 |
| embedder=None | soft_rank/bge → join 原文，不抛 |
| 配置 `refiner.enabled=false` | 等价旧 `context_filter.mode` |

### 4.7 云上对照协议（Boot-R · Arm-Refiner）

| 臂 | 配置 | 测什么 |
|----|------|--------|
| **R0** | `mode=bge`（现状基线） | 100q RAGAS + E2E |
| **R1** | `mode=soft_rank`，`prune_below=null`，`keep_ratio=0.5` | 主对比 |
| **R2**（可选） | `soft_rank` + `prune_below=0.2` | 类 hard，预期风险高 |

冻结：`--skip-index` · colqwen2 · qwen2:7b · CRAG off · Gate2 off · expand/boost 与基线相同。

| 指标 | Go |
|------|-----|
| E2E Correct | R1 ≥ R0 − 0.02；理想 ↑ |
| Faith @100 | ≥ R0 − 0.02 |
| CtxRel | 可写观察，**不作上线否决** |
| latency p50 | R1 ≤ R0 × 1.15（encode 同源应接近） |
| 误拒 | 不显著 ↑ |

**No-Go：** Correct 掉 >2pt → 保持 `bge`；禁止把 prune 臂当默认。

### 4.8 实现顺序（P0-B）

```text
1. score_sentences 抽取 + soft_refiner soft_rank 纯函数单测
2. prepare_context / generator 接线 + 兼容 bge
3. trace + L4 cache 盐
4. 本地 10q 冒烟
5. 云 Boot-R 多臂（可与 P1 同机，见 §7）
```

---

## 5. P1-A — Search Planning（规则自适应选路）

### 5.1 问题

- 固定三路：Visual 弱却常开 → 噪声 + 延迟（综述：compulsive image retrieval 有害）。  
- 现 `VisualRouter` 只管 visual；无统一 `a_none` / text-only 契约。  
- `query_intent` 与 router 正则重复，维护分叉。

### 5.2 设计

**新建** `src/retrieval/search_planner.py`：

```python
@dataclass(frozen=True)
class SearchPlan:
    use_bm25: bool
    use_dense: bool
    use_visual: bool
    skip_retrieval: bool          # a_none：上层可直接生成/拒答
    intent_label: str             # none|table|visual|table+visual|...
    reason: str                   # 可观测短因
    # 预留，本阶段不实现 LLM：
    # sub_queries: list[str] = []
```

**动作空间（MVP）：**

| 动作 | 条件（规则） | 路径 |
|------|--------------|------|
| `a_none` | 可选；`planner.allow_skip_retrieval` 且命中闲聊/明显无文档需求正则 | 不检索（**默认关闭 allow**，防误伤工业问答） |
| `a_text` | 默认；或 table-ish 无 visual cue | bm25+dense，visual=false |
| `a_visual` | visual cue 或 `force_visual` | bm25+dense+visual |
| `a_full` | mode=always 或 planner disabled | 三路全开（兼容现状） |

**配置：**

```yaml
retrieval:
  search_planning:
    enabled: false                 # 总开关；false = 调用方 use_* 原样（现状）
    mode: heuristic                # heuristic | always_full | text_only
    allow_skip_retrieval: false    # 工业场景默认 false
    # heuristic 细则
    visual:                        # 复用/迁移 VisualRouter 词表
      on_cues: true                # 命中 visual cue → use_visual
      default_visual: false        # 未命中时是否开 visual（false=省 Visual）
    table_prefers_text: true       # table cue 不强制 visual
  # 废弃路径：visual_routing 保留 1 版兼容
  # search_planning.enabled=true 时优先 planner，忽略 visual_routing 或桥接
  visual_routing:
    enabled: false
    mode: heuristic
```

### 5.3 接线

| 位置 | 行为 |
|------|------|
| `PrismRAGRetriever.search_with_trace` | 入口：`plan = planner.plan(query, requested_flags)`；用 plan 覆盖 effective bm25/dense/visual |
| 黄金消融 `AblationConfig` | **不走 planner**（或 planner mode=always_full），保证 Full_zerank2 可比 |
| L3 cache key | 加入 `plan_salt`：`v={0/1},t={0/1},...` |
| Trace | `retrieval_trace.search_plan = {asdict}` |
| `/ask` | 响应 `search_plan` 可选字段 |

**合并正则：** `query_intent.detect` 为唯一 cue 源；`VisualRouter` 改为 thin wrapper 调 intent，或标记 deprecated。

### 5.4 单测

| 用例 | 期望 |
|------|------|
| enabled=false | plan 全 true（随请求） |
| "maximum pressure rating" | table intent；visual=false（default_visual=false） |
| "see figure 3 wiring" | visual=true |
| "hello" + allow_skip | skip_retrieval=true（仅当 allow） |
| ablation always_full | 三路开 |

### 5.5 云对照（Boot-R · Arm-Plan）

| 臂 | 配置 |
|----|------|
| **P0** | planning off（基线） |
| **P1** | planning on, heuristic, default_visual=false |

| 指标 | Go |
|------|-----|
| Full 路径 NDCG@10（若评检索） | ≥ 基线 −1pt |
| E2E Correct | ≥ 基线 −0.02；理想持平或↑ |
| 平均 latency | **↓ ≥10%** 或 Visual 调用率↓（主收益） |
| Visual 调用率 | 写入 run README |

**No-Go：** Correct 掉 >2pt → 保持 planning off。

### 5.6 明确不做（P1-A）

- LLM Retrieval Classification  
- 多跳 sub-query 分解（记 P2；Failure Clinic multi-hop 子集再开）  
- HyDE 与 planner 绑定（HyDE 保持独立开关）

---

## 6. P1-B — Cross-ref Expand

### 6.1 问题

手册常见：「见表 3-2」「Fig. 12」「参照 §4.1」。  
命中正文引用句但**未命中表体** → 答不全。  
B1 page/prev_next expand 解决邻块，**不解析引用符号**。

### 6.2 设计

**扩展** `src/retrieval/expand.py`（或 `crossref.py`）：

```yaml
retrieval:
  crossref_expand:
    enabled: false
    stage: post_rerank             # 与 neighbor 一致，省 rerank 成本
    max_extra: 3                   # 每 query 最多扩几块
    max_per_hit: 1
    patterns:                      # 可配置；内置中英
      - "(?i)\\b(?:see\\s+)?(?:table|tbl\\.?)\\s*([A-Z]?\\d+[\\-\\.]?\\d*)"
      - "(?i)\\b(?:see\\s+)?(?:figure|fig\\.?)\\s*([A-Z]?\\d+[\\-\\.]?\\d*)"
      - "(见|参见|参照)\\s*表\\s*([\\d\\.\\-]+)"
      - "(见|参见|参照)\\s*图\\s*([\\d\\.\\-]+)"
    match_fields: [text, caption, table_summary, section_path]
    same_doc_only: true
```

**算法：**

```text
1. 对 top hits 文本跑 patterns → 抽出 refs: [(kind, id_str), ...]
2. 候选检索（pg_store 新方法，批量）:
   get_chunks_matching_ref(doc_id, kind, id_str) 
   实现 MVP：
     SQL: chunk_type IN ('table','image','text')
          AND doc_id = ?
          AND (
            caption ILIKE '%Table 3-2%' OR text ILIKE '%Table 3-2%'
            OR table_summary ILIKE ...
          )
   注意：防过宽；优先 caption/table 头行；limit 3 per ref
3. 去重后 append，标 retrieval_type=crossref_expand，继承 parent score * 0.95
4. cap 截断；trace.crossref = {refs, added_ids}
```

**与 B1 顺序：**

```text
fuse → boost → neighbor_expand? → rerank → neighbor_expand? → crossref_expand?
```

crossref 放 **post_rerank** 更稳（只对最终 top-k 解析引用）。  
若 neighbor 与 crossref 同时开：先 neighbor 再 crossref（引用可能在邻居正文里）。

### 6.3 pg_store API

```python
def find_chunks_by_ref(
    self,
    doc_id: str,
    needles: Sequence[str],
    *,
    limit: int = 5,
) -> List[dict]:
    """ILIKE any needle on caption/text/table_summary; same doc."""
```

单测用 fixture 内存 mock，不强制本机 PG。

### 6.4 单测

| 用例 | 期望 |
|------|------|
| 正文 "See Table 2-1" + 库中有 table caption | 扩入 table chunk_id |
| enabled=false | 结果不变 |
| 无匹配 | added=0 |
| 跨 doc 引用 + same_doc_only | 不扩 |
| 与 neighbor 叠加 | 去重正确 |

### 6.5 云对照

| 臂 | 配置 |
|----|------|
| 基线 | crossref off |
| C1 | crossref on，post_rerank |

优先 **表子集 / 含 see table 的 E2E 子集**；全量 100q 不崩（NDCG −1pt 内）。

**Go：** 子集 Correct 或 miss↓；全量不掉 >2pt Correct；延迟 < +10% p50。

---

## 7. 依赖、排期与云打包

### 7.1 依赖图

```text
P0-A 收口 ──┬──► P0-B Refiner（独立，不依赖 planning）
            │
            ├──► P1-A Planning（独立）
            │
            └──► P1-B Crossref（依赖 pg 可查 caption/text；A3 已满足）
                      └── 可选：与 B1 neighbor 同 stage 协作

本地并行：P0-B ∥ P1-A ∥ P1-B 代码
云：同一 Boot-R 多臂
```

### 7.2 建议工期

| 包 | 本地工程 | 云 |
|----|----------|-----|
| P0-A | 0.5d | 0 |
| P0-B | 1.5–2d | 与 P1 同机 |
| P1-A | 1d | 同机 |
| P1-B | 1–1.5d | 同机 |
| 文档/handoff | 0.5d | — |

### 7.3 Boot-R 臂矩阵（唯一开机）

| 臂 ID | Refiner | Planning | Crossref | Expand B1 | 目的 |
|-------|---------|----------|----------|-----------|------|
| **Base** | bge | off | off | off | 基线 |
| **R1** | soft_rank | off | off | off | P0 |
| **Pl** | bge | on | off | off | P1-A |
| **Cr** | bge | off | on | off | P1-B |
| **Best**（可选） | soft_rank | on | on | 按历史阳性 | 组合 |

任务：每臂 RAGAS 100q + E2E；表子集若有则加。  
`index_version`：本方案 **不强制 re-index**（不改 embed）；若仅开 crossref/planning/refiner → `--skip-index` 即可。

### 7.4 分支 / PR 策略

| 策略 | 说明 |
|------|------|
| 单 PR 串行 | `feat/p0-p1-retrieval-context` 含全部，默认全 false |
| 拆 PR | ① refiner ② planner ③ crossref — 更易 review |
| main | 禁止直接改；合入后默认开关保持 false 直至 Boot-R 阳性写 handoff |

---

## 8. 配置总表（目标态摘录）

```yaml
refiner:
  enabled: true
  mode: bge                      # Boot 阳性后可改 soft_rank
  soft_rank:
    min_sim: 0.25
    prune_below: null
    keep_ratio: 0.5
  protect_table_chunks: true
  adaptive_ratio:
    enabled: false
  emit_trace: true

retrieval:
  search_planning:
    enabled: false
    mode: heuristic
    allow_skip_retrieval: false
    visual:
      on_cues: true
      default_visual: false
    table_prefers_text: true
  crossref_expand:
    enabled: false
    stage: post_rerank
    max_extra: 3
    max_per_hit: 1
    same_doc_only: true
  # 已有，保持
  neighbor_expand: { enabled: false, ... }
  modality_boost: { enabled: false, ... }
  crag: { enabled: false, ... }
  visual_routing: { enabled: false, ... }   # planning 上位后兼容

context_filter:
  mode: bge                      # 兼容别名；与 refiner.mode 冲突时以 refiner 为准

generation:
  self_rag: { enabled: false, ... }
```

---

## 9. 验收与决策纪律

### 9.1 本地退出（开机前）

- [ ] `pytest` 相关模块全绿  
- [ ] 默认配置下黄金消融路径行为 = 现状（planning/crossref/soft 均不改变）  
- [ ] trace 字段有单测或契约测试  
- [ ] Agents.md：无本机大模型下载 / 全量评测  

### 9.2 云退出（Boot-R）

| 决策 | 条件 |
|------|------|
| `soft_rank` 可试开 | R1 Correct/Faith 达标 |
| `search_planning` 可试开 | latency↓ 且 Correct 不掉 |
| `crossref_expand` 可试开 | 子集增益 + 全量不崩 |
| 全部保持 false | 任一否决项触发 |

### 9.3 Failure Clinic 标签（可选增强）

| 现象 | 标签 |
|------|------|
| soft prune 过狠误拒 | P01 / P05 |
| planning 错关 visual 导致图题 miss | P05 |
| crossref 扩错表 | P02 / P11 |
| CtxRel↑ Correct↓ | P09 |

---

## 10. 风险与回滚

| 风险 | 缓解 |
|------|------|
| soft_rank 上下文变长 → 噪声 | keep_ratio + 保序；表保护 |
| prune 变 hard CRAG | 默认 `prune_below: null` |
| planning 漏 visual 图题 | on_cues 词表扩展；force 配置 |
| crossref ILIKE 误匹配 | same_doc + limit + caption 优先 |
| cache 脏 | L4 盐含 mode；改默认时 bump |
| 与 CRAG 叠乘 | 评测禁止同时开；文档声明互斥 |

回滚：全部开关 false / mode=bge → 行为回到合入前。

---

## 11. 任务 Checkbox 总表（执行用）

### P0-A
- [ ] P0-A.1 gap 清单  
- [ ] P0-A.2 单测  
- [ ] P0-A.3 配置纪律  
- [ ] P0-A.4 handoff 链接  

### P0-B Refiner
- [ ] P0-B.1 `score_sentences` + `soft_rank` 纯函数  
- [ ] P0-B.2 `refiner.py` + 配置  
- [ ] P0-B.3 generator / context_filter 接线  
- [ ] P0-B.4 trace + L4 盐  
- [ ] P0-B.5 `tests/test_refiner.py`  
- [ ] P0-B.6 本地 10q  
- [ ] P0-B.7 Boot-R Arm R0/R1  

### P1-A Planning
- [ ] P1-A.1 `SearchPlan` + cue 合并  
- [ ] P1-A.2 `search_with_trace` 接线 + cache 盐  
- [ ] P1-A.3 消融旁路  
- [ ] P1-A.4 单测  
- [ ] P1-A.5 Boot-R Arm Pl  

### P1-B Crossref
- [ ] P1-B.1 正则 + `find_chunks_by_ref`  
- [ ] P1-B.2 expand 流水线 + trace  
- [ ] P1-B.3 单测  
- [ ] P1-B.4 Boot-R Arm Cr  

### 收尾
- [ ] run README 归档 `runs/YYYYMMDD-boot-r/`  
- [ ] handoff 默认建议更新  
- [ ] `docs/architecture/` 补 retrieval / generation 片段  

---

## 12. 参考

- 综述：A Survey on Multimodal RAG, arXiv:2504.08748 §3.2 Planning · §3.3.3 Refiner  
- 阴性实验：`runs/20260722-crag-on/` · `runs/20260721-self-rag-on-only/`  
- 入库主线：`docs/superpowers/plans/2026-07-23-content-pipeline-phase-ab-roadmap.md`  
- 瓶颈：`docs/bottleneck-analysis-2026-07-07.md`  

---

## 附录 A — 与「只写方案不写代码」边界

本文件是 **实现规格 + 任务拆分**，不含生产默认翻 true。  
执行代理应从 **P0-B.1** 起写代码；P0-A 仅验证。  
若只需其中一包：优先 **P0-B**（直接打上下文噪声），其次 P1-A（延迟/Visual 误伤），再次 P1-B（引用类 badcase）。
