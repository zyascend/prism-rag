# PrismRAG Agentic RAG（LangGraph）— 设计文档

> brainstorm 日期: 2026-08-03  
> 分支: `docs/agentic-rag-langgraph-design`（spec）→ 实现切 `feat/agentic-rag-langgraph`  
> 配套: `handoff.md` · `docs/self-rag-closed-loop-design-2026-07-09.md` · `docs/superpowers/specs/2026-06-30-prismrag-longterm-roadmap-design.md`  
> 状态: **已确认**（方案 1′ · 产品纪律 + LangGraph 主特性学习）

---

## 0. 背景与决策（已确认）

### 0.1 现状缺口

PrismRAG 当前是强 **固定流水线**，不是 agent：

```text
query → search_planning → BM25+Dense+Visual → RRF → rerank
     → [可选 CRAG] → context_filter → generate → [可选 Gate2]
```

| 已有「类 agent」能力 | 形态 | 默认 | 结论 |
|----------------------|------|------|------|
| Search Planning | 规则路由 | 开 | latency 友好 |
| CRAG | grade → rewrite → 再检索 | **关** | CtxRel↑ 但 E2E Correct −12pt |
| Self-RAG Gate2 | 生成后忠实性门 | 关 | 可选 |
| Failure Clinic P06 | multi-hop 诊断 | — | 点名多跳缺口 |

长期路线图（`2026-06-30-...-roadmap`）写过 ReACT + 窄工具集，但 **ReACT / LangGraph / KG 均未落地**。handoff 当前也不默认做全量 KG。

### 0.2 脑暴结论一览

| 维度 | 选择 |
|------|------|
| 主目标 | **C** 可控工具编排（LangGraph） |
| 业务能力 | **A** 多跳 / 拆问（依赖拆问 + 多次 search，**不**先建 KG） |
| 成功标准 | **双轨**：先架构可演示 → 再云上子集可辩护；默认 off |
| 工具面 MVP | 极窄：`decompose_query` · `knowledge_search` · `grade_evidence` · `refine_subquery` · `synthesize_answer` |
| 拓扑 | **固定多跳图**（非默认自由 ReAct） |
| 入口 | `agent.enabled` + 请求 `mode=agent` |
| 实现方案 | **方案 1′**：薄封装现有 retriever/generator + **升配** LangGraph 主特性 |
| 学习目标 | 借本能力 **系统覆盖 LangGraph 主特性**（见 §6 Feature Map） |
| HITL | 配置可开，**默认关**（decompose 后 interrupt 审子问） |

### 0.3 方案取舍

| 方案 | 说明 | 决议 |
|------|------|------|
| 1 薄固定图 | 改动小、可控 | 产品骨架保留 |
| 1′ 教学完整固定图 | 1 + 条件边/环/Send/子图/tools/stream/checkpoint/HITL | **采用** |
| 2 厚 Agent 运行时 | 过重，易滑向通用框架 | 否 |
| 3 全量迁 CRAG/Gate2 进一张图 | 重写主路径风险大 | 否 |

---

## 1. 目标与非目标

### 1.1 目标

1. **产品**：在默认 pipeline 不动的前提下，增加可开关的 LangGraph agent 旁路；支持拆问多跳检索与合成答案。  
2. **可控**：动作空间窄、步数/调用硬上限、轨迹可回放；避免 CRAG 式「LLM 乱改写伤 Correct」。  
3. **学习**：在真实 `knowledge_search` 上覆盖 LangGraph 主特性（§6），可按 Feature Map 逐项验证。  
4. **验收双轨**：Phase1 单测 + demo/轨迹；Phase2 云上 ~40–50 子集 pipeline vs agent；**无 Go 不改默认**。

### 1.2 非目标

| 不做 | 原因 |
|------|------|
| 自由 ReAct 作默认生产路径 | 可控性差；CRAG 教训 |
| `web_search` | 路线图与安全边界 |
| 全量 / LightRAG KG | handoff / 工期；多跳先靠 decompose |
| 通用 Agent 平台 / 插件市场 | 作品边界 |
| 多 Agent 协作网 | 评测爆炸；后期可选 |
| Store 业务化长期记忆 | 工业 PDF 单次问答价值低 |
| 默认 `agent.enabled: true` | 无 Phase2 Go 前禁止 |
| 黄金 NDCG / 消融走 agent | 污染检索尺子 |
| 子问依赖链（串行填槽） | v1.1+ |
| 本机全量 283 / 大模型下载 | AGENTS.md |

---

## 2. 架构：状态与图拓扑

### 2.1 主图（教学完整固定图）

```text
START
  → decompose
  → route_complexity
        ├─ atomic  → retrieve_one → synthesize → finalize → END
        └─ multi   → fan_out (Send API)
                        → [retrieval_subgraph × N]
                        → fan_in (reducer 合并 evidence / trajectory)
                        → grade
                             ├─ sufficient → synthesize → finalize → END
                             ├─ insufficient & budget → refine → (有限 cycle 回检索)
                             └─ exhausted → synthesize | abstain → finalize → END
```

可选旁路（默认关）：

| 旁路 | 配置 | 说明 |
|------|------|------|
| HITL | `agent.hitl.review_subqueries` | `decompose` 后 `interrupt`，人工确认/改子问再 resume |
| ReAct demo 图 | `agent.react_demo.enabled` | 仅 `knowledge_search` 的对照自由环；**不进正式评测** |

### 2.2 Graph State

| 字段 | 更新模式 | 说明 |
|------|----------|------|
| `query` | 覆盖 | 用户原问 |
| `subqueries` | 覆盖 | decompose 产出 |
| `evidence` | **`Annotated[list, operator.add]`** | 各路检索追加 |
| `trajectory` | **`Annotated[list, operator.add]`** | `StepRecord` 追加 |
| `messages` | 可选 `add_messages` | tool-calling 节点需要时 |
| `answer` | 覆盖 | 最终答案 |
| `citations` | 覆盖 | 与现有 `/ask` 对齐 |
| `status` | 覆盖 | `ok \| abstain \| error \| degraded \| interrupted` |
| `budget` | 结构体覆盖 | 剩余 search / llm / grade 轮次 |
| `hitl` | 覆盖 | 暂停态、待审子问 |
| `meta` | 合并 | `trace_id`、`thread_id`、counts、degraded 标志 |

`EvidenceItem` / `StepRecord` 全程可 JSON 序列化（API + Trace + 回放）。

#### StepRecord

```text
step, node, tool, input_summary, output_summary,
ok, error, latency_ms,
counts: { searches?, subqueries?, evidence_n? }
```

### 2.3 节点职责

| 节点 | 职责 | MVP |
|------|------|-----|
| `decompose` | 原问 → 1..N 自包含子问 | 必做 |
| `route_complexity` | atomic vs multi（可看 len(subqueries) 或 strategy） | 必做 |
| `retrieve_one` / `retrieval_subgraph` | 单子问 `knowledge_search` | 必做 |
| fan_out / fan_in | Send 并行 + reducer 合并 | 必做（P1b） |
| `grade` | 证据是否足以答 **原问** | 图上默认实现；可用配置关短路 |
| `refine` | 仅改写不足子问 | 随 grade 环 |
| `synthesize` | 合并 evidence 生成 + 拒答口径 | 必做 |
| `finalize` | 填响应元数据、截断 trajectory | 必做 |
| `fallback_single_query` | 拆问失败 → `subqueries=[query]` | 必做 |

**并行假设（MVP）：** 子问互相独立。子问间依赖（用 Q1 答案填 Q2）为 v1.1+。

### 2.4 硬护栏（配置强制，不靠 prompt）

| 配置键 | 默认建议 | 含义 |
|--------|----------|------|
| `agent.enabled` | `false` | 总开关 |
| `agent.max_subqueries` | `3` | 最多子问 |
| `agent.max_search_per_subquery` | `1` | 每子问检索次数 |
| `agent.max_total_searches` | `3` | 全图 search 上限 |
| `agent.max_llm_calls` | `6` | 含 grade/refine 余量 |
| `agent.max_grade_cycles` | `1` | grade⇄refine 环上限 |
| `agent.timeout_ms` | `30000` | 整图软超时 |
| `agent.grade.enabled` | `true` | 学习环路需要；评测可关对照 |
| `agent.checkpoint.enabled` | `true` | MemorySaver（进程内） |
| `agent.hitl.review_subqueries` | `false` | HITL |
| `agent.react_demo.enabled` | `false` | ReAct 对照图 |
| `agent.on_error` | `degrade_pipeline` | 图崩溃策略 |
| `agent.return_trajectory` | `true` | API 是否回传 trajectory |

---

## 3. 工具契约

原则：主路径由 **图结构** 决定何时调工具；模型主要 **填参数**。校验、预算、超时在工具/节点层强制执行。

### 3.1 工具一览

| Tool | 输入要点 | 输出要点 |
|------|----------|----------|
| `decompose_query` | `query`, `max_subqueries` | `subqueries`, `strategy` (`atomic`\|`multi`), `reason` |
| `knowledge_search` | `query`, `subquery_id`, `top_k?` | `hits[]`（chunk/page/score/modality…）, `search_meta` |
| `grade_evidence` | `query`（原问）, `evidence` | `sufficient`, `missing[]`, `score` |
| `refine_subquery` | 不足子问 + missing 提示 | 新子问文本 |
| `synthesize_answer` | 原问 + evidence | `answer`, `citations`, `rejected` |

实现形态：`langchain_core.tools` / LangGraph `@tool`；节点内调用或 ToolNode（ReAct demo）。

### 3.2 行为约定

**decompose**

- 原子题允许返回单元素 `[query]`。  
- 子问必须自包含（禁止「该值是多少」式指代）。  
- 解析失败 / 超时 / 空列表 → fallback 单问；**不**做无效重试（省延迟）。  
- Prompt：`agent_decompose`（PromptRegistry，versioned）。

**knowledge_search**

- **唯一实现**：包装现有检索门面（与 `/ask` pipeline 同核：planning / RRF / rerank / 当前默认 soft_rank、crossref 等）。  
- **不**把 `use_bm25` / `use_visual` 暴露给 LLM；由 search_planning 决定。  
- 触顶 `max_total_searches` → 跳过剩余子问并记 `search_budget_exhausted`。  
- 空结果不重试。  
- **禁止** web；HyDE 不在 agent 层单独发明（与全局配置一致即可）。

**grade / refine**

- 与 CRAG 教训对齐：可整臂关闭；开启必须进 Phase2 对照。  
- refine **禁止** HyDE 式长假文档、禁止 web。  
- `max_grade_cycles` 用尽 → synthesize 或 abstain（按证据是否为空）。

**synthesize**

- 复用 `Generator` + 现有压缩/表格保护。  
- Prompt：`agent_synthesize`（多证据分节）或复用 `answer_generation`。  
- 证据空/不足 → 现有拒答口径（`rejection` / `ABSTAIN`）。  
- citations **从 evidence 构造**，不信任模型自报。  
- Gate2：**MVP 不绑**；可后续后置配置。

### 3.3 错误与降级

| 失败点 | 策略 |
|--------|------|
| decompose 失败 | `subqueries=[query]` |
| 单次 search 异常 | 跳过该子问，其它继续 |
| 全部 search 失败 / 全空 evidence | 拒答 |
| synthesize 异常 | `status=error`；`on_error=degrade_pipeline` 时整单回退 pipeline |
| 超 `timeout_ms` | 尽力用已有 evidence synthesize；否则 abstain/degrade |

---

## 4. API / 配置 / Trace / 运行时

### 4.1 API

**保持单一 `POST /ask`**，不新增默认生产端点 `/ask/agent`（HITL resume 可增加辅助接口，见下）。

#### 请求增量

| 字段 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `mode` | `"pipeline" \| "agent"` | `"pipeline"` | 仅 `agent.enabled=true` 时 `agent` 生效 |

`mode=agent` 时请求级 `use_bm25` 等 **MVP 忽略**（由内部 planning 决定），减少对照噪声。

#### 响应增量

```text
agent: null | {
  used: bool,
  status: "ok" | "abstain" | "error" | "degraded" | "interrupted",
  subqueries: string[],
  trajectory: StepRecord[],
  counts: { subqueries, searches, llm_calls, evidence_n },
  degraded_to_pipeline: bool,
  thread_id: string | null,
  ignored_reason: string | null
}
```

- `mode=pipeline` → `agent=null`。  
- L4 答案缓存 key 必须含 `mode=agent` + `agent_config_hash`，避免与 pipeline 串味。

#### Streaming（学习 / demo）

- `stream=true` 或 demo SSE：`stream_mode` 含 `updates` / `values`。  
- 生产默认仍一次 `invoke` 返回完整 `AskResponse`。

#### HITL resume

- `agent.hitl.review_subqueries=true` 时，decompose 后可 `interrupt`。  
- 首响应 `status=interrupted` + 待审 `subqueries` + `thread_id`。  
- 继续：`POST /ask/resume`（或等价：`thread_id` + 修订子问 / approve）。  
- **评测强制 `hitl=false`**。

### 4.2 配置块（`config/models.yaml` 示意）

```yaml
agent:
  enabled: false
  max_subqueries: 3
  max_search_per_subquery: 1
  max_total_searches: 3
  max_llm_calls: 6
  max_grade_cycles: 1
  timeout_ms: 30000
  on_error: degrade_pipeline   # degrade_pipeline | abstain | error
  return_trajectory: true
  grade:
    enabled: true
  checkpoint:
    enabled: true
  hitl:
    review_subqueries: false
  react_demo:
    enabled: false
  decompose:
    prompt_id: agent_decompose
  synthesize:
    prompt_id: agent_synthesize
```

纪律：

- 默认保守；**禁止**无 Phase2 Go 改 `enabled: true`。  
- `run_eval` / 黄金 NDCG **强制 pipeline**（`apply_search_planning` 等逻辑不变；agent 不介入）。  
- 评测脚本强制 `hitl=false`、可选 `grade.enabled` 双臂。

### 4.3 `/ask` 调用流

```text
POST /ask
  → 校验（现有）
  → L4 lookup（key 含 mode + agent salt）
  → if agent.enabled and mode==agent:
        run_agent(invoke|stream)
        if interrupted → 返回 interrupted（若 HITL）
        if error and on_error==degrade_pipeline:
            pipeline；degraded_to_pipeline=true
        else:
            AskResponse + agent.*
  → else:
        existing pipeline
  → L4 store / Trace / return
```

评测：`answer_for_eval(..., mode="agent")` 或 `agent_answer_for_eval`，默认仍 pipeline。

### 4.4 Checkpoint

- `thread_id = trace_id`（或客户端显式传入）。  
- MVP：`MemorySaver`（进程内）。  
- 多 worker 持久化 checkpoint **本阶段不做**（文档注明扩展点）。  
- 与 `trajectory` 互相印证；resume 依赖 checkpoint。

### 4.5 Trace / 可观测

| 层 | 内容 |
|----|------|
| Parent span | `agent` |
| 子 span | `decompose` · `search[i]` · `grade` · `refine` · `synthesize` · `hitl` |
| metadata | subqueries、counts、budget、error、degraded、thread_id |
| 落盘 | `api_traces.jsonl` 写 trajectory（截断策略对齐 Gate2） |
| 可视化 | `graph.get_graph().draw_mermaid()` → docs 或 demo |
| 指标（可选） | `agent_requests_total` · avg searches · `agent_degrade_total` |

### 4.6 Demo

- Mode：**Pipeline | Agent**（传 `mode`）。  
- 展示：subqueries、trajectory 时间线、citations。  
- 可选：stream 逐步更新；HITL 审子问 UI（配置开时）。  
- Fixture：1～2 条多跳假轨迹，无 GPU 可点。

---

## 5. 模块布局与依赖

### 5.1 目录

```text
src/agent/
  __init__.py
  state.py         # AgentState, EvidenceItem, StepRecord, reducers
  tools.py         # @tool 适配 retriever / generator / LLM
  subgraphs.py     # retrieval_subgraph
  graph.py         # 主 StateGraph 编译
  react_demo.py    # 可选 ReAct 对照图
  runner.py        # invoke / stream / resume；on_error；超时
  config.py        # 读 agent.* 
  checkpoint.py    # MemorySaver 装配

src/prompts/...    # agent_decompose.yaml, agent_synthesize.yaml
src/api/routes.py  # mode 分支、agent 字段、cache salt、resume
config/models.yaml # agent: 块
tests/test_agent_*.py
static/demo/       # Mode + trajectory（P1d）
data/agent_eval_qa.json          # Phase2
scripts/run_agent_eval.py        # Phase2
```

### 5.2 依赖方向

```text
api/routes → agent.runner → agent.graph → agent.tools / subgraphs
                               ↓
         retrieval.* · generation.Generator · prompts · observability
```

- `agent` 依赖现有检索/生成；**禁止** retrieval 反向依赖 agent。  
- 消融 / vidore 主路径不默认挂 agent。

### 5.3 外部依赖

| 包 | 用途 |
|----|------|
| `langgraph` | StateGraph、Send、checkpoint、interrupt 等 |
| `langchain-core`（按需） | `@tool`、messages；**少引** LangChain 全家桶 |

- 钉版本写入 `pyproject.toml` / lock。  
- **懒编译** graph（首次 `mode=agent`），降低纯 pipeline 冷启成本。  
- 节点 **注入** 已有 retriever/generator/`complete_fn`，与 `/ask` 共用实例。

### 5.4 核心接口（示意）

```text
@dataclass
class AgentResult:
    answer: str
    citations: list
    status: Literal["ok", "abstain", "error", "degraded", "interrupted"]
    subqueries: list[str]
    trajectory: list[StepRecord]
    counts: dict
    thread_id: str | None
    error: str | None

def run_agent(query, *, retriever, generator, complete_fn, cfg=None, trace_id=None) -> AgentResult
def stream_agent(...) -> Iterator[agent updates]
def resume_agent(thread_id, command, ...) -> AgentResult
```

### 5.5 与现有闭环关系

| 模块 | Agent MVP |
|------|-----------|
| search_planning / soft_rank / crossref | search 内部照旧 |
| CRAG | **不**接入主图 |
| Self-RAG Gate2 | **不**接入主图；可后置 |
| L3/L4 cache | L3 在 retriever；L4 加 mode+配置盐 |
| Rejection | synthesize 复用 |
| Failure Clinic | Phase2 后可用轨迹增强 P06（非阻塞） |

---

## 6. LangGraph Feature Map（学习清单）

| 特性 | 落点 | 如何验证 |
|------|------|----------|
| StateGraph + 类型 State | `graph.py` / `state.py` | 单测 compile |
| Nodes / Edges | 主图各节点 | mermaid 导出 |
| Conditional edges | `route_complexity`、grade 出口 | 原子 vs 多跳 fixture |
| Cycles + 退出 | grade⇄refine | `max_grade_cycles` 单测 |
| Reducers | `evidence` / `trajectory` | 并行后 list 长度 |
| Send map-reduce | 多子问扇出 | N 子问 N 次 search span |
| Subgraph | `retrieval_subgraph` | 子图独立单测 |
| @tool / ToolNode | `tools.py`、react_demo | schema + 调用 |
| Checkpoint | MemorySaver + `thread_id` | 中断后 resume |
| Stream | updates / values | demo 逐步 UI |
| interrupt HITL | decompose 后 | 配置打开后 resume |
| RetryPolicy | search / LLM 节点 | mock 失败重试 |
| 可视化 | `draw_mermaid` | docs / demo |
| ReAct 对照 | `react_demo.py` | 开关打开；**不进 P2 主表** |
| Store / 多 Agent | — | **明确不做** |

---

## 7. 评测与 Phase 切分

### 7.1 原则

| 原则 | 落地 |
|------|------|
| 不污染检索尺子 | NDCG / Boot 消融永不走 agent |
| 合入门槛 ≥ 检索 | 先 L0/L1，再 L2；禁止只看最终答案 |
| 双轨 | P1 可演示 → P2 可辩护 → 默认仍 off |
| CRAG 教训 | Correct + 误拒 + latency + avg searches 一起看；CtxRel 不作单独决策 |

### 7.2 三层评测

| 层 | 测什么 | Phase |
|----|--------|-------|
| L0 契约 | 编译、护栏、Send 合并、环退出、interrupt mock | P1 必做 |
| L1 轨迹 | 原子不乱拆；多跳 ≥2 自包含子问；search ≤ budget；fallback | P1 fixture + P2 抽检 |
| L2 端到端 | Correct / 拒答 / latency / counts | P2 云上子集 |

暂缓：50 条人工金标完整 Thought 轨迹、异家族 judge 投票（后续增强，不阻塞 MVP）。

### 7.3 Phase2 子集

| 标签 | 规模 | 用途 |
|------|------|------|
| multi_hop | 15–25 | 需多处证据 |
| atomic | 15–25 | 验证不伤简单题 |
| reject | ~10 | 库外/不可答 |
| **合计** | **~40–50** | pipeline vs agent |

存放：`data/agent_eval_qa.json`（或 e2e id 列表 + tag）。

### 7.4 对照协议

| 项 | 约定 |
|----|------|
| 臂 | pipeline（base）vs agent（同索引、同生成模型） |
| 冻结 | `--skip-index` · 记录 env · 与近期 Boot 同 backbone |
| 主指标 | E2E Correct；拒答准确；可答题误拒数 |
| 成本 | p50/p95 latency；avg searches；avg llm_calls；degrade 次数 |
| 通过草案 | ① 原子 Correct 不显著低于 pipeline（建议 Δ ≥ −2pt 或并列）② 多跳 Correct ≥ pipeline ③ avg searches ≤ budget ④ 无大面积 degrade |
| 失败 | 保持 enabled false；`runs/YYYYMMDD-agent-*/README` 写阴性结论 |

可选第三臂：`grade.enabled=false` 隔离 grade 贡献。

### 7.5 Phase 交付

| Phase | 交付 | LangGraph 点 |
|-------|------|----------------|
| **P1a** | `src/agent` invoke + `/ask mode=agent` + 单测 | State、节点、条件边、reducer、基础环 |
| **P1b** | Send 并行 + retrieval 子图 + @tool | Send、subgraph、tools |
| **P1c** | stream + checkpoint + HITL 开关 + mermaid | stream、MemorySaver、interrupt |
| **P1d** | Demo Mode + trajectory；（可选）ReAct 对照 | 端到端体感 |
| **P2** | 子集 + 双臂云跑 + handoff 决议 | 业务验证 |

分支：`feat/agentic-rag-langgraph`。P1 分包可合 main（默认 off）。`enabled: true` 仅 P2 Go 后另开配置 PR。

### 7.6 工作量粗估

| Phase | 量级 |
|-------|------|
| P1a–P1d | 多个中等 PR；本地 mock 可测；禁本机全量 |
| P2 | 半日～1 日云：子集 + 双臂 + 结论 |

---

## 8. 风险与缓解

| 风险 | 缓解 |
|------|------|
| 拆问质量差（小模型 JSON 不稳） | fallback 单问；强解析；0 次无效重试 |
| 延迟 ×N | budget 硬顶；原子单元素；默认 off |
| 简单题被拆坏 | Phase2 原子子集门禁 |
| grade 重蹈 CRAG 阴性 | 可关；P2 含 grade on/off |
| HITL 干扰评测 | 默认关；评测强制 false |
| P1 范围膨胀 | 严格 P1a→P1d；每包可独立合入 |
| 依赖/版本 | 钉 langgraph；CI mock 不拉模型 |
| scope → 通用 Agent 平台 | §1.2 硬边界；web/多租户/插件否决 |

---

## 9. 成功标准（回扣）

| 阶段 | 成功 |
|------|------|
| Phase1 | `mode=agent` 可演示；trajectory /（可选）stream 可见；Feature Map P0/P1 可逐项验证；单测绿；默认 pipeline **零行为变化** |
| Phase2 | 子集上多跳不差于/优于 pipeline，原子不伤，成本在 budget 内；handoff 书面 Go/No-Go |
| 叙事 | 「私有库 + LangGraph 固定图 + 窄工具 + 可回放轨迹 + 框架特性可点名」，不是套壳 ChatBot |

---

## 10. 实现顺序建议（进入 writing-plans 前）

1. 依赖与 `agent.config` / State / 空 graph compile  
2. tools 适配 mock retriever → P1a 直线可跑  
3. 条件边 + grade 环 + 护栏单测  
4. Send + subgraph  
5. `/ask` 接入 + cache salt + Trace  
6. checkpoint + stream + HITL  
7. Demo + mermaid  
8. Phase2 数据与脚本  

下一文档：`docs/superpowers/plans/2026-08-03-agentic-rag-langgraph.md`（由 writing-plans 产出）。

---

## 11. 修订记录

| 日期 | 变更 |
|------|------|
| 2026-08-03 | 初版：脑暴确认方案 1′（C+A、双轨、固定图、窄工具、LangGraph Feature Map、HITL 默认关） |
