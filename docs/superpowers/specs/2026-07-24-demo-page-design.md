# PrismRAG Demo 页 — 设计文档

> brainstorm 日期: 2026-07-24  
> 分支: `docs/demo-page-design`（spec）→ 实现切 `feat/demo-page`  
> 配套: `handoff.md` · `src/api/routes.py` · `README.md` · 简历叙事 `local/resume-prismrag.md`（私有）

---

## 0. 背景与决策（已确认）

| 维度 | 结论 |
|------|------|
| 用户诉求 | 打造可演示的 **demo 页**；先设计再写代码 |
| 形态 | **Hybrid**：默认 Showcase + 可展开「工程师面板」 |
| 实现 | **单页静态前端**（HTML/CSS/JS），无 Node / React / Streamlit |
| 数据 | **Fixture 默认** + **Live API 开关** |
| 布局 | **左右分栏**（左问/答/引用/上传，右三路 Trace + 工程面板） |
| MVP 范围 | ① 问答+Citations ② 预设问句 ③ 三路 retrieval_trace ④ 工程师面板 ⑤ 评测角标 ⑥ PDF 上传 |
| 明确不做 | 纯 `/search` 模式、实时评测管道、默认开 CRAG/Gate2、新 Python 重依赖 |

**动机：** 仓库已有可解释 API（`/ask` 返回 answer、citations、三路 `retrieval_trace`、可选 `self_rag`/`crag`），但 **无 Web 前端**。面试与自用需要「看得见的链路」，且本机不宜全量索引/大模型，故必须支持离线 Fixture。

---

## 1. 目标与非目标

### 1.1 目标

1. 面试约 10 分钟能讲清：三路召回 →（精排叙事）→ 生成引用 → Trace 排障面。
2. 本机 **零 GPU / 零后端** 也能打开页面完成完整 UI 演示（Demo 模式）。
3. 云上或本机 API 就绪后，切 Live 可真实 `/ask`，并可 `/ingest` 上传 PDF。
4. 与现有 `AskResponse` **字段对齐**，尽量 **零 schema 变更**。

### 1.2 非目标（v1）

| 不做 | 原因 |
|------|------|
| 纯 `/search` 模式 | 延后；工程师仍可用 Swagger/curl |
| 接 RAGAS/消融管道 | 角标用静态可辩护数字即可 |
| React/Next/Streamlit | 仓库摩擦与依赖成本 |
| 默认打开 CRAG / Gate2 | 云上实验阴性或默认关；面板只展示字段 |
| 像素级设计系统 | 工业简洁即可 |

---

## 2. 信息架构

```text
┌─ Top bar ──────────────────────────────────────────────────────────┐
│ Logo · 副标  │  metrics chips  │  Demo | Live  │ API base │ health │
├─────────────────────────────┬──────────────────────────────────────┤
│ LEFT                        │ RIGHT                                │
│ · PDF Upload（Live）        │ · BM25 | Dense | Visual top5 列      │
│ · 预设问句 chips            │ · Engineer panel（默认折叠）         │
│ · Query 输入 + Ask          │   self_rag / crag / Trace-Id / JSON  │
│ · Answer                    │                                      │
│ · Citation cards            │                                      │
└─────────────────────────────┴──────────────────────────────────────┘
```

### 2.1 模式行为

| 模式 | 行为 |
|------|------|
| **Demo** | 预设 chips / 自由输入映射到 `fixtures.json`；无网络；上传 disabled + 提示切 Live |
| **Live** | `GET {base}/health`；`POST {base}/ask`；`POST {base}/ingest`（multipart）；解析 `X-Trace-Id` |

模式可用顶栏切换；Live 的 `apiBase` 默认 `""`（同源 `/demo` 挂在 API 上时用相对路径）或 `http://localhost:8000`。

---

## 3. 文件与挂载

```text
static/demo/
  index.html      # 结构 + 样式（可内联 CSS 或 style.css）
  app.js          # 状态、渲染、fetch、上传（推荐与 HTML 拆分，便于审）
  fixtures.json   # 3–5 条录制/合成的 AskResponse 形态
  metrics.json    # 角标数字 + 口径说明文案
src/api/routes.py # StaticFiles mount + 按需 CORS
```

### 3.1 FastAPI 挂载

- `app.mount("/demo", StaticFiles(directory="static/demo", html=True), name="demo")`
- 访问：`http://{host}:8000/demo/`（或 `/demo/index.html`）
- **挂载顺序**：在具体 API 路由注册之后 mount，避免吞掉 `/`；`/demo` 不与现有 path 冲突

### 3.2 CORS

- 同源（页面由同一 uvicorn 提供）时 **不强制** CORS
- 若需 `file://` 或其它 origin 打 Live：增加可配置 `CORSMiddleware`（允许的 origin 列表来自 config 或环境变量，默认仅开发常用 origin）

### 3.3 后端 schema

- **不改** `AskRequest` / `AskResponse` / `Citation` / `RetrievalTrace` / `SelfRAGInfo` / `CRAGInfo`
- 上传响应沿用现有 `IngestResponse`：`doc_id`, `num_pages`, `num_chunks`
- Live 上传成功后：UI 记住 `last_doc_id`；后续 `/ask` **可选** 传 `doc_id`（用户可 toggle「仅当前文档」）

---

## 4. 前端设计

### 4.1 状态

```js
// 概念模型（非强制框架）
{
  mode: "demo" | "live",
  apiBase: string,
  health: { ok: boolean, index_pages?: number } | null,
  query: string,
  loading: boolean,
  error: string | null,
  response: AskResponse | null,  // 与后端 JSON 同形
  traceId: string | null,        // 来自 Live 响应头 X-Trace-Id
  lastDocId: string | null,      // Live 上传
  filterByDoc: boolean,          // ask 是否带 doc_id
  engPanelOpen: boolean
}
```

### 4.2 核心流程

**Ask**

1. 校验非空 query；set loading  
2. Demo：在 `fixtures` 中精确匹配预设键；未命中则提示「Demo 仅支持预设问句」或模糊选最近一条（实现二选一，**推荐精确匹配 + 预设 chips 引导**）  
3. Live：`POST {apiBase}/ask` body `{ query, k: 5, use_rerank: true, doc_id?: lastDocId }`  
4. 解析 JSON → `renderAll`  
5. 错误：展示可读错误条（网络 / 4xx / 5xx detail）

**Upload（仅 Live）**

1. 仅接受 `.pdf`  
2. `FormData` + `POST {apiBase}/ingest`  
3. 成功展示 `doc_id` / pages / chunks；写入 `lastDocId`  
4. 失败展示 detail；不清除已有问答结果

**Health（切 Live 或改 base 时）**

- `GET {apiBase}/health` → 绿/红指示；失败不阻止用户仍尝试 ask（但显示警告）

### 4.3 渲染契约（字段映射）

| UI | 数据 |
|----|------|
| Answer | `response.answer` |
| Citations | `response.citations[]`：`snippet`, `page_number` ?? `page_id`, `chunk_id`, `doc_id` |
| BM25 列 | `response.retrieval_trace.bm25_top5[]`：`chunk_id`, `page_id`, `score` |
| Dense 列 | `dense_top5` 同上 |
| Visual 列 | `visual_top5`；UI 标注 **page 级**（与 chunk 路区分） |
| Self-RAG | `response.self_rag`（缺省按 enabled=false） |
| CRAG | `response.crag` |
| Trace-Id | Live header `X-Trace-Id`；Demo 可用 fixture 旁路字段 `_demo_trace_id`（可选，不进 API schema） |
| Raw JSON | 当前 `response` 美化 dump（工程师面板） |

### 4.4 预设问句

- 来源：`fixtures.json` 的 key 列表，或 `fixtures.meta.presets[]`  
- 至少 **4 条** 建议构成：  
  1. 可答 · 参数/额定类  
  2. 可答 · 表/步骤类  
  3. 多引用  
  4. **拒答样例**（答案为信息不足类文案，citations 空或少）  
- 可选第 5 条：合成 `self_rag.enabled=true` 的 fixture，便于讲工程面板（无需真开 Gate2）

### 4.5 评测角标

`metrics.json` 示例结构：

```json
{
  "chips": [
    {
      "label": "NDCG@10",
      "value": "0.53",
      "detail": "Boot-A · Full_zerank2 · 283q · 协议 v1 · runs/20260720-bootA"
    },
    {
      "label": "E2E Correct",
      "value": "0.66",
      "detail": "Goal-A ON 索引 · 见 handoff / runs/20260723-on-goalA"
    }
  ],
  "footnote": "角标为归档可辩护数字，非本页实时评测。"
}
```

- 点击/hover chip 展示 `detail`  
- **禁止** 使用已废弃口径或 CRAG「质量大涨」类叙事

### 4.6 视觉

- 深色顶栏 + 浅色内容区  
- 等宽三路 Trace 卡片  
- 引用卡片左边线强调  
- 无需外链 UI 库；系统字体栈即可  
- 桌面优先（~1280px）；窄屏允许 RIGHT 折到下方（简单 media query）

---

## 5. Fixture 规范

### 5.1 形状

每条 fixture 必须是合法的 `AskResponse` JSON（字段名与 `src/api/routes.py` 一致）：

```json
{
  "query": "…",
  "answer": "…",
  "citations": [
    {
      "chunk_id": "…",
      "page_id": 12,
      "doc_id": "demo-doc",
      "page_number": 12,
      "snippet": "…"
    }
  ],
  "retrieval_trace": {
    "bm25_top5": [{ "chunk_id": "…", "page_id": 12, "score": 0.9 }],
    "dense_top5": [],
    "visual_top5": []
  },
  "self_rag": { "enabled": false },
  "crag": { "enabled": false, "applied": false }
}
```

### 5.2 录制方式

1. 云上或本地 API：`curl -s -D - -X POST …/ask -d '{"query":"…"}'`  
2. 剥掉多余 header，body 写入 `fixtures.json`  
3. 敏感/过长 snippet 可截断到 ~200 字符，保持可读  

### 5.3 索引文件形态

推荐：

```json
{
  "presets": [
    { "id": "q1", "label": "Rated voltage?", "query": "What is the rated voltage…?" }
  ],
  "responses": {
    "What is the rated voltage…?": { "...": "AskResponse" }
  }
}
```

`presets[].query` 必须是 `responses` 的键。

---

## 6. 与现有系统的接缝

| 接缝 | 说明 |
|------|------|
| `POST /ask` | 主路径；L4 缓存、CRAG、Gate2 行为由 **服务端配置** 决定，前端不伪造开关写回配置 |
| `POST /ingest` | Live 上传；本机全量 visual 重——文档中提示优先云 API 或 `use_visual=false` 配置 |
| `GET /health` | Live 健康与 index_pages |
| `GET /trace/{id}` | v1 **不强制** 在 UI 拉全量 span；工程面板展示 Trace-Id 文本即可，可附「复制 / 打开 /trace/…」链接 |
| Observability | 不改 middleware；Live 请求自然产生 trace |

---

## 7. 测试与验收

### 7.1 自动化（轻）

| 测试 | 内容 |
|------|------|
| 可选 API 烟测 | 客户端 `GET /demo/` 或 `/demo/index.html` → 200 |
| 静态契约 | `fixtures.json` 可被 JSON 解析；每条含 `answer` + `retrieval_trace` |
| 单元 | **不**要求 jsdom 全量；优先后端 mount 烟测 |

### 7.2 手工验收

| # | 标准 |
|---|------|
| 1 | Demo 模式打开 `/demo/`，点预设 ≤3s 渲染答案 + 三路 + 引用 |
| 2 | 工程师面板展开见 self_rag/crag/JSON |
| 3 | Live：填 base → health 绿 → ask 成功 → Trace-Id 非空（若 middleware 注入） |
| 4 | Live：上传小 PDF → 返回 doc_id；可选 filter 后再 ask 不 500 |
| 5 | 角标可展开口径；文案无禁用叙事 |
| 6 | `make test` 现有套件不因 demo 挂载回归失败 |

---

## 8. 实现分期

| 阶段 | 内容 |
|------|------|
| **P0** | `static/demo/*` 骨架 + Demo 模式全 UI + metrics + mount |
| **P1** | Live ask + health + Trace-Id |
| **P2** | Live upload + doc_id 过滤 toggle |
| **P3** | 窄屏 media query、引用 hover 联动（可选） |

---

## 9. 风险与缓解

| 风险 | 缓解 |
|------|------|
| 本机 Live + Visual OOM/慢 | 文档写清用云；本地 `retrieval.use_visual: false` |
| Fixture 与线上答案漂移 | 角标与 fixture 标注录制来源；不声称实时评测 |
| 上传大文件无进度 | v1 spinner + 禁用按钮；失败可读 detail |
| StaticFiles 路径 cwd 不对 | 以仓库根为 cwd 启动 `scripts/run_api.py`；path 相对项目根解析 |
| CORS 误开生产 | 默认关或仅 localhost；配置显式列表 |

---

## 10. 成功标准（一句话）

**打开 `/demo/`，不用 GPU 也能把 PrismRAG 的问答、引用、三路 Trace 和工程字段讲清楚；API 在线时同一页可切 Live 真问真传。**

---

## 11. 决议记录

| 问题 | 选择 |
|------|------|
| 给谁看 | D Hybrid |
| 怎么写 | A 静态单页 |
| 数据从哪来 | C Fixture + Live |
| MVP 能力 | 1–6（含上传；不含纯 search） |
| 布局 | A 左右分栏 |
| 设计批准 | 2026-07-24 用户确认 ok |
