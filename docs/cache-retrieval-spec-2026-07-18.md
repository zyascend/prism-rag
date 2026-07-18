# PrismRAG 检索缓存（L3 检索结果 / L4 Answer）Spec

> 版本：v1.1 ｜ 日期：2026-07-18 ｜ 状态：已落地（L3 + L4，分支 `feat/cache-layers`；L3 于 `f304928`，L4 + K1 修复于后续 commit 同分支）
> 配套文档：`handoff.md`（可观测性/cache 完成项，标记 `[x]`）；复用 PR #27 可观测性闭环（`GET /trace/{id}`、`MetricsCollector`）
> 约束：本地 macOS 32GB 禁全量评测/编码；耗时操作须上云 GPU。本 Spec 仅涉及查询期缓存，无重型依赖。

---

## 0. 现状审计（已代码核实，非推测）

| 能力 | 代码位置 | 状态 |
|------|----------|------|
| HyDE 查询改写缓存（纯内存 `dict`） | `src/retrieval/hyde.py` | ✅ 已实现（**本次 MVP 不动**，见 §1.2） |
| 表格摘要缓存（`lru_cache(2048)`） | `src/generation/table_summary.py` | ✅ 已实现（**入库期**，与查询期优化无关） |
| page embeddings 持久化（pickle） | `src/store/faiss_store.py` | ✅ 已实现（**入库期**） |
| **L3 检索结果缓存**（查询期） | `src/evaluation/vidore_adapter.py` `search_with_trace` | ✅ 本次新增 |
| **L4 Answer 缓存**（查询期） | `src/api/routes.py` `/ask` + `PrismRAGRetriever._answer_cache` / `answer_cache_key` | ✅ 本次新增 |
| 生成确定性守卫 | `src/generation/generator.py` `Generator.cacheable`（`temperature==0`） | ✅ 本次新增 |
| 缓存命中可观测 | `src/observability/collectors.py` `record_cache_event` | ✅ 本次新增（`retrieval` + `answer` 两层） |
| 全局开关 | `src/config.py` `CacheConfig.enabled` | ✅ 本次新增 |
| `index_version` 版本盐失效 | `src/evaluation/vidore_adapter.py` `invalidate_cache` | ✅ 本次新增（同时清 L3 + L4） |
| `POST /cache/invalidate` 端点 | `src/api/routes.py` | ✅ 本次新增 |

**部署形态（决定缓存层级）**：单 `uvicorn` 进程（`Dockerfile` CMD 无 `--workers`、无 gunicorn）。进程内缓存即可保证正确性，无需 Redis。

---

## 1. 目标与非目标

### 1.1 目标
1. **降低重复查询延迟**：L3 命中时跳过 BM25 + Dense + Visual 三路检索、RRF 融合、重排（rerank），直接返回已算好的 Top-K。
2. **降低重复查询算力/成本**：L4 命中时跳过整次 LLM 生成（最大 ROI，省一次大模型调用），直接返回缓存的 `(answer, citations, retrieval_trace)`。
3. **可观测命中情况**：每个 config 维度聚合 `retrieval` 与 `answer` 两层 cache 命中率，并能在单请求 trace（`GET /trace/{id}`）中看到 `cache_hit`。
4. **全局开关**：通过 `cache.enabled` 运行时门控所有缓存层（L3/L4），关闭即穿透。
5. **正确性优先的失效**：以 `index_version` 版本盐保证语料变更后旧 key 自然失效，**不依赖 TTL** 撑更新。
6. **确定性守卫**：仅当生成温度 `temp==0`（确定性）时才读写 L4 缓存，避免缓存不稳定答案。

### 1.2 非目标（范围外）
- **不改造 HyDE 缓存**：用户明确 HyDE 暂不动（`hyde.py` 维持原 `dict`）。
- **不做 L1 BGE embedding 缓存**：本次不缓存 query 向量，列为后续增强（§11）。
- **不引入 Redis / 分布式缓存**：单进程部署，进程内 LRU 足够；`RedisCache` 仅作预留后端接口，不实现。
- **不做磁盘 spill**：检索结果可能较大，但 MVP 内存 LRU 足够；列为后续增强。

---

## 2. 设计原则

| 原则 | 说明 | 本项目落地形式 |
|------|------|----------------|
| 旁路缓存（cache-aside） | 先查缓存，命中返回，未命中查源并回填 | `search_with_trace` 顶部查 `_cache`，未命中走三路后 `put` |
| 版本盐失效优先于 TTL | 正确性由语料版本保证，TTL 仅兜底异常残留 | `index_version` 单调计数；`ttl_seconds=0` 为推荐值 |
| 单进程进程内缓存 | 部署为单 uvicorn worker，进程内状态跨请求持久 | `InMemoryLRUCache` 挂在 retriever 单例上 |
| key 必须含全部影响结果的维度 | 漏任一维度会串结果（脏读） | `_cache_key` 含归一化 query + 全部检索开关 + reranker + index_version |
| 后处理维度不入 key | doc_id 过滤在 API 层确定性后置，无需进 key | 见 §6 正确性约束 C1 |

---

## 3. 目标架构（查询期旁路缓存）

```
   /ask ──▶ ① L4 Answer 缓存检查（answer_cache_key）
                  │
      命中？──────┴──▶ return 缓存的 (answer, citations, retrieval_trace)  ← 跳过 ②③④
                  │ 未命中
                  ▼
   ② PrismRAGRetriever.search_with_trace(query, k, …)
                  │
      ┌───────────▼────────────────┐
      │  cache_on = cfg.cache.enabled ? │
      └───────┬───────────┬──────────┘
         关闭 │           │ 开启
              │           ▼
              │  key = _cache_key(...) ──▶ _cache.get(key) ──┐ 命中：return cached
              │           │                    │            │
              │           │ 未命中             │            │
              │           ▼                    │            │
              │  三路检索(BM25/Dense/Visual)   │            │
              │   → RRF 融合 → Rerank          │            │
              │           │                    │            │
              │           ▼                    │            │
              │  _cache.put(key, result)       │            │
              │  record_cache_event(retrieval) │            │
              └───────────┬────────────────────┘
                          ▼
   ③ 若 doc_id：路由层过滤
   ④ Generator.answer(...)（仅当未命中 L4）
                          │
                          ▼
       若 L4 可写：_answer_cache.put(answer_key, {...})
       record_cache_event(answer)
                          │
                          ▼
                 return (answer, citations, retrieval_trace)

   POST /cache/invalidate ──▶ invalidate_cache() → index_version+=1, _cache.clear(), _answer_cache.clear()
```

L4 命中 = 跳过 ②③④ 全部（检索 + 融合 + 重排 + LLM 生成）；L3 命中 = 跳过 ② 内检索/融合/重排。

---

## 4. 详细设计

### 4.1 缓存后端抽象（`src/cache/store.py`）
- `CacheStore`（抽象基类）：接口 `get(key) -> Optional[Any]`、`put(key, value)`、`clear()`。
- `InMemoryLRUCache(CacheStore)`：
  - `OrderedDict` 实现 LRU，超过 `max_size` 淘汰最久未使用项。
  - 可选 `ttl_seconds`：>0 时 `put` 记录 `expire_at`，`get` 惰性淘汰过期项；`0` 或负 = 不启用 TTL（仅依赖版本盐失效）。
  - 全程 `threading.RLock` 保护（`get_retriever` 单例被多请求共享，需线程安全）。
  - `RedisCache` 仅预留（同接口），本次不实现。

### 4.2 版本盐失效（`invalidate_cache`）
- `PrismRAGRetriever.index_version: int`，初始 `0`。
- `invalidate_cache()`：`index_version += 1`；若 `_cache is not None` 则 `_cache.clear()`。
- 调用点：
  1. `delete_document()`（语料变更，已内联调用）—— 确保已删文档不再被命中。
  2. `POST /cache/invalidate` 端点（重索引 / 批量语料变更后手动触发）。
- **正确性语义**：旧 key 因 `index_version` 不匹配而自然查不到，逻辑失效，零脏读；TTL 不参与正确性。

### 4.3 L3 检索结果缓存（包裹 `search_with_trace`）
- 入口（`vidore_adapter.py:210` 起）：
  - `cache_on = cfg.cache.enabled`；若开启且 `_cache is None` 则惰性创建 `InMemoryLRUCache(max_size=cfg.cache.max_size)`。
  - 计算 `cache_key = self._cache_key(...)`；`cached = self._cache.get(cache_key)`。
  - **命中**：`record_cache_event("retrieval", hit=True, config_label=...)`；若 owns_trace 则发轻量 `retrieval` span（`cache_hit=True, cache_layer="retrieval", num_results=...`），直接返回 `cached`。
  - **未命中**：走三路检索 → 融合 → 重排；在三个返回点（`空结果` / `reranked` / `fused[:k]`）写入 `_cache.put(cache_key, result)` + `record_cache_event("retrieval", hit=False, config_label=...)`。
- 缓存值 = `{"results": [...], "retrieval_trace": {...}}`，即完整检索输出（含三路 top5 trace）。

### 4.4 全局开关（`cache.enabled`）
- `CacheConfig.enabled: bool = True`（`src/config.py:46`）。
- `cfg.cache` 属性从 YAML `cache:` 段加载，缺失回退默认（全开）。
- `search_with_trace` 每请求读取 `cfg.cache.enabled`：关闭时**既不查也不写**缓存，完全穿透（行为与无缓存时一致，便于灰度/回滚）。

### 4.5 可观测命中率（`src/observability/collectors.py`）
- `record_cache_event(layer: str, hit: bool, config_label: str = "api")`：
  - 维护 `_cache_data[config_label][layer] = {hits, misses}`（线程安全）。支持 `"retrieval"`（L3）与 `"answer"`（L4）两层。
- `get_config_metrics()` 聚合：`retrieval_cache_hit_rate` 与 `answer_cache_hit_rate` 分别 = `hits/(hits+misses)`（仅在 total>0 时）。
- `ConfigMetrics.to_dict()["cache"]` 段含 `retrieval_cache_hit_rate` / `answer_cache_hit_rate`（与已有 `hyde_hit_rate` 并列）。
- 单请求可见性：`retrieval` span 的 `cache_hit` / `cache_layer` 元数据 → 复用 PR #27 的 `GET /trace/{id}` 可查单请求是否命中（L4 命中由 `answer` 层事件体现）。

### 4.6 L4 Answer 缓存（包裹 `/ask`）
- 入口（`src/api/routes.py` `ask`，`vidore_adapter.py:253` 起）：
  - L4 受**双重守卫**：`cfg.cache.enabled`（全局开关）**且** `generator.cacheable`（确定性）。
  - `generator.cacheable`（`generator.py`）：`self.temperature == 0.0`。当前 `Generator.__init__` 固定 `self.temperature = 0.0` 且 `answer()` 用 `temperature=self.temperature`，故默认 `cacheable=True`；若未来放开温度，非确定性生成不会被缓存（不读不写）。
  - 计算 `answer_key = retriever.answer_cache_key(query, gen.model, request.k, request.doc_id)`；`cached = retriever._answer_cache.get(answer_key)`。
  - **命中**：`record_cache_event("answer", hit=True, config_label="")`；从缓存的 `retrieval_trace` 重建 `RetrievalTrace`，直接返回 `(answer, citations, retrieval_trace)`，**整次跳过**检索 + 生成。
  - **未命中**：走 `search_with_trace`（内部含 L3）+ `generator.answer`；写入 `retriever._answer_cache.put(answer_key, {"answer", "citations", "retrieval_trace"})` + `record_cache_event("answer", hit=False, config_label="")`。
  - `retriever._answer_cache` 惰性创建（`InMemoryLRUCache(max_size=cfg.cache.max_size)`），与 L3 同后端、同内存上限。
- L4 与 L3 共享 `index_version` 盐：`invalidate_cache()` 同时 `clear()` 两者（§4.2）。

### 4.7 失效反查端点（`src/api/routes.py:130`）
- `POST /cache/invalidate`：调用 `retriever.invalidate_cache()`，返回 `{"status": "ok", "index_version": retriever.index_version}`。
- 用途：重索引、批量 ingest、语料变更后，主动失效服务侧内存缓存。

---

## 5. 配置与数据模型变更

| 对象 | 变更 | 默认值 |
|------|------|--------|
| `src/config.py` `CacheConfig`（新） | `enabled` / `max_size` / `ttl_seconds` | `True` / `2048` / `0` |
| `config/models.yaml` `cache:` 段 | `enabled: true` / `max_size: 2048` / `ttl_seconds: 0` | — |
| `PrismRAGRetriever` 内存 | `index_version: int = 0`、`_cache: InMemoryLRUCache | None = None`、`_answer_cache: InMemoryLRUCache | None = None`（L4） | — |
| `Generator` 内存 | `temperature: float = 0.0`、`cacheable` property（`temperature==0`） | — |
| `MetricsCollector` 内存 | `_cache_data`（命中统计，含 `retrieval` + `answer` 两层） | — |

`ttl_seconds=0` 语义：**仅依赖 index_version 盐失效**（推荐，正确性由版本保证）；`>0` 仅作跨进程异常残留的安全网，**不作为正确性依赖**。

---

## 6. key 构造与正确性约束

`_cache_key`（`vidore_adapter.py:107`）：
```
key = "q=<NFKC+lower+空白折叠>"
    + "|k=<k>"
    + "|bm25=<bool>|dense=<bool>|visual=<bool>|rerank=<bool>|hyde=<bool>"
    + "|rt=<reranker_type>"
    + "|v=<index_version>"
    + "|ve=<visual_query_embedding 的 sha256 前16位 或 'none'>"
```

**正确性约束（实现时必须守）**：

- **C1（L3 的 doc_id 不在 key 中，正确）**：`doc_id` 过滤在 API 路由层（`routes.py:267`，`search_with_trace` 返回后）做确定性后处理。L3 缓存存的是全量检索结果，命中后由路由层按 `doc_id` 过滤，结果正确且不串。若未来将 `doc_id` 过滤下沉到检索层，则**必须**把 `doc_id` 编入 L3 key。
- **C2（L3 key 必须含全部检索开关）**：`use_bm25/dense/visual/rerank/hyde` + `reranker_type` + `k`。漏任一会串不同配置的结果。
- **C3（visual_query_embedding 非 None）**：必须按 tensor 的 `sha256` 编入 key（`ve=...`），否则同一 query 字符串会命中错误的预编码向量结果（API 路径传 `None` → `ve=none`，但 `search` 接口支持预编码，不能假设）。
- **C4（index_version 盐）**：语料任何变更必须触发 `invalidate_cache()`，否则旧结果脏读。L3 与 L4 共享此盐。
- **C5（归一化）**：query 经 `NFKC + lower + 空白折叠`，避免 "How many?" 与 "how many?" 命中不到同一项。
- **C6（L4 key 必须含 doc_id）**：与 L3 不同，L4 缓存的是**最终答案**，`doc_id` 过滤在 L4 命中的答案生成**之前**已应用（路由层后处理在写入 L4 之前完成），因此 `doc_id` 差异会改变最终答案，必须编入 L4 key（`doc=*` 表示无过滤）。L4 key 另含 `model`（不同 LLM 答案不同）与 `k_context`（上下文窗口大小影响引用）。

---

## 7. 实施状态

| 阶段 | 内容 | 状态 |
|------|------|------|
| MVP 后端抽象 | `CacheStore` + `InMemoryLRUCache` | ✅ 已落地 |
| MVP 版本盐失效 | `index_version` + `invalidate_cache` + `delete_document` 钩子 + 端点 | ✅ 已落地 |
| MVP L3 检索缓存 | `search_with_trace` 包裹 + 全局开关门控 | ✅ 已落地 |
| MVP 可观测 | `record_cache_event` + `retrieval_cache_hit_rate` + `answer_cache_hit_rate` + span `cache_hit` | ✅ 已落地 |
| L4 Answer 缓存 | `/ask` 包裹 + `answer_cache_key` + `Generator.cacheable` 守卫 + `invalidate` 清理 | ✅ 已落地 |
| K1 修复 | `vidore_adapter.py:349` 补 `config_label=config_label` | ✅ 已修复 |
| 测试 | `tests/test_retrieval_cache.py`（11 个聚焦单测） | ✅ 全过 |

---

## 8. 验证方案

`tests/test_retrieval_cache.py`（聚焦单测，无重型依赖，符合本地限制）：
- LRU 淘汰：超出 `max_size` 后最久未使用项被淘汰。
- TTL 兜底：过期条目 `get` 返回 `None`（仅当 `ttl_seconds>0`）。
- 命中率聚合：`record_cache_event` 后 `get_config_metrics` 算出正确 `retrieval_cache_hit_rate`。
- key 正确性：`_cache_key` 归一化（大小写/空白）+ `index_version` 盐变化导致旧 key 失效。
- 命中/未命中：`search_with_trace` 重复相同 query 第二次命中（跳过检索），返回与首次一致。
- 全局开关：构造 `cache.enabled=False` 时直接穿透，不读写缓存、命中率为 0。
- L4（K1 修复后）：`answer_cache_key` 归一化等价、`doc_id` 不同则 key 不同、`invalidate_cache` 使旧 key 失效；`collector` 聚合 `answer_cache_hit_rate`；`/ask` 集成测试验证重复 query 下 `generator.answer` 仅被调用一次（L4 命中）。
- K1 回归：`use_rerank=False` 的 fused 末路径 miss 事件计入传递的 `config_label` 而非默认 `"api"`。

---

## 9. 风险与缓解

| 风险 | 缓解 |
|------|------|
| 多 worker 部署（gunicorn） | 进程内缓存不共享，命中率下降且可能不一致；届时实现 `RedisCache` 后端（接口已预留）。当前单 uvicorn，无此风险 |
| key 维度遗漏导致串结果 | §6 约束 C2/C3/C5；新增检索开关时必须同步改 `_cache_key` |
| 语料变更后旧结果脏读 | §6 约束 C4：所有变更路径（delete / ingest / 重索引）必须调 `invalidate_cache` |
| 内存膨胀 | LRU `max_size` 上限；`ttl_seconds` 兜底过期 |
| 单测 label 归类偏差（见 §10 K1） | 修复后命中率按真实 config_label 聚合 |

---

## 10. 已知问题 / 待修复

- **K1（指标归类偏差，非正确性 bug）— ✅ 已修复**：`vidore_adapter.py:349`（fused 末路径的 cache miss 记录）原调用 `record_cache_event("retrieval", hit=False)` **漏传 `config_label=config_label`**，会回退默认 `"api"` label。已补 `config_label=config_label`（一行）。回归测试 `test_retrieval_cache_miss_records_config_label` 固化。
- **K2（设计权衡，非 bug）**：L3 命中返回全量 results，`doc_id` 过滤在 API 层后处理（见 C1）。若未来把 doc_id 过滤下沉检索层，须同步把 doc_id 编入 key。L4 已把 doc_id 编入 key（见 C6）。

---

## 11. 后续增强（非 MVP）

| 增强 | 省下的计算 | 前置条件 |
|------|-----------|----------|
| L1 BGE embedding 缓存 | 一次 BGE 前向 | key=归一化 query |
| HyDE 缓存升级 | 纳入 `CacheStore` + 归一化 + LRU（用户暂要求不动） | — |
| Redis 后端 | 多 worker 共享缓存 | gunicorn 多 worker 部署 |
| 磁盘 spill | 大检索结果不占满内存 | 检索结果体积极大时 |

---

## 12. 与现有文档关系

- 本 Spec 不推翻既有设计，而是把 `feat/cache-layers` 分支的 MVP 实现规格化、可审计化。
- 可观测性部分复用 PR #27（`handoff.md` 可观测性完成项）：`GET /trace/{id}` 单请求可见 `cache_hit`、`MetricsCollector` 已具备聚合能力。
- 后续实现/修复应在此 Spec 框架下更新 `handoff.md` 与对应代码模块；K1 修复后回写本节状态。
