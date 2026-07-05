# Observability 模块设计文档

> 创建日期：2026-07-05 | 状态：待审核

## 1. 概述

为 PrismRAG 项目设计内建的可观测性模块，覆盖**本地调试**和**评测分析**两个场景：

- **请求日志记录与追踪**：每个 query 的完整 Trace（从进入到返回，经过 BM25 → Dense → Visual → Fusion → Rerank 各节点）
- **命中与延迟监控**：按检索配置（ablation config）聚合延迟分布（P50/P95/P99）、命中数、score 分布
- **生成质量监控**：对接 RAGAS Faithfulness / Answer Relevancy 评测结果
- **告警与异常监测**：阈值告警（延迟超限、召回不足、质量低于底线）+ 管道异常捕获（OOM、连接断开等）
- **仪表盘与报表**：终端 `rich` Live 面板（评测运行时实时刷新）+ Markdown/JSON 报告（评测结束后持久化）

不在本期范围：用户反馈系统。

## 2. 关键技术决策

| 决策 | 结论 | 理由 |
|------|------|------|
| 架构方式 | 嵌入式集成（`src/observability/` 核心 + `observability/` 消费端） | pipeline 已有 `retrieval_trace` 概念，升级为正式 Trace 是自然演进 |
| 依赖策略 | `rich` + `structlog`，不引入重量级框架 | 纯 Python、零系统依赖，与项目现有依赖栈兼容 |
| 存储方式 | 评测运行时内存（单例），结束后序列化到 `runs/<id>/observability/` | 匹配「本地调试 + 评测分析」场景，无需持久化数据库 |
| 可视化 | 终端 `rich.Live` 面板 + Markdown 报告 | 开发者最直接的交互方式，无需额外服务 |

## 3. 目录结构

```
src/observability/              # 核心 — 嵌入 pipeline，被 src/ 内模块 import
├── __init__.py                 # 导出公共 API：get_tracer, get_collector, init_logging
│                              #   含 ObservabilityConfig dataclass（YAML 配置的 Python 表示）
├── logging_setup.py            # structlog 统一初始化，输出 JSON → logs/ + 控制台彩色
├── tracer.py                   # Trace / Span 数据模型 + 上下文管理器 + @trace 装饰器
├── collectors.py               # MetricsCollector 单例：延迟/命中/质量指标聚合
├── alerting.py                 # 阈值检测 + 异常分类，产出 AlertEvent
└── middleware.py               # FastAPI 中间件（可选），自动为 HTTP 请求创建 Trace

observability/                  # 消费侧 — 独立模块，读取 collector 数据渲染
├── __init__.py
├── dashboard.py                # rich Live 终端面板（评测运行时）
└── reporter.py                 # Markdown/JSON 报告生成（评测结束后）
```

## 4. 模块详细设计

### 4.1 logging_setup.py — 统一日志初始化

**职责**：替代各脚本中散落的 `logging.basicConfig(...)`，提供一处调用完成全项目日志配置。

**API**：
```python
def init_logging(
    level: str = "INFO",        # DEBUG | INFO | WARNING
    log_file: str | None = "logs/app.jsonl",
    console: bool = True,
) -> None:
    """配置 structlog，输出 JSON 到文件 + 彩色到控制台"""
```

**行为**：
- 控制台：`structlog.dev.ConsoleRenderer`（彩色、简洁）
- 文件：`structlog.processors.JSONRenderer`（每行一条 JSON，方便 `jq` 查询）
- 所有现有 `logging.getLogger(__name__)` 自动桥接到 structlog（通过 `structlog.stdlib.LoggerFactory`）
- 日志自动带 `trace_id`（如果当前上下文有活跃 Trace）

**调用点**：各入口脚本 `main()` 的最开头（`run_eval.py`、`run_ragas_metrics.py`、`run_api.py`、`ingest_vidore.py`）

### 4.2 tracer.py — Trace/Span 模型

**数据模型**：

```
Trace (一个 query 的完整生命周期)
├── trace_id: str (uuid4)
├── query: str
├── config_label: str ("Full_zerank2_HyDE")
├── started_at / finished_at: datetime (UTC)
├── spans: List[Span]
│
Span (一个 pipeline 步骤)
├── span_id: str (uuid4)
├── parent_span_id: str | None
├── name: str ("bm25_search" | "dense_encode" | ...)
├── started_at / finished_at: datetime
├── duration_ms: float
├── metadata: dict
└── status: "ok" | "error"
```

**API**：
```python
# 全局上下文（线程安全，通过 contextvars 实现）
def get_tracer() -> Tracer: ...

class Tracer:
    def start_trace(query: str, config_label: str = "") -> Trace: ...
    def current_trace() -> Trace | None: ...
    def finish_trace() -> None: ...

    def start_span(name: str, metadata: dict | None = None) -> Span: ...
    # Span 用 with 语句自动 finish：
    # with tracer.start_span("bm25_search") as span:
    #     ...
    #     span.set_metadata({"num_results": 20})
```

**集成点（8 个 Span）**：

| Span 名称 | 插入位置 | 记录 |
|-----------|----------|------|
| `bm25_search` | `BM25Retriever.search()` return 前 | 耗时、返回数 |
| `dense_encode` | `DenseRetriever.search()` BGE encode | 编码耗时 |
| `dense_search` | `DenseRetriever.search()` pgvector 查询 | SQL 耗时、返回数 |
| `visual_encode` | `VisualRetriever.search()` ColPali encode | 编码耗时、batch size |
| `visual_search` | `VisualRetriever.search()` FAISS MaxSim | FAISS 耗时、返回 pages 数 |
| `fusion_rerank` | `RRFFusion.fuse()` + `Reranker.rerank()` | 融合输入数、rerank 耗时、top-k |
| `hyde_generate` | `HyDEGenerator.generate()` | Ollama 耗时、cache hit/miss |
| `llm_generate` | `ragas_metrics.generate_answer()` | Ollama 耗时、answer length |

**侵入性**：每个节点 2-3 行：
```python
with tracer.start_span("bm25_search") as span:
    results = self._do_search(query, k)
    span.set_metadata({"num_results": len(results), "k": k})
```

### 4.3 collectors.py — 指标收集器

**数据模型**：

```python
@dataclass
class ConfigMetrics:
    config_label: str
    num_queries: int
    # 延迟 (ms)
    latency_p50_ms, latency_p95_ms, latency_p99_ms: float
    latency_avg_ms, latency_min_ms, latency_max_ms: float
    # 命中
    avg_bm25_hits, avg_dense_hits, avg_visual_hits: float
    avg_fused_count, avg_reranked_count: int
    # 质量
    avg_faithfulness, avg_answer_relevancy: float
    # 缓存
    hyde_hit_rate: float

@dataclass
class AlertEvent:
    timestamp: datetime
    level: "warning" | "error"
    category: str  # "threshold" | "pipeline_error"
    message: str
    config_label: str
    trace_id: str | None
```

**API**：
```python
def get_collector() -> MetricsCollector: ...
# 全局单例，线程安全

class MetricsCollector:
    def reset(self) -> None: ...
    def ingest_trace(self, trace: Trace) -> None: ...
    def record_ragas_score(self, config_label: str,
                           query_id: str,
                           faithfulness: float,
                           answer_relevancy: float) -> None: ...
    def record_alert(self, event: AlertEvent) -> None: ...
    def get_config_metrics(self, config_label: str) -> ConfigMetrics | None: ...
    def get_alerts(self) -> list[AlertEvent]: ...
    def snapshot(self) -> dict: ...
    """返回完整快照：configs + traces + alerts，供 reporter 序列化"""
```

### 4.4 alerting.py — 告警检测

**职责**：消费 collector 中的指标，按配置的阈值触发告警。

**两种告警类型**：

1. **阈值告警**（评测完成后批量检测）：
   - 延迟 P95 > `latency_p95_threshold_ms`（默认 5000ms）
   - Recall@5 < `recall_at_5_min`（默认 0.5）
   - Faithfulness < `faithfulness_min`（默认 0.6）

2. **管道异常**（实时捕获）：
   - 通过 `sys.excepthook` 或 try/except 包装捕获
   - 分类：`OOM`、`ConnectionError`、`ModelLoadFailed`、`IndexCorrupted`
   - 自动记录 stack trace + trace_id

**API**：
```python
class AlertChecker:
    def __init__(self, config: ObservabilityConfig): ...
    def check_thresholds(self, collector: MetricsCollector) -> list[AlertEvent]: ...
    def wrap_exception(self, exc: Exception, trace_id: str | None = None) -> AlertEvent: ...
```

### 4.5 dashboard.py — 终端仪表盘

**职责**：评测运行时启动 `rich.Live`，实时刷新面板。

**面板布局**（单屏，每 0.5s 刷新）：

```
┌── PrismRAG Observability ──────────────────────────────────────────────┐
│ Config: Full_zerank2_HyDE    Queries: 142/283    Runtime: 12m 34s     │
│────────────────────────────────────────────────────────────────────────│
│ Latency (ms)        │  Hits / Query          │  Quality               │
│   P50:    342       │    BM25:   18.2        │   Faithfulness: 0.81   │
│   P95:  1,247       │    Dense:  17.8        │   Relevancy:    0.76   │
│   P99:  3,801       │    Visual: 15.3        │                        │
│   Avg:    512       │    Fused:  40.0        │   Alerts: ⚠ 2 ⛔ 0    │
│────────────────────────────────────────────────────────────────────────│
│ Recent Alerts:                                                        │
│   ⚠ 14:23:01 | Full_zerank2_HyDE | latency_p95 5234ms > threshold    │
│   ⚠ 14:18:45 | Dense_only | recall@5 0.42 < 0.50                     │
└────────────────────────────────────────────────────────────────────────┘
```

**API**：
```python
class Dashboard:
    def start(self, collector: MetricsCollector): ...
    def update(self): ...
    def stop(self): ...
```

### 4.6 reporter.py — 报告生成

**职责**：评测结束后，从 collector 快照生成文件。

**输出**（`runs/<run_id>/observability/`）：

| 文件 | 内容 |
|------|------|
| `metrics.json` | 每个 config 的聚合指标（~10KB） |
| `traces.jsonl` | 逐 query Trace，每行一条 JSON（~500KB / 283 queries） |
| `alerts.json` | 告警事件列表 |
| `report.md` | 人类可读 Markdown 报告：概要表 + 每个 config 的延迟分布 + 告警详情 |

**Markdown 报告示例结构**：
```markdown
# Observability Report — 2026-07-05 15:30

## Summary
| Config | Queries | P50 | P95 | Avg Latency | Recall@5 | Faithfulness |
|--------|---------|-----|-----|-------------|----------|--------------|
| BM25_only | 283 | 12ms | 45ms | 18ms | 0.62 | — |
| Full_zerank2_HyDE | 283 | 342ms | 1247ms | 512ms | 0.84 | 0.81 |

## Alerts (3)
- ⚠ Full_zerank2_HyDE: latency_p95 5234ms > 5000ms threshold
- ...

## Per-Config Detail: Full_zerank2_HyDE
### Latency Distribution
| Span | P50 | P95 | P99 |
|------|-----|-----|-----|
| bm25_search | 8ms | 22ms | 45ms |
| dense_encode | 45ms | 89ms | 120ms |
...
```

### 4.7 middleware.py — FastAPI 中间件（可选）

对于 API 场景（`src/api/routes.py`），通过 FastAPI middleware 自动为每个 HTTP 请求创建 Trace：

```python
@app.middleware("http")
async def observability_middleware(request: Request, call_next):
    trace = tracer.start_trace(query="(from API)")
    response = await call_next(request)
    response.headers["X-Trace-Id"] = trace.trace_id
    tracer.finish_trace()
    collector.ingest_trace(trace)
    return response
```

## 5. 配置

在 `config/models.yaml` 新增：

```yaml
observability:
  log_level: "INFO"
  log_file: "logs/app.jsonl"
  trace_enabled: true
  dashboard_enabled: true
  alerting:
    latency_p95_threshold_ms: 5000
    recall_at_5_min: 0.5
    faithfulness_min: 0.6
```

`src/config.py` 新增 `ObservabilityConfig` 属性（带默认值，避免改动现有 YAML 导致启动失败）。

## 6. 数据流

```
入口脚本 main()
  ├─ init_logging()                       # 一次性配置
  ├─ collector.reset()                    # 新评测开始
  │
  ├─ for each query:
  │     tracer.start_trace(query, config)
  │       ├─ span: bm25_search
  │       ├─ span: dense_encode
  │       ├─ span: dense_search
  │       ├─ span: visual_encode
  │       ├─ span: visual_search
  │       ├─ span: hyde_generate
  │       ├─ span: fusion_rerank
  │       └─ span: llm_generate
  │     tracer.finish_trace()
  │     collector.ingest_trace(trace)      # 聚合到内存
  │     dashboard.update()                 # 刷新 rich 面板
  │
  ├─ collector.record_ragas_score(...)     # RAGAS 评测完成
  ├─ alert_checker.check_thresholds()      # 阈值检测
  │
  └─ reporter.generate(snapshot, run_id)   # 序列化到 runs/
       ├─ metrics.json
       ├─ traces.jsonl
       ├─ alerts.json
       └─ report.md
```

## 7. 测试策略

| 测试范围 | 内容 |
|---------|------|
| 单元测试 | `tracer.py` Trace/Span 生命周期、`collectors.py` 聚合逻辑、`alerting.py` 阈值检测 |
| 集成测试 | 端到端：创建 Trace → 注入 collector → 生成报告，验证数据完整性 |
| 兼容性 | 不启用 observability 时（`trace_enabled: false`），所有 hook 零开销返回 no-op 对象 |
| 烟雾测试 | 运行 `run_eval.py --max-queries 5`，验证 dashboard 不崩溃、report.md 生成正常 |

## 8. 风险 & 注意事项

- **线程安全**：`contextvars` 确保 Trace 在 async FastAPI 环境下正确隔离
- **内存占用**：283 queries 的完整 Trace 约 5-10MB 内存，可接受
- **零开销模式**：`trace_enabled: false` 时 `tracer.start_span()` 返回 no-op Span（不做计时），避免评测性能偏差
- **不破坏现有代码**：`retrieval_trace` 保持不变，Trace 是独立新层
- **向后兼容**：YAML 配置缺失 `observability` 段时使用代码内默认值，不报错
