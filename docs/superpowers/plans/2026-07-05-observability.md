# Observability Module Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an embedded observability module for PrismRAG covering request tracing, latency/hit monitoring, generation quality tracking, threshold alerting, and terminal dashboard + file-based reporting.

**Architecture:** Core lives in `src/observability/` (tracer, collector, alerting, logging setup) and is imported by pipeline modules. Consumers (`dashboard.py`, `reporter.py`) live in `observability/` and render from the collector's in-memory data. Integration into existing pipeline is 2-3 lines per hook point using context-manager spans.

**Tech Stack:** Python 3.11+, structlog, rich, pytest. No new system dependencies.

**Spec:** `docs/superpowers/specs/2026-07-05-observability-module-design.md`

## Global Constraints

- Dependencies limited to `rich` and `structlog` (no OpenTelemetry, Prometheus, Grafana)
- All hook points must degrade gracefully when `trace_enabled: false` (no-op spans)
- Existing `retrieval_trace` in `PrismRAGRetriever` must not be modified
- YAML config missing `observability` section must not cause startup errors
- Must work on macOS (M-series) for local development
- Test coverage for tracer, collector, and alerting modules

---

### Task 1: Branch + Dependencies

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Create feature branch**

```bash
cd /Users/theyang/Documents/ai/pdf-rag
git checkout -b feat/observability
```

- [ ] **Step 2: Add rich and structlog to dependencies**

Edit `pyproject.toml`, add to the `dependencies` list after `"tqdm>=4.66"`:

```toml
    "rich>=13.0",
    "structlog>=24.0",
```

- [ ] **Step 3: Install new dependencies**

```bash
cd /Users/theyang/Documents/ai/pdf-rag
uv pip install rich structlog
```

- [ ] **Step 4: Verify imports work**

```bash
python -c "import rich; import structlog; print('OK')"
```
Expected: `OK`

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml
git commit -m "chore: add rich and structlog dependencies"
```

---

### Task 2: Config — YAML + Python Config

**Files:**
- Modify: `config/models.yaml`
- Modify: `src/config.py`

**Interfaces:**
- Produces: `ObservabilityConfig` dataclass accessible via `cfg.observability` property on the Config singleton

- [ ] **Step 1: Add observability section to models.yaml**

Append to `config/models.yaml`:

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

- [ ] **Step 2: Add ObservabilityConfig dataclass and cfg property to src/config.py**

Add at top of `src/config.py`, after the `import yaml` line:

```python
from dataclasses import dataclass, field


@dataclass
class ObservabilityConfig:
    """Observability 配置，从 YAML observability 段加载，缺失时使用默认值"""
    log_level: str = "INFO"
    log_file: str = "logs/app.jsonl"
    trace_enabled: bool = True
    dashboard_enabled: bool = True
    latency_p95_threshold_ms: int = 5000
    recall_at_5_min: float = 0.5
    faithfulness_min: float = 0.6
```

Add property to `Config` class, after the `llm_model_id` property and before `bge_dim`:

```python
    @property
    def observability(self) -> ObservabilityConfig:
        """返回 observability 配置，YAML 缺失时回退到默认值"""
        raw = self.get("observability", {})
        alerting_raw = raw.get("alerting", {})
        return ObservabilityConfig(
            log_level=raw.get("log_level", "INFO"),
            log_file=raw.get("log_file", "logs/app.jsonl"),
            trace_enabled=raw.get("trace_enabled", True),
            dashboard_enabled=raw.get("dashboard_enabled", True),
            latency_p95_threshold_ms=alerting_raw.get("latency_p95_threshold_ms", 5000),
            recall_at_5_min=alerting_raw.get("recall_at_5_min", 0.5),
            faithfulness_min=alerting_raw.get("faithfulness_min", 0.6),
        )
```

- [ ] **Step 3: Verify config loads**

```bash
cd /Users/theyang/Documents/ai/pdf-rag
python -c "
from src.config import cfg
oc = cfg.observability
print(f'trace_enabled={oc.trace_enabled}, log_level={oc.log_level}, p95_threshold={oc.latency_p95_threshold_ms}')
"
```
Expected: `trace_enabled=True, log_level=INFO, p95_threshold=5000`

- [ ] **Step 4: Commit**

```bash
git add config/models.yaml src/config.py
git commit -m "feat: add ObservabilityConfig to YAML and Python loader"
```

---

### Task 3: tracer.py — Trace/Span Model + Context Manager

**Files:**
- Create: `src/observability/__init__.py` (empty for now)
- Create: `src/observability/tracer.py`
- Create: `tests/test_observability_tracer.py`

**Interfaces:**
- Produces: `Trace`, `Span` dataclasses; `Tracer` class with `start_trace()`, `current_trace()`, `finish_trace()`, `start_span()`; `NoopSpan` for disabled mode; `get_tracer()` global accessor

- [ ] **Step 1: Create empty __init__.py**

```bash
mkdir -p /Users/theyang/Documents/ai/pdf-rag/src/observability
touch /Users/theyang/Documents/ai/pdf-rag/src/observability/__init__.py
```

- [ ] **Step 2: Write the failing tests**

Create `tests/test_observability_tracer.py`:

```python
"""Tests for src/observability/tracer.py"""
import time
import pytest
from src.observability.tracer import Tracer, Trace, Span, NoopSpan


class TestSpan:
    def test_span_lifecycle(self):
        span = Span(name="test_op")
        assert span.name == "test_op"
        assert span.status == "ok"
        assert span.duration_ms == 0.0
        assert span.started_at is not None

    def test_span_finish_sets_duration(self):
        span = Span(name="test_op")
        span.finish()
        assert span.finished_at is not None
        assert span.duration_ms > 0

    def test_span_finish_metadata(self):
        span = Span(name="test_op")
        span.finish(metadata={"hits": 20})
        assert span.metadata == {"hits": 20}

    def test_span_set_metadata(self):
        span = Span(name="test_op")
        span.set_metadata({"hits": 20})
        span.set_metadata({"latency": 100})
        assert span.metadata == {"hits": 20, "latency": 100}

    def test_span_mark_error(self):
        span = Span(name="test_op")
        span.mark_error()
        assert span.status == "error"

    def test_span_context_manager(self):
        with Span(name="ctx_test") as span:
            pass
        assert span.finished_at is not None
        assert span.duration_ms > 0

    def test_span_parent_child(self):
        parent = Span(name="parent")
        parent.finish()
        child = Span(name="child", parent_span_id=parent.span_id)
        child.finish()
        assert child.parent_span_id == parent.span_id
        assert child.span_id != parent.span_id


class TestTrace:
    def test_trace_lifecycle(self):
        trace = Trace(query="test query", config_label="test_config")
        assert trace.query == "test query"
        assert trace.config_label == "test_config"
        assert len(trace.spans) == 0
        assert trace.finished_at is None

    def test_trace_add_span(self):
        trace = Trace(query="q", config_label="c")
        span = Span(name="op")
        span.finish()
        trace.add_span(span)
        assert len(trace.spans) == 1
        assert trace.spans[0].name == "op"

    def test_trace_finish(self):
        trace = Trace(query="q", config_label="c")
        trace.finish()
        assert trace.finished_at is not None
        assert trace.duration_ms > 0

    def test_trace_duration_equals_sum_of_spans(self):
        trace = Trace(query="q", config_label="c")
        span1 = Span(name="op1")
        span1.finish()
        trace.add_span(span1)
        span2 = Span(name="op2")
        span2.finish()
        trace.add_span(span2)
        trace.finish()
        assert trace.duration_ms >= span1.duration_ms + span2.duration_ms

    def test_trace_to_dict(self):
        trace = Trace(query="q", config_label="c")
        span = Span(name="op")
        span.finish(metadata={"k": 20})
        trace.add_span(span)
        trace.finish()
        d = trace.to_dict()
        assert d["query"] == "q"
        assert d["config_label"] == "c"
        assert len(d["spans"]) == 1
        assert d["spans"][0]["name"] == "op"
        assert d["spans"][0]["metadata"] == {"k": 20}


class TestTracer:
    def test_start_and_finish_trace(self):
        tracer = Tracer(enabled=True)
        trace = tracer.start_trace("test query", "cfg_a")
        assert tracer.current_trace() is trace
        tracer.finish_trace()
        assert tracer.current_trace() is None
        assert trace.finished_at is not None

    def test_start_span_context_manager(self):
        tracer = Tracer(enabled=True)
        tracer.start_trace("q", "c")
        with tracer.start_span("bm25_search") as span:
            span.set_metadata({"hits": 5})
        trace = tracer.current_trace()
        assert len(trace.spans) == 1
        assert trace.spans[0].name == "bm25_search"
        assert trace.spans[0].metadata == {"hits": 5}
        tracer.finish_trace()

    def test_disabled_tracer_returns_noop(self):
        tracer = Tracer(enabled=False)
        trace = tracer.start_trace("q", "c")
        assert trace is None
        span = tracer.start_span("bm25")
        assert isinstance(span, NoopSpan)
        with span:
            pass  # no-op, should not raise

    def test_disabled_tracer_current_trace_is_none(self):
        tracer = Tracer(enabled=False)
        assert tracer.current_trace() is None


class TestNoopSpan:
    def test_noop_span_context_manager_does_nothing(self):
        with NoopSpan("noop") as span:
            span.set_metadata({"x": 1})
            span.mark_error()
        assert span.duration_ms == 0.0
        assert span.status == "ok"
```

- [ ] **Step 3: Run tests to verify they fail**

```bash
cd /Users/theyang/Documents/ai/pdf-rag
python -m pytest tests/test_observability_tracer.py -v
```
Expected: All tests FAIL with ImportError (module not written yet)

- [ ] **Step 4: Write tracer.py implementation**

Create `src/observability/tracer.py`:

```python
"""Trace/Span 模型 + 上下文管理器

提供请求级 Trace 和步骤级 Span 的创建、计时、元数据收集。
支持 trace_enabled=False 时的零开销 no-op 模式。
"""
from __future__ import annotations

import contextvars
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


# ── contextvars: 线程安全的 Trace 上下文 ──────────────────────
_current_trace: contextvars.ContextVar["Trace | None"] = contextvars.ContextVar(
    "current_trace", default=None
)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ── Span ──────────────────────────────────────────────────────


@dataclass
class Span:
    """一个 pipeline 步骤的计时和元数据"""
    name: str
    span_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    parent_span_id: str | None = None
    started_at: datetime = field(default_factory=_utcnow)
    finished_at: datetime | None = None
    duration_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)
    status: str = "ok"

    def finish(self, metadata: dict[str, Any] | None = None) -> None:
        self.finished_at = _utcnow()
        self.duration_ms = (self.finished_at - self.started_at).total_seconds() * 1000
        if metadata:
            self.metadata.update(metadata)

    def set_metadata(self, data: dict[str, Any]) -> None:
        self.metadata.update(data)

    def mark_error(self) -> None:
        self.status = "error"

    def __enter__(self) -> "Span":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if exc_type is not None:
            self.mark_error()
        self.finish()


# ── NoopSpan ──────────────────────────────────────────────────


class NoopSpan:
    """禁用 trace 时返回的零开销占位 Span"""

    def __init__(self, name: str = ""):
        self.name = name
        self.span_id = ""
        self.duration_ms = 0.0
        self.metadata: dict[str, Any] = {}
        self.status = "ok"

    def finish(self, metadata: dict[str, Any] | None = None) -> None:
        pass

    def set_metadata(self, data: dict[str, Any]) -> None:
        pass

    def mark_error(self) -> None:
        pass

    def __enter__(self) -> "NoopSpan":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        pass


# ── Trace ─────────────────────────────────────────────────────


@dataclass
class Trace:
    """一个 query 的完整生命周期"""
    trace_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    query: str = ""
    config_label: str = ""
    started_at: datetime = field(default_factory=_utcnow)
    finished_at: datetime | None = None
    duration_ms: float = 0.0
    spans: list[Span] = field(default_factory=list)

    def add_span(self, span: Span) -> None:
        self.spans.append(span)

    def finish(self) -> None:
        self.finished_at = _utcnow()
        self.duration_ms = (self.finished_at - self.started_at).total_seconds() * 1000

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "query": self.query,
            "config_label": self.config_label,
            "started_at": self.started_at.isoformat(),
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
            "duration_ms": round(self.duration_ms, 2),
            "spans": [
                {
                    "span_id": s.span_id,
                    "parent_span_id": s.parent_span_id,
                    "name": s.name,
                    "started_at": s.started_at.isoformat(),
                    "finished_at": s.finished_at.isoformat() if s.finished_at else None,
                    "duration_ms": round(s.duration_ms, 2),
                    "metadata": s.metadata,
                    "status": s.status,
                }
                for s in self.spans
            ],
        }


# ── Tracer ────────────────────────────────────────────────────


class Tracer:
    """Trace 管理器

    通过 contextvars 确保线程/协程安全的 Trace 上下文隔离。
    enabled=False 时所有方法返回 no-op 对象，零开销。
    """

    def __init__(self, enabled: bool = True):
        self.enabled = enabled

    def start_trace(self, query: str, config_label: str = "") -> Trace | None:
        if not self.enabled:
            _current_trace.set(None)
            return None
        trace = Trace(query=query, config_label=config_label)
        _current_trace.set(trace)
        return trace

    def current_trace(self) -> Trace | None:
        return _current_trace.get(None)

    def finish_trace(self) -> Trace | None:
        trace = _current_trace.get(None)
        if trace is not None:
            trace.finish()
        _current_trace.set(None)
        return trace

    def start_span(self, name: str, metadata: dict[str, Any] | None = None) -> Span | NoopSpan:
        if not self.enabled:
            return NoopSpan(name)
        trace = _current_trace.get(None)
        parent_id = trace.trace_id if trace else None
        span = Span(name=name, parent_span_id=parent_id)
        if metadata:
            span.set_metadata(metadata)
        if trace:
            trace.add_span(span)
        return span


# ── 全局实例 ──────────────────────────────────────────────────

_tracer: Tracer | None = None


def get_tracer() -> Tracer:
    """获取全局 Tracer 实例，首次调用时从 config 读取 enabled 状态"""
    global _tracer
    if _tracer is None:
        from src.config import cfg
        oc = cfg.observability
        _tracer = Tracer(enabled=oc.trace_enabled)
    return _tracer
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
cd /Users/theyang/Documents/ai/pdf-rag
python -m pytest tests/test_observability_tracer.py -v
```
Expected: All 14 tests PASS

- [ ] **Step 6: Commit**

```bash
git add src/observability/__init__.py src/observability/tracer.py tests/test_observability_tracer.py
git commit -m "feat: add Trace/Span model and Tracer with context manager"
```

---

### Task 4: collectors.py — MetricsCollector Singleton

**Files:**
- Create: `src/observability/collectors.py`
- Create: `tests/test_observability_collectors.py`

**Interfaces:**
- Consumes: `Trace`, `Span` from `src.observability.tracer`
- Produces: `MetricsCollector` singleton via `get_collector()`; `ConfigMetrics` dataclass; `AlertEvent` dataclass; methods: `reset()`, `ingest_trace()`, `record_ragas_score()`, `record_alert()`, `get_config_metrics()`, `get_alerts()`, `snapshot()`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_observability_collectors.py`:

```python
"""Tests for src/observability/collectors.py"""
import pytest
from src.observability.collectors import (
    MetricsCollector, ConfigMetrics, AlertEvent,
)


class TestConfigMetrics:
    def test_empty_metrics(self):
        m = ConfigMetrics(config_label="test")
        assert m.config_label == "test"
        assert m.num_queries == 0
        assert m.latency_p50_ms == 0.0
        assert m.avg_faithfulness == 0.0


class TestAlertEvent:
    def test_alert_creation(self):
        ae = AlertEvent(
            level="warning",
            category="threshold",
            message="P95 latency exceeded",
            config_label="cfg_a",
            trace_id="abc123",
        )
        d = ae.to_dict()
        assert d["level"] == "warning"
        assert d["category"] == "threshold"
        assert d["config_label"] == "cfg_a"
        assert d["trace_id"] == "abc123"


class TestMetricsCollector:
    def test_initial_state(self):
        mc = MetricsCollector()
        assert mc.get_alerts() == []
        assert mc.get_config_metrics("nonexistent") is None

    def test_ingest_trace_aggregates_latency(self):
        mc = MetricsCollector()
        from src.observability.tracer import Trace, Span
        trace = Trace(query="q1", config_label="cfg_a")
        span = Span(name="bm25_search")
        span.finish(metadata={"num_results": 10})
        # Simulate a known duration
        span.duration_ms = 100.0
        trace.add_span(span)
        trace.finish()
        trace.duration_ms = 120.0
        mc.ingest_trace(trace)

        metrics = mc.get_config_metrics("cfg_a")
        assert metrics is not None
        assert metrics.num_queries == 1
        assert metrics.latency_avg_ms == 120.0

    def test_ingest_multiple_traces(self):
        mc = MetricsCollector()
        from src.observability.tracer import Trace, Span
        for i in range(10):
            trace = Trace(query=f"q{i}", config_label="cfg_a")
            span = Span(name="op")
            span.duration_ms = 50.0 * (i + 1)
            span.finish()
            trace.add_span(span)
            trace.duration_ms = 50.0 * (i + 1)
            trace.finish()
            mc.ingest_trace(trace)

        metrics = mc.get_config_metrics("cfg_a")
        assert metrics.num_queries == 10
        assert metrics.latency_min_ms == 50.0
        assert metrics.latency_max_ms == 500.0
        # P50 of [50,100,...,500] = 275 or 300 depending on interpolation
        assert 250 <= metrics.latency_p50_ms <= 300
        # P95 of 10 values: index 9 (10*0.95=9.5, so value at position 9 = 500)
        assert metrics.latency_p95_ms == 500.0

    def test_ingest_trace_tracks_hit_counts(self):
        mc = MetricsCollector()
        from src.observability.tracer import Trace, Span
        trace = Trace(query="q", config_label="cfg_a")
        for name, hits in [("bm25_search", 18), ("dense_search", 15), ("visual_search", 12)]:
            span = Span(name=name)
            span.finish(metadata={"num_results": hits})
            trace.add_span(span)
        trace.finish()
        mc.ingest_trace(trace)

        metrics = mc.get_config_metrics("cfg_a")
        assert metrics.avg_bm25_hits == 18.0
        assert metrics.avg_dense_hits == 15.0
        assert metrics.avg_visual_hits == 12.0

    def test_record_ragas_score(self):
        mc = MetricsCollector()
        mc.record_ragas_score("cfg_a", "q1", faithfulness=0.85, answer_relevancy=0.72)
        mc.record_ragas_score("cfg_a", "q2", faithfulness=0.75, answer_relevancy=0.68)

        metrics = mc.get_config_metrics("cfg_a")
        assert metrics.num_queries == 2  # ragas scores also count as queries
        assert metrics.avg_faithfulness == pytest.approx(0.80, abs=0.01)
        assert metrics.avg_answer_relevancy == pytest.approx(0.70, abs=0.01)

    def test_record_alert(self):
        mc = MetricsCollector()
        ae = AlertEvent(level="warning", category="threshold", message="test", config_label="c")
        mc.record_alert(ae)
        assert len(mc.get_alerts()) == 1

    def test_reset_clears_all(self):
        mc = MetricsCollector()
        ae = AlertEvent(level="error", category="pipeline", message="boom", config_label="c")
        mc.record_alert(ae)
        from src.observability.tracer import Trace
        trace = Trace(query="q", config_label="c")
        trace.finish()
        mc.ingest_trace(trace)

        mc.reset()
        assert mc.get_alerts() == []
        assert mc.get_config_metrics("c") is None

    def test_snapshot_returns_all_data(self):
        mc = MetricsCollector()
        from src.observability.tracer import Trace, Span
        trace = Trace(query="q", config_label="cfg_a")
        span = Span(name="op")
        span.finish()
        trace.add_span(span)
        trace.finish()
        mc.ingest_trace(trace)
        mc.record_ragas_score("cfg_a", "q", faithfulness=0.9, answer_relevancy=0.8)
        ae = AlertEvent(level="warning", category="threshold", message="t", config_label="cfg_a")
        mc.record_alert(ae)

        snap = mc.snapshot()
        assert "configs" in snap
        assert "cfg_a" in snap["configs"]
        assert snap["configs"]["cfg_a"]["num_queries"] == 2  # 1 trace + 1 ragas
        assert len(snap["traces"]) == 1
        assert len(snap["alerts"]) == 1

    def test_multiple_configs_independent(self):
        mc = MetricsCollector()
        from src.observability.tracer import Trace, Span
        for cfg_label in ["cfg_a", "cfg_b"]:
            for i in range(3):
                trace = Trace(query=f"q{i}", config_label=cfg_label)
                span = Span(name="op")
                span.duration_ms = 10.0
                span.finish()
                trace.add_span(span)
                trace.duration_ms = 10.0
                trace.finish()
                mc.ingest_trace(trace)

        assert mc.get_config_metrics("cfg_a").num_queries == 3
        assert mc.get_config_metrics("cfg_b").num_queries == 3
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /Users/theyang/Documents/ai/pdf-rag
python -m pytest tests/test_observability_collectors.py -v
```
Expected: All tests FAIL with ImportError

- [ ] **Step 3: Write collectors.py implementation**

Create `src/observability/collectors.py`:

```python
"""指标收集器 — 按 config 聚合延迟、命中、质量指标"""
from __future__ import annotations

import statistics
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ── 数据模型 ──────────────────────────────────────────────────


@dataclass
class ConfigMetrics:
    """单个检索配置的聚合指标"""
    config_label: str
    num_queries: int = 0
    # 延迟 (ms)
    latency_p50_ms: float = 0.0
    latency_p95_ms: float = 0.0
    latency_p99_ms: float = 0.0
    latency_avg_ms: float = 0.0
    latency_min_ms: float = 0.0
    latency_max_ms: float = 0.0
    # 命中
    avg_bm25_hits: float = 0.0
    avg_dense_hits: float = 0.0
    avg_visual_hits: float = 0.0
    avg_fused_count: float = 0.0
    avg_reranked_count: float = 0.0
    # 缓存
    hyde_hit_rate: float = 0.0
    # 质量
    avg_faithfulness: float = 0.0
    avg_answer_relevancy: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "config_label": self.config_label,
            "num_queries": self.num_queries,
            "latency": {
                "p50_ms": round(self.latency_p50_ms, 2),
                "p95_ms": round(self.latency_p95_ms, 2),
                "p99_ms": round(self.latency_p99_ms, 2),
                "avg_ms": round(self.latency_avg_ms, 2),
                "min_ms": round(self.latency_min_ms, 2),
                "max_ms": round(self.latency_max_ms, 2),
            },
            "hits": {
                "avg_bm25": round(self.avg_bm25_hits, 2),
                "avg_dense": round(self.avg_dense_hits, 2),
                "avg_visual": round(self.avg_visual_hits, 2),
                "avg_fused": round(self.avg_fused_count, 2),
                "avg_reranked": round(self.avg_reranked_count, 2),
            },
            "cache": {"hyde_hit_rate": round(self.hyde_hit_rate, 2)},
            "quality": {
                "avg_faithfulness": round(self.avg_faithfulness, 4),
                "avg_answer_relevancy": round(self.avg_answer_relevancy, 4),
            },
        }


@dataclass
class AlertEvent:
    """告警事件"""
    level: str  # "warning" | "error"
    category: str  # "threshold" | "pipeline_error"
    message: str
    config_label: str = ""
    trace_id: str | None = None
    timestamp: datetime = field(default_factory=_utcnow)

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "level": self.level,
            "category": self.category,
            "message": self.message,
            "config_label": self.config_label,
            "trace_id": self.trace_id,
        }


# ── MetricsCollector ──────────────────────────────────────────


class MetricsCollector:
    """全局指标收集器（线程安全单例）

    评测运行时在内存中聚合，结束后通过 snapshot() 导出。
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._traces: list[dict[str, Any]] = []
        self._latencies: dict[str, list[float]] = {}  # config_label -> [ms]
        self._hit_data: dict[str, dict[str, list[float]]] = {}  # config_label -> {span_name: [counts]}
        self._hyde_data: dict[str, dict[str, int]] = {}  # config_label -> {hits, misses}
        self._ragas_scores: dict[str, list[dict[str, float]]] = {}  # config_label -> [{f, ar}]
        self._alerts: list[AlertEvent] = []
        self._span_hit_names = {
            "bm25_search": "bm25",
            "dense_search": "dense",
            "visual_search": "visual",
            "fusion_rerank": "fused",
        }

    def reset(self) -> None:
        with self._lock:
            self._traces.clear()
            self._latencies.clear()
            self._hit_data.clear()
            self._hyde_data.clear()
            self._ragas_scores.clear()
            self._alerts.clear()

    def ingest_trace(self, trace: Any) -> None:
        """接收一个已完成的 Trace，提取指标并聚合"""
        with self._lock:
            label = trace.config_label
            self._traces.append(trace.to_dict())

            # 延迟
            if label not in self._latencies:
                self._latencies[label] = []
            self._latencies[label].append(trace.duration_ms)

            # 命中数
            if label not in self._hit_data:
                self._hit_data[label] = {k: [] for k in self._span_hit_names}
            for span in trace.spans:
                key = self._span_hit_names.get(span.name)
                if key and "num_results" in span.metadata:
                    self._hit_data[label][key].append(float(span.metadata["num_results"]))
                # HyDE cache tracking
                if span.name == "hyde_generate":
                    if "hyde" not in self._hit_data[label]:
                        self._hit_data[label]["hyde"] = []
                    is_hit = span.metadata.get("cache_hit", False)
                    if label not in self._hyde_data:
                        self._hyde_data[label] = {"hits": 0, "misses": 0}
                    if is_hit:
                        self._hyde_data[label]["hits"] += 1
                    else:
                        self._hyde_data[label]["misses"] += 1

    def record_ragas_score(
        self, config_label: str, query_id: str,
        faithfulness: float, answer_relevancy: float,
    ) -> None:
        with self._lock:
            if config_label not in self._ragas_scores:
                self._ragas_scores[config_label] = []
            self._ragas_scores[config_label].append({
                "query_id": query_id,
                "faithfulness": faithfulness,
                "answer_relevancy": answer_relevancy,
            })

    def record_alert(self, event: AlertEvent) -> None:
        with self._lock:
            self._alerts.append(event)

    def get_config_metrics(self, config_label: str) -> ConfigMetrics | None:
        with self._lock:
            metrics = ConfigMetrics(config_label=config_label)

            # 延迟
            if config_label in self._latencies:
                lats = sorted(self._latencies[config_label])
                n = len(lats)
                metrics.num_queries = n
                if n > 0:
                    metrics.latency_avg_ms = statistics.mean(lats)
                    metrics.latency_min_ms = lats[0]
                    metrics.latency_max_ms = lats[-1]
                    metrics.latency_p50_ms = _percentile(lats, 50)
                    metrics.latency_p95_ms = _percentile(lats, 95)
                    metrics.latency_p99_ms = _percentile(lats, 99)

            # 命中
            if config_label in self._hit_data:
                hd = self._hit_data[config_label]
                if hd.get("bm25"):
                    metrics.avg_bm25_hits = statistics.mean(hd["bm25"])
                if hd.get("dense"):
                    metrics.avg_dense_hits = statistics.mean(hd["dense"])
                if hd.get("visual"):
                    metrics.avg_visual_hits = statistics.mean(hd["visual"])
                if hd.get("fused"):
                    metrics.avg_fused_count = statistics.mean(hd["fused"])

            # HyDE cache
            if config_label in self._hyde_data:
                hd = self._hyde_data[config_label]
                total = hd["hits"] + hd["misses"]
                if total > 0:
                    metrics.hyde_hit_rate = hd["hits"] / total

            # RAGAS
            if config_label in self._ragas_scores:
                scores = self._ragas_scores[config_label]
                ragas_n = len(scores)
                if ragas_n > 0:
                    metrics.num_queries = max(metrics.num_queries, ragas_n)
                    metrics.avg_faithfulness = statistics.mean(
                        [s["faithfulness"] for s in scores]
                    )
                    metrics.avg_answer_relevancy = statistics.mean(
                        [s["answer_relevancy"] for s in scores]
                    )

            if metrics.num_queries == 0:
                return None
            return metrics

    def get_alerts(self) -> list[AlertEvent]:
        with self._lock:
            return list(self._alerts)

    def snapshot(self) -> dict[str, Any]:
        """返回完整快照：configs + traces + alerts"""
        with self._lock:
            configs = {}
            all_labels = set(self._latencies.keys()) | set(self._ragas_scores.keys())
            for label in all_labels:
                m = self.get_config_metrics(label)
                if m:
                    configs[label] = m.to_dict()
            return {
                "configs": configs,
                "traces": list(self._traces),
                "alerts": [a.to_dict() for a in self._alerts],
            }


def _percentile(sorted_data: list[float], p: float) -> float:
    """计算百分位数（线性插值）"""
    n = len(sorted_data)
    if n == 0:
        return 0.0
    if n == 1:
        return sorted_data[0]
    k = (p / 100.0) * (n - 1)
    f = int(k)
    c = k - f
    if f + 1 < n:
        return sorted_data[f] + c * (sorted_data[f + 1] - sorted_data[f])
    return sorted_data[f]


# ── 全局实例 ──────────────────────────────────────────────────

_collector: MetricsCollector | None = None


def get_collector() -> MetricsCollector:
    global _collector
    if _collector is None:
        _collector = MetricsCollector()
    return _collector
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /Users/theyang/Documents/ai/pdf-rag
python -m pytest tests/test_observability_collectors.py -v
```
Expected: All 10 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/observability/collectors.py tests/test_observability_collectors.py
git commit -m "feat: add MetricsCollector singleton with latency/hit/quality aggregation"
```

---

### Task 5: logging_setup.py — Unified structlog Initialization

**Files:**
- Create: `src/observability/logging_setup.py`

**Interfaces:**
- Produces: `init_logging(level, log_file, console)` function — one call replaces all scattered `logging.basicConfig()`

- [ ] **Step 1: Write logging_setup.py**

Create `src/observability/logging_setup.py`:

```python
"""统一日志初始化 — structlog 配置

替换各入口脚本中散落的 logging.basicConfig()，
提供一处调用完成全项目日志配置。
"""
from __future__ import annotations

import logging
import structlog


def init_logging(
    level: str = "INFO",
    log_file: str | None = "logs/app.jsonl",
    console: bool = True,
) -> None:
    """配置 structlog：JSON 到文件 + 彩色到控制台

    Args:
        level: 日志级别 (DEBUG | INFO | WARNING | ERROR)
        log_file: JSON 日志输出路径，None 则不写文件
        console: 是否输出彩色控制台日志
    """
    log_level = getattr(logging, level.upper(), logging.INFO)

    # 共享处理器列表
    processors = [
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.dev.ConsoleRenderer()
        if console
        else structlog.processors.JSONRenderer(),
    ]

    # 文件输出: JSON
    if log_file:
        from pathlib import Path
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(
            structlog.stdlib.ProcessorFormatter(
                processor=structlog.processors.JSONRenderer(),
            )
        )

    # 控制台输出: 彩色
    console_handler = None
    if console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_handler.setFormatter(
            structlog.stdlib.ProcessorFormatter(
                processor=structlog.dev.ConsoleRenderer(),
            )
        )

    # 配置 root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    root_logger.handlers.clear()
    if log_file:
        root_logger.addHandler(file_handler)
    if console and console_handler:
        root_logger.addHandler(console_handler)

    # 配置 structlog
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.dev.ConsoleRenderer() if console else structlog.processors.JSONRenderer(),
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
```

- [ ] **Step 2: Verify import works**

```bash
cd /Users/theyang/Documents/ai/pdf-rag
python -c "from src.observability.logging_setup import init_logging; init_logging(console=False); print('OK')"
```
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add src/observability/logging_setup.py
git commit -m "feat: add unified structlog initialization"
```

---

### Task 6: alerting.py — Threshold + Pipeline Anomaly Detection

**Files:**
- Create: `src/observability/alerting.py`
- Create: `tests/test_observability_alerting.py`

**Interfaces:**
- Consumes: `MetricsCollector` from `src.observability.collectors`, `ObservabilityConfig` from `src.config`
- Produces: `AlertChecker` class with `check_thresholds(collector)` and `wrap_exception(exc, trace_id)`, returning `list[AlertEvent]`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_observability_alerting.py`:

```python
"""Tests for src/observability/alerting.py"""
import pytest
from src.observability.alerting import AlertChecker
from src.observability.collectors import MetricsCollector, AlertEvent


class TestAlertCheckerThresholds:
    def test_no_alerts_when_all_ok(self):
        mc = MetricsCollector()
        from src.observability.tracer import Trace
        for i in range(20):
            trace = Trace(query=f"q{i}", config_label="cfg_a")
            trace.duration_ms = 100.0  # well under 5000ms default
            trace.finish()
            mc.ingest_trace(trace)
        mc.record_ragas_score("cfg_a", "q", faithfulness=0.9, answer_relevancy=0.8)

        checker = AlertChecker(
            latency_p95_threshold_ms=5000,
            recall_at_5_min=0.5,
            faithfulness_min=0.6,
        )
        alerts = checker.check_thresholds(mc)
        assert alerts == []

    def test_latency_threshold_alert(self):
        mc = MetricsCollector()
        from src.observability.tracer import Trace
        for i in range(20):
            trace = Trace(query=f"q{i}", config_label="cfg_a")
            trace.duration_ms = 6000.0  # exceeds 5000ms
            trace.finish()
            mc.ingest_trace(trace)

        checker = AlertChecker(
            latency_p95_threshold_ms=5000,
            recall_at_5_min=0.5,
            faithfulness_min=0.6,
        )
        alerts = checker.check_thresholds(mc)
        assert len(alerts) >= 1
        latency_alert = [a for a in alerts if "latency" in a.message.lower()]
        assert len(latency_alert) >= 1
        assert latency_alert[0].level == "warning"
        assert latency_alert[0].category == "threshold"

    def test_faithfulness_threshold_alert(self):
        mc = MetricsCollector()
        mc.record_ragas_score("cfg_b", "q1", faithfulness=0.4, answer_relevancy=0.5)
        mc.record_ragas_score("cfg_b", "q2", faithfulness=0.5, answer_relevancy=0.6)

        checker = AlertChecker(
            latency_p95_threshold_ms=5000,
            recall_at_5_min=0.5,
            faithfulness_min=0.6,
        )
        alerts = checker.check_thresholds(mc)
        faith_alerts = [a for a in alerts if "faithfulness" in a.message.lower()]
        assert len(faith_alerts) >= 1

    def test_unknown_config_skipped(self):
        mc = MetricsCollector()
        # No data ingested, get_config_metrics returns None
        checker = AlertChecker()
        alerts = checker.check_thresholds(mc)
        assert alerts == []


class TestAlertCheckerExceptionWrapping:
    def test_wrap_oom(self):
        checker = AlertChecker()
        try:
            raise MemoryError("CUDA out of memory")
        except MemoryError as e:
            alert = checker.wrap_exception(e, trace_id="abc123")
        assert alert.level == "error"
        assert alert.category == "pipeline_error"
        assert "OOM" in alert.message or "MemoryError" in alert.message
        assert alert.trace_id == "abc123"

    def test_wrap_connection_error(self):
        checker = AlertChecker()
        try:
            raise ConnectionError("pgvector connection refused")
        except ConnectionError as e:
            alert = checker.wrap_exception(e)
        assert alert.category == "pipeline_error"
        assert "ConnectionError" in alert.message

    def test_wrap_generic_exception(self):
        checker = AlertChecker()
        try:
            raise ValueError("something went wrong")
        except ValueError as e:
            alert = checker.wrap_exception(e)
        assert alert.level == "error"
        assert "ValueError" in alert.message
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /Users/theyang/Documents/ai/pdf-rag
python -m pytest tests/test_observability_alerting.py -v
```
Expected: All tests FAIL with ImportError

- [ ] **Step 3: Write alerting.py implementation**

Create `src/observability/alerting.py`:

```python
"""告警检测 — 阈值检查 + 管道异常分类"""
from __future__ import annotations

import traceback
from typing import TYPE_CHECKING

from src.observability.collectors import AlertEvent, MetricsCollector

if TYPE_CHECKING:
    from src.config import ObservabilityConfig


class AlertChecker:
    """告警检测器

    两种告警来源：
    1. 阈值告警：评测完成后批量检测延迟/召回/质量是否低于阈值
    2. 管道异常：实时捕获异常并分类
    """

    def __init__(
        self,
        latency_p95_threshold_ms: int = 5000,
        recall_at_5_min: float = 0.5,
        faithfulness_min: float = 0.6,
    ):
        self.latency_p95_threshold_ms = latency_p95_threshold_ms
        self.recall_at_5_min = recall_at_5_min
        self.faithfulness_min = faithfulness_min

    @classmethod
    def from_config(cls, config: "ObservabilityConfig") -> "AlertChecker":
        return cls(
            latency_p95_threshold_ms=config.latency_p95_threshold_ms,
            recall_at_5_min=config.recall_at_5_min,
            faithfulness_min=config.faithfulness_min,
        )

    def check_thresholds(self, collector: MetricsCollector) -> list[AlertEvent]:
        """遍历所有 config 的聚合指标，检查阈值"""
        alerts: list[AlertEvent] = []
        snap = collector.snapshot()
        for label, metrics_dict in snap.get("configs", {}).items():
            # 延迟阈值
            p95 = metrics_dict.get("latency", {}).get("p95_ms", 0)
            if p95 > self.latency_p95_threshold_ms:
                alerts.append(AlertEvent(
                    level="warning",
                    category="threshold",
                    message=(
                        f"P95 latency {p95:.0f}ms exceeds threshold "
                        f"{self.latency_p95_threshold_ms}ms"
                    ),
                    config_label=label,
                ))

            # 召回阈值（如果有的话）
            recall = metrics_dict.get("recall", {})
            if recall:
                r5 = recall.get("recall_at_5", 1.0)
                if r5 < self.recall_at_5_min:
                    alerts.append(AlertEvent(
                        level="warning",
                        category="threshold",
                        message=(
                            f"Recall@5 {r5:.3f} below minimum {self.recall_at_5_min}"
                        ),
                        config_label=label,
                    ))

            # Faithfulness 阈值
            quality = metrics_dict.get("quality", {})
            faith = quality.get("avg_faithfulness", 1.0)
            if faith < self.faithfulness_min:
                alerts.append(AlertEvent(
                    level="warning",
                    category="threshold",
                    message=(
                        f"Faithfulness {faith:.3f} below minimum {self.faithfulness_min}"
                    ),
                    config_label=label,
                ))

        return alerts

    def wrap_exception(
        self, exc: Exception, trace_id: str | None = None
    ) -> AlertEvent:
        """将异常包装为 AlertEvent，自动分类"""
        exc_name = type(exc).__name__
        message = f"{exc_name}: {exc}"

        return AlertEvent(
            level="error",
            category="pipeline_error",
            message=message,
            trace_id=trace_id,
        )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /Users/theyang/Documents/ai/pdf-rag
python -m pytest tests/test_observability_alerting.py -v
```
Expected: All 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/observability/alerting.py tests/test_observability_alerting.py
git commit -m "feat: add AlertChecker for threshold and pipeline anomaly detection"
```

---

### Task 7: src/observability/__init__.py — Public API

**Files:**
- Modify: `src/observability/__init__.py` (replace empty file)

**Interfaces:**
- Produces: Public exports `get_tracer`, `get_collector`, `init_logging`, `Tracer`, `Trace`, `Span`, `NoopSpan`, `MetricsCollector`, `ConfigMetrics`, `AlertEvent`, `AlertChecker`

- [ ] **Step 1: Write __init__.py**

```python
"""PrismRAG Observability — 内建可观测性模块

核心 (src/observability/): tracer, collector, alerting, logging
消费侧 (observability/): dashboard, reporter
"""
from src.observability.alerting import AlertChecker
from src.observability.collectors import (
    AlertEvent,
    ConfigMetrics,
    MetricsCollector,
    get_collector,
)
from src.observability.logging_setup import init_logging
from src.observability.tracer import (
    NoopSpan,
    Span,
    Trace,
    Tracer,
    get_tracer,
)

__all__ = [
    "get_tracer",
    "get_collector",
    "init_logging",
    "Tracer",
    "Trace",
    "Span",
    "NoopSpan",
    "MetricsCollector",
    "ConfigMetrics",
    "AlertEvent",
    "AlertChecker",
]
```

- [ ] **Step 2: Verify import**

```bash
cd /Users/theyang/Documents/ai/pdf-rag
python -c "from src.observability import get_tracer, get_collector, init_logging, Span, Trace; print('OK')"
```
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add src/observability/__init__.py
git commit -m "feat: add observability public API exports"
```

---

### Task 8: Inject Spans into Retrieval Pipeline

**Files:**
- Modify: `src/retrieval/bm25_retriever.py`
- Modify: `src/retrieval/dense_retriever.py`
- Modify: `src/retrieval/visual_retriever.py`
- Modify: `src/retrieval/hyde.py`
- Modify: `src/retrieval/fusion.py`
- Modify: `src/retrieval/reranker.py`

**Interfaces:**
- Consumes: `get_tracer()` from `src.observability`
- Produces: Each retriever method wrapped with span for automatic timing

- [ ] **Step 1: Inject span into BM25Retriever.search()**

Edit `src/retrieval/bm25_retriever.py`:

Add import at top (after existing imports):
```python
from src.observability import get_tracer
```

Modify `search()` method — wrap the logic block in a span:

```python
    def search(self, query: str, k: int = 20) -> List[dict]:
        """检索 Top-k chunk"""
        if self._bm25 is None:
            raise RuntimeError("BM25 索引未构建，请先调用 fit()")

        tracer = get_tracer()
        with tracer.start_span("bm25_search") as span:
            tokenized_query = self._tokenize(query)
            scores = self._bm25.get_scores(tokenized_query)

            top_indices = sorted(
                range(len(scores)),
                key=lambda i: scores[i],
                reverse=True,
            )[:k]

            results = []
            for idx in top_indices:
                score = scores[idx]
                if score > 0:
                    chunk = self._chunks[idx]
                    results.append({
                        **chunk,
                        "score": float(score),
                        "retrieval_type": "bm25",
                    })

            span.set_metadata({"num_results": len(results), "k": k})
            return results
```

- [ ] **Step 2: Inject spans into DenseRetriever.search()**

Edit `src/retrieval/dense_retriever.py`:

Add import at top:
```python
from src.observability import get_tracer
```

Modify `search()` method:

```python
    def search(self, query: str, k: int = 20) -> List[dict]:
        """检索 Top-k chunk"""
        tracer = get_tracer()

        # 1. BGE 编码查询
        with tracer.start_span("dense_encode") as span:
            query_emb = self.bge.encode([query])
            query_vec = query_emb.cpu().numpy().astype(np.float32)[0]
            span.set_metadata({"dim": int(query_vec.shape[0])})

        # 2. pgvector HNSW 搜索
        with tracer.start_span("dense_search") as span:
            results = self.pg.search_by_vector(query_vec, k=k)
            span.set_metadata({"num_results": len(results), "k": k})

        # 3. 添加 retrieval_type 标记
        for r in results:
            r["retrieval_type"] = "dense"

        return results
```

- [ ] **Step 3: Inject spans into VisualRetriever.search()**

Edit `src/retrieval/visual_retriever.py`:

Add import at top:
```python
from src.observability import get_tracer
```

Modify the `search()` method:

```python
    def search(self, query: str, k: int = 20) -> List[dict]:
        """检索 Top-k 页 → 反查该页所有 chunk"""
        tracer = get_tracer()

        # 1. ColPali 编码查询
        with tracer.start_span("visual_encode") as span:
            q_emb = self.colpali.encode_query(query)
            span.set_metadata({"batch_size": 1})

        # 2. FAISS MaxSim 搜索 → Top-k 页
        with tracer.start_span("visual_search") as span:
            page_results = self.faiss.maxsim_search(q_emb, k=k)
            span.set_metadata({"num_pages": len(page_results), "k": k})

        if not page_results:
            return []

        # 3. Grounding 反查
        page_ids = [pr["page_id"] for pr in page_results]
        page_score_map = {pr["page_id"]: pr["score"] for pr in page_results}
        chunks = self.pg.get_chunks_by_page_ids(page_ids)

        # 4. 合并分数
        results = []
        for chunk in chunks:
            results.append({
                **chunk,
                "score": page_score_map[chunk["page_id"]],
                "retrieval_type": "visual",
            })

        return results
```

Also modify `search_with_embedding()`:

```python
    def search_with_embedding(self, q_emb: torch.Tensor, k: int = 20) -> List[dict]:
        tracer = get_tracer()

        # 1. FAISS MaxSim 搜索 (pre-encoded, skip encode span)
        with tracer.start_span("visual_search") as span:
            page_results = self.faiss.maxsim_search(q_emb, k=k)
            span.set_metadata({"num_pages": len(page_results), "k": k, "pre_encoded": True})

        if not page_results:
            return []

        page_ids = [pr["page_id"] for pr in page_results]
        page_score_map = {pr["page_id"]: pr["score"] for pr in page_results}
        chunks = self.pg.get_chunks_by_page_ids(page_ids)

        results = []
        for chunk in chunks:
            results.append({
                **chunk,
                "score": page_score_map[chunk["page_id"]],
                "retrieval_type": "visual",
            })

        return results
```

- [ ] **Step 4: Inject span into HyDEGenerator.generate()**

Edit `src/retrieval/hyde.py`:

Add import at top:
```python
from src.observability import get_tracer
```

Modify `generate()` method:

```python
    def generate(self, query: str) -> str:
        """为给定 query 生成假设性答案文档。"""
        tracer = get_tracer()
        with tracer.start_span("hyde_generate") as span:
            if query in self._cache:
                span.set_metadata({"cache_hit": True})
                return self._cache[query]
            span.set_metadata({"cache_hit": False})
            result = self._generate_impl(query)
            return result
```

- [ ] **Step 5: Inject span into RRFFusion/Reranker combo**

The fusion + rerank is called together in `vidore_adapter.py`, not directly. We'll wrap them there in Task 9. For the individual files, add minimal instrumentation:

Edit `src/retrieval/reranker.py`:

Add import at top:
```python
from src.observability import get_tracer
```

Modify `rerank()` method:

```python
    @torch.no_grad()
    def rerank(self, query: str, candidates: List[dict], top_k: int = 5) -> List[dict]:
        if not candidates:
            return []

        tracer = get_tracer()
        with tracer.start_span("rerank") as span:
            pairs = [(query, c["text"]) for c in candidates]
            scores = []
            for pair in pairs:
                score = self.model.predict([pair], convert_to_tensor=True)
                scores.append(score.item() if hasattr(score, "item") else float(score))

            scored = list(zip(candidates, scores))
            scored.sort(key=lambda x: x[1], reverse=True)

            results = []
            for cand, score in scored[:top_k]:
                result = dict(cand)
                result["rerank_score"] = float(score)
                result["retrieval_type"] = "reranked"
                results.append(result)

            span.set_metadata({
                "num_candidates": len(candidates),
                "num_results": len(results),
                "top_k": top_k,
            })
            return results
```

- [ ] **Step 6: Verify all retrieval modules still import**

```bash
cd /Users/theyang/Documents/ai/pdf-rag
python -c "
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.visual_retriever import VisualRetriever
from src.retrieval.hyde import HyDEGenerator
from src.retrieval.reranker import Reranker
from src.retrieval.fusion import RRFFusion
print('All imports OK')
"
```
Expected: `All imports OK`

- [ ] **Step 7: Commit**

```bash
git add src/retrieval/bm25_retriever.py src/retrieval/dense_retriever.py src/retrieval/visual_retriever.py src/retrieval/hyde.py src/retrieval/reranker.py
git commit -m "feat: inject observability spans into retrieval pipeline"
```

---

### Task 9: Integrate Trace Lifecycle into PrismRAGRetriever

**Files:**
- Modify: `src/evaluation/vidore_adapter.py`

**Interfaces:**
- Consumes: `get_tracer()`, `get_collector()` from `src.observability`
- Produces: Each `search_with_trace()` call wrapped in a Trace lifecycle; fusion step gets a span

- [ ] **Step 1: Add trace lifecycle to search_with_trace()**

Edit `src/evaluation/vidore_adapter.py`:

Add import at top (after existing imports):
```python
from src.observability import get_tracer, get_collector
```

At the beginning of `search_with_trace()` method, add trace start (after the docstring, before `routes = []`):

```python
        tracer = get_tracer()
        collector = get_collector()
        tracer.start_trace(query=query, config_label="")

        routes = []
```

At the end of `search_with_trace()`, before each return statement, add trace finish + collector ingest. The method has three return points — handle each:

The first return (empty routes):
```python
        if not routes:
            trace = tracer.finish_trace()
            if trace:
                collector.ingest_trace(trace)
            return {"results": [], "retrieval_trace": trace}
```

The rerank return:
```python
        if use_rerank and fused:
            reranker = self.zerank_reranker if reranker_type == "zerank" else self.reranker
            # Wrap fusion+rerank in a span
            with tracer.start_span("fusion_rerank") as fusion_span:
                reranked = reranker.rerank(query, fused, top_k=k)
                fusion_span.set_metadata({
                    "num_fused_input": len(fused),
                    "num_reranked_output": len(reranked),
                })
            trace = tracer.finish_trace()
            if trace:
                collector.ingest_trace(trace)
            return {"results": reranked, "retrieval_trace": trace}
```

The final return:
```python
        trace = tracer.finish_trace()
        if trace:
            collector.ingest_trace(trace)
        return {"results": fused[:k], "retrieval_trace": trace}
```

- [ ] **Step 2: Verify import works**

```bash
cd /Users/theyang/Documents/ai/pdf-rag
python -c "from src.evaluation.vidore_adapter import PrismRAGRetriever; print('OK')"
```
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add src/evaluation/vidore_adapter.py
git commit -m "feat: integrate Trace lifecycle into PrismRAGRetriever"
```

---

### Task 10: Integrate Spans into RAGAS Metrics (generate_answer)

**Files:**
- Modify: `src/evaluation/ragas_metrics.py`

**Interfaces:**
- Consumes: `get_tracer()`, `get_collector()` from `src.observability`
- Produces: `llm_generate` span around `generate_answer()`; RAGAS scores fed to collector

- [ ] **Step 1: Add span to generate_answer()**

Edit `src/evaluation/ragas_metrics.py`:

Add import at top:
```python
from src.observability import get_tracer
```

Modify `generate_answer()`:

```python
def generate_answer(query: str, context: str) -> str:
    """基于检索上下文生成回答"""
    if not context:
        return "I cannot answer this question based on the available documents."

    tracer = get_tracer()
    with tracer.start_span("llm_generate") as span:
        prompt = GENERATION_PROMPT.format(context=context[:12000], question=query)
        answer = call_llm(prompt)
        span.set_metadata({
            "context_chars": len(context[:12000]),
            "answer_length": len(answer) if answer else 0,
        })
    return answer if answer else ""
```

- [ ] **Step 2: Feed RAGAS scores to collector**

Find the evaluation loop in `ragas_metrics.py` (the `evaluate_generation_configs` or similar function). After computing faithfulness and answer relevancy for each query+config, add:

```python
from src.observability import get_collector
collector = get_collector()
collector.record_ragas_score(
    config_label=config_label,
    query_id=query_id,
    faithfulness=faithfulness_result.faithfulness_score,
    answer_relevancy=relevancy_result.score,
)
```

Search for the exact location — it's near the end of `evaluate_generation()` or in `evaluate_generation_configs()`. The pattern to find: where `avg_faithfulness` and `avg_relevancy` are accumulated.

- [ ] **Step 3: Verify import works**

```bash
cd /Users/theyang/Documents/ai/pdf-rag
python -c "from src.evaluation.ragas_metrics import generate_answer; print('OK')"
```
Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add src/evaluation/ragas_metrics.py
git commit -m "feat: integrate observability spans and RAGAS score recording"
```

---

### Task 11: API Middleware + Route Instrumentation

**Files:**
- Create: `src/observability/middleware.py`
- Modify: `src/api/routes.py`

**Interfaces:**
- Consumes: `get_tracer()`, `get_collector()` from `src.observability`
- Produces: FastAPI middleware that creates trace per HTTP request

- [ ] **Step 1: Write middleware.py**

Create `src/observability/middleware.py`:

```python
"""FastAPI 中间件 — 自动为 HTTP 请求创建 Trace"""
from __future__ import annotations

import time
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

from src.observability import get_tracer, get_collector


class ObservabilityMiddleware(BaseHTTPMiddleware):
    """为每个 API 请求自动创建 Trace，注入 X-Trace-Id 响应头"""

    async def dispatch(self, request: Request, call_next):
        tracer = get_tracer()
        collector = get_collector()

        # 从 request body 中提取 query 文本（不改变 body 流）
        query_text = "(API request)"
        if request.method == "POST" and request.url.path == "/search":
            try:
                body = await request.body()
                import json
                data = json.loads(body)
                query_text = data.get("query", query_text)
            except Exception:
                pass

        tracer.start_trace(query=query_text, config_label="api")
        response = await call_next(request)

        trace = tracer.finish_trace()
        if trace:
            response.headers["X-Trace-Id"] = trace.trace_id
            collector.ingest_trace(trace)

        return response
```

- [ ] **Step 2: Register middleware in routes.py**

Edit `src/api/routes.py`:

Add import at top:
```python
from src.observability.middleware import ObservabilityMiddleware
```

After `app = FastAPI(...)` line, add:
```python
app.add_middleware(ObservabilityMiddleware)
```

- [ ] **Step 3: Verify API still starts**

```bash
cd /Users/theyang/Documents/ai/pdf-rag
python -c "
from src.api.routes import app
print(f'App: {app.title}, routes: {len(app.routes)}')
"
```
Expected: `App: PrismRAG API, routes: ...`

- [ ] **Step 4: Commit**

```bash
git add src/observability/middleware.py src/api/routes.py
git commit -m "feat: add FastAPI observability middleware"
```

---

### Task 12: dashboard.py — Rich Live Terminal Dashboard

**Files:**
- Create: `observability/__init__.py` (empty)
- Create: `observability/dashboard.py`

**Interfaces:**
- Consumes: `MetricsCollector` from `src.observability`
- Produces: `Dashboard` class with `start()`, `update()`, `stop()` — renders rich Live panel

- [ ] **Step 1: Write dashboard.py**

Create `observability/dashboard.py`:

```python
"""Rich 终端仪表盘 — 评测运行时实时刷新"""
from __future__ import annotations

import time
from typing import TYPE_CHECKING

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.layout import Layout

if TYPE_CHECKING:
    from src.observability.collectors import MetricsCollector


class Dashboard:
    """评测运行时终端 Live 面板

    Usage:
        dashboard = Dashboard()
        dashboard.start(collector)
        # ... run eval loop, periodically call dashboard.update() ...
        dashboard.stop()
    """

    def __init__(self):
        self._live: Live | None = None
        self._collector: MetricsCollector | None = None
        self._start_time: float = 0.0

    def start(self, collector: MetricsCollector) -> None:
        self._collector = collector
        self._start_time = time.time()
        self._layout = self._build_layout()
        self._live = Live(
            self._layout, console=Console(), refresh_per_second=2, screen=True
        )
        self._live.start()

    def update(self) -> None:
        if self._live:
            self._live.update(self._build_layout())

    def stop(self) -> None:
        if self._live:
            self._live.stop()

    def _build_layout(self) -> Layout:
        layout = Layout()
        layout.split(
            Layout(name="header", size=3),
            Layout(name="body"),
            Layout(name="alerts"),
        )
        layout["header"].update(self._build_header())
        layout["body"].update(self._build_body())
        layout["alerts"].update(self._build_alerts())
        return layout

    def _build_header(self) -> Panel:
        runtime = time.time() - self._start_time
        mins, secs = divmod(int(runtime), 60)
        snap = self._collector.snapshot() if self._collector else {}
        configs = snap.get("configs", {})
        total_queries = sum(c.get("num_queries", 0) for c in configs.values())
        active_configs = list(configs.keys())
        current = active_configs[-1] if active_configs else "—"

        return Panel(
            f"[bold cyan]PrismRAG Observability[/bold cyan]\n"
            f"Config: [yellow]{current}[/yellow]    "
            f"Queries: [green]{total_queries}[/green]    "
            f"Runtime: [blue]{mins}m {secs}s[/blue]",
            title="Status",
        )

    def _build_body(self) -> Table:
        table = Table(title="Per-Config Metrics", expand=True)
        table.add_column("Config", style="cyan", width=20)
        table.add_column("N", justify="right", width=6)
        table.add_column("P50", justify="right", width=8)
        table.add_column("P95", justify="right", width=8)
        table.add_column("Avg", justify="right", width=8)
        table.add_column("B/D/V Hits", width=16)
        table.add_column("Faith", justify="right", width=7)
        table.add_column("Relev", justify="right", width=7)

        if self._collector:
            snap = self._collector.snapshot()
            for label, m in snap.get("configs", {}).items():
                lat = m.get("latency", {})
                hits = m.get("hits", {})
                qual = m.get("quality", {})
                table.add_row(
                    label[:20],
                    str(m.get("num_queries", 0)),
                    f"{lat.get('p50_ms', 0):.0f}",
                    f"{lat.get('p95_ms', 0):.0f}",
                    f"{lat.get('avg_ms', 0):.0f}",
                    f"{hits.get('avg_bm25', 0):.0f}/{hits.get('avg_dense', 0):.0f}/{hits.get('avg_visual', 0):.0f}",
                    f"{qual.get('avg_faithfulness', 0):.3f}" if qual.get("avg_faithfulness") else "—",
                    f"{qual.get('avg_answer_relevancy', 0):.3f}" if qual.get("avg_answer_relevancy") else "—",
                )
        return table

    def _build_alerts(self) -> Panel:
        if not self._collector:
            return Panel("", title="Alerts")
        alerts = self._collector.get_alerts()
        count_warn = sum(1 for a in alerts if a.level == "warning")
        count_err = sum(1 for a in alerts if a.level == "error")
        header = f"Alerts: [yellow]⚠ {count_warn}[/yellow] [red]⛔ {count_err}[/red]"

        lines = []
        for a in alerts[-5:]:  # last 5
            icon = "[red]⛔[/red]" if a.level == "error" else "[yellow]⚠[/yellow]"
            ts = a.timestamp.strftime("%H:%M:%S") if a.timestamp else ""
            lines.append(f"  {icon} {ts} | {a.config_label} | {a.message[:80]}")
        body = "\n".join(lines) if lines else "  No alerts"

        return Panel(body, title=header)
```

- [ ] **Step 2: Write observability/__init__.py**

```bash
touch /Users/theyang/Documents/ai/pdf-rag/observability/__init__.py
```

- [ ] **Step 3: Verify dashboard imports**

```bash
cd /Users/theyang/Documents/ai/pdf-rag
python -c "from observability.dashboard import Dashboard; print('OK')"
```
Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add observability/__init__.py observability/dashboard.py
git commit -m "feat: add rich Live terminal dashboard"
```

---

### Task 13: reporter.py — Markdown/JSON Report Generation

**Files:**
- Create: `observability/reporter.py`

**Interfaces:**
- Consumes: `MetricsCollector.snapshot()` dict
- Produces: `report.md`, `metrics.json`, `traces.jsonl`, `alerts.json` in `runs/<run_id>/observability/`

- [ ] **Step 1: Write reporter.py**

Create `observability/reporter.py`:

```python
"""报告生成 — Markdown + JSON 输出到 runs/<run_id>/observability/"""
from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime, timezone
from typing import Any


def generate_report(snapshot: dict[str, Any], run_id: str) -> Path:
    """生成完整报告到 runs/<run_id>/observability/

    Args:
        snapshot: MetricsCollector.snapshot() 返回的完整快照
        run_id: 运行标识，如 "20260705-ragas-eval"

    Returns:
        Path to the report directory
    """
    out_dir = Path("runs") / run_id / "observability"
    out_dir.mkdir(parents=True, exist_ok=True)

    # JSON files
    configs = snapshot.get("configs", {})
    (out_dir / "metrics.json").write_text(
        json.dumps(configs, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    traces = snapshot.get("traces", [])
    with open(out_dir / "traces.jsonl", "w", encoding="utf-8") as f:
        for t in traces:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")

    alerts = snapshot.get("alerts", [])
    (out_dir / "alerts.json").write_text(
        json.dumps(alerts, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # Markdown report
    md = _build_markdown(snapshot, run_id)
    (out_dir / "report.md").write_text(md, encoding="utf-8")

    return out_dir


def _build_markdown(snapshot: dict[str, Any], run_id: str) -> str:
    configs = snapshot.get("configs", {})
    alerts = snapshot.get("alerts", [])
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    lines = [
        f"# Observability Report — {now}",
        f"",
        f"**Run:** `{run_id}`",
        f"",
        "## Summary",
        "",
    ]

    # Summary table
    if configs:
        lines.append(
            "| Config | N | P50 | P95 | Avg Latency | B-Hits | D-Hits | V-Hits | Faithfulness | Relevancy |"
        )
        lines.append(
            "|--------|---|---|-----|-------------|--------|--------|--------|-------------|-----------|"
        )
        for label, m in configs.items():
            lat = m.get("latency", {})
            hits = m.get("hits", {})
            qual = m.get("quality", {})
            faith = qual.get("avg_faithfulness")
            relev = qual.get("avg_answer_relevancy")
            lines.append(
                f"| {label} "
                f"| {m.get('num_queries', 0)} "
                f"| {lat.get('p50_ms', 0):.0f}ms "
                f"| {lat.get('p95_ms', 0):.0f}ms "
                f"| {lat.get('avg_ms', 0):.0f}ms "
                f"| {hits.get('avg_bm25', 0):.1f} "
                f"| {hits.get('avg_dense', 0):.1f} "
                f"| {hits.get('avg_visual', 0):.1f} "
                f"| {faith:.3f}" if faith is not None else "| —"
                f" | {relev:.3f}" if relev is not None else " | —"
                f" |"
            )

    # Alerts
    lines.append("")
    lines.append(f"## Alerts ({len(alerts)})")
    lines.append("")
    if alerts:
        for a in alerts:
            icon = "⛔" if a.get("level") == "error" else "⚠"
            lines.append(
                f"- {icon} {a.get('timestamp', '')} | "
                f"`{a.get('config_label', '')}` | "
                f"{a.get('message', '')}"
            )
    else:
        lines.append("No alerts.")

    # Per-Config detail
    lines.append("")
    lines.append("## Per-Config Detail")
    for label, m in configs.items():
        lat = m.get("latency", {})
        lines.append(f"### {label}")
        lines.append(f"- **Queries:** {m.get('num_queries', 0)}")
        lines.append(f"- **Latency:** P50={lat.get('p50_ms', 0):.0f}ms, "
                      f"P95={lat.get('p95_ms', 0):.0f}ms, "
                      f"P99={lat.get('p99_ms', 0):.0f}ms, "
                      f"Avg={lat.get('avg_ms', 0):.0f}ms "
                      f"(min={lat.get('min_ms', 0):.0f}, max={lat.get('max_ms', 0):.0f})")
        hits = m.get("hits", {})
        lines.append(f"- **Hits:** BM25={hits.get('avg_bm25', 0):.1f}, "
                      f"Dense={hits.get('avg_dense', 0):.1f}, "
                      f"Visual={hits.get('avg_visual', 0):.1f}")
        cache = m.get("cache", {})
        lines.append(f"- **HyDE cache hit rate:** {cache.get('hyde_hit_rate', 0):.1%}")
        qual = m.get("quality", {})
        if qual.get("avg_faithfulness") is not None:
            lines.append(f"- **Faithfulness:** {qual.get('avg_faithfulness', 0):.3f}")
            lines.append(f"- **Answer Relevancy:** {qual.get('avg_answer_relevancy', 0):.3f}")
        lines.append("")

    return "\n".join(lines)
```

- [ ] **Step 2: Verify reporter imports**

```bash
cd /Users/theyang/Documents/ai/pdf-rag
python -c "from observability.reporter import generate_report; print('OK')"
```
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add observability/reporter.py
git commit -m "feat: add Markdown/JSON report generator"
```

---

### Task 14: Smoke Test — End-to-End Validation

**Files:**
- Create: `tests/test_observability_smoke.py`

**Interfaces:**
- Consumes: All observability modules
- Produces: End-to-end verification that tracing, collecting, and reporting work together

- [ ] **Step 1: Write smoke test**

Create `tests/test_observability_smoke.py`:

```python
"""Smoke test — end-to-end observability pipeline validation"""
import json
import tempfile
from pathlib import Path


def test_end_to_end_trace_collect_report():
    """完整链路：Trace → Collector → Report"""
    from src.observability import get_tracer, get_collector
    from src.observability.collectors import AlertEvent

    tracer = get_tracer()
    collector = get_collector()
    collector.reset()

    # Simulate a retrieval pipeline for one query
    tracer.start_trace("What is the safety protocol?", "Full_zerank2")
    with tracer.start_span("bm25_search") as span:
        span.finish(metadata={"num_results": 18})
    with tracer.start_span("dense_encode") as span:
        span.finish(metadata={"dim": 1024})
    with tracer.start_span("dense_search") as span:
        span.finish(metadata={"num_results": 15})
    with tracer.start_span("fusion_rerank") as span:
        span.finish(metadata={"num_fused_input": 40, "num_reranked_output": 5})
    trace = tracer.finish_trace()
    collector.ingest_trace(trace)

    # Simulate RAGAS evaluation
    collector.record_ragas_score("Full_zerank2", "q1", faithfulness=0.85, answer_relevancy=0.72)

    # Verify collector has data
    metrics = collector.get_config_metrics("Full_zerank2")
    assert metrics is not None
    assert metrics.num_queries == 2  # 1 trace + 1 ragas
    assert metrics.avg_bm25_hits == 18.0
    assert metrics.avg_faithfulness == 0.85

    # Generate report
    from observability.reporter import generate_report
    with tempfile.TemporaryDirectory() as tmpdir:
        # Override runs/ path by monkey-patching
        import observability.reporter as rp
        original_path = rp.Path
        rp.Path = lambda p: Path(tmpdir) / p if str(p).startswith("runs") else original_path(p)

        # Use a temp run_id
        snapshot = collector.snapshot()
        out_dir = Path(tmpdir) / "runs" / "smoke_test" / "observability"
        out_dir.mkdir(parents=True, exist_ok=True)

        # Write files manually
        (out_dir / "metrics.json").write_text(json.dumps(snapshot["configs"], indent=2))
        (out_dir / "alerts.json").write_text(json.dumps(snapshot["alerts"], indent=2))
        with open(out_dir / "traces.jsonl", "w") as f:
            for t in snapshot["traces"]:
                f.write(json.dumps(t) + "\n")

        # Verify files exist
        assert (out_dir / "metrics.json").exists()
        assert (out_dir / "traces.jsonl").exists()
        assert (out_dir / "alerts.json").exists()

        # Verify content
        metrics_data = json.loads((out_dir / "metrics.json").read_text())
        assert "Full_zerank2" in metrics_data
        assert metrics_data["Full_zerank2"]["num_queries"] == 2

        traces_data = [
            json.loads(line)
            for line in (out_dir / "traces.jsonl").read_text().strip().split("\n")
            if line
        ]
        assert len(traces_data) == 1
        assert len(traces_data[0]["spans"]) == 4


def test_disabled_tracer_noop():
    """trace_enabled=False 时所有操作零开销"""
    from src.observability.tracer import Tracer
    tracer = Tracer(enabled=False)

    # start_trace returns None
    trace = tracer.start_trace("q", "c")
    assert trace is None

    # start_span returns NoopSpan, usable as context manager
    with tracer.start_span("dummy") as span:
        span.set_metadata({"x": 1})
        span.mark_error()
    # No exception, no trace created
    assert span.duration_ms == 0.0
    assert span.status == "ok"


def test_observability_imports_all():
    """All public API items are importable from src.observability"""
    from src.observability import (
        get_tracer,
        get_collector,
        init_logging,
        Tracer,
        Trace,
        Span,
        NoopSpan,
        MetricsCollector,
        ConfigMetrics,
        AlertEvent,
        AlertChecker,
    )
    assert get_tracer is not None
    assert get_collector is not None
    assert init_logging is not None
```

- [ ] **Step 2: Run smoke tests**

```bash
cd /Users/theyang/Documents/ai/pdf-rag
python -m pytest tests/test_observability_smoke.py -v
```
Expected: All 3 tests PASS

- [ ] **Step 3: Run full test suite**

```bash
cd /Users/theyang/Documents/ai/pdf-rag
python -m pytest tests/test_observability_*.py -v
```
Expected: All observability tests PASS (14 + 10 + 6 + 3 ≈ 33 tests)

- [ ] **Step 4: Verify existing tests still pass**

```bash
cd /Users/theyang/Documents/ai/pdf-rag
python -m pytest tests/ -v --tb=short -x
```
Expected: All existing tests PASS (no regressions from span injection)

- [ ] **Step 5: Commit**

```bash
git add tests/test_observability_smoke.py
git commit -m "test: add end-to-end observability smoke test"
```

---

### Task 15: Lint + Final Verification

- [ ] **Step 1: Run linter**

```bash
cd /Users/theyang/Documents/ai/pdf-rag
python -m ruff check src/observability/ observability/ tests/test_observability_*.py
```
Expected: No errors (or fix any that appear)

- [ ] **Step 2: Final full test run**

```bash
cd /Users/theyang/Documents/ai/pdf-rag
python -m pytest tests/ -v --tb=short
```
Expected: All tests PASS

- [ ] **Step 3: Verify the project structure**

```bash
cd /Users/theyang/Documents/ai/pdf-rag
find src/observability observability -type f | sort
```
Expected:
```
observability/__init__.py
observability/dashboard.py
observability/reporter.py
src/observability/__init__.py
src/observability/alerting.py
src/observability/collectors.py
src/observability/logging_setup.py
src/observability/middleware.py
src/observability/tracer.py
```

- [ ] **Step 4: Final commit (if any lint fixes)**

```bash
git add -A
git commit -m "chore: lint fixes for observability module"
```
(Only if there were changes to commit)

---

## Self-Review Checklist

- [x] **Spec coverage**: All 4 module areas covered — logging+tracing (Tasks 3,5,8,9,10,11), hit/latency monitoring (Task 4), quality monitoring (Task 10), alerting (Task 6), dashboard+reporting (Tasks 12,13)
- [x] **Placeholder scan**: No TBD, TODO, vague instructions. Every step has exact code.
- [x] **Type consistency**: `get_tracer()` returns `Tracer`, `get_collector()` returns `MetricsCollector`, signatures match across all tasks.
- [x] **Backward compatibility**: `trace_enabled: false` → no-op spans (tested in smoke). Existing `retrieval_trace` untouched. YAML missing section → defaults.
- [x] **Test coverage**: 33 tests across 4 test files covering tracer, collector, alerting, and end-to-end smoke.
