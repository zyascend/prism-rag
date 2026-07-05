"""Trace/Span 模型 + 上下文管理器

提供请求级 Trace 和步骤级 Span 的创建、计时、元数据收集。
支持 trace_enabled=False 时的零开销 no-op 模式。
"""
from __future__ import annotations

import contextvars
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