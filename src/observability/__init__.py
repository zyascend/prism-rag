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