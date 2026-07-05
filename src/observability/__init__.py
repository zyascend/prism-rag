"""PrismRAG Observability — 内建可观测性模块

核心 (src/observability/): tracer, collector, alerting, logging
消费侧 (observability/): dashboard, reporter
"""
import logging
from datetime import datetime, timezone

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

logger = logging.getLogger(__name__)


def dump_collector(run_name: str) -> str | None:
    """将 Collector 中的追踪数据落盘到 runs/<run_name>/observability/

    供 CLI 评测脚本在 main() 末尾调用。无追踪数据时静默跳过。

    Args:
        run_name: 运行标识，如 "e2e_qa_20260705"

    Returns:
        报告目录路径，或 None（无数据/失败）
    """
    collector = get_collector()
    snap = collector.snapshot()
    if not snap.get("traces"):
        return None
    try:
        from observability.reporter import generate_report

        report_dir = generate_report(snap, run_name)
        logger.info("  可观测性报告: %s", report_dir)
        return str(report_dir)
    except Exception as e:
        logger.warning("  可观测性报告生成失败: %s", e)
        return None


__all__ = [
    "dump_collector",
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