"""告警检测 — 阈值检查 + 管道异常分类"""
from __future__ import annotations

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
        rerank_score_min: float = 0.0,
        context_relevancy_min: float = 0.05,
    ):
        self.latency_p95_threshold_ms = latency_p95_threshold_ms
        self.recall_at_5_min = recall_at_5_min
        self.faithfulness_min = faithfulness_min
        self.rerank_score_min = rerank_score_min
        self.context_relevancy_min = context_relevancy_min

    @classmethod
    def from_config(cls, config: "ObservabilityConfig") -> "AlertChecker":
        return cls(
            latency_p95_threshold_ms=config.latency_p95_threshold_ms,
            recall_at_5_min=config.recall_at_5_min,
            faithfulness_min=config.faithfulness_min,
            rerank_score_min=config.rerank_score_min,
            context_relevancy_min=config.context_relevancy_min,
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

            # Context Relevance 阈值
            ctxrel = quality.get("avg_context_relevancy", 1.0)
            if ctxrel > 0 and ctxrel < self.context_relevancy_min:
                alerts.append(AlertEvent(
                    level="warning",
                    category="threshold",
                    message=(
                        f"Context Relevance {ctxrel:.3f} below minimum "
                        f"{self.context_relevancy_min}"
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
