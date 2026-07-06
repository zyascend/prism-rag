"""指标收集器 — 按 config 聚合延迟、命中、质量指标"""
from __future__ import annotations

import statistics
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from src.observability.tracer import Trace


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
    avg_context_relevancy: float = 0.0

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
                "avg_context_relevancy": round(self.avg_context_relevancy, 4),
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
        self._lock = threading.RLock()
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

    def ingest_trace(self, trace: Trace) -> None:
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
                self._hit_data[label] = {v: [] for v in self._span_hit_names.values()}
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
        context_relevancy: float = 0.0,
    ) -> None:
        with self._lock:
            if config_label not in self._ragas_scores:
                self._ragas_scores[config_label] = []
            self._ragas_scores[config_label].append({
                "query_id": query_id,
                "faithfulness": faithfulness,
                "answer_relevancy": answer_relevancy,
                "context_relevancy": context_relevancy,
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
                    metrics.num_queries = metrics.num_queries + ragas_n
                    metrics.avg_faithfulness = statistics.mean(
                        [s["faithfulness"] for s in scores]
                    )
                    metrics.avg_answer_relevancy = statistics.mean(
                        [s["answer_relevancy"] for s in scores]
                    )
                    # Context Relevancy (may be missing from older records)
                    cr_scores = [s.get("context_relevancy", 0.0) for s in scores]
                    if any(cr_scores):
                        metrics.avg_context_relevancy = statistics.mean(cr_scores)

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