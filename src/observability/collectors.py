"""指标收集器 — 按 config 聚合延迟、命中、质量指标"""
from __future__ import annotations

import json
import statistics
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.observability.tracer import Trace

# 项目根目录（collectors.py 位于 src/observability/）
_ROOT = Path(__file__).resolve().parent.parent.parent
# 内存中保留的可反查 Trace 上限（FIFO），防止长时间运行撑爆内存；
# 超过后最旧的从内存淘汰，但仍可在磁盘 JSONL 中查到。
_TRACE_MEM_CAP = 2000


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
    retrieval_cache_hit_rate: float = 0.0
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
            "cache": {
                "hyde_hit_rate": round(self.hyde_hit_rate, 2),
                "retrieval_cache_hit_rate": round(self.retrieval_cache_hit_rate, 2),
            },
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
        self._ragas_details: dict[str, list[dict[str, Any]]] = {}  # config_label -> [per-query detail]
        self._alerts: list[AlertEvent] = []
        # 缓存命中统计：config_label -> layer -> {hits, misses}
        self._cache_data: dict[str, dict[str, dict[str, int]]] = {}
        # 单条 Trace 反查索引（trace_id -> trace dict）与磁盘持久化状态
        self._trace_by_id: dict[str, dict[str, Any]] = {}
        self._trace_id_order: list[str] = []  # FIFO 顺序，配合 _TRACE_MEM_CAP 淘汰
        self._trace_log_path: Path | None = None  # 懒加载：首次 ingest 时按 config 解析
        self._span_hit_names = {
            "bm25_search": "bm25",
            "dense_search": "dense",
            "visual_search": "visual",
            "fusion_rerank": "fused",
            "rerank": "reranked",
        }

    def reset(self) -> None:
        with self._lock:
            self._traces.clear()
            self._latencies.clear()
            self._hit_data.clear()
            self._hyde_data.clear()
            self._ragas_scores.clear()
            self._ragas_details.clear()
            self._alerts.clear()
            self._cache_data.clear()
            self._trace_by_id.clear()
            self._trace_id_order.clear()

    def ingest_trace(self, trace: Trace) -> None:
        """接收一个已完成的 Trace，提取指标并聚合"""
        with self._lock:
            label = trace.config_label
            trace_dict = trace.to_dict()
            self._traces.append(trace_dict)

            # ── 单条反查索引（内存）───
            tid = trace_dict["trace_id"]
            if tid in self._trace_by_id:
                self._trace_id_order.remove(tid)
            self._trace_by_id[tid] = trace_dict
            self._trace_id_order.append(tid)
            # FIFO 淘汰最旧的（磁盘仍保留）
            while len(self._trace_id_order) > _TRACE_MEM_CAP:
                old = self._trace_id_order.pop(0)
                self._trace_by_id.pop(old, None)

            # ── 磁盘持久化（进程重启后仍可反查）───
            self._persist_trace(trace_dict)

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

    def record_cache_event(self, layer: str, hit: bool, config_label: str = "api") -> None:
        """记录一次缓存命中/未命中事件，用于聚合各层 cache 命中率。

        Args:
            layer: 缓存层标识，如 "retrieval"（L3 检索结果缓存）、"hyde"（已有）。
            hit: True=命中，False=未命中。
            config_label: 检索配置标签；在线请求默认 "api"，评测时继承 config_label。
        """
        with self._lock:
            if config_label not in self._cache_data:
                self._cache_data[config_label] = {}
            layer_data = self._cache_data[config_label]
            if layer not in layer_data:
                layer_data[layer] = {"hits": 0, "misses": 0}
            if hit:
                layer_data[layer]["hits"] += 1
            else:
                layer_data[layer]["misses"] += 1

    def _resolve_trace_log_path(self) -> Path | None:
        """按 config 解析磁盘持久化路径；空字符串表示关闭持久化。"""
        if self._trace_log_path is not None:
            return self._trace_log_path  # 可能是 None（已关闭）
        from src.config import cfg
        raw = cfg.observability.trace_persist_path
        if not raw:
            self._trace_log_path = None
            return None
        p = Path(raw)
        if not p.is_absolute():
            p = _ROOT / p
        self._trace_log_path = p
        return p

    def _persist_trace(self, trace_dict: dict[str, Any]) -> None:
        """将单条 Trace 追加写入磁盘 JSONL。任何异常都被吞掉，绝不影响主流程。"""
        try:
            path = self._resolve_trace_log_path()
            if path is None:
                return
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(trace_dict, ensure_ascii=False) + "\n")
        except Exception:
            # 持久化是排查辅助能力，失败不影响请求
            pass

    def get_trace(self, trace_id: str) -> dict[str, Any] | None:
        """按 trace_id 反查单条 Trace。

        优先命中内存索引；未命中时回退扫描磁盘 JSONL（覆盖进程重启后的场景）。
        返回完整 trace dict（含各 span 的 metadata，如 generation 的 context/citations）。
        """
        with self._lock:
            hit = self._trace_by_id.get(trace_id)
        if hit is not None:
            return hit
        # 内存未命中 → 扫描磁盘（仅用于重启后或已被 FIFO 淘汰的旧 trace）
        try:
            path = self._resolve_trace_log_path()
            if path is None or not path.exists():
                return None
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if rec.get("trace_id") == trace_id:
                        return rec
        except Exception:
            pass
        return None

    def record_ragas_score(
        self, config_label: str, query_id: str,
        faithfulness: float, answer_relevancy: float,
        context_relevancy: float = 0.0,
        context_relevancy_details: dict[str, Any] | None = None,
        faithfulness_details: dict[str, Any] | None = None,
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
            # Per-query details for snapshot persistence
            if config_label not in self._ragas_details:
                self._ragas_details[config_label] = []
            detail: dict[str, Any] = {
                "query_id": query_id,
                "faithfulness": faithfulness,
                "answer_relevancy": answer_relevancy,
                "context_relevancy": context_relevancy,
            }
            if context_relevancy_details is not None:
                detail["context_relevancy_per_sentence"] = context_relevancy_details
            if faithfulness_details is not None:
                detail["faithfulness_details"] = faithfulness_details
            self._ragas_details[config_label].append(detail)

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

            # Retrieval cache hit rate
            if config_label in self._cache_data:
                cd = self._cache_data[config_label]
                if "retrieval" in cd:
                    total = cd["retrieval"]["hits"] + cd["retrieval"]["misses"]
                    if total > 0:
                        metrics.retrieval_cache_hit_rate = cd["retrieval"]["hits"] / total

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
                "ragas_details": dict(self._ragas_details),
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