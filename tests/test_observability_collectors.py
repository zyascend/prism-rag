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
            trace.finish()
            trace.duration_ms = 50.0 * (i + 1)
            mc.ingest_trace(trace)

        metrics = mc.get_config_metrics("cfg_a")
        assert metrics.num_queries == 10
        assert metrics.latency_min_ms == 50.0
        assert metrics.latency_max_ms == 500.0
        # P50 of [50,100,...,500] = 275 or 300 depending on interpolation
        assert 250 <= metrics.latency_p50_ms <= 300
        # P95 of 10 values: linear interpolation between indices 8 and 9 = 477.5
        assert metrics.latency_p95_ms == pytest.approx(477.5, abs=0.1)

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

    def test_record_ragas_with_details(self):
        """验证 record_ragas_score 存储 per-sentence 和 faithfulness 细节"""
        mc = MetricsCollector()
        mc.record_ragas_score(
            "cfg_a", "q1",
            faithfulness=0.80, answer_relevancy=0.70, context_relevancy=0.10,
            context_relevancy_details={
                "num_sentences": 10,
                "num_relevant": 1,
                "per_sentence": [{"id": 0, "text": "hello world test", "relevant": True}],
            },
            faithfulness_details={
                "num_claims": 4,
                "num_supported": 3,
                "claims": ["c1", "c2", "c3", "c4"],
                "supported": [True, True, True, False],
            },
        )

        snap = mc.snapshot()
        assert "ragas_details" in snap
        assert "cfg_a" in snap["ragas_details"]
        details = snap["ragas_details"]["cfg_a"][0]
        assert details["faithfulness"] == 0.80
        assert details["context_relevancy"] == 0.10
        assert "context_relevancy_per_sentence" in details
        assert details["context_relevancy_per_sentence"]["num_sentences"] == 10
        assert "faithfulness_details" in details
        assert details["faithfulness_details"]["num_claims"] == 4

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