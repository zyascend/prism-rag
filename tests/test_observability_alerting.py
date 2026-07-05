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
            trace.finish()
            trace.duration_ms = 100.0  # well under 5000ms default
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
            trace.finish()
            trace.duration_ms = 6000.0  # exceeds 5000ms
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
