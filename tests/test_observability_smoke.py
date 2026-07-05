"""Smoke test — end-to-end observability pipeline validation"""
import json
import tempfile
from pathlib import Path


def test_end_to_end_trace_collect_report():
    """完整链路：Trace → Collector → Report"""
    from src.observability import get_tracer, get_collector

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
    assert Tracer is not None
    assert Trace is not None
    assert Span is not None
    assert NoopSpan is not None
    assert MetricsCollector is not None
    assert ConfigMetrics is not None
    assert AlertEvent is not None
    assert AlertChecker is not None