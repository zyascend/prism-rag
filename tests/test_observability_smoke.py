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
        span.finish(metadata={"num_results": 18, "k": 20})
    with tracer.start_span("dense_encode") as span:
        span.finish(metadata={"dim": 1024})
    with tracer.start_span("dense_search") as span:
        span.finish(metadata={"num_results": 15, "k": 20})
    with tracer.start_span("fusion_rerank") as span:
        span.finish(metadata={
            "num_fused_input": 40,
            "num_reranked_output": 5,
            "num_results": 5,
            "max_rerank_score": 0.95,
            "min_rerank_score": 0.12,
            "mean_rerank_score": 0.53,
        })
    trace = tracer.finish_trace()
    collector.ingest_trace(trace)

    # Simulate RAGAS evaluation with details
    collector.record_ragas_score(
        "Full_zerank2", "q1", faithfulness=0.85, answer_relevancy=0.72,
        context_relevancy=0.08,
        context_relevancy_details={
            "num_sentences": 25,
            "num_relevant": 2,
            "per_sentence": [{"id": 0, "text": "s1", "relevant": False}],
        },
        faithfulness_details={
            "num_claims": 5,
            "num_supported": 4,
            "claims": ["c1", "c2"],
            "supported": [True, True, True, True, False],
        },
    )

    # Verify collector has data
    metrics = collector.get_config_metrics("Full_zerank2")
    assert metrics is not None
    assert metrics.num_queries == 2  # 1 trace + 1 ragas
    assert metrics.avg_bm25_hits == 18.0
    assert metrics.avg_faithfulness == 0.85

    # Verify snapshot includes ragas_details
    snap = collector.snapshot()
    assert "ragas_details" in snap
    assert "Full_zerank2" in snap["ragas_details"]
    details = snap["ragas_details"]["Full_zerank2"]
    assert len(details) == 1
    assert details[0]["faithfulness"] == 0.85
    assert "context_relevancy_per_sentence" in details[0]
    assert "faithfulness_details" in details[0]

    # Generate report files (manual write, verifying new fields)
    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir) / "runs" / "smoke_test" / "observability"
        out_dir.mkdir(parents=True, exist_ok=True)

        (out_dir / "metrics.json").write_text(json.dumps(snap["configs"], indent=2))
        (out_dir / "alerts.json").write_text(json.dumps(snap["alerts"], indent=2))
        (out_dir / "ragas_details.json").write_text(
            json.dumps(snap["ragas_details"], indent=2)
        )
        with open(out_dir / "traces.jsonl", "w") as f:
            for t in snap["traces"]:
                f.write(json.dumps(t) + "\n")

        # Verify all files exist
        assert (out_dir / "metrics.json").exists()
        assert (out_dir / "traces.jsonl").exists()
        assert (out_dir / "alerts.json").exists()
        assert (out_dir / "ragas_details.json").exists()

        # Verify ragas_details content
        details_data = json.loads((out_dir / "ragas_details.json").read_text())
        assert "Full_zerank2" in details_data
        assert details_data["Full_zerank2"][0]["context_relevancy"] == 0.08

        # Verify fusion_rerank span has rerank_score metadata
        traces_data = [
            json.loads(line) for line in
            (out_dir / "traces.jsonl").read_text().strip().split("\n") if line
        ]
        rerank_span = [s for s in traces_data[0]["spans"] if s["name"] == "fusion_rerank"][0]
        assert rerank_span["metadata"]["max_rerank_score"] == 0.95
        assert rerank_span["metadata"]["num_results"] == 5


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