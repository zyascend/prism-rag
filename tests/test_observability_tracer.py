"""Tests for src/observability/tracer.py"""
from src.observability.tracer import Tracer, Trace, Span, NoopSpan


class TestSpan:
    def test_span_lifecycle(self):
        span = Span(name="test_op")
        assert span.name == "test_op"
        assert span.status == "ok"
        assert span.duration_ms == 0.0
        assert span.started_at is not None

    def test_span_finish_sets_duration(self):
        import time

        span = Span(name="test_op")
        time.sleep(0.001)
        span.finish()
        assert span.finished_at is not None
        assert span.duration_ms > 0

    def test_span_finish_metadata(self):
        span = Span(name="test_op")
        span.finish(metadata={"hits": 20})
        assert span.metadata == {"hits": 20}

    def test_span_set_metadata(self):
        span = Span(name="test_op")
        span.set_metadata({"hits": 20})
        span.set_metadata({"latency": 100})
        assert span.metadata == {"hits": 20, "latency": 100}

    def test_span_mark_error(self):
        span = Span(name="test_op")
        span.mark_error()
        assert span.status == "error"

    def test_span_context_manager(self):
        with Span(name="ctx_test") as span:
            pass
        assert span.finished_at is not None
        assert span.duration_ms > 0

    def test_span_parent_child(self):
        parent = Span(name="parent")
        parent.finish()
        child = Span(name="child", parent_span_id=parent.span_id)
        child.finish()
        assert child.parent_span_id == parent.span_id
        assert child.span_id != parent.span_id


class TestTrace:
    def test_trace_lifecycle(self):
        trace = Trace(query="test query", config_label="test_config")
        assert trace.query == "test query"
        assert trace.config_label == "test_config"
        assert len(trace.spans) == 0
        assert trace.finished_at is None

    def test_trace_add_span(self):
        trace = Trace(query="q", config_label="c")
        span = Span(name="op")
        span.finish()
        trace.add_span(span)
        assert len(trace.spans) == 1
        assert trace.spans[0].name == "op"

    def test_trace_finish(self):
        import time

        trace = Trace(query="q", config_label="c")
        time.sleep(0.001)
        trace.finish()
        assert trace.finished_at is not None
        assert trace.duration_ms > 0

    def test_trace_duration_equals_sum_of_spans(self):
        trace = Trace(query="q", config_label="c")
        span1 = Span(name="op1")
        span1.finish()
        trace.add_span(span1)
        span2 = Span(name="op2")
        span2.finish()
        trace.add_span(span2)
        trace.finish()
        assert trace.duration_ms >= span1.duration_ms + span2.duration_ms

    def test_trace_to_dict(self):
        trace = Trace(query="q", config_label="c")
        span = Span(name="op")
        span.finish(metadata={"k": 20})
        trace.add_span(span)
        trace.finish()
        d = trace.to_dict()
        assert d["query"] == "q"
        assert d["config_label"] == "c"
        assert len(d["spans"]) == 1
        assert d["spans"][0]["name"] == "op"
        assert d["spans"][0]["metadata"] == {"k": 20}


class TestTracer:
    def test_start_and_finish_trace(self):
        tracer = Tracer(enabled=True)
        trace = tracer.start_trace("test query", "cfg_a")
        assert tracer.current_trace() is trace
        tracer.finish_trace()
        assert tracer.current_trace() is None
        assert trace.finished_at is not None

    def test_start_span_context_manager(self):
        tracer = Tracer(enabled=True)
        tracer.start_trace("q", "c")
        with tracer.start_span("bm25_search") as span:
            span.set_metadata({"hits": 5})
        trace = tracer.current_trace()
        assert len(trace.spans) == 1
        assert trace.spans[0].name == "bm25_search"
        assert trace.spans[0].metadata == {"hits": 5}
        tracer.finish_trace()

    def test_disabled_tracer_returns_noop(self):
        tracer = Tracer(enabled=False)
        trace = tracer.start_trace("q", "c")
        assert trace is None
        span = tracer.start_span("bm25")
        assert isinstance(span, NoopSpan)
        with span:
            pass  # no-op, should not raise

    def test_disabled_tracer_current_trace_is_none(self):
        tracer = Tracer(enabled=False)
        assert tracer.current_trace() is None

    def test_parent_trace_detection(self):
        """验证 current_trace() 可用于检测父 Trace 是否存在"""
        tracer = Tracer(enabled=True)
        # 初始无活跃 Trace
        assert tracer.current_trace() is None

        # 创建 Trace 后可以检测到
        trace = tracer.start_trace("test query", "test_config")
        assert tracer.current_trace() is not None
        assert tracer.current_trace().trace_id == trace.trace_id

        # 结束 Trace 后恢复为 None
        tracer.finish_trace()
        assert tracer.current_trace() is None

    def test_child_spans_attach_to_parent_trace(self):
        """子 span 自动挂载到当前活跃的父 Trace"""
        tracer = Tracer(enabled=True)
        trace = tracer.start_trace("query", "config")

        # 创建子 span — 应挂载到 trace
        with tracer.start_span("child_span") as span:
            span.set_metadata({"key": "value"})

        assert len(trace.spans) == 1
        assert trace.spans[0].name == "child_span"
        assert trace.spans[0].metadata == {"key": "value"}

        tracer.finish_trace()


class TestNoopSpan:
    def test_noop_span_context_manager_does_nothing(self):
        with NoopSpan("noop") as span:
            span.set_metadata({"x": 1})
            span.mark_error()
        assert span.duration_ms == 0.0
        assert span.status == "ok"