"""FastAPI 中间件 — 自动为 HTTP 请求创建 Trace"""
from __future__ import annotations

import time
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

from src.observability import get_tracer, get_collector


class ObservabilityMiddleware(BaseHTTPMiddleware):
    """为每个 API 请求自动创建 Trace，注入 X-Trace-Id 响应头"""

    async def dispatch(self, request: Request, call_next):
        tracer = get_tracer()
        collector = get_collector()

        # 从 request body 中提取 query 文本（不改变 body 流）
        query_text = "(API request)"
        if request.method == "POST" and request.url.path == "/search":
            try:
                body = await request.body()
                import json
                data = json.loads(body)
                query_text = data.get("query", query_text)
            except Exception:
                pass

        tracer.start_trace(query=query_text, config_label="api")
        response = await call_next(request)

        trace = tracer.finish_trace()
        if trace:
            response.headers["X-Trace-Id"] = trace.trace_id
            collector.ingest_trace(trace)

        return response