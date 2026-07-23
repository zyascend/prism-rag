"""Phase A1: 上下文感知 TableSummarizer."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from src.ingestion.table_summarizer import TableSummarizer
from src.prompts import get_active, get_prompt


SAMPLE_TABLE = """| Param | Max |
| --- | --- |
| PSI | 40 |
"""


def test_active_prompt_still_v1_without_context_placeholder():
    """默认 active 仍为 v1，保证迁移单测与现网无上下文路径不变。"""
    active = get_active("table_summary")
    assert active.version == 1
    assert "{table}" in active.template
    assert "{context}" not in active.template


def test_v2_prompt_has_context_slot():
    prompt = get_prompt("table_summary")
    v2 = next(v for v in prompt.versions if v.version == 2)
    assert not v2.active
    assert "{context}" in v2.template
    assert "{table}" in v2.template


def test_summarize_disabled_returns_empty():
    s = TableSummarizer(enabled=False, context_enabled=True)
    assert s.summarize(SAMPLE_TABLE, context="Section 3 Cooling") == ""


def test_context_disabled_ignores_context_uses_v1_prompt():
    captured: list[str] = []

    def fake_llm(prompt: str, max_retries: int = 2) -> str:
        captured.append(prompt)
        return "A pressure table with max PSI 40."

    s = TableSummarizer(enabled=True, context_enabled=False)
    with patch("src.ingestion.table_summarizer.call_llm", side_effect=fake_llm):
        out = s.summarize(SAMPLE_TABLE, context="Section 3 Cooling System")
    assert "PSI" in out or "pressure" in out.lower() or out
    assert len(captured) == 1
    assert "Surrounding context" not in captured[0]
    assert "Section 3" not in captured[0]
    assert SAMPLE_TABLE.strip() in captured[0]


def test_context_enabled_includes_context_in_prompt():
    captured: list[str] = []

    def fake_llm(prompt: str, max_retries: int = 2) -> str:
        captured.append(prompt)
        return "Cooling system pressure limits; Max PSI is 40."

    s = TableSummarizer(enabled=True, context_enabled=True, context_max_chars=1500)
    with patch("src.ingestion.table_summarizer.call_llm", side_effect=fake_llm):
        out = s.summarize(SAMPLE_TABLE, context="## 3.2 Cooling System\nOperating limits.")
    assert out
    assert len(captured) == 1
    assert "Surrounding context" in captured[0]
    assert "3.2 Cooling" in captured[0]
    assert "PSI" in captured[0]


def test_context_truncation():
    captured: list[str] = []

    def fake_llm(prompt: str, max_retries: int = 2) -> str:
        captured.append(prompt)
        return "summary"

    s = TableSummarizer(enabled=True, context_enabled=True, context_max_chars=20)
    long_ctx = "A" * 100
    with patch("src.ingestion.table_summarizer.call_llm", side_effect=fake_llm):
        s.summarize(SAMPLE_TABLE, context=long_ctx)
    # 截断后带 ...
    assert "..." in captured[0]
    assert long_ctx not in captured[0]


def test_build_page_context_skips_tables_and_respects_flag():
    chunks = [
        SimpleNamespace(chunk_id="t1", chunk_type="table", text="| A | B |\n|---|---|\n| 1 | 2 |"),
        SimpleNamespace(chunk_id="x1", chunk_type="text", text="Section: Pressure limits for rinse."),
        SimpleNamespace(chunk_id="x2", chunk_type="text", text="Use clean water only."),
    ]
    off = TableSummarizer(context_enabled=False)
    assert off.build_page_context(chunks) == ""

    on = TableSummarizer(context_enabled=True, context_max_chars=1500)
    ctx = on.build_page_context(chunks)
    assert "Pressure limits" in ctx
    assert "clean water" in ctx
    assert "| A | B |" not in ctx


def test_cache_keys_separate_by_context():
    calls = {"n": 0}

    def fake_llm(prompt: str, max_retries: int = 2) -> str:
        calls["n"] += 1
        return f"summary-{calls['n']}"

    s = TableSummarizer(enabled=True, context_enabled=True)
    with patch("src.ingestion.table_summarizer.call_llm", side_effect=fake_llm):
        a = s.summarize(SAMPLE_TABLE, context="ctx-a")
        b = s.summarize(SAMPLE_TABLE, context="ctx-b")
        a2 = s.summarize(SAMPLE_TABLE, context="ctx-a")
    assert a != b
    assert a == a2
    assert calls["n"] == 2


def test_llm_failure_degrades_to_empty():
    s = TableSummarizer(enabled=True, context_enabled=True)

    def boom(prompt: str, max_retries: int = 2) -> str:
        raise RuntimeError("llm down")

    with patch("src.ingestion.table_summarizer.call_llm", side_effect=boom):
        assert s.summarize(SAMPLE_TABLE, context="x") == ""
