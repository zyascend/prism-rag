# tests/test_demo_fixtures.py
"""static/demo JSON 契约：可解析且与 AskResponse 字段对齐。"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "static" / "demo"


def test_fixtures_file_exists_and_parses():
    path = DEMO / "fixtures.json"
    assert path.is_file(), f"missing {path}"
    data = json.loads(path.read_text(encoding="utf-8"))
    assert "presets" in data and "responses" in data
    assert len(data["presets"]) >= 4


def test_each_preset_has_matching_response():
    data = json.loads((DEMO / "fixtures.json").read_text(encoding="utf-8"))
    for p in data["presets"]:
        q = p["query"]
        assert q in data["responses"], f"preset query missing response: {q!r}"
        resp = data["responses"][q]
        assert "answer" in resp
        assert "citations" in resp
        assert "retrieval_trace" in resp
        rt = resp["retrieval_trace"]
        for key in ("bm25_top5", "dense_top5", "visual_top5"):
            assert key in rt
            assert isinstance(rt[key], list)


def test_has_reject_style_fixture():
    data = json.loads((DEMO / "fixtures.json").read_text(encoding="utf-8"))
    answers = [r["answer"].lower() for r in data["responses"].values()]
    assert any(
        "enough information" in a or "cannot answer" in a or "not enough" in a
        for a in answers
    ), "need at least one abstain/reject-style answer for demo storytelling"


def test_metrics_chips():
    path = DEMO / "metrics.json"
    assert path.is_file()
    data = json.loads(path.read_text(encoding="utf-8"))
    assert len(data["chips"]) >= 2
    for c in data["chips"]:
        assert c.get("label") and c.get("value") and c.get("detail")
