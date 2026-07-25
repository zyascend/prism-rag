"""P1-A Search Planning 单测。"""
from __future__ import annotations

from src.retrieval.search_planner import SearchPlan, plan_search


def test_planning_disabled_passthrough():
    plan = plan_search(
        "see figure 3",
        use_bm25=True,
        use_dense=True,
        use_visual=True,
        cfg={"enabled": False},
    )
    assert plan.use_visual is True
    assert plan.skip_retrieval is False
    assert plan.reason == "planning_disabled"


def test_heuristic_table_query_skips_visual():
    plan = plan_search(
        "What is the maximum pressure rating in the specification table?",
        use_visual=True,
        cfg={
            "enabled": True,
            "mode": "heuristic",
            "allow_skip_retrieval": False,
            "table_prefers_text": True,
            "visual": {"on_cues": True, "default_visual": False},
        },
    )
    assert plan.use_bm25 and plan.use_dense
    assert plan.use_visual is False
    assert "table" in plan.intent_label or plan.reason in (
        "table_prefers_text",
        "no_visual_cue",
    )


def test_heuristic_figure_query_enables_visual():
    plan = plan_search(
        "See figure 3 for the wiring diagram layout",
        use_visual=True,
        cfg={
            "enabled": True,
            "mode": "heuristic",
            "allow_skip_retrieval": False,
            "table_prefers_text": True,
            "visual": {"on_cues": True, "default_visual": False},
        },
    )
    assert plan.use_visual is True
    assert plan.reason == "visual_cue"


def test_always_full_respects_request_flags():
    plan = plan_search(
        "hello",
        use_bm25=True,
        use_dense=False,
        use_visual=True,
        cfg={"enabled": True, "mode": "always_full", "visual": {}},
    )
    assert plan.use_dense is False
    assert plan.use_visual is True
    assert plan.reason == "always_full"


def test_chitchat_skip_only_when_allowed():
    cfg_base = {
        "enabled": True,
        "mode": "heuristic",
        "visual": {"on_cues": True, "default_visual": False},
    }
    no_skip = plan_search("hello", cfg={**cfg_base, "allow_skip_retrieval": False})
    assert no_skip.skip_retrieval is False

    skip = plan_search("hello", cfg={**cfg_base, "allow_skip_retrieval": True})
    assert skip.skip_retrieval is True
    assert skip.use_bm25 is False


def test_text_only_mode():
    plan = plan_search(
        "see figure 1",
        use_visual=True,
        cfg={"enabled": True, "mode": "text_only", "visual": {}},
    )
    assert plan.use_visual is False
    assert plan.reason == "text_only_mode"


def test_plan_as_dict():
    p = SearchPlan(True, True, False, False, "none", "x")
    d = p.as_dict()
    assert d["use_visual"] is False
    assert d["reason"] == "x"
