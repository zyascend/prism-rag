"""Agent 路径配置（默认关闭，与 CRAG/Gate2 同纪律）。"""
from __future__ import annotations

from typing import Any, Callable, Dict, Optional


def agent_config(get_cfg: Optional[Callable] = None) -> Dict[str, Any]:
    if get_cfg is None:
        from src.config import cfg
        get_cfg = cfg.get
    raw = get_cfg("agent", {}) or {}
    if not isinstance(raw, dict):
        raw = {}
    grade = raw.get("grade") if isinstance(raw.get("grade"), dict) else {}
    checkpoint = raw.get("checkpoint") if isinstance(raw.get("checkpoint"), dict) else {}
    hitl = raw.get("hitl") if isinstance(raw.get("hitl"), dict) else {}
    react = raw.get("react_demo") if isinstance(raw.get("react_demo"), dict) else {}
    decompose = raw.get("decompose") if isinstance(raw.get("decompose"), dict) else {}
    synthesize = raw.get("synthesize") if isinstance(raw.get("synthesize"), dict) else {}
    return {
        "enabled": bool(raw.get("enabled", False)),
        "max_subqueries": int(raw.get("max_subqueries", 3)),
        "max_search_per_subquery": int(raw.get("max_search_per_subquery", 1)),
        "max_total_searches": int(raw.get("max_total_searches", 3)),
        "max_llm_calls": int(raw.get("max_llm_calls", 6)),
        "max_grade_cycles": int(raw.get("max_grade_cycles", 1)),
        "timeout_ms": int(raw.get("timeout_ms", 30000)),
        "on_error": str(raw.get("on_error") or "degrade_pipeline"),
        "return_trajectory": bool(raw.get("return_trajectory", True)),
        "grade": {"enabled": bool(grade.get("enabled", True))},
        "checkpoint": {"enabled": bool(checkpoint.get("enabled", True))},
        "hitl": {"review_subqueries": bool(hitl.get("review_subqueries", False))},
        "react_demo": {"enabled": bool(react.get("enabled", False))},
        "decompose": {"prompt_id": str(decompose.get("prompt_id") or "agent_decompose")},
        "synthesize": {"prompt_id": str(synthesize.get("prompt_id") or "agent_synthesize")},
    }


def agent_cache_salt(get_cfg: Optional[Callable] = None) -> str:
    """L4 Answer 缓存盐：agent 开/关与护栏参数变化不得串答案。"""
    c = agent_config(get_cfg)
    if not c["enabled"]:
        return "ag=off"
    return (
        f"ag=on"
        f"|msq={c['max_subqueries']}"
        f"|mts={c['max_total_searches']}"
        f"|mlc={c['max_llm_calls']}"
        f"|mgc={c['max_grade_cycles']}"
        f"|gr={int(c['grade']['enabled'])}"
        f"|hitl={int(c['hitl']['review_subqueries'])}"
    )
