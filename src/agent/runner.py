"""Agent runner — opt-in entry for LangGraph agent path.

Callers inject search/complete/generate; ``agent.enabled`` stays false in yaml.
Errors honor ``on_error``: degrade_pipeline | abstain | re-raise.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

__all__ = ["AgentResult", "run_agent", "merge_agent_cfg"]


@dataclass
class AgentResult:
    answer: str
    citations: List[Dict[str, Any]] = field(default_factory=list)
    status: str = "ok"
    subqueries: List[str] = field(default_factory=list)
    trajectory: List[Dict[str, Any]] = field(default_factory=list)
    counts: Dict[str, Any] = field(default_factory=dict)
    thread_id: Optional[str] = None
    error: Optional[str] = None
    context: str = ""
    evidence: List[Dict[str, Any]] = field(default_factory=list)


_NESTED_KEYS = ("grade", "hitl", "checkpoint", "react_demo", "decompose", "synthesize")


def merge_agent_cfg(
    base: Optional[Dict[str, Any]] = None,
    override: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Shallow-merge agent cfg; deep-merge nested grade/hitl/checkpoint/... dicts.

    Ensures build_agent_graph and empty_agent_state see one unified budget/cfg.
    """
    out: Dict[str, Any] = dict(base or {})
    ov = dict(override or {})
    for key, val in ov.items():
        if key in _NESTED_KEYS and isinstance(val, dict):
            prev = out.get(key) if isinstance(out.get(key), dict) else {}
            out[key] = {**prev, **val}
        else:
            out[key] = val
    return out


def _build_counts(out: Dict[str, Any]) -> Dict[str, Any]:
    meta = out.get("meta") or {}
    evidence = out.get("evidence") or []
    subqueries = out.get("subqueries") or []
    return {
        "subqueries": int(meta.get("n_subqueries") or len(subqueries)),
        "searches": int(meta.get("searches") or 0),
        "llm_calls": int(meta.get("llm_calls") or 0),
        "evidence_n": int(meta.get("n_evidence") or len(evidence)),
    }


def _state_to_result(
    out: Dict[str, Any],
    *,
    cfg: Dict[str, Any],
    thread_id: Optional[str],
    error: Optional[str] = None,
) -> AgentResult:
    meta = out.get("meta") or {}
    traj = list(out.get("trajectory") or [])
    if not cfg.get("return_trajectory", True):
        traj = []
    context = ""
    if isinstance(meta.get("context"), str):
        context = meta["context"]
    return AgentResult(
        answer=str(out.get("answer") or ""),
        citations=list(out.get("citations") or []),
        status=str(out.get("status") or "ok"),
        subqueries=list(out.get("subqueries") or []),
        trajectory=traj,
        counts=_build_counts(out),
        thread_id=thread_id,
        error=error,
        context=context,
        evidence=list(out.get("evidence") or []),
    )


def run_agent(
    query: str,
    *,
    search_fn: Callable,
    complete_fn: Callable,
    generate_fn: Callable,
    cfg: Optional[Dict] = None,
    trace_id: Optional[str] = None,
    pipeline_fallback_fn: Optional[Callable[[], Dict[str, Any]]] = None,
) -> AgentResult:
    """Run the agent graph once and map state → AgentResult.

    ``cfg`` overrides ``agent_config()`` (deep-merge nested sections). Same merged
    dict is used for both ``build_agent_graph`` and ``empty_agent_state`` so budgets
    stay consistent.
    """
    from src.agent.config import agent_config
    from src.agent.graph import build_agent_graph
    from src.agent.state import empty_agent_state

    c = merge_agent_cfg(agent_config(), cfg)
    thread_id = trace_id or "local"

    try:
        graph = build_agent_graph(
            search_fn=search_fn,
            complete_fn=complete_fn,
            generate_fn=generate_fn,
            cfg=c,
        )
        init = empty_agent_state(query, cfg=c)
        init.setdefault("meta", {})
        if isinstance(init["meta"], dict):
            init["meta"] = dict(init["meta"])
            init["meta"]["trace_id"] = trace_id
        out = graph.invoke(
            init,
            config={"configurable": {"thread_id": thread_id}},
        )
        if not isinstance(out, dict):
            out = {}
        return _state_to_result(out, cfg=c, thread_id=thread_id)
    except Exception as e:
        err = str(e)[:300]
        on_error = str(c.get("on_error") or "degrade_pipeline")
        if on_error == "degrade_pipeline" and pipeline_fallback_fn is not None:
            fb = pipeline_fallback_fn() or {}
            return AgentResult(
                answer=str(fb.get("answer") or ""),
                citations=list(fb.get("citations") or []),
                status="degraded",
                error=err,
                context=str(fb.get("context") or ""),
                evidence=list(fb.get("evidence") or []),
                thread_id=thread_id,
                trajectory=[] if not c.get("return_trajectory", True) else list(fb.get("trajectory") or []),
                counts=dict(fb.get("counts") or {}),
                subqueries=list(fb.get("subqueries") or []),
            )
        if on_error == "abstain":
            from src.rejection import abstain_message

            return AgentResult(
                answer=abstain_message(),
                status="error",
                error=err,
                thread_id=thread_id,
            )
        raise
