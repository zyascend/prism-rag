"""Agent runner — opt-in entry for LangGraph agent path.

Callers inject search/complete/generate; ``agent.enabled`` stays false in yaml.
Errors honor ``on_error``: degrade_pipeline | abstain | re-raise.

P1c: ``stream_agent``, ``resume_agent``, HITL interrupt → status=interrupted.

Isolation: each ``run_agent`` / ``stream_agent`` uses a **unique** ``thread_id``
unless the caller passes ``trace_id`` (HITL resume must reuse the same id).
Reusing a fixed id with MemorySaver merges reducer channels (evidence,
trajectory) across questions — see Phase2 NO_GO ``evidence_n`` 20→611.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, List, Optional

__all__ = [
    "AgentResult",
    "run_agent",
    "resume_agent",
    "stream_agent",
    "merge_agent_cfg",
    "new_thread_id",
]


def new_thread_id(prefix: str = "agent") -> str:
    """Fresh thread id for one graph invocation (checkpoint isolation)."""
    return f"{prefix}-{uuid.uuid4().hex}"


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


def _extract_interrupt_payload(out: Dict[str, Any]) -> Optional[Any]:
    """Return first interrupt value if graph paused via ``interrupt()``."""
    raw = out.get("__interrupt__")
    if not raw:
        return None
    # list[Interrupt] or tuple
    items = list(raw) if isinstance(raw, (list, tuple)) else [raw]
    if not items:
        return None
    first = items[0]
    if hasattr(first, "value"):
        return first.value
    return first


def _subqueries_from_interrupt(
    payload: Any, state: Dict[str, Any]
) -> List[str]:
    if isinstance(payload, dict):
        subs = payload.get("subqueries")
        if isinstance(subs, list):
            return [str(s) for s in subs]
    if isinstance(payload, list):
        return [str(s) for s in payload]
    return list(state.get("subqueries") or [])


def _state_to_result(
    out: Dict[str, Any],
    *,
    cfg: Dict[str, Any],
    thread_id: Optional[str],
    error: Optional[str] = None,
) -> AgentResult:
    inter = _extract_interrupt_payload(out)
    if inter is not None:
        subs = _subqueries_from_interrupt(inter, out)
        traj = list(out.get("trajectory") or [])
        if not cfg.get("return_trajectory", True):
            traj = []
        return AgentResult(
            answer="",
            citations=[],
            status="interrupted",
            subqueries=subs,
            trajectory=traj,
            counts=_build_counts(out),
            thread_id=thread_id,
            error=None,
            context="",
            evidence=list(out.get("evidence") or []),
        )

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


def _handle_error(
    e: Exception,
    *,
    c: Dict[str, Any],
    thread_id: Optional[str],
    pipeline_fallback_fn: Optional[Callable[[], Dict[str, Any]]],
) -> AgentResult:
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
            trajectory=[]
            if not c.get("return_trajectory", True)
            else list(fb.get("trajectory") or []),
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

    When HITL interrupts, returns ``status="interrupted"`` with pending
    ``subqueries`` and ``thread_id`` (not an error).

    Pass ``trace_id`` only when you need a stable id (HITL pause/resume).
    Batch eval must **not** share one id across questions.
    """
    from src.agent.config import agent_config
    from src.agent.graph import build_agent_graph
    from src.agent.state import empty_agent_state

    c = merge_agent_cfg(agent_config(), cfg)
    # Never default to a shared "local" id under MemorySaver — reducer state leaks.
    thread_id = (str(trace_id).strip() if trace_id else "") or new_thread_id()

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
            init["meta"]["trace_id"] = thread_id
            init["meta"]["thread_id"] = thread_id
        out = graph.invoke(
            init,
            config={"configurable": {"thread_id": thread_id}},
        )
        if not isinstance(out, dict):
            out = {}
        return _state_to_result(out, cfg=c, thread_id=thread_id)
    except Exception as e:
        return _handle_error(
            e, c=c, thread_id=thread_id, pipeline_fallback_fn=pipeline_fallback_fn
        )


def resume_agent(
    thread_id: str,
    *,
    approved_subqueries: List[str],
    search_fn: Callable,
    complete_fn: Callable,
    generate_fn: Callable,
    cfg: Optional[Dict] = None,
    pipeline_fallback_fn: Optional[Callable[[], Dict[str, Any]]] = None,
) -> AgentResult:
    """Resume a HITL-paused graph with approved (or revised) subqueries.

    Requires the same process-wide MemorySaver used at interrupt time and the
    same ``thread_id``.
    """
    from langgraph.types import Command

    from src.agent.config import agent_config
    from src.agent.graph import build_agent_graph

    c = merge_agent_cfg(agent_config(), cfg)
    # Resume always needs checkpointer
    if not isinstance(c.get("checkpoint"), dict):
        c["checkpoint"] = {"enabled": True}
    else:
        c["checkpoint"] = {**c["checkpoint"], "enabled": True}
    # Keep HITL on so graph topology matches the interrupted graph
    if not isinstance(c.get("hitl"), dict):
        c["hitl"] = {"review_subqueries": True}
    else:
        c["hitl"] = {**c["hitl"], "review_subqueries": True}

    try:
        graph = build_agent_graph(
            search_fn=search_fn,
            complete_fn=complete_fn,
            generate_fn=generate_fn,
            cfg=c,
        )
        out = graph.invoke(
            Command(resume=list(approved_subqueries)),
            config={"configurable": {"thread_id": thread_id}},
        )
        if not isinstance(out, dict):
            out = {}
        return _state_to_result(out, cfg=c, thread_id=thread_id)
    except Exception as e:
        return _handle_error(
            e, c=c, thread_id=thread_id, pipeline_fallback_fn=pipeline_fallback_fn
        )


def stream_agent(
    query: str,
    *,
    search_fn: Callable,
    complete_fn: Callable,
    generate_fn: Callable,
    cfg: Optional[Dict] = None,
    trace_id: Optional[str] = None,
    stream_mode: Optional[List[str]] = None,
) -> Iterator[Any]:
    """Yield LangGraph stream events for the agent run.

    Default ``stream_mode`` is ``["updates", "values"]``. Does not catch
    application errors — callers handle them. HITL may yield an interrupt
    update and stop.
    """
    from src.agent.config import agent_config
    from src.agent.graph import build_agent_graph
    from src.agent.state import empty_agent_state

    c = merge_agent_cfg(agent_config(), cfg)
    thread_id = (str(trace_id).strip() if trace_id else "") or new_thread_id("stream")
    modes = stream_mode if stream_mode is not None else ["updates", "values"]

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
        init["meta"]["trace_id"] = thread_id
        init["meta"]["thread_id"] = thread_id

    yield from graph.stream(
        init,
        config={"configurable": {"thread_id": thread_id}},
        stream_mode=modes,
    )
