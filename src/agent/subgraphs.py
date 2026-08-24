"""Retrieval subgraph for Send map-reduce workers (Task 6 / P1b).

Workers must return **deltas only** for reducer fields (`evidence`, `trajectory`).
Budget accounting is pre-allocated on the parent graph before fan-out.
"""
from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional

from langgraph.graph import END, StateGraph

from src.agent.state import AgentState, next_step, step_record
from src.agent.tools import AgentToolBox

SearchFn = Callable[..., List[dict]]

__all__ = ["build_retrieval_subgraph", "retrieval_worker_delta"]


def build_retrieval_subgraph(
    search_fn: Optional[SearchFn] = None,
    toolbox_cfg: Optional[Dict[str, Any]] = None,
    *,
    box: Optional[AgentToolBox] = None,
    top_k: int = 5,
    max_search_per_subquery: int = 1,
    node_name: str = "retrieval_worker",
) -> Any:
    """Compile a single-path retrieval subgraph.

    Input state (via Send) should include:
      - ``active_subquery`` / ``active_subquery_id``
      - optional ``active_search_allowance`` (pre-allocated search count for this path)
      - empty ``evidence`` / ``trajectory`` so reducer merges only this path's deltas

    Returns only ``evidence`` + ``trajectory`` updates (no budget/meta writes).
    """
    if box is None:
        if search_fn is None:
            raise ValueError("build_retrieval_subgraph requires search_fn or box")
        box = AgentToolBox(
            search_fn=search_fn,
            complete_fn=lambda _p: "{}",
            generate_fn=lambda q, h: {
                "answer": "",
                "citations": [],
                "rejected": True,
            },
            cfg=dict(toolbox_cfg or {}),
        )
    top_k = max(1, int(top_k))
    default_allow = max(1, int(max_search_per_subquery))

    def retrieve(state: Dict[str, Any]) -> Dict[str, Any]:
        t0 = time.perf_counter()
        sq = (state.get("active_subquery") or state.get("query") or "").strip()
        sq_id = int(state.get("active_subquery_id") or 0)
        allow = state.get("active_search_allowance")
        n_allow = max(0, int(allow if allow is not None else default_allow))

        evidence_delta: List[dict] = []
        n_searches = 0
        arms = state.get("active_arms") or None
        for _ in range(n_allow):
            if not sq:
                break
            out = box.knowledge_search(
                sq, subquery_id=sq_id, top_k=top_k, arms=arms
            )
            hits = list(out.get("hits") or [])
            evidence_delta.extend(hits)
            n_searches += 1

        latency = (time.perf_counter() - t0) * 1000
        step = next_step(state)
        traj = [
            step_record(
                step=step,
                node=node_name,
                tool="knowledge_search",
                input_summary=(sq or "")[:200],
                output_summary=f"searches={n_searches} hits={len(evidence_delta)}",
                ok=True,
                latency_ms=latency,
                counts={
                    "searches": n_searches,
                    "hits": len(evidence_delta),
                    "subquery_id": sq_id,
                },
            )
        ]
        # Deltas only — parent reducers merge across Send workers.
        return {
            "evidence": evidence_delta,
            "trajectory": traj,
        }

    g = StateGraph(AgentState)
    g.add_node("retrieve", retrieve)
    g.set_entry_point("retrieve")
    g.add_edge("retrieve", END)
    return g.compile()


def retrieval_worker_delta(
    subgraph: Any,
    state: Dict[str, Any],
) -> Dict[str, Any]:
    """Invoke retrieval subgraph and strip non-reducer keys for concurrent safety.

    Compiled subgraphs return full state channels; parallel Send workers must not
    write last-value keys like ``active_subquery`` concurrently.
    """
    inv = dict(state)
    inv["evidence"] = []
    inv["trajectory"] = []
    out = subgraph.invoke(inv)
    return {
        "evidence": list(out.get("evidence") or []),
        "trajectory": list(out.get("trajectory") or []),
    }
