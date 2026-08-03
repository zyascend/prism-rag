"""Agent eval entry — thin wrapper around ``run_agent`` for E2E / smoke scripts.

Mirrors ``answer_for_eval`` shape: returns answer/citations/context plus an
``agent`` metadata block (status, subqueries, trajectory, counts).
HITL is forced off for batch eval.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from src.agent.config import agent_config
from src.agent.runner import merge_agent_cfg, run_agent

__all__ = [
    "agent_answer_for_eval",
    "load_agent_eval_qa",
    "DEFAULT_AGENT_EVAL_QA",
]

DEFAULT_AGENT_EVAL_QA = Path("data/agent_eval_qa.json")


def agent_answer_for_eval(
    query: str,
    *,
    retriever,
    generator,
    k_context: int = 5,
    use_rerank: bool = True,
    use_visual: bool = True,
    cfg: Optional[Dict[str, Any]] = None,
    pipeline_fallback_fn: Optional[Callable[[], Dict[str, Any]]] = None,
    **search_kwargs,
) -> dict:
    """Run the agent graph for one query; return dict for E2E / dual-arm scripts.

    Returns
    -------
    dict
        ``{answer, citations, context, agent: {status, subqueries, trajectory,
        counts, error, thread_id}}`` — same top-level keys as ``Generator.answer``
        / ``answer_for_eval`` so callers can swap modes.

    Notes
    -----
    - Forces ``hitl.review_subqueries=False`` for unattended batch runs.
    - Does **not** require ``agent.enabled`` in yaml (caller opts into this path).
    - Extra ``search_kwargs`` are forwarded to ``retriever.search`` (e.g. flags).
    """

    def search_fn(q, k=None):
        kk = int(k) if k is not None else k_context
        return retriever.search(
            q,
            k=kk,
            use_rerank=use_rerank,
            use_visual=use_visual,
            **search_kwargs,
        )

    def complete_fn(prompt: str) -> str:
        if hasattr(generator, "complete") and callable(generator.complete):
            return generator.complete(prompt) or ""
        # Fallback: injected private complete (tests / older Generator)
        injected = getattr(generator, "_complete_fn", None)
        if callable(injected):
            return injected(prompt) or ""
        raise AttributeError(
            "generator has no complete(); add Generator.complete or inject _complete_fn"
        )

    def generate_fn(q, hits):
        return generator.answer(q, hits, k_context=k_context)

    agent_cfg = merge_agent_cfg(
        agent_config(),
        {
            "hitl": {"review_subqueries": False},
            **(cfg or {}),
        },
    )
    # ensure nested hitl stays off even if cfg overwrote hitl partially
    hitl = dict(agent_cfg.get("hitl") or {})
    hitl["review_subqueries"] = False
    agent_cfg["hitl"] = hitl

    res = run_agent(
        query,
        search_fn=search_fn,
        complete_fn=complete_fn,
        generate_fn=generate_fn,
        cfg=agent_cfg,
        pipeline_fallback_fn=pipeline_fallback_fn,
    )
    return {
        "answer": res.answer,
        "citations": res.citations,
        "context": res.context,
        "agent": {
            "status": res.status,
            "subqueries": res.subqueries,
            "trajectory": res.trajectory,
            "counts": res.counts,
            "error": res.error,
            "thread_id": res.thread_id,
        },
        "evidence": res.evidence,
    }


def load_agent_eval_qa(
    path: Optional[Union[str, Path]] = None,
    *,
    tags: Optional[List[str]] = None,
    max_items: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Load ``data/agent_eval_qa.json`` items; optional tag filter and max_items.

    Accepts either ``{"version": N, "items": [...]}`` or a bare list.
    """
    p = Path(path) if path else DEFAULT_AGENT_EVAL_QA
    raw = json.loads(p.read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        items = list(raw.get("items") or [])
    elif isinstance(raw, list):
        items = list(raw)
    else:
        raise ValueError(f"unsupported agent_eval_qa format in {p}")

    if tags:
        tag_set = {t.strip().lower() for t in tags if t and str(t).strip()}
        items = [
            it
            for it in items
            if str(it.get("tag") or "").strip().lower() in tag_set
        ]
    if max_items is not None:
        items = items[: max(0, int(max_items))]
    return items
