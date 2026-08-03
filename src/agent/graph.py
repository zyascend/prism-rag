"""Agent StateGraph P1a — sequential decompose → retrieve → grade ⇄ refine → synthesize.

No Send parallel yet (Task 6). HITL interrupt / checkpointer deferred (Task 7).
"""
from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional

from langgraph.graph import END, StateGraph

from src.agent.state import AgentState, next_step, step_record
from src.agent.tools import AgentToolBox

SearchFn = Callable[..., List[dict]]
CompleteFn = Callable[[str], str]
GenerateFn = Callable[..., dict]

__all__ = [
    "build_agent_graph",
    "route_after_decompose",
    "route_after_grade",
]


def route_after_decompose(state: Dict[str, Any]) -> str:
    """atomic → retrieve_one; multi (or multi-subquery) → retrieve_multi."""
    strategy = str(state.get("strategy") or "atomic").strip().lower()
    subs = state.get("subqueries") or []
    if strategy == "multi" or len(subs) > 1:
        return "retrieve_multi"
    return "retrieve_one"


def route_after_grade(state: Dict[str, Any]) -> str:
    """sufficient → synthesize; else refine if grade cycles + searches remain."""
    grade = state.get("grade") or {}
    budget = state.get("budget") or {}
    if grade.get("sufficient"):
        return "synthesize"
    cycles_left = int(budget.get("grade_cycles_left") or 0)
    searches_left = int(budget.get("searches_left") or 0)
    if cycles_left > 0 and searches_left > 0:
        return "refine"
    # Exhausted refine budget → synthesize (empty evidence handled as abstain inside)
    return "synthesize"


def _budget_copy(state: Dict[str, Any]) -> Dict[str, Any]:
    b = state.get("budget") or {}
    return {
        "searches_left": int(b.get("searches_left") or 0),
        "llm_calls_left": int(b.get("llm_calls_left") or 0),
        "grade_cycles_left": int(b.get("grade_cycles_left") or 0),
        "max_subqueries": int(b.get("max_subqueries") or 0),
    }


def _meta_copy(state: Dict[str, Any]) -> Dict[str, Any]:
    return dict(state.get("meta") or {})


def _dec_llm(budget: Dict[str, Any]) -> None:
    budget["llm_calls_left"] = max(0, int(budget.get("llm_calls_left") or 0) - 1)


def _grade_enabled(cfg: Dict[str, Any]) -> bool:
    g = cfg.get("grade")
    if isinstance(g, dict):
        return bool(g.get("enabled", True))
    return True


def build_agent_graph(
    *,
    search_fn: SearchFn,
    complete_fn: CompleteFn,
    generate_fn: GenerateFn,
    cfg: Optional[Dict[str, Any]] = None,
) -> Any:
    """Compile sequential agent graph with injected toolbox deps."""
    cfg = dict(cfg or {})
    box = AgentToolBox(
        search_fn=search_fn,
        complete_fn=complete_fn,
        generate_fn=generate_fn,
        cfg=cfg,
    )
    max_per_sq = max(1, int(cfg.get("max_search_per_subquery") or 1))
    top_k = int(cfg.get("search_top_k") or 5)

    def decompose_node(state: Dict[str, Any]) -> Dict[str, Any]:
        t0 = time.perf_counter()
        query = state.get("query") or ""
        budget = _budget_copy(state)
        meta = _meta_copy(state)
        step = next_step(state)
        meta["step"] = step

        result = box.decompose_query(query)
        _dec_llm(budget)
        meta["llm_calls"] = int(meta.get("llm_calls") or 0) + 1

        latency = (time.perf_counter() - t0) * 1000
        traj = [
            step_record(
                step=step,
                node="decompose",
                tool="decompose_query",
                input_summary=query[:200],
                output_summary=f"strategy={result.get('strategy')} n={len(result.get('subqueries') or [])}",
                ok=True,
                latency_ms=latency,
                counts={"subqueries": len(result.get("subqueries") or [])},
            )
        ]
        return {
            "subqueries": list(result.get("subqueries") or [query]),
            "strategy": str(result.get("strategy") or "atomic"),
            "budget": budget,
            "meta": meta,
            "trajectory": traj,  # delta only
            "pending_subqueries": [],
        }

    def _retrieve(state: Dict[str, Any], *, use_pending: bool) -> Dict[str, Any]:
        t0 = time.perf_counter()
        budget = _budget_copy(state)
        meta = _meta_copy(state)
        step = next_step(state)
        meta["step"] = step

        pending = list(state.get("pending_subqueries") or [])
        all_subs = list(state.get("subqueries") or [])
        if use_pending and pending:
            queries = pending
            clear_pending = True
        else:
            queries = all_subs if all_subs else [state.get("query") or ""]
            clear_pending = False

        # Map subquery text → stable id (index in subqueries, or pending offset)
        evidence_delta: List[dict] = []
        n_searches = 0
        for i, sq in enumerate(queries):
            if not (sq or "").strip():
                continue
            # Prefer index into canonical subqueries list
            if sq in all_subs:
                sq_id = all_subs.index(sq)
            else:
                sq_id = i
            for _ in range(max_per_sq):
                if int(budget.get("searches_left") or 0) <= 0:
                    break
                out = box.knowledge_search(sq, subquery_id=sq_id, top_k=top_k)
                hits = list(out.get("hits") or [])
                evidence_delta.extend(hits)
                budget["searches_left"] = max(
                    0, int(budget.get("searches_left") or 0) - 1
                )
                n_searches += 1
                meta["searches"] = int(meta.get("searches") or 0) + 1
            if int(budget.get("searches_left") or 0) <= 0:
                break

        latency = (time.perf_counter() - t0) * 1000
        node_name = "retrieve_multi" if use_pending or len(queries) > 1 else "retrieve_one"
        # Caller names the node via wrapper; set via state flag
        node_name = state.get("_retrieve_node") or node_name
        traj = [
            step_record(
                step=step,
                node=node_name,
                tool="knowledge_search",
                input_summary="; ".join(str(q)[:80] for q in queries)[:200],
                output_summary=f"searches={n_searches} hits={len(evidence_delta)}",
                ok=True,
                latency_ms=latency,
                counts={"searches": n_searches, "hits": len(evidence_delta)},
            )
        ]
        update: Dict[str, Any] = {
            "evidence": evidence_delta,  # delta only (operator.add)
            "budget": budget,
            "meta": meta,
            "trajectory": traj,
        }
        if clear_pending:
            update["pending_subqueries"] = []
        return update

    def retrieve_one_node(state: Dict[str, Any]) -> Dict[str, Any]:
        # Single subquery path — still honor budget
        st = dict(state)
        st["_retrieve_node"] = "retrieve_one"
        subs = list(state.get("subqueries") or [])
        if not subs:
            subs = [state.get("query") or ""]
        # Only first subquery for atomic path
        st = {**st, "subqueries": subs[:1], "pending_subqueries": []}
        return _retrieve(st, use_pending=False)

    def retrieve_multi_node(state: Dict[str, Any]) -> Dict[str, Any]:
        st = dict(state)
        st["_retrieve_node"] = "retrieve_multi"
        # Prefer pending (post-refine); else all subqueries
        return _retrieve(st, use_pending=True)

    def grade_node(state: Dict[str, Any]) -> Dict[str, Any]:
        t0 = time.perf_counter()
        budget = _budget_copy(state)
        meta = _meta_copy(state)
        step = next_step(state)
        meta["step"] = step
        query = state.get("query") or ""
        evidence = list(state.get("evidence") or [])

        if not _grade_enabled(cfg):
            grade = {"sufficient": True, "missing": "", "score": 1.0, "skipped": True}
            used_llm = False
        else:
            grade = box.grade_evidence(query, evidence)
            _dec_llm(budget)
            meta["llm_calls"] = int(meta.get("llm_calls") or 0) + 1
            used_llm = True

        latency = (time.perf_counter() - t0) * 1000
        traj = [
            step_record(
                step=step,
                node="grade",
                tool="grade_evidence" if used_llm else None,
                input_summary=f"n_evidence={len(evidence)}",
                output_summary=(
                    f"sufficient={grade.get('sufficient')} "
                    f"score={grade.get('score')}"
                ),
                ok=True,
                latency_ms=latency,
                counts={"llm": int(used_llm)},
            )
        ]
        return {
            "grade": grade,
            "budget": budget,
            "meta": meta,
            "trajectory": traj,
        }

    def refine_node(state: Dict[str, Any]) -> Dict[str, Any]:
        t0 = time.perf_counter()
        budget = _budget_copy(state)
        meta = _meta_copy(state)
        step = next_step(state)
        meta["step"] = step
        query = state.get("query") or ""
        grade = state.get("grade") or {}
        missing = grade.get("missing") or ""
        if isinstance(missing, list):
            missing_s = "; ".join(str(m) for m in missing)
        else:
            missing_s = str(missing)

        subs = list(state.get("subqueries") or [])
        if not subs:
            subs = [query]

        # Consume one grade cycle when actually refining
        budget["grade_cycles_left"] = max(
            0, int(budget.get("grade_cycles_left") or 0) - 1
        )

        pending: List[str] = []
        for sq in subs:
            if int(budget.get("llm_calls_left") or 0) <= 0:
                pending.append(sq)
                continue
            rewritten = box.refine_subquery(query, sq, missing_s)
            _dec_llm(budget)
            meta["llm_calls"] = int(meta.get("llm_calls") or 0) + 1
            pending.append(rewritten or sq)

        # Cap pending by remaining searches and max_subqueries
        max_sq = int(budget.get("max_subqueries") or len(pending) or 3)
        searches_left = int(budget.get("searches_left") or 0)
        pending = pending[: max(0, min(max_sq, searches_left if searches_left > 0 else max_sq))]

        latency = (time.perf_counter() - t0) * 1000
        traj = [
            step_record(
                step=step,
                node="refine",
                tool="refine_subquery",
                input_summary=f"missing={missing_s[:120]}",
                output_summary=f"pending={pending!r}"[:200],
                ok=True,
                latency_ms=latency,
                counts={"pending": len(pending)},
            )
        ]
        return {
            "pending_subqueries": pending,
            "budget": budget,
            "meta": meta,
            "trajectory": traj,
        }

    def synthesize_node(state: Dict[str, Any]) -> Dict[str, Any]:
        t0 = time.perf_counter()
        budget = _budget_copy(state)
        meta = _meta_copy(state)
        step = next_step(state)
        meta["step"] = step
        query = state.get("query") or ""
        evidence = list(state.get("evidence") or [])

        out = box.synthesize_answer(query, evidence)
        # generate_fn is LLM-backed when evidence non-empty
        if evidence:
            _dec_llm(budget)
            meta["llm_calls"] = int(meta.get("llm_calls") or 0) + 1

        rejected = bool(out.get("rejected"))
        status = "abstain" if rejected else "ok"
        answer = out.get("answer") or ""
        citations = list(out.get("citations") or [])

        latency = (time.perf_counter() - t0) * 1000
        traj = [
            step_record(
                step=step,
                node="synthesize",
                tool="synthesize_answer",
                input_summary=f"n_evidence={len(evidence)}",
                output_summary=f"status={status} answer_len={len(answer)}",
                ok=True,
                latency_ms=latency,
                counts={"citations": len(citations)},
            )
        ]
        update: Dict[str, Any] = {
            "answer": answer,
            "citations": citations,
            "status": status,
            "budget": budget,
            "meta": meta,
            "trajectory": traj,
        }
        if "context" in out:
            # keep optional context in meta for runner
            meta = dict(meta)
            meta["context"] = out["context"]
            update["meta"] = meta
        return update

    def finalize_node(state: Dict[str, Any]) -> Dict[str, Any]:
        meta = _meta_copy(state)
        traj = list(state.get("trajectory") or [])
        searches = int(meta.get("searches") or 0)
        llm_calls = int(meta.get("llm_calls") or 0)
        if not searches:
            searches = sum(
                int((t.get("counts") or {}).get("searches") or 0)
                for t in traj
                if t.get("tool") == "knowledge_search"
            )
        if not llm_calls:
            llm_calls = sum(
                1
                for t in traj
                if t.get("tool")
                in ("decompose_query", "grade_evidence", "refine_subquery", "synthesize_answer")
            )
        meta["searches"] = searches
        meta["llm_calls"] = llm_calls
        meta["n_trajectory"] = len(traj)
        meta["n_evidence"] = len(state.get("evidence") or [])
        meta["finalized"] = True
        # no trajectory delta — avoid duplicate step records
        return {"meta": meta}

    builder = StateGraph(AgentState)
    builder.add_node("decompose", decompose_node)
    builder.add_node("retrieve_one", retrieve_one_node)
    builder.add_node("retrieve_multi", retrieve_multi_node)
    builder.add_node("grade", grade_node)
    builder.add_node("refine", refine_node)
    builder.add_node("synthesize", synthesize_node)
    builder.add_node("finalize", finalize_node)

    builder.set_entry_point("decompose")
    builder.add_conditional_edges(
        "decompose",
        route_after_decompose,
        {
            "retrieve_one": "retrieve_one",
            "retrieve_multi": "retrieve_multi",
        },
    )
    builder.add_edge("retrieve_one", "grade")
    builder.add_edge("retrieve_multi", "grade")
    builder.add_conditional_edges(
        "grade",
        route_after_grade,
        {
            "refine": "refine",
            "synthesize": "synthesize",
            # alias accepted by route_after_grade contract
            "abstain_or_synthesize": "synthesize",
        },
    )
    builder.add_edge("refine", "retrieve_multi")
    builder.add_edge("synthesize", "finalize")
    builder.add_edge("finalize", END)

    # checkpointer optional — Task 7; compile without for P1a
    return builder.compile()
