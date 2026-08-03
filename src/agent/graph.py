"""Agent StateGraph — decompose → retrieve (seq or Send) → grade ⇄ refine → synthesize.

P1a: sequential multi-retrieve. P1b: optional Send map-reduce via use_send.
HITL interrupt / checkpointer deferred (Task 7).
"""
from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional, Union

from langgraph.graph import END, StateGraph
from langgraph.types import Send

from src.agent.state import AgentState, next_step, step_record
from src.agent.subgraphs import build_retrieval_subgraph, retrieval_worker_delta
from src.agent.tools import AgentToolBox

SearchFn = Callable[..., List[dict]]
CompleteFn = Callable[[str], str]
GenerateFn = Callable[..., dict]

__all__ = [
    "build_agent_graph",
    "route_after_decompose",
    "route_after_grade",
    "fan_out_searches",
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


def fan_out_searches(state: Dict[str, Any]) -> Union[List[Send], str]:
    """Map pending (or empty) work list to Send workers; empty → grade."""
    pending = list(state.get("pending_subqueries") or [])
    all_subs = list(state.get("subqueries") or [])
    if not pending:
        return "grade"
    meta = state.get("meta") or {}
    # list[int] parallel to pending, staged by prepare_multi (meta is a real channel)
    allow_list = list(meta.get("fan_allowances") or [])
    sends: List[Send] = []
    for i, sq in enumerate(pending):
        if not (sq or "").strip():
            continue
        if sq in all_subs:
            sq_id = all_subs.index(sq)
        else:
            sq_id = i
        n_allow = int(allow_list[i]) if i < len(allow_list) else 1
        payload = {
            **dict(state),
            "active_subquery": sq,
            "active_subquery_id": sq_id,
            "active_search_allowance": max(0, n_allow),
            # Critical: worker deltas only — avoid reducer double-count of parent lists
            "evidence": [],
            "trajectory": [],
        }
        payload.pop("_retrieve_node", None)
        sends.append(Send("retrieval_worker", payload))
    if not sends:
        return "grade"
    return sends


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


def _select_multi_queries(state: Dict[str, Any]) -> tuple[List[str], bool]:
    """Return (queries, clear_pending) for multi retrieve / fan-out."""
    pending = list(state.get("pending_subqueries") or [])
    all_subs = list(state.get("subqueries") or [])
    if pending:
        return pending, True
    queries = all_subs if all_subs else [state.get("query") or ""]
    return queries, False


def build_agent_graph(
    *,
    search_fn: SearchFn,
    complete_fn: CompleteFn,
    generate_fn: GenerateFn,
    cfg: Optional[Dict[str, Any]] = None,
) -> Any:
    """Compile agent graph with injected toolbox deps.

    When ``cfg['use_send']`` is true, multi-subquery retrieval fans out via
    LangGraph ``Send`` + ``retrieval_subgraph``; otherwise sequential loop.
    """
    cfg = dict(cfg or {})
    use_send = bool(cfg.get("use_send", False))
    box = AgentToolBox(
        search_fn=search_fn,
        complete_fn=complete_fn,
        generate_fn=generate_fn,
        cfg=cfg,
    )
    max_per_sq = max(1, int(cfg.get("max_search_per_subquery") or 1))
    top_k = int(cfg.get("search_top_k") or 5)

    retrieval_sg = build_retrieval_subgraph(
        box=box,
        top_k=top_k,
        max_search_per_subquery=max_per_sq,
        node_name="retrieval_worker",
    )

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

    def prepare_multi_node(state: Dict[str, Any]) -> Dict[str, Any]:
        """Pre-allocate search budget, stage pending work list for Send fan-out."""
        t0 = time.perf_counter()
        budget = _budget_copy(state)
        meta = _meta_copy(state)
        step = next_step(state)
        meta["step"] = step

        raw_queries, _ = _select_multi_queries(state)
        all_subs = list(state.get("subqueries") or [])
        max_sq = int(budget.get("max_subqueries") or len(raw_queries) or 3)
        remaining = int(budget.get("searches_left") or 0)

        work: List[str] = []
        allow_list: List[int] = []
        n_allocated = 0
        for i, sq in enumerate(raw_queries):
            if not (sq or "").strip():
                continue
            if len(work) >= max_sq:
                break
            if remaining <= 0:
                break
            n_allow = min(max_per_sq, remaining)
            if n_allow <= 0:
                break
            work.append(sq)
            allow_list.append(n_allow)
            remaining -= n_allow
            n_allocated += n_allow

        budget["searches_left"] = remaining
        meta["searches"] = int(meta.get("searches") or 0) + n_allocated
        meta["send_workers"] = len(work)
        meta["fan_allowances"] = allow_list

        latency = (time.perf_counter() - t0) * 1000
        traj = [
            step_record(
                step=step,
                node="prepare_multi",
                tool=None,
                input_summary=f"n_raw={len(raw_queries)} n_work={len(work)}",
                output_summary=f"allocated={n_allocated} left={remaining}",
                ok=True,
                latency_ms=latency,
                counts={
                    "workers": len(work),
                    "searches_allocated": n_allocated,
                    "canonical_subs": len(all_subs),
                },
            )
        ]
        return {
            "budget": budget,
            "meta": meta,
            "pending_subqueries": work,
            "trajectory": traj,
        }

    def retrieval_worker_node(state: Dict[str, Any]) -> Dict[str, Any]:
        return retrieval_worker_delta(retrieval_sg, state)

    def fan_in_node(state: Dict[str, Any]) -> Dict[str, Any]:
        """Clear staged work list after parallel merge."""
        return {"pending_subqueries": []}

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
        meta["use_send"] = use_send
        # no trajectory delta — avoid duplicate step records
        return {"meta": meta}

    builder = StateGraph(AgentState)
    builder.add_node("decompose", decompose_node)
    builder.add_node("retrieve_one", retrieve_one_node)
    builder.add_node("grade", grade_node)
    builder.add_node("refine", refine_node)
    builder.add_node("synthesize", synthesize_node)
    builder.add_node("finalize", finalize_node)

    if use_send:
        builder.add_node("prepare_multi", prepare_multi_node)
        builder.add_node("retrieval_worker", retrieval_worker_node)
        builder.add_node("fan_in", fan_in_node)
        multi_entry = "prepare_multi"
    else:
        builder.add_node("retrieve_multi", retrieve_multi_node)
        multi_entry = "retrieve_multi"

    builder.set_entry_point("decompose")
    builder.add_conditional_edges(
        "decompose",
        route_after_decompose,
        {
            "retrieve_one": "retrieve_one",
            "retrieve_multi": multi_entry,
        },
    )
    builder.add_edge("retrieve_one", "grade")

    if use_send:
        builder.add_conditional_edges(
            "prepare_multi",
            fan_out_searches,
            ["retrieval_worker", "grade"],
        )
        builder.add_edge("retrieval_worker", "fan_in")
        builder.add_edge("fan_in", "grade")
        builder.add_edge("refine", "prepare_multi")
    else:
        builder.add_edge("retrieve_multi", "grade")
        builder.add_edge("refine", "retrieve_multi")

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
    builder.add_edge("synthesize", "finalize")
    builder.add_edge("finalize", END)

    # checkpointer optional — Task 7; compile without for P1a/P1b
    return builder.compile()
