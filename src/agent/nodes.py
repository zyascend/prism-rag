"""Phase 1: Agent 角色子图化 — 各业务节点独立 subgraph 工厂。

设计（对齐 docs/architecture/agent.md §4 分层 + subgraphs.py 范式）：
- 每个角色（decompose / hitl_review / retrieve / grade / refine / synthesize / finalize）
  是一个独立编译的 StateGraph 子图，外层 graph.py 只做路由编排。
- 子图返回 **delta only**（reducer 字段 evidence/trajectory 靠外层 operator.add 合并；
  预算/meta 是 last-value 覆盖，返回完整 dict 即可）。
- 零行为变化：budget / meta / trajectory 记账逻辑与 graph.py 原闭包逐字段一致，
  仅把闭包提升为模块级工厂（测试契约：trajectory node 名、reducer 合并、meta 键不变）。

子图通道契约（state.py 未定义 subgraph 输入/输出 schema，LangGraph 会按引用自动注入；
这里显式声明，避免隐式通道漂移）。所有通道 total=False，与 AgentState 同 style。
"""
from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional, TypedDict, Union

from langgraph.graph import END, StateGraph

from src.agent.state import AgentState, next_step, step_record
from src.agent.tools import AgentToolBox

SearchFn = Callable[..., List[dict]]
CompleteFn = Callable[[str], str]
GenerateFn = Callable[..., dict]

__all__ = [
    "DecomposeChannel",
    "HitlReviewChannel",
    "RetrieveOneChannel",
    "RetrieveMultiChannel",
    "PrepareMultiChannel",
    "GradeChannel",
    "RefineChannel",
    "SynthesizeChannel",
    "FinalizeChannel",
    "build_decompose_subgraph",
    "build_hitl_review_subgraph",
    "build_retrieve_one_subgraph",
    "build_retrieve_multi_subgraph",
    "build_prepare_multi_subgraph",
    "build_grade_subgraph",
    "build_refine_subgraph",
    "build_synthesize_subgraph",
    "build_finalize_subgraph",
]


class _SharedInput(TypedDict, total=False):
    """子图共享输入通道（非 reducer）。

    关键：**不声明 reducer 通道（evidence / trajectory）为输入** —— 子图 schema
    声明 reducer 通道会导致子图 invoke 输出时把输入值原样带回（echo），外层
    operator.add 再合并 → 每次外层节点更新都叠加一次（Phase1 修复：evidence
    0→2→4→8 翻倍 / trajectory 两整轮）。子图节点仍可**写** evidence/trajectory
    （未声明键作为普通输出合并回外层），只是不把输入带回来。
    """

    query: str
    subqueries: List[str]
    strategy: str
    grade: Dict[str, Any]
    pending_subqueries: List[str]
    budget: Dict[str, Any]
    meta: Dict[str, Any]


class DecomposeChannel(_SharedInput, TypedDict, total=False):
    subqueries: List[str]
    strategy: str
    budget: Dict[str, Any]
    meta: Dict[str, Any]
    trajectory: List[Dict[str, Any]]
    pending_subqueries: List[str]


class HitlReviewChannel(_SharedInput, TypedDict, total=False):
    subqueries: List[str]
    strategy: str
    meta: Dict[str, Any]
    trajectory: List[Dict[str, Any]]
    pending_subqueries: List[str]


class RetrieveOneChannel(_SharedInput, TypedDict, total=False):
    evidence: List[Dict[str, Any]]
    budget: Dict[str, Any]
    meta: Dict[str, Any]
    trajectory: List[Dict[str, Any]]


class RetrieveMultiChannel(_SharedInput, TypedDict, total=False):
    evidence: List[Dict[str, Any]]
    budget: Dict[str, Any]
    meta: Dict[str, Any]
    trajectory: List[Dict[str, Any]]
    pending_subqueries: List[str]


class PrepareMultiChannel(_SharedInput, TypedDict, total=False):
    budget: Dict[str, Any]
    meta: Dict[str, Any]
    pending_subqueries: List[str]
    trajectory: List[Dict[str, Any]]


class GradeChannel(_SharedInput, TypedDict, total=False):
    grade: Dict[str, Any]
    budget: Dict[str, Any]
    meta: Dict[str, Any]
    trajectory: List[Dict[str, Any]]


class RefineChannel(_SharedInput, TypedDict, total=False):
    pending_subqueries: List[str]
    budget: Dict[str, Any]
    meta: Dict[str, Any]
    trajectory: List[Dict[str, Any]]


class SynthesizeChannel(_SharedInput, TypedDict, total=False):
    answer: str
    citations: List[Dict[str, Any]]
    status: str
    budget: Dict[str, Any]
    meta: Dict[str, Any]
    trajectory: List[Dict[str, Any]]


class FinalizeChannel(_SharedInput, TypedDict, total=False):
    meta: Dict[str, Any]


# --- budget / meta 记账原语（graph.py 内联逻辑提升为模块级，语义不变） ---


def budget_copy(state: Dict[str, Any]) -> Dict[str, Any]:
    b = state.get("budget") or {}
    return {
        "searches_left": int(b.get("searches_left") or 0),
        "llm_calls_left": int(b.get("llm_calls_left") or 0),
        "grade_cycles_left": int(b.get("grade_cycles_left") or 0),
        "max_subqueries": int(b.get("max_subqueries") or 0),
    }


def meta_copy(state: Dict[str, Any]) -> Dict[str, Any]:
    return dict(state.get("meta") or {})


def dec_llm(budget: Dict[str, Any]) -> None:
    budget["llm_calls_left"] = max(0, int(budget.get("llm_calls_left") or 0) - 1)


# --- 各角色 subgraph 工厂 ---


def build_decompose_subgraph(box: AgentToolBox, *, node_name: str = "decompose") -> Any:
    def decompose(state: Dict[str, Any]) -> Dict[str, Any]:
        t0 = time.perf_counter()
        query = state.get("query") or ""
        budget = budget_copy(state)
        meta = meta_copy(state)
        step = next_step(state)
        meta["step"] = step

        result = box.decompose_query(query)
        dec_llm(budget)
        meta["llm_calls"] = int(meta.get("llm_calls") or 0) + 1

        latency = (time.perf_counter() - t0) * 1000
        traj = [
            step_record(
                step=step,
                node=node_name,
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

    g = StateGraph(DecomposeChannel)
    g.add_node("decompose", decompose)
    g.set_entry_point("decompose")
    g.add_edge("decompose", END)
    return g.compile()


def build_hitl_review_subgraph(
    box: AgentToolBox, *, node_name: str = "hitl_review"
) -> Any:
    """HITL review 子图。interrupt 必须与 resume 在同一 checkpointer 线程下执行。

    subgraph 只声明自身输出通道；interrupt 用的 ``__interrupt__`` 由 LangGraph 注入，
    与外层状态不冲突（测试：test_interrupt_and_resume / test_resume_with_revised_subqueries）。
    """

    def hitl_review(state: Dict[str, Any]) -> Dict[str, Any]:
        from langgraph.types import interrupt

        t0 = time.perf_counter()
        meta = meta_copy(state)
        step = next_step(state)
        meta["step"] = step
        subs = list(state.get("subqueries") or [])
        if not subs:
            q = state.get("query") or ""
            subs = [q] if q else []

        approved = interrupt(
            {
                "subqueries": subs,
                "strategy": state.get("strategy") or "atomic",
                "query": state.get("query") or "",
            }
        )
        from src.agent.tools import (
            normalize_subquery_list,
            normalize_subquery_text,
        )

        def _normalize_approved(approved: Any, fallback: List[str]) -> List[str]:
            if isinstance(approved, list):
                out = normalize_subquery_list(approved, max_n=max(1, len(approved)))
                return out if out else list(fallback)
            if isinstance(approved, dict):
                raw = approved.get("subqueries")
                if raw is None and any(
                    k in approved for k in ("query", "subquery", "text", "q")
                ):
                    one = normalize_subquery_text(approved)
                    return [one] if one else list(fallback)
                out = normalize_subquery_list(
                    raw if raw is not None else approved, max_n=16
                )
                return out if out else list(fallback)
            if isinstance(approved, str) and approved.strip():
                one = normalize_subquery_text(approved)
                return [one] if one else list(fallback)
            return list(fallback)

        new_subs = _normalize_approved(approved, subs)
        strategy = str(state.get("strategy") or "atomic")
        if len(new_subs) > 1:
            strategy = "multi"
        elif len(new_subs) == 1 and strategy == "multi":
            strategy = "atomic"

        latency = (time.perf_counter() - t0) * 1000
        traj = [
            step_record(
                step=step,
                node=node_name,
                tool=None,
                input_summary=f"n_pending={len(subs)}",
                output_summary=f"n_approved={len(new_subs)} strategy={strategy}",
                ok=True,
                latency_ms=latency,
                counts={"subqueries": len(new_subs)},
            )
        ]
        return {
            "subqueries": new_subs,
            "strategy": strategy,
            "meta": meta,
            "trajectory": traj,
            "pending_subqueries": [],
        }

    g = StateGraph(HitlReviewChannel)
    g.add_node("hitl_review", hitl_review)
    g.set_entry_point("hitl_review")
    g.add_edge("hitl_review", END)
    return g.compile()


def build_retrieve_one_subgraph(box: AgentToolBox, *, node_name: str = "retrieve_one") -> Any:
    def retrieve(state: Dict[str, Any]) -> Dict[str, Any]:
        t0 = time.perf_counter()
        budget = budget_copy(state)
        meta = meta_copy(state)
        step = next_step(state)
        meta["step"] = step

        subs = list(state.get("subqueries") or [])
        if not subs:
            subs = [state.get("query") or ""]

        evidence_delta: List[dict] = []
        n_searches = 0
        for i, sq in enumerate(subs[:1]):
            if not (sq or "").strip():
                continue
            for _ in range(int(box.cfg.get("max_search_per_subquery") or 1)):
                if int(budget.get("searches_left") or 0) <= 0:
                    break
                out = box.knowledge_search(sq, subquery_id=i, top_k=int(box.cfg.get("search_top_k") or 5))
                hits = list(out.get("hits") or [])
                evidence_delta.extend(hits)
                budget["searches_left"] = max(0, int(budget.get("searches_left") or 0) - 1)
                n_searches += 1
                meta["searches"] = int(meta.get("searches") or 0) + 1
            if int(budget.get("searches_left") or 0) <= 0:
                break

        latency = (time.perf_counter() - t0) * 1000
        traj = [
            step_record(
                step=step,
                node=node_name,
                tool="knowledge_search",
                input_summary=(sq or "")[:200],
                output_summary=f"searches={n_searches} hits={len(evidence_delta)}",
                ok=True,
                latency_ms=latency,
                counts={"searches": n_searches, "hits": len(evidence_delta)},
            )
        ]
        return {
            "evidence": evidence_delta,
            "budget": budget,
            "meta": meta,
            "trajectory": traj,
        }

    g = StateGraph(RetrieveOneChannel)
    g.add_node("retrieve", retrieve)
    g.set_entry_point("retrieve")
    g.add_edge("retrieve", END)
    return g.compile()


def build_retrieve_multi_subgraph(
    box: AgentToolBox, *, node_name: str = "retrieve_multi"
) -> Any:
    def retrieve(state: Dict[str, Any]) -> Dict[str, Any]:
        t0 = time.perf_counter()
        budget = budget_copy(state)
        meta = meta_copy(state)
        step = next_step(state)
        meta["step"] = step

        pending = list(state.get("pending_subqueries") or [])
        all_subs = list(state.get("subqueries") or [])
        if pending:
            queries = pending
            clear_pending = True
        else:
            queries = all_subs if all_subs else [state.get("query") or ""]
            clear_pending = False

        evidence_delta: List[dict] = []
        n_searches = 0
        for i, sq in enumerate(queries):
            if not (sq or "").strip():
                continue
            if sq in all_subs:
                sq_id = all_subs.index(sq)
            else:
                sq_id = i
            for _ in range(int(box.cfg.get("max_search_per_subquery") or 1)):
                if int(budget.get("searches_left") or 0) <= 0:
                    break
                out = box.knowledge_search(sq, subquery_id=sq_id, top_k=int(box.cfg.get("search_top_k") or 5))
                hits = list(out.get("hits") or [])
                evidence_delta.extend(hits)
                budget["searches_left"] = max(0, int(budget.get("searches_left") or 0) - 1)
                n_searches += 1
                meta["searches"] = int(meta.get("searches") or 0) + 1
            if int(budget.get("searches_left") or 0) <= 0:
                break

        latency = (time.perf_counter() - t0) * 1000
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
            "evidence": evidence_delta,
            "budget": budget,
            "meta": meta,
            "trajectory": traj,
        }
        if clear_pending:
            update["pending_subqueries"] = []
        return update

    g = StateGraph(RetrieveMultiChannel)
    g.add_node("retrieve", retrieve)
    g.set_entry_point("retrieve")
    g.add_edge("retrieve", END)
    return g.compile()


def build_prepare_multi_subgraph(
    box: AgentToolBox, *, node_name: str = "prepare_multi"
) -> Any:
    def prepare(state: Dict[str, Any]) -> Dict[str, Any]:
        t0 = time.perf_counter()
        budget = budget_copy(state)
        meta = meta_copy(state)
        step = next_step(state)
        meta["step"] = step

        pending = list(state.get("pending_subqueries") or [])
        all_subs = list(state.get("subqueries") or [])
        if pending:
            raw_queries = pending
        else:
            raw_queries = all_subs if all_subs else [state.get("query") or ""]

        max_sq = int(budget.get("max_subqueries") or len(raw_queries) or 3)
        remaining = int(budget.get("searches_left") or 0)
        max_per_sq = max(1, int(box.cfg.get("max_search_per_subquery") or 1))

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
                node=node_name,
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

    g = StateGraph(PrepareMultiChannel)
    g.add_node("prepare", prepare)
    g.set_entry_point("prepare")
    g.add_edge("prepare", END)
    return g.compile()


def build_grade_subgraph(box: AgentToolBox, *, node_name: str = "grade") -> Any:
    def _grade_enabled(cfg: Dict[str, Any]) -> bool:
        g = cfg.get("grade")
        if isinstance(g, dict):
            return bool(g.get("enabled", True))
        return True

    def grade(state: Dict[str, Any]) -> Dict[str, Any]:
        t0 = time.perf_counter()
        budget = budget_copy(state)
        meta = meta_copy(state)
        step = next_step(state)
        meta["step"] = step
        query = state.get("query") or ""
        evidence = list(state.get("evidence") or [])

        if not _grade_enabled(box.cfg):
            grade = {"sufficient": True, "missing": "", "score": 1.0, "skipped": True}
            used_llm = False
        else:
            grade = box.grade_evidence(query, evidence)
            dec_llm(budget)
            meta["llm_calls"] = int(meta.get("llm_calls") or 0) + 1
            used_llm = True

        latency = (time.perf_counter() - t0) * 1000
        traj = [
            step_record(
                step=step,
                node=node_name,
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

    g = StateGraph(GradeChannel)
    g.add_node("grade", grade)
    g.set_entry_point("grade")
    g.add_edge("grade", END)
    return g.compile()


def build_refine_subgraph(box: AgentToolBox, *, node_name: str = "refine") -> Any:
    def refine(state: Dict[str, Any]) -> Dict[str, Any]:
        t0 = time.perf_counter()
        budget = budget_copy(state)
        meta = meta_copy(state)
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

        budget["grade_cycles_left"] = max(
            0, int(budget.get("grade_cycles_left") or 0) - 1
        )

        pending: List[str] = []
        for sq in subs:
            if int(budget.get("llm_calls_left") or 0) <= 0:
                pending.append(sq)
                continue
            rewritten = box.refine_subquery(query, sq, missing_s)
            dec_llm(budget)
            meta["llm_calls"] = int(meta.get("llm_calls") or 0) + 1
            pending.append(rewritten or sq)

        max_sq = int(budget.get("max_subqueries") or len(pending) or 3)
        searches_left = int(budget.get("searches_left") or 0)
        pending = pending[: max(0, min(max_sq, searches_left if searches_left > 0 else max_sq))]

        latency = (time.perf_counter() - t0) * 1000
        traj = [
            step_record(
                step=step,
                node=node_name,
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

    g = StateGraph(RefineChannel)
    g.add_node("refine", refine)
    g.set_entry_point("refine")
    g.add_edge("refine", END)
    return g.compile()


def build_synthesize_subgraph(box: AgentToolBox, *, node_name: str = "synthesize") -> Any:
    def synthesize(state: Dict[str, Any]) -> Dict[str, Any]:
        t0 = time.perf_counter()
        budget = budget_copy(state)
        meta = meta_copy(state)
        step = next_step(state)
        meta["step"] = step
        query = state.get("query") or ""
        evidence = list(state.get("evidence") or [])

        out = box.synthesize_answer(query, evidence)
        if evidence:
            dec_llm(budget)
            meta["llm_calls"] = int(meta.get("llm_calls") or 0) + 1

        rejected = bool(out.get("rejected"))
        status = "abstain" if rejected else "ok"
        answer = out.get("answer") or ""
        citations = list(out.get("citations") or [])

        latency = (time.perf_counter() - t0) * 1000
        traj = [
            step_record(
                step=step,
                node=node_name,
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
            meta = dict(meta)
            meta["context"] = out["context"]
            update["meta"] = meta
        return update

    g = StateGraph(SynthesizeChannel)
    g.add_node("synthesize", synthesize)
    g.set_entry_point("synthesize")
    g.add_edge("synthesize", END)
    return g.compile()


def build_finalize_subgraph(
    *,
    use_send: bool,
    use_hitl: bool,
    use_checkpoint: bool,
    node_name: str = "finalize",
) -> Any:
    def finalize(state: Dict[str, Any]) -> Dict[str, Any]:
        meta = meta_copy(state)
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
        meta["hitl"] = use_hitl
        meta["checkpoint"] = use_checkpoint
        # no trajectory delta — avoid duplicate step records
        return {"meta": meta}

    g = StateGraph(FinalizeChannel)
    g.add_node("finalize", finalize)
    g.set_entry_point("finalize")
    g.add_edge("finalize", END)
    return g.compile()
