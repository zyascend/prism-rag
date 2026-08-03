# src/agent/state.py
from __future__ import annotations

import operator
from typing import Annotated, Any, Dict, List, Optional, TypedDict


class EvidenceItem(TypedDict, total=False):
    chunk_id: str
    doc_id: str
    page_id: Any
    text: str
    score: float
    modality: str
    subquery_id: int
    rank: int


class StepRecord(TypedDict, total=False):
    step: int
    node: str
    tool: Optional[str]
    input_summary: str
    output_summary: str
    ok: bool
    error: Optional[str]
    latency_ms: float
    counts: Dict[str, Any]


class Budget(TypedDict):
    searches_left: int
    llm_calls_left: int
    grade_cycles_left: int
    max_subqueries: int


class AgentState(TypedDict, total=False):
    query: str
    subqueries: List[str]
    strategy: str  # atomic | multi
    evidence: Annotated[List[EvidenceItem], operator.add]
    trajectory: Annotated[List[StepRecord], operator.add]
    answer: str
    citations: List[Dict[str, Any]]
    status: str  # ok | abstain | error | degraded | interrupted
    budget: Budget
    grade: Dict[str, Any]
    pending_subqueries: List[str]  # refine 后待检索
    meta: Dict[str, Any]
    # Send 扇出时单路可带
    active_subquery: str
    active_subquery_id: int


def empty_agent_state(query: str, *, cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    from src.agent.config import agent_config
    c = cfg or agent_config()
    return {
        "query": query,
        "subqueries": [],
        "strategy": "atomic",
        "evidence": [],
        "trajectory": [],
        "answer": "",
        "citations": [],
        "status": "ok",
        "budget": {
            "searches_left": int(c["max_total_searches"]),
            "llm_calls_left": int(c["max_llm_calls"]),
            "grade_cycles_left": int(c["max_grade_cycles"]),
            "max_subqueries": int(c["max_subqueries"]),
        },
        "grade": {},
        "pending_subqueries": [],
        "meta": {"searches": 0, "llm_calls": 0, "step": 0},
    }


def merge_evidence(
    left: Optional[List[EvidenceItem]], right: Optional[List[EvidenceItem]]
) -> List[EvidenceItem]:
    return list(left or []) + list(right or [])


def step_record(
    *,
    step: int,
    node: str,
    tool: Optional[str] = None,
    input_summary: str = "",
    output_summary: str = "",
    ok: bool = True,
    error: Optional[str] = None,
    latency_ms: float = 0.0,
    counts: Optional[Dict[str, Any]] = None,
) -> StepRecord:
    return {
        "step": step,
        "node": node,
        "tool": tool,
        "input_summary": (input_summary or "")[:500],
        "output_summary": (output_summary or "")[:500],
        "ok": ok,
        "error": error,
        "latency_ms": float(latency_ms),
        "counts": counts or {},
    }


def next_step(state: Dict[str, Any]) -> int:
    meta = state.get("meta") or {}
    return int(meta.get("step") or 0) + 1
