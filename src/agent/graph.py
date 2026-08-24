"""Agent StateGraph — decompose → [HITL] → retrieve (seq or Send) → grade ⇄ refine → synthesize.

Phase 1: 各角色节点抽为独立 subgraph（nodes.py 的 build_*_subgraph），
外层 StateGraph 只保留路由 + Send fan-out。行为零变化（trajectory node 名、
reducer 合并、meta 键、路由函数签名与 Phase 0 完全一致）。

P1a: sequential multi-retrieve. P1b: optional Send map-reduce via use_send.
P1c: MemorySaver checkpoint, stream, HITL interrupt after decompose.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from langgraph.graph import END, StateGraph
from langgraph.types import Send

from src.agent.nodes import (
    build_decompose_subgraph,
    build_finalize_subgraph,
    build_grade_subgraph,
    build_hitl_review_subgraph,
    build_prepare_multi_subgraph,
    build_refine_subgraph,
    build_retrieve_multi_subgraph,
    build_retrieve_one_subgraph,
    build_supervise_subgraph,
    build_synthesize_subgraph,
)
from src.agent.state import AgentState
from src.agent.subgraphs import build_retrieval_subgraph, retrieval_worker_delta
from src.agent.tools import AgentToolBox

SearchFn = Callable[..., List[dict]]
CompleteFn = Callable[[str], str]
GenerateFn = Callable[..., dict]

__all__ = [
    "build_agent_graph",
    "route_after_decompose",
    "route_after_grade",
    "route_after_supervise",
    "fan_out_searches",
    "export_graph_mermaid",
    "sanitize_langgraph_mermaid",
    "hitl_enabled",
    "checkpoint_enabled",
]


def sanitize_langgraph_mermaid(mermaid: str) -> str:
    """Make LangGraph ``draw_mermaid()`` output portable for common renderers.

    Raw LangGraph output often breaks VS Code / Cursor preview and SVG→image
    pipelines because it embeds HTML in labels (``<p>…</p>``), HTML entities
    (``&nbsp;``), and CSS properties some Mermaid builds reject
    (e.g. ``line-height`` on ``classDef``).
    """
    s = mermaid.replace("\t", "    ")
    # Strip <p> wrappers LangGraph puts inside stadium/rounded labels
    s = re.sub(r"</?p>", "", s, flags=re.IGNORECASE)
    # Edge labels use &nbsp; as padding — plain spaces work in Mermaid
    s = s.replace("&nbsp;", " ")
    # Collapse " -.   .-> " empty-ish dotted edges after nbsp strip
    s = re.sub(r"-\.\s+\.->", "-.->", s)
    # Normalize labeled dotted edges: -.  label  .->  →  -. label .->
    s = re.sub(r"-\.\s+(\S(?:.*?\S)?)\s+\.->", r"-. \1 .->", s)
    # classDef: drop unsupported / noisy props that can fail SVG renderers
    s = re.sub(r",\s*line-height:[^,\n;]+", "", s)
    # fill-opacity:0 makes start invisible and trips some SVG pipelines
    s = re.sub(
        r"(classDef\s+first\s+)[^\n]+",
        r"\1fill:#ffffff,stroke:#999999,color:#333333",
        s,
    )
    return s


def hitl_enabled(cfg: Optional[Dict[str, Any]]) -> bool:
    h = (cfg or {}).get("hitl")
    if isinstance(h, dict):
        return bool(h.get("review_subqueries"))
    return False


def checkpoint_enabled(cfg: Optional[Dict[str, Any]]) -> bool:
    """True when MemorySaver should be attached (checkpoint flag or HITL)."""
    c = cfg or {}
    cp = c.get("checkpoint")
    if isinstance(cp, dict) and bool(cp.get("enabled")):
        return True
    # HITL resume requires a checkpointer even if flag is off
    return hitl_enabled(c)


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


def route_after_supervise(state: Dict[str, Any]) -> str:
    """Supervisor 派单后路由：plan 有效且 mode=atomic → retrieve_one；否则 multi。

    fallback plan（无有效派单）保持 decompose 的 strategy —— 走规则对应分支。
    """
    plan = state.get("plan") or {}
    mode = ""
    if isinstance(plan, dict):
        mode = str(plan.get("mode") or "").strip().lower()
        if not mode or plan.get("fallback"):
            mode = str(state.get("strategy") or "atomic").strip().lower()
    if mode == "atomic":
        return "retrieve_one"
    return "retrieve_multi"


def fan_out_searches(state: Dict[str, Any]) -> Union[List[Send], str]:
    """Map pending (or empty) work list to Send workers; empty → grade."""
    pending = list(state.get("pending_subqueries") or [])
    all_subs = list(state.get("subqueries") or [])
    if not pending:
        return "grade"
    meta = state.get("meta") or {}
    # list[dict] parallel to pending, staged by prepare_multi (supervisor arms + quota)
    allow_list = list(meta.get("fan_allowances") or [])
    sends: List[Send] = []
    for i, sq in enumerate(pending):
        if not (sq or "").strip():
            continue
        if sq in all_subs:
            sq_id = all_subs.index(sq)
        else:
            sq_id = i
        allow = allow_list[i] if i < len(allow_list) else {}
        if isinstance(allow, dict):
            n_allow = max(0, int(allow.get("searches") or 0))
            arms = list(allow.get("arms") or [])
        else:  # 兼容旧 int 形态
            n_allow = max(0, int(allow or 1))
            arms = []
        payload = {
            **dict(state),
            "active_subquery": sq,
            "active_subquery_id": sq_id,
            "active_search_allowance": n_allow,
            "active_arms": arms,
            # Critical: worker deltas only — avoid reducer double-count of parent lists
            "evidence": [],
            "trajectory": [],
        }
        payload.pop("_retrieve_node", None)
        sends.append(Send("retrieval_worker", payload))
    if not sends:
        return "grade"
    return sends


def build_agent_graph(
    *,
    search_fn: SearchFn,
    complete_fn: CompleteFn,
    generate_fn: GenerateFn,
    cfg: Optional[Dict[str, Any]] = None,
    search_fns: Optional[Dict[str, SearchFn]] = None,
) -> Any:
    """Compile agent graph with injected toolbox deps.

    When ``cfg['use_send']`` is true, multi-subquery retrieval fans out via
    LangGraph ``Send`` + ``retrieval_subgraph``; otherwise sequential loop.

    When ``checkpoint.enabled`` or HITL is on, compile with process-wide
    ``MemorySaver``. HITL inserts ``hitl_review`` after ``decompose``.

    Phase 1: 角色节点全部委托 nodes.py 的独立 subgraph；本函数只做装配与路由。
    """
    cfg = dict(cfg or {})
    use_send = bool(cfg.get("use_send", False))
    use_hitl = hitl_enabled(cfg)
    use_checkpoint = checkpoint_enabled(cfg)
    # supervise 开关：cfg 里可传 supervise.enabled 或平铺 supervise_enabled（对齐 agent_config）
    sup = cfg.get("supervise")
    if isinstance(sup, dict):
        use_supervise = bool(sup.get("enabled", False))
    else:
        use_supervise = bool(cfg.get("supervise_enabled", False))
    box = AgentToolBox(
        search_fn=search_fn,
        complete_fn=complete_fn,
        generate_fn=generate_fn,
        cfg=cfg,
        search_fns=search_fns or {},
    )
    max_per_sq = max(1, int(cfg.get("max_search_per_subquery") or 1))
    top_k = int(cfg.get("search_top_k") or 5)

    # 角色子图（nodes.py）：各自编译，接收外层 state，返回 delta
    decompose_sg = build_decompose_subgraph(box)
    retrieve_one_sg = build_retrieve_one_subgraph(box)
    retrieve_multi_sg = build_retrieve_multi_subgraph(box)
    prepare_multi_sg = build_prepare_multi_subgraph(box)
    grade_sg = build_grade_subgraph(box)
    refine_sg = build_refine_subgraph(box)
    synthesize_sg = build_synthesize_subgraph(box)
    finalize_sg = build_finalize_subgraph(
        use_send=use_send,
        use_hitl=use_hitl,
        use_checkpoint=use_checkpoint,
    )
    retrieval_sg = build_retrieval_subgraph(
        box=box,
        top_k=top_k,
        max_search_per_subquery=max_per_sq,
        node_name="retrieval_worker",
    )

    def invoke_subgraph(sg: Any, state: Dict[str, Any]) -> Dict[str, Any]:
        """Invoke a compiled role subgraph with outer state, return only its delta.

        只对 ``evidence`` 剥回显：grade/synthesize 子图 schema 声明 evidence 为
        只读输入，LangGraph 会把它回显到子图输出（子图不写 evidence，输出 = 输入
        echo）。外层 reducer 若把 echo 当 delta 再合并 → 证据双计。这里剥掉输入
        部分。``trajectory`` 不做切片——各子图的 trajectory 是纯输出 delta，
        切片会误截（prepare_multi 的 1 条 delta 被外层已有长度截空）。
        """
        out = sg.invoke(state)
        if not isinstance(out, dict):
            return {}
        delta = {}
        for k, v in out.items():
            if k == "evidence" and k in state and isinstance(state.get(k), list):
                cur = state[k]
                # 子图输出 evidence = 输入 echo + 节点 delta；echo 部分剥掉
                new = v if not isinstance(v, list) else v[len(cur):]
                delta[k] = new
            else:
                delta[k] = v
        return delta

    builder = StateGraph(AgentState)
    builder.add_node("decompose", lambda s: invoke_subgraph(decompose_sg, s))
    builder.add_node("retrieve_one", lambda s: invoke_subgraph(retrieve_one_sg, s))
    builder.add_node("grade", lambda s: invoke_subgraph(grade_sg, s))
    builder.add_node("refine", lambda s: invoke_subgraph(refine_sg, s))
    builder.add_node("synthesize", lambda s: invoke_subgraph(synthesize_sg, s))
    builder.add_node("finalize", lambda s: invoke_subgraph(finalize_sg, s))

    if use_send:
        builder.add_node("prepare_multi", lambda s: invoke_subgraph(prepare_multi_sg, s))
        builder.add_node("retrieval_worker", _make_retrieval_worker_node(retrieval_sg))
        builder.add_node("fan_in", fan_in_node)
        multi_entry = "prepare_multi"
    else:
        builder.add_node("retrieve_multi", lambda s: invoke_subgraph(retrieve_multi_sg, s))
        multi_entry = "retrieve_multi"

    builder.set_entry_point("decompose")

    # HITL: pause after decompose so human can edit subqueries before retrieve
    if use_hitl:
        hitl_sg = build_hitl_review_subgraph(box)
        builder.add_node("hitl_review", lambda s: invoke_subgraph(hitl_sg, s))
        builder.add_edge("decompose", "hitl_review")
        route_source = "hitl_review"
    else:
        route_source = "decompose"

    # Supervisor（Phase 2）：multi 路径先派单再检索；默认关 → 图拓扑与 Phase 1 完全一致
    if use_supervise:
        supervise_sg = build_supervise_subgraph(box)
        builder.add_node("supervise", lambda s: invoke_subgraph(supervise_sg, s))
        builder.add_edge("supervise", multi_entry)
        # 条件边：supervise 之后按 plan.mode 路由；fallback（无有效 plan）→ 规则路由
        builder.add_conditional_edges(
            "supervise",
            route_after_supervise,
            {
                "retrieve_one": "retrieve_one",
                "retrieve_multi": multi_entry,
            },
        )
        multi_source = "supervise"
    else:
        multi_source = "decompose"  # 保持 Phase 1 路由

    builder.add_conditional_edges(
        route_source,
        route_after_decompose,
        {
            "retrieve_one": "retrieve_one",
            # supervise 关 → 直连 multi 入口；开 → 先进 supervise 派单
            "retrieve_multi": multi_source if use_supervise else multi_entry,
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

    if use_checkpoint:
        from src.agent.checkpoint import get_memory_saver

        return builder.compile(checkpointer=get_memory_saver())
    return builder.compile()


def _make_retrieval_worker_node(retrieval_sg: Any) -> Callable[[Dict[str, Any]], Dict[str, Any]]:
    """Bind a compiled retrieval subgraph to a Send worker entry (deltas only)."""

    def retrieval_worker_node(state: Dict[str, Any]) -> Dict[str, Any]:
        return retrieval_worker_delta(retrieval_sg, state)

    return retrieval_worker_node


def fan_in_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Clear staged work list after parallel merge."""
    return {"pending_subqueries": []}


def export_graph_mermaid(
    path: str = "docs/architecture/agent-graph.mmd",
    *,
    cfg: Optional[Dict[str, Any]] = None,
) -> str:
    """Export compiled graph topology as Mermaid; write to ``path`` if given."""
    # Use noop deps — topology only
    def _search(q: str, k: int = 5) -> List[dict]:
        return []

    def _complete(p: str) -> str:
        return '{"subqueries": ["q"], "strategy": "atomic", "reason": "export"}'

    def _generate(q: str, hits: Any) -> dict:
        return {"answer": "", "citations": [], "rejected": True}

    export_cfg = {
        "enabled": True,
        "grade": {"enabled": True},
        "hitl": {"review_subqueries": False},
        "checkpoint": {"enabled": False},
        "use_send": True,
        "max_subqueries": 3,
        "max_total_searches": 3,
        "max_llm_calls": 6,
        "max_grade_cycles": 1,
    }
    if cfg:
        export_cfg.update(cfg)
    g = build_agent_graph(
        search_fn=_search,
        complete_fn=_complete,
        generate_fn=_generate,
        cfg=export_cfg,
    )
    mermaid = sanitize_langgraph_mermaid(g.get_graph().draw_mermaid())
    if path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(mermaid, encoding="utf-8")
    return mermaid
