"""Optional ReAct demo graph — learning /对照 only.

Default off (``agent.react_demo.enabled=false``). Production path remains the
fixed StateGraph in ``graph.py``; API does **not** route here by default.

Uses ``langchain_core`` @tool wrappers + a lightweight hand-written agent↔ToolNode
loop driven by ``complete_fn`` (string in/out) so tests need no ChatModel deps.
"""
from __future__ import annotations

import json
import logging
import re
import uuid
from typing import Annotated, Any, Callable, Dict, List, Optional, Sequence, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from src.agent.tools import make_agent_lc_tools, parse_json_object

logger = logging.getLogger(__name__)

SearchFn = Callable[..., List[dict]]
CompleteFn = Callable[[str], str]

__all__ = [
    "ReactDemoState",
    "build_react_demo_graph",
    "run_react_demo",
    "react_demo_enabled",
    "parse_react_complete",
    "DEFAULT_MAX_STEPS",
    "DEFAULT_MAX_SEARCHES",
]

DEFAULT_MAX_STEPS = 4
DEFAULT_MAX_SEARCHES = 2

_SYSTEM = """You are a ReAct retrieval agent over a private PDF knowledge base.
Tools: knowledge_search(query, top_k?).

Respond with EXACTLY one of:
1) A tool call as JSON only:
   {"tool": "knowledge_search", "args": {"query": "...", "top_k": 5}}
2) A final answer:
   FINAL: <your answer>

Rules:
- Prefer at least one knowledge_search before FINAL when facts may be in the docs.
- No web search. Keep queries self-contained.
- If evidence is empty or insufficient, FINAL with a short abstention.
"""


class ReactDemoState(TypedDict, total=False):
    messages: Annotated[list, add_messages]
    steps: int
    searches: int
    answer: str
    status: str  # ok | max_steps | error


def react_demo_enabled(cfg: Optional[Dict[str, Any]] = None) -> bool:
    """True only when ``agent.react_demo.enabled`` is set (default false)."""
    if cfg is None:
        from src.agent.config import agent_config

        cfg = agent_config()
    rd = (cfg or {}).get("react_demo")
    if isinstance(rd, dict):
        return bool(rd.get("enabled", False))
    return False


def parse_react_complete(raw: str) -> Dict[str, Any]:
    """Parse complete_fn output → final answer or tool call.

    Returns:
      {"kind": "final", "answer": str} or
      {"kind": "tool", "name": str, "args": dict}
    """
    text = (raw or "").strip()
    if not text:
        return {"kind": "final", "answer": ""}

    # FINAL: ...
    m = re.match(r"^FINAL\s*:\s*(.*)$", text, re.I | re.S)
    if m:
        return {"kind": "final", "answer": m.group(1).strip()}

    # JSON tool call (whole or fenced)
    try:
        data = parse_json_object(text)
    except Exception:
        data = None
    if isinstance(data, dict):
        tool = data.get("tool") or data.get("action") or data.get("name")
        if tool:
            args = data.get("args") or data.get("action_input") or data.get("input")
            if not isinstance(args, dict):
                # flat: {"tool": "...", "query": "..."}
                args = {
                    k: v
                    for k, v in data.items()
                    if k not in ("tool", "action", "name", "args", "action_input", "input")
                }
            return {
                "kind": "tool",
                "name": str(tool),
                "args": args if isinstance(args, dict) else {},
            }
        if "answer" in data or "final" in data:
            ans = data.get("answer") if data.get("answer") is not None else data.get("final")
            return {"kind": "final", "answer": str(ans or "")}

    # TOOL name {json}
    m2 = re.match(r"^TOOL\s+(\w+)\s+(\{.*\})\s*$", text, re.I | re.S)
    if m2:
        try:
            args = json.loads(m2.group(2))
        except json.JSONDecodeError:
            args = {}
        return {
            "kind": "tool",
            "name": m2.group(1),
            "args": args if isinstance(args, dict) else {},
        }

    # Bare text → treat as final (graceful)
    return {"kind": "final", "answer": text}


def _messages_to_prompt(messages: Sequence[BaseMessage]) -> str:
    parts: List[str] = []
    for msg in messages:
        role = getattr(msg, "type", None) or msg.__class__.__name__
        content = getattr(msg, "content", "") or ""
        if role in ("human", "HumanMessage"):
            parts.append(f"User: {content}")
        elif role in ("ai", "AIMessage"):
            tcs = getattr(msg, "tool_calls", None) or []
            if tcs:
                parts.append(f"Assistant(tool_call): {json.dumps(tcs, ensure_ascii=False)}")
            else:
                parts.append(f"Assistant: {content}")
        elif role in ("tool", "ToolMessage"):
            name = getattr(msg, "name", "tool")
            parts.append(f"Observation({name}): {content}")
        elif role in ("system", "SystemMessage"):
            parts.append(f"System: {content}")
        else:
            parts.append(f"{role}: {content}")
    parts.append(
        "Next: reply with either "
        '{"tool": "knowledge_search", "args": {"query": "..."}} '
        "or FINAL: <answer>"
    )
    return "\n\n".join(parts)


def build_react_demo_graph(
    *,
    search_fn: SearchFn,
    complete_fn: CompleteFn,
    max_steps: int = DEFAULT_MAX_STEPS,
    max_searches: int = DEFAULT_MAX_SEARCHES,
    top_k: int = 5,
):
    """Compile a learning-only ReAct StateGraph (agent ↔ tools) with hard step caps.

    ``complete_fn`` is a plain ``str -> str`` completer (not a ChatModel). Tool
    selection is parsed from its text (see ``parse_react_complete``).
    """
    tools = make_agent_lc_tools(search_fn, top_k_default=int(top_k))
    tool_by_name = {t.name: t for t in tools}
    tool_node = ToolNode(tools)
    max_steps_i = max(1, int(max_steps))
    max_searches_i = max(0, int(max_searches))

    def agent_node(state: ReactDemoState) -> Dict[str, Any]:
        steps = int(state.get("steps") or 0) + 1
        searches = int(state.get("searches") or 0)
        messages = list(state.get("messages") or [])

        # Hard stop before another LLM call if already at cap
        if steps > max_steps_i:
            ans = str(state.get("answer") or "")
            if not ans:
                ans = "max_steps reached without FINAL"
            return {
                "messages": [AIMessage(content=f"FINAL: {ans}")],
                "steps": steps,
                "answer": ans,
                "status": "max_steps",
            }

        # Budget: force final if search budget exhausted and we already searched
        force_final = searches >= max_searches_i and steps > 1

        prompt_msgs = [SystemMessage(content=_SYSTEM)] + messages
        if force_final:
            prompt_msgs = prompt_msgs + [
                HumanMessage(
                    content="Search budget exhausted. Reply FINAL: <answer> only."
                )
            ]
        prompt = _messages_to_prompt(prompt_msgs)
        try:
            raw = complete_fn(prompt)
        except Exception as e:
            logger.warning("react_demo complete_fn failed: %s", e)
            return {
                "messages": [AIMessage(content=f"FINAL: error: {e}")],
                "steps": steps,
                "answer": f"error: {e}",
                "status": "error",
            }

        parsed = parse_react_complete(raw if isinstance(raw, str) else str(raw))
        if force_final and parsed.get("kind") == "tool":
            # Ignore further tool calls when search budget is gone
            ans = str(state.get("answer") or "search budget exhausted")
            return {
                "messages": [AIMessage(content=f"FINAL: {ans}")],
                "steps": steps,
                "answer": ans,
                "status": "max_steps",
            }

        if parsed.get("kind") == "tool":
            name = str(parsed.get("name") or "knowledge_search")
            if name not in tool_by_name:
                name = "knowledge_search"
            args = dict(parsed.get("args") or {})
            # Cap searches: if already at max, convert to final
            if searches >= max_searches_i:
                ans = "search budget exhausted"
                return {
                    "messages": [AIMessage(content=f"FINAL: {ans}")],
                    "steps": steps,
                    "answer": ans,
                    "status": "max_steps",
                }
            tc_id = f"call_{uuid.uuid4().hex[:10]}"
            msg = AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": name,
                        "args": args,
                        "id": tc_id,
                        "type": "tool_call",
                    }
                ],
            )
            return {
                "messages": [msg],
                "steps": steps,
                "searches": searches + 1,
                "status": state.get("status") or "ok",
            }

        answer = str(parsed.get("answer") or "")
        return {
            "messages": [AIMessage(content=f"FINAL: {answer}" if answer else "FINAL:")],
            "steps": steps,
            "answer": answer,
            "status": "ok",
        }

    def route_after_agent(state: ReactDemoState) -> str:
        if int(state.get("steps") or 0) >= max_steps_i:
            # Allow one more tools hop only if last message has tool_calls and steps==max
            last = (state.get("messages") or [None])[-1]
            if (
                isinstance(last, AIMessage)
                and getattr(last, "tool_calls", None)
                and int(state.get("steps") or 0) == max_steps_i
            ):
                return "tools"
            return END
        last = (state.get("messages") or [None])[-1]
        if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
            return "tools"
        return END

    g: StateGraph = StateGraph(ReactDemoState)
    g.add_node("agent", agent_node)
    g.add_node("tools", tool_node)
    g.add_edge(START, "agent")
    g.add_conditional_edges(
        "agent",
        route_after_agent,
        {"tools": "tools", END: END},
    )
    g.add_edge("tools", "agent")
    return g.compile()


def run_react_demo(
    query: str,
    *,
    search_fn: SearchFn,
    complete_fn: CompleteFn,
    max_steps: int = DEFAULT_MAX_STEPS,
    max_searches: int = DEFAULT_MAX_SEARCHES,
    top_k: int = 5,
    recursion_limit: Optional[int] = None,
) -> Dict[str, Any]:
    """Invoke the ReAct demo graph once; returns answer/status/messages/steps.

    Learning entrypoint only — not used by production ``run_agent`` / ``/ask``.
    """
    graph = build_react_demo_graph(
        search_fn=search_fn,
        complete_fn=complete_fn,
        max_steps=max_steps,
        max_searches=max_searches,
        top_k=top_k,
    )
    limit = recursion_limit if recursion_limit is not None else max(10, int(max_steps) * 4)
    init: ReactDemoState = {
        "messages": [HumanMessage(content=query or "")],
        "steps": 0,
        "searches": 0,
        "answer": "",
        "status": "ok",
    }
    out = graph.invoke(init, config={"recursion_limit": limit})
    if not isinstance(out, dict):
        out = {}
    return {
        "answer": str(out.get("answer") or ""),
        "status": str(out.get("status") or "ok"),
        "steps": int(out.get("steps") or 0),
        "searches": int(out.get("searches") or 0),
        "messages": list(out.get("messages") or []),
    }
