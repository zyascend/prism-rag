"""ReAct demo graph — learning对照 only; production path stays fixed StateGraph."""
from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage

from src.agent.config import agent_config
from src.agent.react_demo import (
    build_react_demo_graph,
    parse_react_complete,
    react_demo_enabled,
    run_react_demo,
)
from src.agent.tools import make_agent_lc_tools, make_knowledge_search_tool


def test_react_demo_enabled_default_false():
    assert agent_config().get("react_demo", {}).get("enabled") is False
    assert react_demo_enabled() is False
    assert react_demo_enabled({"react_demo": {"enabled": False}}) is False
    assert react_demo_enabled({"react_demo": {"enabled": True}}) is True


def test_knowledge_search_tool_is_structured_tool():
    hits = [{"chunk_id": "c1", "text": "175 PSI", "score": 0.9, "doc_id": "d1"}]
    t = make_knowledge_search_tool(lambda q, k=5: hits, top_k_default=3)
    assert t.name == "knowledge_search"
    assert "knowledge" in (t.description or "").lower() or "search" in (t.description or "").lower()
    out = t.invoke({"query": "pressure", "top_k": 2})
    assert "c1" in out
    assert "175 PSI" in out
    tools = make_agent_lc_tools(lambda q, k=5: [])
    assert len(tools) == 1
    assert tools[0].name == "knowledge_search"


def test_parse_react_complete_final_and_tool():
    assert parse_react_complete("FINAL: idk") == {"kind": "final", "answer": "idk"}
    assert parse_react_complete("FINAL:  ")["kind"] == "final"
    p = parse_react_complete(
        '{"tool": "knowledge_search", "args": {"query": "psi", "top_k": 3}}'
    )
    assert p["kind"] == "tool"
    assert p["name"] == "knowledge_search"
    assert p["args"]["query"] == "psi"
    p2 = parse_react_complete('TOOL knowledge_search {"query": "x"}')
    assert p2["kind"] == "tool" and p2["args"]["query"] == "x"


def test_react_demo_compiles_and_stops():
    from src.agent.react_demo import build_react_demo_graph

    g = build_react_demo_graph(
        search_fn=lambda q, k=5: [],
        complete_fn=lambda p: "FINAL: idk",
        max_steps=3,
    )
    # Compiled graph has invoke
    assert hasattr(g, "invoke")
    out = g.invoke(
        {
            "messages": [HumanMessage(content="what is the pressure?")],
            "steps": 0,
            "searches": 0,
            "answer": "",
            "status": "ok",
        },
        config={"recursion_limit": 8},
    )
    assert out.get("answer") == "idk"
    assert int(out.get("steps") or 0) <= 3
    assert out.get("status") == "ok"
    # Last AI message is final (no pending tool_calls)
    last = out["messages"][-1]
    assert isinstance(last, AIMessage)
    assert not getattr(last, "tool_calls", None)


def test_react_demo_stops_with_max_steps_when_model_loops():
    """If complete_fn always requests tools, max_steps / search budget must exit."""
    calls = {"n": 0}

    def always_tool(prompt: str) -> str:
        calls["n"] += 1
        return '{"tool": "knowledge_search", "args": {"query": "loop"}}'

    result = run_react_demo(
        "endless?",
        search_fn=lambda q, k=5: [{"chunk_id": "c", "text": "t", "score": 1.0}],
        complete_fn=always_tool,
        max_steps=3,
        max_searches=2,
        recursion_limit=20,
    )
    assert result["steps"] <= 4  # agent steps capped
    assert result["searches"] <= 2
    assert result["status"] in ("max_steps", "ok")
    # Must terminate (not hang); answer may be budget message
    assert isinstance(result["answer"], str)
    assert calls["n"] >= 1
    assert calls["n"] <= 6


def test_react_demo_tool_then_final():
    seq = iter(
        [
            '{"tool": "knowledge_search", "args": {"query": "PSI rinsing"}}',
            "FINAL: max 175 PSI",
        ]
    )

    def complete(prompt: str) -> str:
        return next(seq)

    hits = [
        {
            "chunk_id": "c9",
            "text": "Manual rinsing max water pressure is 175 PSI.",
            "score": 0.95,
            "doc_id": "manual",
            "page_number": 12,
        }
    ]
    result = run_react_demo(
        "What is max rinsing PSI?",
        search_fn=lambda q, k=5: hits,
        complete_fn=complete,
        max_steps=4,
        max_searches=2,
    )
    assert result["answer"] == "max 175 PSI"
    assert result["status"] == "ok"
    assert result["searches"] == 1
    assert result["steps"] == 2
    # ToolMessage present in history
    types = [type(m).__name__ for m in result["messages"]]
    assert "ToolMessage" in types or any("tool" in t.lower() for t in types)
