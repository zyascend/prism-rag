# Agentic RAG（LangGraph）Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在默认 pipeline 零行为变化的前提下，交付可开关的 LangGraph agent 旁路：固定图拆问多跳检索 + 窄工具 + 可回放 trajectory；并覆盖 LangGraph 主特性（条件边/环/Send/子图/tools/stream/checkpoint/HITL）。

**Architecture:** `src/agent/` 单向依赖现有 `PrismRAGRetriever` 与 `Generator`。主路径为固定 StateGraph（非默认自由 ReAct）；`POST /ask` 增加 `mode=agent`（需 `agent.enabled`）；L4 cache 加 agent 盐；默认 `enabled: false`。Phase2 用 `data/agent_eval_qa.json` 子集双臂对照后再决议是否开默认。

**Tech Stack:** Python 3.11+、LangGraph、langchain-core（仅 tools/messages）、FastAPI、现有 Ollama/OpenAI 兼容 client、pytest。

**Spec:** [`docs/superpowers/specs/2026-08-03-agentic-rag-langgraph-design.md`](../specs/2026-08-03-agentic-rag-langgraph-design.md)

**Branch:** 从 `docs/agentic-rag-langgraph-design` 或 `main` 切 **`feat/agentic-rag-langgraph`**（**禁止在 `main` 直接改代码**）。

**DoD（Phase1）:**
1. `agent.enabled=false` 时 `/ask` 与现网完全一致（`agent` 字段为 `null`）
2. `enabled=true` + `mode=agent`：可返回 answer + `agent.trajectory` + subqueries
3. 护栏单测：max_subqueries / max_total_searches / decompose fallback / 空证据拒答 / degrade
4. Feature Map P0/P1 可点名验证（见 Task 验收表）
5. `pytest tests/test_agent_*.py tests/test_api.py -q` 绿；黄金消融路径不碰 agent

**DoD（Phase2，独立 PR/云跑）:**
6. `data/agent_eval_qa.json` + 双臂脚本产物进 `runs/`
7. handoff 书面 Go/No-Go；**无 Go 不改 `enabled: true`**

**明确不做:** web_search、KG、默认开 agent、NDCG 走 agent、子问依赖链、多 Agent、Store 业务记忆。

**本地约束（AGENTS.md）:** 单测用 mock；**允许本机轻量真链路（≤10 query）**，前提：小语料（local-demo）+ 模型已在缓存、不触发新下载 / 不全量 ingest。禁止本机 283 / 全量 RAGAS。可辩护质量结论仍上云 Phase2。详见 [architecture/agent.md §14.1](../../architecture/agent.md)。

---

## File map

| 路径 | 职责 |
|------|------|
| `src/agent/__init__.py` | 导出 `run_agent` / `AgentResult` / `agent_config` |
| `src/agent/config.py` | 读 `agent.*`，带默认值；`agent_cache_salt()` |
| `src/agent/state.py` | `AgentState` TypedDict、`StepRecord`、`EvidenceItem`、reducers |
| `src/agent/tools.py` | 可注入 deps 的 tool 实现（非全局单例） |
| `src/agent/subgraphs.py` | `retrieval_subgraph` 编译 |
| `src/agent/graph.py` | 主 StateGraph 编译（懒单例） |
| `src/agent/checkpoint.py` | MemorySaver 装配 |
| `src/agent/runner.py` | `run_agent` / `stream_agent` / `resume_agent` / on_error |
| `src/agent/react_demo.py` | 可选 ReAct 对照图（默认关） |
| `src/prompts/prompts/agent_decompose.yaml` | 拆问 prompt |
| `src/prompts/prompts/agent_grade_evidence.yaml` | 证据充分性 |
| `src/prompts/prompts/agent_refine_subquery.yaml` | 子问改写 |
| `src/prompts/prompts/agent_synthesize.yaml` | 多证据合成（或复用 answer_generation） |
| `config/models.yaml` | `agent:` 块，**全部默认保守 / enabled false** |
| `src/api/routes.py` | `AskRequest.mode`、`AgentInfo`、`/ask` 分支、`/ask/resume` |
| `src/evaluation/vidore_adapter.py` | `answer_cache_key` 加 `agent_cache_salt` |
| `src/generation/self_rag.py` 或新 `src/agent/eval.py` | `answer_for_eval` 可选 `mode=agent`（薄封装） |
| `static/demo/*` | Mode Pipeline\|Agent + trajectory 展示 |
| `tests/test_agent_config.py` | 配置默认 |
| `tests/test_agent_state.py` | reducer / 序列化 |
| `tests/test_agent_tools.py` | mock search/decompose/synthesize |
| `tests/test_agent_graph.py` | 图编译、条件边、budget、环退出 |
| `tests/test_agent_runner.py` | run_agent 端到端 mock |
| `tests/test_agent_api.py` | `/ask` mode 与 cache salt / resume |
| `tests/test_agent_checkpoint_hitl.py` | interrupt + resume |
| `data/agent_eval_qa.json` | Phase2 子集（可先骨架） |
| `scripts/run_agent_eval.py` | Phase2 双臂（可后置） |
| `docs/architecture/agent.md` | 可选：实现后短文档 + mermaid |

**依赖方向（强制）:** `api` → `agent.runner` → `graph/tools` → `retrieval` / `generation`。禁止 `retrieval` import `agent`。

---

## Task 0: 分支与依赖

**Files:**
- Modify: `pyproject.toml`
- Modify: `uv.lock`（若项目用 uv sync）

- [ ] **Step 1: 确认分支**

```bash
git branch --show-current
# 若在 main 或 docs/*：
git checkout main && git pull --ff-only
git checkout -b feat/agentic-rag-langgraph
# 可 cherry-pick 或 merge docs 分支上的 spec/plan（若尚未在 main）
```

- [ ] **Step 2: 加入依赖**

在 `pyproject.toml` 的 `dependencies` 列表追加（版本以安装时 PyPI 稳定版为准，写入后 lock）：

```toml
"langgraph>=0.2.0",
"langchain-core>=0.3.0",
```

- [ ] **Step 3: 安装并验证 import**

```bash
# 优先用项目既有方式，例如：
uv sync
# 或: pip install -e ".[dev]"
.venv/bin/python -c "import langgraph; import langchain_core; print('ok')"
```

Expected: 打印 `ok`，无 ImportError。

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "chore: add langgraph + langchain-core for agent path"
```

---

## Task 1: 配置与 cache salt

**Files:**
- Create: `src/agent/__init__.py`
- Create: `src/agent/config.py`
- Modify: `config/models.yaml`
- Create: `tests/test_agent_config.py`

- [ ] **Step 1: 写失败测试**

```python
# tests/test_agent_config.py
from src.agent.config import agent_config, agent_cache_salt


def test_agent_defaults_disabled():
    c = agent_config()
    assert c["enabled"] is False
    assert c["max_subqueries"] == 3
    assert c["max_total_searches"] == 3
    assert c["max_search_per_subquery"] == 1
    assert c["max_llm_calls"] == 6
    assert c["max_grade_cycles"] == 1
    assert c["on_error"] == "degrade_pipeline"
    assert c["grade"]["enabled"] is True
    assert c["checkpoint"]["enabled"] is True
    assert c["hitl"]["review_subqueries"] is False
    assert c["react_demo"]["enabled"] is False


def test_cache_salt_stable_and_changes_with_enabled(monkeypatch):
    from src import config as config_mod

    # 无 agent 段时 salt 仍可生成
    s_off = agent_cache_salt()
    assert "ag=" in s_off
    assert "off" in s_off or "enabled=False" in s_off or "en=0" in s_off
```

- [ ] **Step 2: 跑测确认失败**

```bash
.venv/bin/python -m pytest tests/test_agent_config.py -v
```

Expected: FAIL（模块不存在）

- [ ] **Step 3: 实现 config**

```python
# src/agent/config.py
"""Agent 路径配置（默认关闭，与 CRAG/Gate2 同纪律）。"""
from __future__ import annotations

from typing import Any, Callable, Dict, Optional


def agent_config(get_cfg: Optional[Callable] = None) -> Dict[str, Any]:
    if get_cfg is None:
        from src.config import cfg
        get_cfg = cfg.get
    raw = get_cfg("agent", {}) or {}
    if not isinstance(raw, dict):
        raw = {}
    grade = raw.get("grade") if isinstance(raw.get("grade"), dict) else {}
    checkpoint = raw.get("checkpoint") if isinstance(raw.get("checkpoint"), dict) else {}
    hitl = raw.get("hitl") if isinstance(raw.get("hitl"), dict) else {}
    react = raw.get("react_demo") if isinstance(raw.get("react_demo"), dict) else {}
    decompose = raw.get("decompose") if isinstance(raw.get("decompose"), dict) else {}
    synthesize = raw.get("synthesize") if isinstance(raw.get("synthesize"), dict) else {}
    return {
        "enabled": bool(raw.get("enabled", False)),
        "max_subqueries": int(raw.get("max_subqueries", 3)),
        "max_search_per_subquery": int(raw.get("max_search_per_subquery", 1)),
        "max_total_searches": int(raw.get("max_total_searches", 3)),
        "max_llm_calls": int(raw.get("max_llm_calls", 6)),
        "max_grade_cycles": int(raw.get("max_grade_cycles", 1)),
        "timeout_ms": int(raw.get("timeout_ms", 30000)),
        "on_error": str(raw.get("on_error") or "degrade_pipeline"),
        "return_trajectory": bool(raw.get("return_trajectory", True)),
        "grade": {"enabled": bool(grade.get("enabled", True))},
        "checkpoint": {"enabled": bool(checkpoint.get("enabled", True))},
        "hitl": {"review_subqueries": bool(hitl.get("review_subqueries", False))},
        "react_demo": {"enabled": bool(react.get("enabled", False))},
        "decompose": {"prompt_id": str(decompose.get("prompt_id") or "agent_decompose")},
        "synthesize": {"prompt_id": str(synthesize.get("prompt_id") or "agent_synthesize")},
    }


def agent_cache_salt(get_cfg: Optional[Callable] = None) -> str:
    """L4 Answer 缓存盐：agent 开/关与护栏参数变化不得串答案。"""
    c = agent_config(get_cfg)
    if not c["enabled"]:
        return "ag=off"
    return (
        f"ag=on"
        f"|msq={c['max_subqueries']}"
        f"|mts={c['max_total_searches']}"
        f"|mlc={c['max_llm_calls']}"
        f"|mgc={c['max_grade_cycles']}"
        f"|gr={int(c['grade']['enabled'])}"
        f"|hitl={int(c['hitl']['review_subqueries'])}"
    )
```

```python
# src/agent/__init__.py
from src.agent.config import agent_cache_salt, agent_config
from src.agent.runner import AgentResult, run_agent  # runner 在 Task 5 补齐前可先注释后打开

__all__ = ["agent_config", "agent_cache_salt", "run_agent", "AgentResult"]
```

> 若 `runner` 尚未存在，`__init__.py` 先只导出 config，Task 5 再补导出。

在 `config/models.yaml` **文件末尾**追加：

```yaml
# Agentic RAG（LangGraph）；默认关。见 docs/superpowers/specs/2026-08-03-agentic-rag-langgraph-design.md
agent:
  enabled: false
  max_subqueries: 3
  max_search_per_subquery: 1
  max_total_searches: 3
  max_llm_calls: 6
  max_grade_cycles: 1
  timeout_ms: 30000
  on_error: degrade_pipeline
  return_trajectory: true
  grade:
    enabled: true
  checkpoint:
    enabled: true
  hitl:
    review_subqueries: false
  react_demo:
    enabled: false
  decompose:
    prompt_id: agent_decompose
  synthesize:
    prompt_id: agent_synthesize
```

- [ ] **Step 4: 跑测通过**

```bash
.venv/bin/python -m pytest tests/test_agent_config.py -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/agent/config.py src/agent/__init__.py config/models.yaml tests/test_agent_config.py
git commit -m "feat(agent): config defaults and cache salt"
```

---

## Task 2: State 与 StepRecord

**Files:**
- Create: `src/agent/state.py`
- Create: `tests/test_agent_state.py`

- [ ] **Step 1: 写失败测试**

```python
# tests/test_agent_state.py
from src.agent.state import (
    empty_agent_state,
    merge_evidence,
    step_record,
)


def test_empty_state_shape():
    s = empty_agent_state("what is X?")
    assert s["query"] == "what is X?"
    assert s["subqueries"] == []
    assert s["evidence"] == []
    assert s["trajectory"] == []
    assert s["status"] == "ok"
    assert s["budget"]["searches_left"] >= 1


def test_merge_evidence_appends():
    a = [{"chunk_id": "1", "text": "a", "subquery_id": 0}]
    b = [{"chunk_id": "2", "text": "b", "subquery_id": 1}]
    assert merge_evidence(a, b) == a + b


def test_step_record_jsonable():
    rec = step_record(
        step=1,
        node="decompose",
        tool="decompose_query",
        input_summary="q",
        output_summary="2 subqs",
        ok=True,
        latency_ms=12.5,
        counts={"subqueries": 2},
    )
    import json
    json.dumps(rec)
    assert rec["node"] == "decompose"
```

- [ ] **Step 2: 跑测确认失败**

```bash
.venv/bin/python -m pytest tests/test_agent_state.py -v
```

- [ ] **Step 3: 实现 state**

```python
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
```

- [ ] **Step 4: 跑测通过并 commit**

```bash
.venv/bin/python -m pytest tests/test_agent_state.py -v
git add src/agent/state.py tests/test_agent_state.py
git commit -m "feat(agent): AgentState and trajectory records"
```

---

## Task 3: Tools（可注入 deps）+ Prompts

**Files:**
- Create: `src/agent/tools.py`
- Create: `src/prompts/prompts/agent_decompose.yaml`
- Create: `src/prompts/prompts/agent_grade_evidence.yaml`
- Create: `src/prompts/prompts/agent_refine_subquery.yaml`
- Create: `src/prompts/prompts/agent_synthesize.yaml`
- Create: `tests/test_agent_tools.py`

- [ ] **Step 1: 写失败测试（mock 无真模型）**

```python
# tests/test_agent_tools.py
from src.agent.tools import AgentToolBox, parse_json_object


def test_parse_json_object_fence():
    raw = '```json\n{"subqueries": ["a", "b"], "strategy": "multi"}\n```'
    data = parse_json_object(raw)
    assert data["strategy"] == "multi"
    assert len(data["subqueries"]) == 2


def test_decompose_fallback_on_bad_json():
    box = AgentToolBox(
        search_fn=lambda q, k=5: [],
        complete_fn=lambda p: "NOT JSON",
        generate_fn=lambda q, hits: {"answer": "x", "citations": [], "rejected": False},
        cfg={"max_subqueries": 3, "max_total_searches": 3},
    )
    out = box.decompose_query("simple question?")
    assert out["subqueries"] == ["simple question?"]
    assert out["strategy"] == "atomic"
    assert out.get("fallback") is True


def test_knowledge_search_tags_subquery_id():
    hits_in = [{"chunk_id": "c1", "text": "hello", "score": 0.9, "doc_id": "d1"}]
    box = AgentToolBox(
        search_fn=lambda q, k=5: hits_in,
        complete_fn=lambda p: "{}",
        generate_fn=lambda q, hits: {"answer": "", "citations": [], "rejected": True},
    )
    out = box.knowledge_search("q1", subquery_id=2, top_k=5)
    assert out["hits"][0]["subquery_id"] == 2
    assert out["hits"][0]["chunk_id"] == "c1"


def test_synthesize_empty_evidence_rejects():
    box = AgentToolBox(
        search_fn=lambda q, k=5: [],
        complete_fn=lambda p: "{}",
        generate_fn=lambda q, hits: {"answer": "should not call", "citations": [], "rejected": False},
    )
    out = box.synthesize_answer("q", evidence=[])
    assert out["rejected"] is True
    assert out["answer"]  # 拒答文案非空
```

- [ ] **Step 2: 实现 `tools.py` 核心**

要点（完整实现时按此契约）：

```python
# src/agent/tools.py — 结构纲要
class AgentToolBox:
    def __init__(self, *, search_fn, complete_fn, generate_fn, cfg=None, prompt_get_active=None):
        ...

    def decompose_query(self, query: str) -> dict:
        # render agent_decompose; parse JSON; clamp to max_subqueries;
        # on failure: return {subqueries:[query], strategy:"atomic", fallback:True}

    def knowledge_search(self, query: str, *, subquery_id: int, top_k: int = 5) -> dict:
        # hits = search_fn(query, k=top_k); tag subquery_id + rank

    def grade_evidence(self, query: str, evidence: list) -> dict:
        # LLM JSON: sufficient, missing, score; on error: sufficient=True pass-through（与 CRAG on_grade_error 类似，避免误杀）

    def refine_subquery(self, query: str, subquery: str, missing: str) -> str:
        # LLM 返回单行改写；失败返回原 subquery

    def synthesize_answer(self, query: str, evidence: list) -> dict:
        # if not evidence: abstain via src.rejection.abstain_message
        # else generate_fn(query, evidence_as_retrieved_dicts)
```

`parse_json_object`：复用 `self_rag._parse_verdict_json` 同类逻辑（可复制小函数到 tools，避免循环 import）。

**Prompt YAML** 最小模板示例（`agent_decompose.yaml`）：

```yaml
id: agent_decompose
description: Split a user question into independent sub-queries for multi-hop RAG
versions:
  - version: 1
    created_at: "2026-08-03"
    author: yang
    changelog: "MVP agent decompose"
    active: true
    template: |-
      You plan retrieval for an industrial PDF QA system.
      Split the QUESTION into at most {max_subqueries} self-contained sub-queries.
      If the question is already atomic, return a single sub-query equal to the question.
      Each sub-query must be understandable alone (no "the above value").
      Return ONLY JSON:
      {{"subqueries": ["..."], "strategy": "atomic"|"multi", "reason": "short"}}

      QUESTION:
      {query}
```

`agent_grade_evidence.yaml`：对 evidence 列表判断 `sufficient` / `missing` / `score`。  
`agent_refine_subquery.yaml`：根据 missing 改写一个子问。  
`agent_synthesize.yaml`：可薄包装「根据分条 EVIDENCE 回答 QUESTION；不足则拒答」；实现上 **优先** 调 `Generator.answer`（citations 已由 Generator 构造），prompt 仅在 Generator 未覆盖多段时使用。

- [ ] **Step 3: 跑测通过并 commit**

```bash
.venv/bin/python -m pytest tests/test_agent_tools.py -v
git add src/agent/tools.py src/prompts/prompts/agent_*.yaml tests/test_agent_tools.py
git commit -m "feat(agent): toolbox + prompts for decompose/search/grade/synthesize"
```

---

## Task 4: Graph P1a — 顺序图（无 Send）+ 条件边 + grade 环

**Files:**
- Create: `src/agent/graph.py`
- Create: `tests/test_agent_graph.py`

本 Task **先做顺序版**（多子问 for-loop 检索），保证可测闭环；Task 6 再换成 Send 并行。

- [ ] **Step 1: 写图行为测试（全 mock toolbox）**

```python
# tests/test_agent_graph.py
from src.agent.graph import build_agent_graph, route_after_decompose, route_after_grade
from src.agent.state import empty_agent_state


def test_route_atomic():
    assert route_after_decompose({"strategy": "atomic", "subqueries": ["q"]}) == "retrieve_one"
    assert route_after_decompose({"strategy": "multi", "subqueries": ["a", "b"]}) == "retrieve_multi"


def test_route_grade():
    st = {"grade": {"sufficient": True}, "budget": {"grade_cycles_left": 1, "searches_left": 2}}
    assert route_after_grade(st) == "synthesize"
    st2 = {
        "grade": {"sufficient": False},
        "budget": {"grade_cycles_left": 1, "searches_left": 2},
        "meta": {},
    }
    assert route_after_grade(st2) == "refine"
    st3 = {
        "grade": {"sufficient": False},
        "budget": {"grade_cycles_left": 0, "searches_left": 2},
    }
    assert route_after_grade(st3) in ("synthesize", "abstain_or_synthesize")


def test_graph_compile_and_invoke_atomic():
    def search_fn(q, k=5):
        return [{"chunk_id": "1", "text": f"fact about {q}", "score": 1.0, "doc_id": "d"}]

    def complete_fn(p):
        if "Split" in p or "sub-queries" in p or "subqueries" in p.lower():
            return '{"subqueries": ["what is torque?"], "strategy": "atomic", "reason": "simple"}'
        if "sufficient" in p.lower() or "Grade" in p or "evidence" in p.lower():
            return '{"sufficient": true, "missing": "", "score": 0.9}'
        return "{}"

    def generate_fn(q, hits):
        return {"answer": "42 Nm", "citations": [{"chunk_id": "1"}], "rejected": False}

    graph = build_agent_graph(
        search_fn=search_fn,
        complete_fn=complete_fn,
        generate_fn=generate_fn,
        cfg={
            "enabled": True,
            "max_subqueries": 3,
            "max_total_searches": 3,
            "max_search_per_subquery": 1,
            "max_llm_calls": 6,
            "max_grade_cycles": 1,
            "grade": {"enabled": True},
            "hitl": {"review_subqueries": False},
            "checkpoint": {"enabled": False},
        },
    )
    out = graph.invoke(empty_agent_state("what is torque?", cfg={
        "max_subqueries": 3, "max_total_searches": 3, "max_llm_calls": 6, "max_grade_cycles": 1,
    }))
    assert out["answer"]
    assert out["status"] in ("ok", "abstain")
    assert any(t.get("node") == "decompose" for t in out.get("trajectory") or [])
```

- [ ] **Step 2: 实现 `build_agent_graph`**

节点伪代码：

```text
decompose_node:
  box.decompose_query → set subqueries/strategy; append trajectory; dec llm budget
  # hitl 时 interrupt — Task 7

retrieve_one_node / retrieve_multi_node:
  for each subquery (cap by searches_left):
    box.knowledge_search → evidence += hits; searches_left -= 1; trajectory

grade_node:
  if not grade.enabled: set sufficient=True; skip LLM
  else: box.grade_evidence; dec grade_cycles on refine path

refine_node:
  rewrite pending; set pending_subqueries; loop to retrieve_multi for those only

synthesize_node:
  box.synthesize_answer → answer/citations/status

finalize_node:
  fill meta counts from trajectory
```

使用：

```python
from langgraph.graph import END, StateGraph
from src.agent.state import AgentState

builder = StateGraph(AgentState)
# add_node / add_conditional_edges / set_entry_point
# return builder.compile(checkpointer=...)  # checkpointer Task 7
```

`grade.enabled=false` 时 grade 节点短路 `sufficient=True`。

- [ ] **Step 3: 增测护栏**

```python
def test_search_budget_caps_calls():
    calls = {"n": 0}
    def search_fn(q, k=5):
        calls["n"] += 1
        return [{"chunk_id": str(calls["n"]), "text": q, "score": 1.0, "doc_id": "d"}]

    def complete_fn(p):
        return '{"subqueries": ["a", "b", "c"], "strategy": "multi", "reason": "x"}'

    def generate_fn(q, hits):
        return {"answer": "ok", "citations": [], "rejected": False}

    g = build_agent_graph(
        search_fn=search_fn, complete_fn=complete_fn, generate_fn=generate_fn,
        cfg={
            "max_subqueries": 3, "max_total_searches": 2, "max_search_per_subquery": 1,
            "max_llm_calls": 6, "max_grade_cycles": 0, "grade": {"enabled": False},
            "hitl": {"review_subqueries": False}, "checkpoint": {"enabled": False},
        },
    )
    st = empty_agent_state("q", cfg={"max_subqueries": 3, "max_total_searches": 2, "max_llm_calls": 6, "max_grade_cycles": 0})
    g.invoke(st)
    assert calls["n"] <= 2
```

- [ ] **Step 4: 跑测通过并 commit**

```bash
.venv/bin/python -m pytest tests/test_agent_graph.py -v
git add src/agent/graph.py tests/test_agent_graph.py
git commit -m "feat(agent): StateGraph decompose-retrieve-grade-synthesize"
```

---

## Task 5: Runner + L4 salt 接线

**Files:**
- Create: `src/agent/runner.py`
- Modify: `src/agent/__init__.py`
- Modify: `src/evaluation/vidore_adapter.py`（`answer_cache_key`）
- Create: `tests/test_agent_runner.py`

- [ ] **Step 1: 测试 AgentResult 与 on_error**

```python
# tests/test_agent_runner.py
from src.agent.runner import run_agent, AgentResult


def test_run_agent_ok():
    res = run_agent(
        "what is X?",
        search_fn=lambda q, k=5: [{"chunk_id": "1", "text": "X is 1", "score": 1.0, "doc_id": "d"}],
        complete_fn=lambda p: '{"subqueries": ["what is X?"], "strategy": "atomic", "reason": "a"}',
        generate_fn=lambda q, hits: {"answer": "1", "citations": [{"chunk_id": "1"}], "rejected": False},
        cfg={"enabled": True, "grade": {"enabled": False}, "max_total_searches": 3,
             "max_subqueries": 3, "max_llm_calls": 6, "max_grade_cycles": 0,
             "checkpoint": {"enabled": False}, "hitl": {"review_subqueries": False},
             "on_error": "abstain", "return_trajectory": True},
    )
    assert isinstance(res, AgentResult)
    assert res.status in ("ok", "abstain")
    assert res.answer
    assert res.trajectory is not None
```

- [ ] **Step 2: 实现 runner**

```python
# src/agent/runner.py
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

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
    from src.agent.config import agent_config
    from src.agent.graph import build_agent_graph
    from src.agent.state import empty_agent_state

    c = {**agent_config(), **(cfg or {})}
    # deep-merge nested grade/hitl if needed
    try:
        graph = build_agent_graph(
            search_fn=search_fn, complete_fn=complete_fn, generate_fn=generate_fn, cfg=c
        )
        init = empty_agent_state(query, cfg=c)
        init.setdefault("meta", {})["trace_id"] = trace_id
        out = graph.invoke(init, config={"configurable": {"thread_id": trace_id or "local"}})
        # map out → AgentResult
        ...
    except Exception as e:
        if c.get("on_error") == "degrade_pipeline" and pipeline_fallback_fn:
            fb = pipeline_fallback_fn()
            return AgentResult(
                answer=fb.get("answer", ""),
                citations=fb.get("citations") or [],
                status="degraded",
                error=str(e)[:300],
                ...
            )
        if c.get("on_error") == "abstain":
            from src.rejection import abstain_message
            return AgentResult(answer=abstain_message(), status="error", error=str(e)[:300])
        raise
```

`counts` 从 `meta` / trajectory 汇总：`subqueries`, `searches`, `llm_calls`, `evidence_n`。

- [ ] **Step 3: answer_cache_key 加盐**

在 `src/evaluation/vidore_adapter.py` 的 `answer_cache_key` 中：

```python
from src.agent.config import agent_cache_salt
# parts 列表追加:
agent_cache_salt(),
```

- [ ] **Step 4: 单测 salt 变化（可选轻量）**

```python
def test_answer_cache_key_includes_agent_salt(monkeypatch):
    # 构造最小 PrismRAGRetriever mock 或直接测 agent_cache_salt 已在 Task1
    pass
```

- [ ] **Step 5: 跑测 commit**

```bash
.venv/bin/python -m pytest tests/test_agent_runner.py tests/test_agent_config.py -v
git add src/agent/runner.py src/agent/__init__.py src/evaluation/vidore_adapter.py tests/test_agent_runner.py
git commit -m "feat(agent): run_agent runner and L4 agent cache salt"
```

---

## Task 6: P1b — Send 并行 + retrieval_subgraph

**Files:**
- Create: `src/agent/subgraphs.py`
- Modify: `src/agent/graph.py`
- Modify: `tests/test_agent_graph.py`

- [ ] **Step 1: 测试并行扇出次数**

```python
def test_multi_uses_send_or_equivalent_n_searches():
    """N 个子问应触发 N 次 search（受 budget 限制）。"""
    calls = []
    def search_fn(q, k=5):
        calls.append(q)
        return [{"chunk_id": q, "text": q, "score": 1.0, "doc_id": "d"}]
    def complete_fn(p):
        if "subqueries" in p.lower() or "Split" in p or "sub-queries" in p:
            return '{"subqueries": ["q1", "q2"], "strategy": "multi", "reason": "m"}'
        return '{"sufficient": true, "missing": "", "score": 1}'
    g = build_agent_graph(
        search_fn=search_fn,
        complete_fn=complete_fn,
        generate_fn=lambda q, h: {"answer": "a", "citations": [], "rejected": False},
        cfg={
            "max_subqueries": 3, "max_total_searches": 3, "max_llm_calls": 6,
            "max_grade_cycles": 1, "grade": {"enabled": False},
            "hitl": {"review_subqueries": False}, "checkpoint": {"enabled": False},
            "use_send": True,
        },
    )
    from src.agent.state import empty_agent_state
    g.invoke(empty_agent_state("combo?", cfg={
        "max_subqueries": 3, "max_total_searches": 3, "max_llm_calls": 6, "max_grade_cycles": 1,
    }))
    assert len(calls) == 2
```

- [ ] **Step 2: 实现 subgraph + Send**

```python
# src/agent/subgraphs.py
from langgraph.graph import END, StateGraph
from src.agent.state import AgentState

def build_retrieval_subgraph(search_fn, toolbox_cfg):
    def retrieve(state):
        # active_subquery / active_subquery_id → knowledge_search → return partial state
        # return {"evidence": hits, "trajectory": [step], "meta": ...}  # reducers merge
        ...
    g = StateGraph(AgentState)
    g.add_node("retrieve", retrieve)
    g.set_entry_point("retrieve")
    g.add_edge("retrieve", END)
    return g.compile()
```

主图 fan_out 节点：

```python
from langgraph.types import Send

def fan_out_searches(state):
    sends = []
    left = state["budget"]["searches_left"]
    for i, sq in enumerate(state["subqueries"][:left]):
        sends.append(Send("retrieval_worker", {
            **state,
            "active_subquery": sq,
            "active_subquery_id": i,
            "evidence": [],  # 重要：worker 只返回本路 evidence，靠 reducer 合并
            "trajectory": [],
        }))
    return sends
```

注意：LangGraph 版本 API 可能是 `langgraph.constants.Send` 或 `langgraph.types.Send`——**以已安装版本文档为准**，单测锁定行为。

- [ ] **Step 3: 跑测 commit**

```bash
.venv/bin/python -m pytest tests/test_agent_graph.py -v
git add src/agent/subgraphs.py src/agent/graph.py tests/test_agent_graph.py
git commit -m "feat(agent): Send map-reduce retrieval subgraph"
```

---

## Task 7: P1c — Checkpoint + Stream + HITL

**Files:**
- Create: `src/agent/checkpoint.py`
- Modify: `src/agent/graph.py` / `runner.py`
- Create: `tests/test_agent_checkpoint_hitl.py`

- [ ] **Step 1: HITL 测试**

```python
# tests/test_agent_checkpoint_hitl.py
from src.agent.runner import run_agent, resume_agent


def test_interrupt_and_resume():
    # complete_fn: decompose 返回 multi
    # cfg hitl.review_subqueries=True, checkpoint.enabled=True
    res = run_agent(
        "multi hop?",
        search_fn=lambda q, k=5: [{"chunk_id": "1", "text": "t", "score": 1.0, "doc_id": "d"}],
        complete_fn=lambda p: '{"subqueries": ["a", "b"], "strategy": "multi", "reason": "m"}',
        generate_fn=lambda q, h: {"answer": "done", "citations": [], "rejected": False},
        cfg={
            "enabled": True, "hitl": {"review_subqueries": True},
            "checkpoint": {"enabled": True}, "grade": {"enabled": False},
            "max_subqueries": 3, "max_total_searches": 3, "max_llm_calls": 6,
            "max_grade_cycles": 0, "on_error": "error", "return_trajectory": True,
        },
        trace_id="thread-test-1",
    )
    assert res.status == "interrupted"
    assert res.thread_id == "thread-test-1"
    assert len(res.subqueries) == 2

    res2 = resume_agent(
        thread_id="thread-test-1",
        approved_subqueries=["a", "b"],  # 或修订
        search_fn=...,
        complete_fn=...,
        generate_fn=...,
        cfg=...,
    )
    assert res2.status in ("ok", "abstain")
    assert res2.answer
```

- [ ] **Step 2: 实现**

```python
# src/agent/checkpoint.py
from langgraph.checkpoint.memory import MemorySaver

_SAVER = None

def get_memory_saver():
    global _SAVER
    if _SAVER is None:
        _SAVER = MemorySaver()
    return _SAVER
```

- decompose 后节点：若 `hitl.review_subqueries`，调用 `interrupt({"subqueries": ...})`（`langgraph.types.interrupt`）。  
- `compile(checkpointer=get_memory_saver())`。  
- `resume_agent`：`graph.invoke(Command(resume=approved_subqueries), config={"configurable": {"thread_id": ...}})` —— **以当前 langgraph 版本 HITL 文档为准** 调整 API。

- [ ] **Step 3: stream_agent**

```python
def stream_agent(query, **kwargs):
    graph = build_agent_graph(...)
    for event in graph.stream(init, stream_mode=["updates", "values"], config=...):
        yield event
```

单测：至少收到 1 个 update 事件。

- [ ] **Step 4: mermaid 导出 helper**

```python
def export_graph_mermaid(path: str = "docs/architecture/agent-graph.mmd") -> str:
    g = build_agent_graph(...)  # mock fns ok
    mermaid = g.get_graph().draw_mermaid()
    Path(path).write_text(mermaid, encoding="utf-8")
    return mermaid
```

- [ ] **Step 5: 跑测 commit**

```bash
.venv/bin/python -m pytest tests/test_agent_checkpoint_hitl.py tests/test_agent_graph.py -v
git add src/agent/checkpoint.py src/agent/runner.py src/agent/graph.py tests/test_agent_checkpoint_hitl.py docs/architecture/agent-graph.mmd
git commit -m "feat(agent): checkpoint, stream, HITL interrupt/resume"
```

---

## Task 8: API 接入 `/ask` + `/ask/resume`

**Files:**
- Modify: `src/api/routes.py`
- Create: `tests/test_agent_api.py`
- Modify: `tests/test_api.py`（若有通用 ask 契约，确保不破）

- [ ] **Step 1: API 契约测试**

```python
# tests/test_agent_api.py
from fastapi.testclient import TestClient
from src.api.routes import app
from unittest.mock import patch

client = TestClient(app)


def test_ask_pipeline_agent_null_by_default():
    # mock retriever/generator 与现有 test_api 相同风格
    ...
    r = client.post("/ask", json={"query": "hi", "k": 3})
    assert r.status_code == 200
    body = r.json()
    assert body.get("agent") is None


def test_ask_mode_agent_ignored_when_disabled(monkeypatch):
    monkeypatch.setitem(...)  # 或 patch agent_config enabled False
    r = client.post("/ask", json={"query": "hi", "mode": "agent", "k": 3})
    assert r.status_code == 200
    # agent 可为 null 或 used=false + ignored_reason
    ag = r.json().get("agent")
    assert ag is None or ag.get("used") is False
```

- [ ] **Step 2: 扩展 schema**

```python
class AskRequest(BaseModel):
    query: str
    doc_id: Optional[str] = None
    k: int = 5
    use_rerank: bool = True
    mode: str = "pipeline"  # pipeline | agent


class AgentStepInfo(BaseModel):
    step: int = 0
    node: str = ""
    tool: Optional[str] = None
    input_summary: str = ""
    output_summary: str = ""
    ok: bool = True
    error: Optional[str] = None
    latency_ms: Optional[float] = None
    counts: Dict[str, Any] = {}


class AgentInfo(BaseModel):
    used: bool = False
    status: Optional[str] = None
    subqueries: List[str] = []
    trajectory: List[AgentStepInfo] = []
    counts: Dict[str, Any] = {}
    degraded_to_pipeline: bool = False
    thread_id: Optional[str] = None
    ignored_reason: Optional[str] = None


class AskResponse(BaseModel):
    ...
    agent: Optional[AgentInfo] = None
```

- [ ] **Step 3: `/ask` 分支逻辑（插在 L4 miss 之后、现有 search 之前）**

```python
from src.agent.config import agent_config
from src.agent.runner import run_agent

acfg = agent_config()
if request.mode == "agent":
    if not acfg.get("enabled"):
        # 继续 pipeline，响应 agent=AgentInfo(used=False, ignored_reason="agent.enabled=false")
        pass
    else:
        def search_fn(q, k=request.k):
            return retriever.search(q, k=k, use_rerank=request.use_rerank)
        def complete_fn(prompt: str) -> str:
            # 与 Generator 同源 client
            return gen._complete(prompt)  # 若无私有方法，抽 public complete 或复用 client.chat
        def generate_fn(q, hits):
            return gen.answer(q, hits, k_context=request.k)

        def pipeline_fallback():
            # 现有 search+generate 最短路径，返回 dict
            ...

        result = run_agent(
            request.query,
            search_fn=search_fn,
            complete_fn=complete_fn,
            generate_fn=generate_fn,
            pipeline_fallback_fn=pipeline_fallback,
            trace_id=...,  # 从 tracer
        )
        return AskResponse(
            query=request.query,
            answer=result.answer,
            citations=...,
            retrieval_trace=...,  # 可从 evidence 合成精简 trace 或空壳
            agent=AgentInfo(
                used=True,
                status=result.status,
                subqueries=result.subqueries,
                trajectory=[AgentStepInfo(**t) for t in (result.trajectory or [])]
                    if acfg.get("return_trajectory") else [],
                counts=result.counts,
                degraded_to_pipeline=(result.status == "degraded"),
                thread_id=result.thread_id,
            ),
            context=result.context or "",
        )
```

**注意：**
- L4 key 构造必须包含 mode；在 `answer_cache_key` 调用处额外传入 mode 或拼进 query 旁路盐：`parts` 已有 `agent_cache_salt`；**mode=agent 请求**还应在 key 中加 `mode=agent`（即使 enabled 变化已在 salt 中）。建议 `answer_cache_key(..., mode="agent")` 扩展可选参数。
- `doc_id` 过滤：search_fn 内过滤，与 pipeline 一致。
- 评测强制 hitl off：runner 读 config 即可。

- [ ] **Step 4: `POST /ask/resume`**

```python
class AskResumeRequest(BaseModel):
    thread_id: str
    subqueries: Optional[List[str]] = None  # None = approve as-is
    k: int = 5
    use_rerank: bool = True

@app.post("/ask/resume", response_model=AskResponse)
async def ask_resume(request: AskResumeRequest):
    ...
```

仅当 `agent.enabled` 且 checkpoint 可用；否则 400。

- [ ] **Step 5: 跑测 commit**

```bash
.venv/bin/python -m pytest tests/test_agent_api.py tests/test_api.py -q
git add src/api/routes.py src/evaluation/vidore_adapter.py tests/test_agent_api.py
git commit -m "feat(api): /ask mode=agent and /ask/resume"
```

---

## Task 9: `@tool` 包装 + react_demo（学习对照）

**Files:**
- Modify: `src/agent/tools.py`（导出 StructuredTool / @tool 工厂）
- Create: `src/agent/react_demo.py`
- Create: `tests/test_agent_react_demo.py`

- [ ] **Step 1: 测试 react demo 图可编译且 max_steps 有限**

```python
def test_react_demo_compiles_and_stops():
    from src.agent.react_demo import build_react_demo_graph
    g = build_react_demo_graph(search_fn=lambda q, k=5: [], complete_fn=lambda p: "FINAL: idk")
    # invoke with low recursion_limit
    ...
```

- [ ] **Step 2: 实现**

用 `langchain_core.tools.tool` 包装 `knowledge_search`；`create_react_agent` **或** 手写 ToolNode 环（若 create_react_agent 依赖过重则手写）。  
配置 `agent.react_demo.enabled` 时 runner 可选走 demo 图；**默认 false，API 不默认暴露**。可加 `mode=agent_react` 仅 debug，或仅单元测试调用。

- [ ] **Step 3: Commit**

```bash
git commit -m "feat(agent): @tool wrappers and optional ReAct demo graph"
```

---

## Task 10: P1d — Demo UI + fixtures

**Files:**
- Modify: `static/demo/app.js` / `index.html` / `styles.css`
- Modify: `static/demo/fixtures.json`（增加 agent 假轨迹）
- Modify: `tests/test_demo_fixtures.py`

- [ ] **Step 1: UI**

- 顶栏或工程师面板：`Mode: Pipeline | Agent`  
- Live ask body 增加 `"mode": "agent"|"pipeline"`  
- 若 `response.agent` 存在：渲染 subqueries 列表 + trajectory 时间线（node/tool/latency）  
- Fixture 模式：在 `fixtures.json` 增加一条带 `agent` 字段的 response

- [ ] **Step 2: 契约测试扩展**

```python
def test_agent_fixture_has_trajectory():
    data = json.loads((DEMO / "fixtures.json").read_text(encoding="utf-8"))
    agent_resps = [r for r in data["responses"].values() if r.get("agent")]
    assert agent_resps, "need at least one fixture with agent trajectory"
    ag = agent_resps[0]["agent"]
    assert "subqueries" in ag and "trajectory" in ag
```

- [ ] **Step 3: Commit**

```bash
git commit -m "feat(demo): Pipeline|Agent mode and trajectory panel"
```

---

## Task 11: 评测入口薄封装 + 本机轻量真链路 + Phase2 骨架

**Files:**
- Create: `src/agent/eval.py`
- Create: `data/agent_eval_qa.json`（骨架 **先 5～10 条**，足够本机 smoke；Phase2 再扩到 40–50）
- Create: `scripts/run_agent_local_smoke.py`（本机 ≤10 条 pipeline vs agent）
- Create: `scripts/run_agent_eval.py`（云上双臂框架，后置）
- Modify: `handoff.md`（实现完成后更新状态）

### 本机轻量真链路（Task 11 必做脚本能力）

- [ ] **Step 0: 约定**

| 项 | 值 |
|----|-----|
| 默认条数 | **5**（CLI `--max-queries` 上限 **10**） |
| 配置 | `CONFIG_PROFILE=local-dev` + 已有 local-demo 索引 |
| 模型 | Ollama 等 **已 list 可见**；脚本检测缺失则 **exit 提示，禁止自动 pull** |
| 输出 | `runs/local-agent-smoke-<ts>/results.json` + 打印 counts/trajectory 摘要 |
| 失败策略 | 单条异常记 error 继续；总失败率写入 summary |

- [ ] **Step 1: `agent_answer_for_eval`**

```python
# src/agent/eval.py
def agent_answer_for_eval(query, *, retriever, generator, k_context=5, **kwargs) -> dict:
    """返回 {answer, citations, context, agent, ...}；供 E2E 子集使用。"""
    def search_fn(q, k=k_context):
        return retriever.search(q, k=k, **kwargs)
    def complete_fn(prompt):
        return generator.complete(prompt)  # 若无此方法则统一抽 Generator.complete
    def generate_fn(q, hits):
        return generator.answer(q, hits, k_context=k_context)
    res = run_agent(query, search_fn=search_fn, complete_fn=complete_fn, generate_fn=generate_fn,
                    cfg={**agent_config(), "hitl": {"review_subqueries": False}})
    return {
        "answer": res.answer,
        "citations": res.citations,
        "context": res.context,
        "agent": {
            "status": res.status,
            "subqueries": res.subqueries,
            "trajectory": res.trajectory,
            "counts": res.counts,
        },
    }
```

若 `Generator` 无 `complete`，本 Task **先加** 小方法：

```python
def complete(self, prompt: str) -> str:
    # 现有 chat completion 路径
```

- [ ] **Step 2: 数据骨架**

```json
{
  "version": 1,
  "items": [
    {
      "id": "mh_001",
      "tag": "multi_hop",
      "query": "...",
      "gold_answer": "...",
      "notes": "needs two evidence loci"
    },
    {
      "id": "at_001",
      "tag": "atomic",
      "query": "...",
      "gold_answer": "..."
    },
    {
      "id": "rj_001",
      "tag": "reject",
      "query": "What is the capital of Mars?",
      "gold_answer": null,
      "expect_reject": true
    }
  ]
}
```

- [ ] **Step 3: `run_agent_eval.py` 骨架**

- 读 JSON，对每条跑 pipeline（现有 e2e 生成路径）与 agent  
- 输出 `results.json`：per-item + 汇总 Correct 占位（可先字符串匹配/人工字段）  
- **不在本机跑全量**；文档写清云上命令

- [ ] **Step 4: Commit**

```bash
git commit -m "feat(agent): eval entry and agent_eval_qa skeleton"
```

---

## Task 12: 全量本地验收 + 文档

- [ ] **Step 1: 测试**

```bash
.venv/bin/python -m pytest tests/test_agent_*.py tests/test_api.py tests/test_demo_fixtures.py -q
```

Expected: 全绿。

- [ ] **Step 2: Feature Map 自检表（写入 PR 描述或 `docs/architecture/agent.md`）**

| 特性 | 验证命令/操作 | 状态 |
|------|----------------|------|
| StateGraph | `test_graph_compile_and_invoke_atomic` | ☐ |
| Conditional edges | `test_route_*` | ☐ |
| Cycles | grade disabled / refine 测例 | ☐ |
| Reducers | multi evidence len | ☐ |
| Send | `test_multi_uses_send_*` | ☐ |
| Subgraph | subgraphs 单测 | ☐ |
| @tool / ReAct demo | `test_react_demo_*` | ☐ |
| Checkpoint+HITL | `test_interrupt_and_resume` | ☐ |
| Stream | stream 单测 | ☐ |
| Mermaid | 文件存在 | ☐ |

- [ ] **Step 3: 更新 handoff.md**

记录：分支、默认仍 false、如何 `mode=agent` 试用、测试命令、Phase2 未做。

- [ ] **Step 4: 最终 commit**

```bash
git add docs/architecture/agent.md handoff.md
git commit -m "docs: agent architecture notes and handoff for LangGraph MVP"
```

---

## Phase2（云上 · 独立 checklist，不阻塞 Phase1 合入）

> 有 GPU 时再做；遵守 AGENTS.md：先查缓存、HF 镜像、禁止无预算下载。

- [ ] 扩 `data/agent_eval_qa.json` 至 ~40–50（multi_hop / atomic / reject）
- [ ] `scripts/cloud_agent_eval.sh`：`enabled` 环境开 agent 配置覆盖 + skip-index
- [ ] 双臂：pipeline vs agent（可选 grade off）
- [ ] 指标：Correct、误拒、latency、avg searches、degrade 次数
- [ ] 产物：`runs/YYYYMMDD-agent-eval/` + README 决议
- [ ] **仅 Go 后** 另 PR 讨论是否启发式进 agent；**默认 enabled 仍建议 false**

---

## 实现顺序与 PR 切分建议

| PR | Tasks | 可合入条件 |
|----|-------|------------|
| PR1 | 0–2 | 依赖 + config + state |
| PR2 | 3–5 | tools + graph 顺序版 + runner + salt |
| PR3 | 6–7 | Send/subgraph + checkpoint/HITL/stream |
| PR4 | 8 | API |
| PR5 | 9–10 | react_demo + demo UI |
| PR6 | 11–12 | eval 骨架 + 文档 |

每 PR：`enabled` 保持 false；pipeline 回归绿。

---

## Spec 覆盖自检

| Spec 章节 | 对应 Task |
|-----------|-----------|
| §1 目标/非目标 | 全文纪律 + Task 8 默认 null |
| §2 拓扑/状态/护栏 | Task 2, 4, 6 |
| §3 工具契约 | Task 3 |
| §4 API/配置/Trace/运行时 | Task 1, 5, 7, 8 |
| §5 模块布局 | File map |
| §6 Feature Map | Task 4–7, 9, 12 |
| §7 评测 Phase | Task 11 + Phase2 |
| HITL 默认关 | Task 7, 8 |
| ReAct 非默认 | Task 9 |
| L4 cache salt | Task 1, 5, 8 |
| 不碰 NDCG | 无 Task 改 run_eval 默认 |

**已知有意延后（spec 允许）:** 持久化 checkpoint 跨进程、SSE 生产端点（demo stream 即可）、Gate2 挂接、子问依赖链。

---

## 风险提醒（实现时）

1. **LangGraph API 漂移**（Send / interrupt / Command）：以安装版 docstring 为准，单测锁行为。  
2. **`Generator.complete` 抽取**：避免 agent 复制一份 chat 调用。  
3. **evidence reducer + Send**：worker 返回的 `evidence`/`trajectory` 必须是「增量列表」，勿回传全量 state 导致重复。  
4. **CRAG 教训**：grade 可关；Phase2 再定默认。  
5. **禁止** 在本机 `uv add` 触发无关大模型下载；仅装 langgraph 栈。
