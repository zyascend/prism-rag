# 最小 ReACT Agent + Demo UI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 交付可面试投屏的「最小 ReACT Agent + 单页 Demo」：用户提问后可见 Thought/Action/Observation 轨迹、三路检索贡献、最终答案与引用；**不**做 GraphRAG / Neo4j / 多租户 / 生产级前端。

**Architecture:** Agent 是薄编排层，只调用既有 `PrismRAGRetriever.search_with_trace` 与 `Generator.answer`，不改检索核心。循环用 **OpenAI tool-calling** 实现可靠动作选择，同时把每步映射为 ReACT 轨迹字段供 UI 展示。Demo 为 `ui/` 静态页，由 FastAPI `StaticFiles` 挂载，零 Node 构建。

**Tech Stack:** Python 3.11、FastAPI、OpenAI SDK（已有）、既有 retrieval/generation、纯 HTML/CSS/JS Demo、pytest。

**面试成功标准（DoD）:**
1. `POST /agent/ask` 返回 `answer` + `steps[]`（每步含 thought/action/observation）+ `citations` + 末次 `retrieval_trace`
2. 浏览器打开 Demo：输入问题 → 动画式展示轨迹 → 答案+引用卡片 → 三路 top 列表
3. `make demo` 一条命令起服务（local-dev profile，可无 Visual）
4. 单测覆盖 Agent 循环（mock LLM + mock search），不依赖 GPU/全量索引
5. 明确 **不做**：`web_search`、图谱工具、Self-RAG 双门、React/Vite 构建链

**工期感:** 约 4–6 个有效工作日（Agent 2–3 天 + Demo 1–2 天 + 胶水/话术 0.5 天）。

**分支:** `feat/minimal-react-demo`（禁止直接在 `main` 改）

---

## 0. 范围与产品契约

### 0.1 工具集（刻意收窄）

| 工具名 | 作用 | 底层 |
|--------|------|------|
| `knowledge_search` | 私有库检索，返回 top chunks 摘要 + 三路 trace | `PrismRAGRetriever.search_with_trace` |
| `refine_query` | 根据已有 observation 改写下一跳查询（无多轮对话状态机，仅 Agent 内） | LLM 短调用（同一 client） |
| `finish` | 基于累计证据生成最终答案并结束 | `Generator.answer` |

**禁止工具:** `web_search`、任何外网检索、任意 shell/code 执行。

### 0.2 循环护栏

| 参数 | 默认 | 说明 |
|------|------|------|
| `max_steps` | 4 | Thought-Action 轮次上限（含 finish） |
| `max_searches` | 2 | `knowledge_search` 最多 2 次，防死循环 |
| `k` | 5 | 检索 top-k |

超限未 `finish` → 用最后一次检索结果强制 `Generator.answer`（graceful degrade），`status="max_steps"`。

### 0.3 API 契约

```http
POST /agent/ask
Content-Type: application/json

{
  "query": "What is the max water pressure for manual rinsing?",
  "k": 5,
  "max_steps": 4,
  "doc_id": null
}
```

```json
{
  "query": "...",
  "answer": "...",
  "status": "finished | max_steps | abstained",
  "steps": [
    {
      "step": 1,
      "thought": "Need exact PSI from TO manual...",
      "action": "knowledge_search",
      "action_input": {"query": "..."},
      "observation": "Found 5 chunks. Top: [page 12] max 175 PSI..."
    }
  ],
  "citations": [
    {"chunk_id": "...", "page_id": 12, "doc_id": "...", "page_number": 12, "snippet": "..."}
  ],
  "retrieval_trace": {
    "bm25_top5": [{"chunk_id": "...", "page_id": 1, "score": 0.9}],
    "dense_top5": [],
    "visual_top5": []
  },
  "num_searches": 1,
  "latency_ms": 1234
}
```

### 0.4 文件地图

| 路径 | 职责 |
|------|------|
| `src/agent/__init__.py` | 导出 `ReactAgent`, `AgentResult` |
| `src/agent/models.py` | `AgentStep`, `AgentResult` dataclass |
| `src/agent/tools.py` | 工具 schema + `ToolContext` 执行器 |
| `src/agent/react_agent.py` | 主循环 |
| `src/prompts/prompts/react_system.yaml` | Agent system prompt（版本化） |
| `src/api/routes.py` | `POST /agent/ask` + 挂载 `ui/` |
| `ui/index.html` | Demo 单页 |
| `ui/app.js` | 调用 API、渲染轨迹 |
| `ui/styles.css` | 深色技术风（面试投屏可读） |
| `tests/test_react_agent.py` | Agent 单测（全 mock） |
| `tests/test_agent_api.py` | FastAPI TestClient 路由测 |
| `scripts/run_demo.py` | 本地起 API + 打印 Demo URL |
| `Makefile` | `demo` target |
| `docs/interview-demo.md` | 投屏脚本 + 示例问题 |

---

## Task 1: Agent 数据模型 + 空壳

**Files:**
- Create: `src/agent/models.py`
- Create: `src/agent/__init__.py`
- Create: `tests/test_react_agent.py`（先写模型序列化测）

- [ ] **Step 1: 写失败测试（模型 to_dict）**

```python
# tests/test_react_agent.py
from src.agent.models import AgentStep, AgentResult


def test_agent_step_to_dict():
    step = AgentStep(
        step=1,
        thought="Need PSI limit",
        action="knowledge_search",
        action_input={"query": "manual rinsing PSI"},
        observation="top chunk mentions 175 PSI",
    )
    d = step.to_dict()
    assert d["step"] == 1
    assert d["action"] == "knowledge_search"
    assert d["action_input"]["query"] == "manual rinsing PSI"


def test_agent_result_to_dict_includes_steps():
    result = AgentResult(
        query="q",
        answer="175 PSI",
        status="finished",
        steps=[],
        citations=[{"chunk_id": "c1", "page_id": 1, "snippet": "175"}],
        retrieval_trace={"bm25_top5": [], "dense_top5": [], "visual_top5": []},
        num_searches=1,
        latency_ms=10,
    )
    d = result.to_dict()
    assert d["status"] == "finished"
    assert d["answer"] == "175 PSI"
    assert "retrieval_trace" in d
```

- [ ] **Step 2: 跑测确认失败**

```bash
cd /Users/theyang/Documents/ai/pdf-rag
source .venv/bin/activate  # 或 project venv
pytest tests/test_react_agent.py -v
```

Expected: `ModuleNotFoundError` or import error for `src.agent.models`

- [ ] **Step 3: 实现 models**

```python
# src/agent/models.py
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


@dataclass
class AgentStep:
    step: int
    thought: str
    action: str
    action_input: Dict[str, Any]
    observation: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step": self.step,
            "thought": self.thought,
            "action": self.action,
            "action_input": self.action_input,
            "observation": self.observation,
        }


@dataclass
class AgentResult:
    query: str
    answer: str
    status: str  # finished | max_steps | abstained
    steps: List[AgentStep] = field(default_factory=list)
    citations: List[Dict[str, Any]] = field(default_factory=list)
    retrieval_trace: Dict[str, Any] = field(
        default_factory=lambda: {"bm25_top5": [], "dense_top5": [], "visual_top5": []}
    )
    num_searches: int = 0
    latency_ms: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "answer": self.answer,
            "status": self.status,
            "steps": [s.to_dict() for s in self.steps],
            "citations": self.citations,
            "retrieval_trace": self.retrieval_trace,
            "num_searches": self.num_searches,
            "latency_ms": self.latency_ms,
        }
```

```python
# src/agent/__init__.py
from src.agent.models import AgentResult, AgentStep

__all__ = ["AgentResult", "AgentStep"]
```

- [ ] **Step 4: 跑测通过**

```bash
pytest tests/test_react_agent.py::test_agent_step_to_dict tests/test_react_agent.py::test_agent_result_to_dict_includes_steps -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git checkout -b feat/minimal-react-demo  # if not already
git add src/agent/models.py src/agent/__init__.py tests/test_react_agent.py
git commit -m "feat(agent): add AgentStep/AgentResult models for ReACT trajectory"
```

---

## Task 2: 工具层 `tools.py`（mock 可测）

**Files:**
- Create: `src/agent/tools.py`
- Modify: `tests/test_react_agent.py`

### 设计要点

- `ToolContext` 持有 `retriever`, `generator`, 以及运行时状态：`evidence: list[dict]`, `last_trace`, `search_count`
- `knowledge_search` 把 results **append** 到 evidence（去重 by chunk_id），observation 截断文本（每 chunk ≤180 字，最多 5 条）
- `refine_query` 用 client 做一次短 chat（可注入 mock client）
- `finish` 调 `generator.answer(original_query, evidence, k_context=k)`；若 evidence 空则 abstain 话术

OpenAI tools schema（供 Agent 传 `tools=`）:

```python
TOOL_SCHEMAS = [
  {
    "type": "function",
    "function": {
      "name": "knowledge_search",
      "description": "Search the private industrial PDF knowledge base. Use for factual lookups.",
      "parameters": {
        "type": "object",
        "properties": {
          "query": {"type": "string", "description": "Search query, English preferred"}
        },
        "required": ["query"],
      },
    },
  },
  {
    "type": "function",
    "function": {
      "name": "refine_query",
      "description": "Rewrite the search query to be more specific given prior observations.",
      "parameters": {
        "type": "object",
        "properties": {
          "reason": {"type": "string"},
          "new_query": {"type": "string"},
        },
        "required": ["new_query"],
      },
    },
  },
  {
    "type": "function",
    "function": {
      "name": "finish",
      "description": "Produce the final answer from gathered evidence and stop.",
      "parameters": {
        "type": "object",
        "properties": {
          "reason": {"type": "string", "description": "Why evidence is sufficient or not"}
        },
        "required": [],
      },
    },
  },
]
```

- [ ] **Step 1: 写失败测试**

```python
# tests/test_react_agent.py (append)
from unittest.mock import MagicMock
from src.agent.tools import ToolContext, execute_tool, TOOL_SCHEMAS


def test_tool_schemas_include_three_tools():
    names = {t["function"]["name"] for t in TOOL_SCHEMAS}
    assert names == {"knowledge_search", "refine_query", "finish"}


def test_knowledge_search_appends_evidence_and_dedupes():
    retriever = MagicMock()
    retriever.search_with_trace.return_value = {
        "results": [
            {
                "chunk_id": "c1",
                "page_id": 1,
                "doc_id": "d1",
                "page_number": 1,
                "text": "Maximum water pressure is 175 PSI for manual rinsing.",
                "chunk_type": "text",
                "score": 0.9,
                "retrieval_type": "bm25",
            },
            {
                "chunk_id": "c1",  # dup
                "page_id": 1,
                "doc_id": "d1",
                "page_number": 1,
                "text": "Maximum water pressure is 175 PSI for manual rinsing.",
                "chunk_type": "text",
                "score": 0.8,
                "retrieval_type": "dense",
            },
        ],
        "retrieval_trace": {
            "bm25_top5": [{"chunk_id": "c1", "page_id": 1, "score": 0.9}],
            "dense_top5": [],
            "visual_top5": [],
        },
    }
    ctx = ToolContext(retriever=retriever, generator=MagicMock(), k=5)
    obs = execute_tool(ctx, "knowledge_search", {"query": "rinsing PSI"})
    assert ctx.search_count == 1
    assert len(ctx.evidence) == 1
    assert "175 PSI" in obs
    assert ctx.last_trace["bm25_top5"]


def test_knowledge_search_respects_max_searches():
    retriever = MagicMock()
    retriever.search_with_trace.return_value = {
        "results": [],
        "retrieval_trace": {"bm25_top5": [], "dense_top5": [], "visual_top5": []},
    }
    ctx = ToolContext(retriever=retriever, generator=MagicMock(), k=5, max_searches=1)
    execute_tool(ctx, "knowledge_search", {"query": "a"})
    obs = execute_tool(ctx, "knowledge_search", {"query": "b"})
    assert "max_searches" in obs.lower() or "limit" in obs.lower()
    assert retriever.search_with_trace.call_count == 1


def test_finish_with_evidence_calls_generator():
    gen = MagicMock()
    gen.answer.return_value = {
        "answer": "175 PSI",
        "citations": [{"chunk_id": "c1", "page_id": 1, "snippet": "175"}],
        "context": "ctx",
    }
    ctx = ToolContext(retriever=MagicMock(), generator=gen, k=5)
    ctx.evidence = [{
        "chunk_id": "c1", "page_id": 1, "doc_id": "d", "page_number": 1,
        "text": "175 PSI", "chunk_type": "text", "score": 1.0, "retrieval_type": "bm25",
    }]
    obs = execute_tool(ctx, "finish", {}, original_query="what PSI?")
    assert ctx.final_answer == "175 PSI"
    assert ctx.citations[0]["chunk_id"] == "c1"
    gen.answer.assert_called_once()


def test_finish_without_evidence_abstains():
    gen = MagicMock()
    ctx = ToolContext(retriever=MagicMock(), generator=gen, k=5)
    execute_tool(ctx, "finish", {}, original_query="what PSI?")
    assert "enough information" in ctx.final_answer.lower() or "cannot" in ctx.final_answer.lower()
    gen.answer.assert_not_called()
```

- [ ] **Step 2: 跑测确认失败**

```bash
pytest tests/test_react_agent.py -k "tool_ or knowledge_ or finish_" -v
```

Expected: import / attribute errors

- [ ] **Step 3: 实现 `src/agent/tools.py`**

实现要求（完整代码在实现时按下列行为写，勿省略边界）:

```python
# src/agent/tools.py — 关键结构
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

TOOL_SCHEMAS = [ ... ]  # 见上

@dataclass
class ToolContext:
    retriever: Any
    generator: Any
    k: int = 5
    max_searches: int = 2
    use_rerank: bool = True
    use_visual: Optional[bool] = None  # None → 读 cfg
    doc_id: Optional[str] = None
    evidence: List[Dict[str, Any]] = field(default_factory=list)
    last_trace: Dict[str, Any] = field(
        default_factory=lambda: {"bm25_top5": [], "dense_top5": [], "visual_top5": []}
    )
    search_count: int = 0
    final_answer: Optional[str] = None
    citations: List[Dict[str, Any]] = field(default_factory=list)
    finished: bool = False
    abstained: bool = False

def _snippet(text: str, n: int = 180) -> str:
    text = (text or "").replace("\n", " ").strip()
    return text if len(text) <= n else text[: n - 3] + "..."

def execute_tool(
    ctx: ToolContext,
    name: str,
    args: Dict[str, Any],
    *,
    original_query: str = "",
) -> str:
    if name == "knowledge_search":
        return _knowledge_search(ctx, args.get("query") or original_query)
    if name == "refine_query":
        new_q = (args.get("new_query") or "").strip()
        reason = args.get("reason") or ""
        if not new_q:
            return "refine_query failed: new_query is empty"
        return f"Refined query ready: {new_q}" + (f" (reason: {reason})" if reason else "")
    if name == "finish":
        return _finish(ctx, original_query)
    return f"Unknown tool: {name}"

def _knowledge_search(ctx: ToolContext, query: str) -> str:
    if ctx.search_count >= ctx.max_searches:
        return f"Search limit reached (max_searches={ctx.max_searches}). Call finish with current evidence."
    # use_visual: if ctx.use_visual is None, pass through retriever default via kwargs only if supported
    result = ctx.retriever.search_with_trace(
        query=query, k=ctx.k, use_rerank=ctx.use_rerank,
    )
    results = result["results"]
    if ctx.doc_id:
        results = [r for r in results if r.get("doc_id") == ctx.doc_id]
    ctx.last_trace = result.get("retrieval_trace") or ctx.last_trace
    ctx.search_count += 1
    seen = {e["chunk_id"] for e in ctx.evidence}
    added = 0
    for r in results:
        cid = r.get("chunk_id")
        if not cid or cid in seen:
            continue
        ctx.evidence.append(r)
        seen.add(cid)
        added += 1
    if not results:
        return "No results found. Consider refine_query or finish (abstain)."
    lines = [f"Found {len(results)} chunks ({added} new). Top:"]
    for r in results[:5]:
        lines.append(
            f"- [{r.get('chunk_id')}] page={r.get('page_id')} "
            f"score={r.get('score', 0):.3f}: {_snippet(r.get('text', ''))}"
        )
    return "\n".join(lines)

def _finish(ctx: ToolContext, original_query: str) -> str:
    ctx.finished = True
    if not ctx.evidence:
        ctx.abstained = True
        ctx.final_answer = "I don't have enough information to answer that question."
        ctx.citations = []
        return "Finished with abstain (no evidence)."
    out = ctx.generator.answer(original_query, ctx.evidence, k_context=ctx.k)
    ctx.final_answer = out["answer"]
    ctx.citations = out.get("citations") or []
    return f"Finished. Answer length={len(ctx.final_answer or '')}."
```

- [ ] **Step 4: 跑测通过**

```bash
pytest tests/test_react_agent.py -v
```

Expected: 全部 PASS

- [ ] **Step 5: Commit**

```bash
git add src/agent/tools.py tests/test_react_agent.py
git commit -m "feat(agent): tool executors for knowledge_search/refine_query/finish"
```

---

## Task 3: ReACT 主循环 `react_agent.py`

**Files:**
- Create: `src/agent/react_agent.py`
- Create: `src/prompts/prompts/react_system.yaml`
- Modify: `src/agent/__init__.py`
- Modify: `tests/test_react_agent.py`

### 循环伪代码

```
messages = [system, user(query)]
for step in 1..max_steps:
  resp = client.chat.completions.create(model, messages, tools=TOOL_SCHEMAS, tool_choice=auto)
  msg = resp.choices[0].message
  if not msg.tool_calls:
    # 模型直接文本结束：把文本当 answer，status=finished
    break
  for each tool_call (本最小实现：每轮只处理第一个 tool_call):
    thought = msg.content or f"Calling {name}"
    observation = execute_tool(...)
    append step to steps
    append assistant tool_call + tool result to messages
    if finish: return AgentResult
force finish with evidence
```

### Prompt YAML

```yaml
# src/prompts/prompts/react_system.yaml
id: react_system
description: Minimal ReACT agent system prompt for private PDF QA
versions:
  - version: 1
    created_at: "2026-07-19"
    author: yang
    changelog: "Initial interview-demo ReACT prompt"
    active: true
    system: |
      You are a ReACT agent for industrial PDF question answering over a PRIVATE knowledge base.

      Tools:
      - knowledge_search: retrieve evidence from the PDF index
      - refine_query: rewrite the search query when results are weak
      - finish: answer from gathered evidence or abstain if insufficient

      Rules:
      1. Never invent facts not present in tool observations.
      2. Prefer at least one knowledge_search before finish, unless the question is clearly unanswerable.
      3. At most 2 searches. If still weak, finish and abstain honestly.
      4. Do not claim access to the public internet.
      5. When finishing, rely on evidence; the finish tool will generate the user-facing answer.
    user: |
      Question: {query}
```

> 若 `PromptRegistry` 要求固定字段与其它 yaml 一致，对齐 `answer_generation.yaml` 的 schema；`get_active("react_system").system` 必须可用。`user` 模板用 `render("user", query=...)`。

### Mock LLM 策略（单测）

用 `MagicMock` client：

**场景 A — 一跳成功:**
1. 第一次 `create` 返回 tool_call `knowledge_search`
2. 第二次 `create` 返回 tool_call `finish`

**场景 B — 超限强制 finish:**
1. 每次返回 `knowledge_search` 直到 max_steps
2. Agent 应强制 finish

**场景 C — 无 tool_call 直接文本:**
1. 返回 content="I cannot answer..." 无 tool_calls → status finished，answer 用 content

- [ ] **Step 1: 写失败测试**

```python
import json
from unittest.mock import MagicMock, patch
from src.agent.react_agent import ReactAgent


def _msg(content=None, tool_calls=None):
    m = MagicMock()
    m.content = content
    m.tool_calls = tool_calls
    return m


def _tool_call(name: str, args: dict, id: str = "call_1"):
    tc = MagicMock()
    tc.id = id
    tc.function.name = name
    tc.function.arguments = json.dumps(args)
    return tc


def test_react_one_search_then_finish():
    retriever = MagicMock()
    retriever.search_with_trace.return_value = {
        "results": [{
            "chunk_id": "c1", "page_id": 1, "doc_id": "d", "page_number": 1,
            "text": "Maximum water pressure is 175 PSI.",
            "chunk_type": "text", "score": 0.9, "retrieval_type": "bm25",
        }],
        "retrieval_trace": {
            "bm25_top5": [{"chunk_id": "c1", "page_id": 1, "score": 0.9}],
            "dense_top5": [], "visual_top5": [],
        },
    }
    generator = MagicMock()
    generator.answer.return_value = {
        "answer": "The maximum is 175 PSI.",
        "citations": [{"chunk_id": "c1", "page_id": 1, "snippet": "175 PSI"}],
        "context": "Maximum water pressure is 175 PSI.",
    }

    client = MagicMock()
    client.chat.completions.create.side_effect = [
        MagicMock(choices=[MagicMock(message=_msg(
            content="I should search the manuals.",
            tool_calls=[_tool_call("knowledge_search", {"query": "manual rinsing PSI"})],
        ))]),
        MagicMock(choices=[MagicMock(message=_msg(
            content="Evidence is sufficient.",
            tool_calls=[_tool_call("finish", {"reason": "found PSI"})],
        ))]),
    ]

    agent = ReactAgent(
        retriever=retriever, generator=generator, client=client, model="test-model",
        max_steps=4, max_searches=2, k=5,
    )
    result = agent.run("What is max manual rinsing PSI?")
    assert result.status == "finished"
    assert "175" in result.answer
    assert result.num_searches == 1
    assert len(result.steps) == 2
    assert result.steps[0].action == "knowledge_search"
    assert result.steps[1].action == "finish"
    assert result.retrieval_trace["bm25_top5"]


def test_react_max_steps_force_finish():
    retriever = MagicMock()
    retriever.search_with_trace.return_value = {
        "results": [{
            "chunk_id": "c1", "page_id": 1, "doc_id": "d", "page_number": 1,
            "text": "some text", "chunk_type": "text", "score": 0.5, "retrieval_type": "bm25",
        }],
        "retrieval_trace": {"bm25_top5": [], "dense_top5": [], "visual_top5": []},
    }
    generator = MagicMock()
    generator.answer.return_value = {
        "answer": "forced", "citations": [], "context": "some text",
    }
    client = MagicMock()
    # Always search — agent must stop
    client.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=_msg(
            content="search more",
            tool_calls=[_tool_call("knowledge_search", {"query": "x"}, id="c")],
        ))]
    )
    agent = ReactAgent(
        retriever=retriever, generator=generator, client=client, model="m",
        max_steps=2, max_searches=5, k=5,
    )
    result = agent.run("q")
    assert result.status == "max_steps"
    assert result.answer == "forced"
    generator.answer.assert_called()
```

- [ ] **Step 2: 跑测失败**

```bash
pytest tests/test_react_agent.py::test_react_one_search_then_finish -v
```

Expected: cannot import ReactAgent

- [ ] **Step 3: 实现 `ReactAgent`**

```python
# src/agent/react_agent.py — 核心骨架（实现时补全 import / 错误处理）
from __future__ import annotations
import json
import time
from typing import Any, Optional

from src.agent.models import AgentResult, AgentStep
from src.agent.tools import TOOL_SCHEMAS, ToolContext, execute_tool
from src.prompts import get_active


class ReactAgent:
    def __init__(
        self,
        retriever,
        generator,
        client=None,
        model: Optional[str] = None,
        max_steps: int = 4,
        max_searches: int = 2,
        k: int = 5,
        use_rerank: bool = True,
        doc_id: Optional[str] = None,
    ):
        if client is None:
            import os
            from openai import OpenAI
            from src.config import cfg
            client = OpenAI(
                base_url=cfg.get("llm.base_url", "https://api.openai.com/v1"),
                api_key=cfg.get("llm.api_key", "") or os.environ.get("OPENAI_API_KEY", ""),
            )
            model = model or cfg.get("llm.model", "gpt-4o-mini")
        self.client = client
        self.model = model or "gpt-4o-mini"
        self.retriever = retriever
        self.generator = generator
        self.max_steps = max_steps
        self.max_searches = max_searches
        self.k = k
        self.use_rerank = use_rerank
        self.doc_id = doc_id

    def run(self, query: str) -> AgentResult:
        t0 = time.perf_counter()
        pv = get_active("react_system")
        system = pv.system
        user = pv.render("user", query=query) if hasattr(pv, "render") else f"Question: {query}"
        # 若 PromptVersion API 与假设不一致，对齐 src/prompts/models.py 的实际接口

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        ctx = ToolContext(
            retriever=self.retriever,
            generator=self.generator,
            k=self.k,
            max_searches=self.max_searches,
            use_rerank=self.use_rerank,
            doc_id=self.doc_id,
        )
        steps: list[AgentStep] = []

        for i in range(1, self.max_steps + 1):
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=TOOL_SCHEMAS,
                temperature=0,
            )
            msg = resp.choices[0].message
            tool_calls = getattr(msg, "tool_calls", None) or []

            if not tool_calls:
                text = (msg.content or "").strip()
                steps.append(AgentStep(
                    step=i, thought=text or "No tool call",
                    action="respond", action_input={},
                    observation=text,
                ))
                answer = text or "I don't have enough information to answer that question."
                return AgentResult(
                    query=query, answer=answer, status="finished", steps=steps,
                    citations=ctx.citations, retrieval_trace=ctx.last_trace,
                    num_searches=ctx.search_count,
                    latency_ms=int((time.perf_counter() - t0) * 1000),
                )

            # 最小实现：每轮只执行第一个 tool_call
            tc = tool_calls[0]
            name = tc.function.name
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            thought = (msg.content or "").strip() or f"Action: {name}"
            observation = execute_tool(ctx, name, args, original_query=query)
            steps.append(AgentStep(
                step=i, thought=thought, action=name,
                action_input=args, observation=observation,
            ))

            # 维护多轮 tool 对话
            messages.append({
                "role": "assistant",
                "content": msg.content,
                "tool_calls": [{
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": name, "arguments": tc.function.arguments or "{}"},
                }],
            })
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": observation,
            })

            if ctx.finished:
                status = "abstained" if ctx.abstained else "finished"
                return AgentResult(
                    query=query,
                    answer=ctx.final_answer or "",
                    status=status,
                    steps=steps,
                    citations=ctx.citations,
                    retrieval_trace=ctx.last_trace,
                    num_searches=ctx.search_count,
                    latency_ms=int((time.perf_counter() - t0) * 1000),
                )

        # max_steps: force finish
        if not ctx.finished:
            execute_tool(ctx, "finish", {"reason": "max_steps"}, original_query=query)
            steps.append(AgentStep(
                step=len(steps) + 1,
                thought="Reached max_steps; forcing finish.",
                action="finish",
                action_input={"reason": "max_steps"},
                observation="forced finish",
            ))
        return AgentResult(
            query=query,
            answer=ctx.final_answer or "I don't have enough information to answer that question.",
            status="max_steps",
            steps=steps,
            citations=ctx.citations,
            retrieval_trace=ctx.last_trace,
            num_searches=ctx.search_count,
            latency_ms=int((time.perf_counter() - t0) * 1000),
        )
```

**PromptRegistry 注意:** 新增 yaml 后确认 `src/prompts/registry.py` 启动扫描能加载 `react_system`。若 registry 是 lazy 扫目录，无需改代码；若白名单，把 `react_system` 加进去。加一个小测：

```python
def test_react_prompt_registered():
    from src.prompts import get_active
    pv = get_active("react_system")
    assert pv.system
    assert "knowledge_search" in pv.system
```

- [ ] **Step 4: 跑测通过**

```bash
pytest tests/test_react_agent.py -v
```

Expected: PASS（含 prompt 注册）

- [ ] **Step 5: 更新 `__init__.py` 导出并 commit**

```python
# src/agent/__init__.py
from src.agent.models import AgentResult, AgentStep
from src.agent.react_agent import ReactAgent

__all__ = ["AgentResult", "AgentStep", "ReactAgent"]
```

```bash
git add src/agent/react_agent.py src/agent/__init__.py src/prompts/prompts/react_system.yaml tests/test_react_agent.py
git commit -m "feat(agent): minimal ReACT loop with tool-calling and trajectory"
```

---

## Task 4: API `POST /agent/ask` + 静态 Demo 挂载

**Files:**
- Modify: `src/api/routes.py`
- Create: `tests/test_agent_api.py`

- [ ] **Step 1: 写 API 失败测试**

```python
# tests/test_agent_api.py
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

from src.api.routes import app, set_retriever, set_generator
from src.agent.models import AgentResult, AgentStep


def test_agent_ask_returns_trajectory(monkeypatch):
    fake = AgentResult(
        query="q",
        answer="175 PSI",
        status="finished",
        steps=[AgentStep(
            step=1, thought="search", action="knowledge_search",
            action_input={"query": "q"}, observation="175 PSI",
        )],
        citations=[{"chunk_id": "c1", "page_id": 1, "snippet": "175"}],
        retrieval_trace={"bm25_top5": [], "dense_top5": [], "visual_top5": []},
        num_searches=1,
        latency_ms=42,
    )

    class FakeAgent:
        def __init__(self, *a, **k):
            pass
        def run(self, query: str):
            assert query == "What PSI?"
            return fake

    monkeypatch.setattr("src.api.routes.ReactAgent", FakeAgent)
    set_retriever(MagicMock())
    set_generator(MagicMock())

    client = TestClient(app)
    r = client.post("/agent/ask", json={"query": "What PSI?", "k": 5})
    assert r.status_code == 200
    body = r.json()
    assert body["answer"] == "175 PSI"
    assert body["steps"][0]["action"] == "knowledge_search"
    assert body["num_searches"] == 1


def test_agent_ask_requires_query():
    client = TestClient(app)
    r = client.post("/agent/ask", json={})
    assert r.status_code == 422
```

- [ ] **Step 2: 跑测失败**

```bash
pytest tests/test_agent_api.py -v
```

Expected: 404 on `/agent/ask` or validation

- [ ] **Step 3: 在 `routes.py` 增加端点与静态资源**

在文件合适位置（imports + 模型 + 路由）加入：

```python
from pathlib import Path
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from src.agent import ReactAgent

class AgentAskRequest(BaseModel):
    query: str
    k: int = 5
    max_steps: int = 4
    max_searches: int = 2
    use_rerank: bool = True
    doc_id: Optional[str] = None

class AgentStepOut(BaseModel):
    step: int
    thought: str
    action: str
    action_input: dict
    observation: str

class AgentAskResponse(BaseModel):
    query: str
    answer: str
    status: str
    steps: List[AgentStepOut]
    citations: List[Citation] = []
    retrieval_trace: RetrievalTrace = RetrievalTrace()
    num_searches: int = 0
    latency_ms: int = 0

@app.post("/agent/ask", response_model=AgentAskResponse)
async def agent_ask(request: AgentAskRequest):
    retriever = get_retriever()
    gen = get_generator(retriever.bge)
    agent = ReactAgent(
        retriever=retriever,
        generator=gen,
        max_steps=request.max_steps,
        max_searches=request.max_searches,
        k=request.k,
        use_rerank=request.use_rerank,
        doc_id=request.doc_id,
    )
    try:
        result = agent.run(request.query)
    except Exception as e:
        logger.error(f"agent error: {e}")
        raise HTTPHTTPException(status_code=500, detail="Agent failed")
    # 映射 result.to_dict() → AgentAskResponse
    ...
```

**静态 UI 挂载（放在所有 API 路由定义之后，避免遮蔽）:**

```python
UI_DIR = Path(__file__).resolve().parents[2] / "ui"
if UI_DIR.is_dir():
    @app.get("/")
    async def demo_index():
        return FileResponse(UI_DIR / "index.html")
    app.mount("/ui", StaticFiles(directory=str(UI_DIR), html=True), name="ui")
```

注意：`HTTPHTTPException` 是笔误，实现时用已有的 `HTTPException`。

- [ ] **Step 4: 跑测通过**

```bash
pytest tests/test_agent_api.py tests/test_react_agent.py -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/api/routes.py tests/test_agent_api.py
git commit -m "feat(api): POST /agent/ask and static demo mount"
```

---

## Task 5: Demo UI（单页，面试投屏）

**Files:**
- Create: `ui/index.html`
- Create: `ui/styles.css`
- Create: `ui/app.js`

### UI 布局（必须全部有）

1. **顶栏:** PrismRAG Demo · ReACT Agent
2. **输入区:** query textarea +「Ask」按钮 +「Direct /ask」切换（可选，默认 Agent）
3. **轨迹区:** 时间线 steps — 每步显示 Thought / Action pill / Observation（等宽字体、可折叠长 observation）
4. **答案区:** 大号 answer + status badge
5. **引用区:** citations 卡片列表（page / snippet）
6. **三路 Trace:** 三列 BM25 / Dense / Visual（chunk_id + score）
7. **空态 / loading / error** 三种状态

### 视觉

- 深色背景（`#0b0f14`）、高对比正文、accent 青/琥珀
- 字号投屏友好：正文 ≥15px，标题 ≥20px
- 不引入构建工具；原生 fetch

### `app.js` 核心

```javascript
const API = window.location.origin; // same host as FastAPI

async function runAgent(query) {
  setLoading(true);
  try {
    const res = await fetch(`${API}/agent/ask`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, k: 5, max_steps: 4 }),
    });
    if (!res.ok) throw new Error(await res.text());
    const data = await res.json();
    renderResult(data);
  } catch (e) {
    showError(e.message);
  } finally {
    setLoading(false);
  }
}
```

`renderResult` 按 step 顺序 DOM 插入，可用 `requestAnimationFrame` 或 150ms stagger 做轻动画（面试「看得见思考」）。

示例预设问题按钮（3 个）:
1. `What is the maximum water pressure for manual rinsing of aircraft?`
2. （域外）`What is the capital of France?` → 期望 abstain / 拒答
3. 若本地 sample PDF 不同，在 `docs/interview-demo.md` 写真实可答问题

- [ ] **Step 1: 创建三文件最小可用版**（先静态 mock 数据渲染，再接 API）

可在 `app.js` 顶部留：

```javascript
const USE_MOCK = new URLSearchParams(location.search).has("mock");
```

`?mock=1` 时用内置假轨迹，**无后端也能投屏讲 UI**（面试救命开关）。

- [ ] **Step 2: 本地手测**

```bash
# terminal 1 — 若索引/PG 不齐，至少用 mock
python -c "from fastapi.testclient import TestClient; from src.api.routes import app; c=TestClient(app); print(c.get('/').status_code)"
```

或:

```bash
CONFIG_PROFILE=local-dev uvicorn src.api.routes:app --reload --port 8000
# 浏览器打开 http://127.0.0.1:8000/?mock=1
```

Expected: 页面渲染 mock 轨迹

- [ ] **Step 3: Commit**

```bash
git add ui/
git commit -m "feat(ui): interview demo page for ReACT trajectory and citations"
```

---

## Task 6: `make demo` + 启动脚本 + 面试文档

**Files:**
- Create: `scripts/run_demo.py`
- Create: `docs/interview-demo.md`
- Modify: `Makefile`
- Modify: `README.md`（加 Demo 一小节，3–5 行 + 链接）

- [ ] **Step 1: `scripts/run_demo.py`**

```python
"""Start PrismRAG demo API. Usage: python scripts/run_demo.py"""
import os
import uvicorn

def main():
    os.environ.setdefault("CONFIG_PROFILE", "local-dev")
    host = os.environ.get("DEMO_HOST", "127.0.0.1")
    port = int(os.environ.get("DEMO_PORT", "8000"))
    print(f"Demo UI: http://{host}:{port}/")
    print(f"Mock UI: http://{host}:{port}/?mock=1")
    print("API docs: http://{host}:{port}/docs")
    uvicorn.run("src.api.routes:app", host=host, port=port, reload=False)

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Makefile**

```makefile
demo: ## 启动面试 Demo（local-dev，UI at /）
	CONFIG_PROFILE=local-dev python scripts/run_demo.py
```

把 `demo` 加入 `.PHONY`。

- [ ] **Step 3: `docs/interview-demo.md` 必含**

- 启动命令 `make demo` / `?mock=1`
- 2 分钟投屏脚本（开场白 → 点预设问题 → 指轨迹 → 指三路 → 指拒答）
- 架构一句话 + 工具边界（无 web_search）
- 故障表：无 OPENAI_API_KEY、无 PG、无索引 → 用 mock
- 与 `/ask` 直出的区别（Agent 多跳可解释）

- [ ] **Step 4: README 补一小节**

```markdown
## 面试 Demo

```bash
make demo
# 浏览器打开 http://127.0.0.1:8000/  （无索引时用 /?mock=1）
```

最小 ReACT Agent：`POST /agent/ask` 返回 Thought/Action/Observation 轨迹。详见 [docs/interview-demo.md](docs/interview-demo.md)。
```

- [ ] **Step 5: Commit**

```bash
git add scripts/run_demo.py Makefile README.md docs/interview-demo.md
git commit -m "docs: interview demo runbook and make demo target"
```

---

## Task 7: 端到端冒烟 + 收尾

**Files:**
- 可能小改：CORS（若将来分端口；同 origin 则不需要）
- `handoff.md` 补一节「面试 Demo 状态」

- [ ] **Step 1: 全量相关测试**

```bash
pytest tests/test_react_agent.py tests/test_agent_api.py tests/test_prompt_registry.py -v
ruff check src/agent src/api/routes.py
```

Expected: PASS / 无新 lint 错误

- [ ] **Step 2: 手工冒烟清单**

| # | 操作 | 期望 |
|---|------|------|
| 1 | `/?mock=1` | 轨迹+答案渲染，无 API 依赖 |
| 2 | 有 key + 有索引：真实 `/agent/ask` | steps≥1，answer 非空 |
| 3 | 域外问题 | abstain 或「不够信息」 |
| 4 | `GET /docs` 可见 `/agent/ask` | schema 正确 |
| 5 | 旧 `POST /ask` 仍可用 | 无回归 |

- [ ] **Step 3: 更新 handoff.md**（简短）

记录：分支、DoD 勾选、已知限制（无 Graph 工具、max 2 search）。

- [ ] **Step 4: Final commit + PR 准备**

```bash
git add -A
git status
git commit -m "chore: react demo smoke checklist and handoff note"
```

PR 标题建议: `feat: minimal ReACT agent + interview demo UI`

PR 正文要点:
- 面试向最小 Agent（3 tools，无私网/web）
- UI 轨迹可视化 + mock 模式
- 测试：`test_react_agent` / `test_agent_api`
- 非目标：GraphRAG、Self-RAG、生产前端

---

## 自检（Plan Review）

| 需求 | 对应 Task |
|------|-----------|
| ReACT 轨迹可返回 | T1–T3 |
| 工具收窄、无 web | T2 schema + prompt |
| 复用检索/生成核心 | T2/T3 只调 retriever/generator |
| Demo 可投屏 | T5–T6 |
| 无索引也能演示 | T5 `?mock=1` |
| 单测不依赖 GPU | T1–T4 mock |
| make demo | T6 |
| 面试话术文档 | T6 `interview-demo.md` |

**明确不做（防 scope 蔓延）:** GraphRAG、query_knowledge_graph、Self-RAG 双门、Vite/React 工程、多用户 Auth、流式 SSE（可选 follow-up）、Redis。

**可选 follow-up（本 plan 结束后）:**  
- SSE 流式推送 steps（更炫）  
- Self-RAG Gate2 作为第四工具 `verify_answer`  
- 真实 sample 索引打成 GitHub Release 供 clone 即 demo  

---

## 执行顺序与并行

```
T1 models ──► T2 tools ──► T3 loop ──► T4 API ──► T5 UI ──► T6 make/docs ──► T7 smoke
                              │
                              └── prompt yaml 可与 T2 并行准备
```

UI（T5）可在 T4 契约冻结后与 T3 收尾并行（用 mock）。
