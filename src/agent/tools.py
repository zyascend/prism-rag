"""Agent tool surface — dependency-injected, no global retrieval/generation singletons.

Tools call search / LLM / generate only via injected callables so unit tests stay mock-only.

Also exports langchain ``@tool`` / StructuredTool factories for the optional ReAct demo
(``react_demo``). Production fixed StateGraph calls ``AgentToolBox`` methods directly.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Callable, Dict, List, Optional, Sequence

from src.rejection import abstain_message, is_rejection

logger = logging.getLogger(__name__)

SearchFn = Callable[..., List[dict]]
CompleteFn = Callable[[str], str]
GenerateFn = Callable[..., dict]
PromptGetActive = Callable[[str], Any]

__all__ = [
    "AgentToolBox",
    "parse_json_object",
    "normalize_subquery_text",
    "normalize_subquery_list",
    "make_knowledge_search_tool",
    "make_agent_lc_tools",
]


def make_knowledge_search_tool(
    search_fn: SearchFn,
    *,
    subquery_id: int = 0,
    top_k_default: int = 5,
    name: str = "knowledge_search",
):
    """Wrap a retrieval callable as a langchain StructuredTool (@tool).

    Used by ``react_demo`` ToolNode loops. Production graph still calls
    ``AgentToolBox.knowledge_search`` directly (no ReAct).
    """
    from langchain_core.tools import tool

    default_k = int(top_k_default)
    sid = int(subquery_id)

    @tool(name)
    def knowledge_search(query: str, top_k: int = default_k) -> str:
        """Search the private PDF knowledge base. Returns JSON with hits[].

        Args:
            query: Self-contained search query (no anaphora).
            top_k: Max hits to return.
        """
        k = int(top_k) if top_k is not None else default_k
        try:
            hits_raw = search_fn(query, k=k)
        except TypeError:
            hits_raw = search_fn(query)
        hits: List[dict] = []
        for i, h in enumerate(hits_raw or []):
            if not isinstance(h, dict):
                continue
            item = dict(h)
            item.setdefault("subquery_id", sid)
            item.setdefault("rank", i + 1)
            # Bound observation size for the ReAct loop
            text = (item.get("text") or "")[:400]
            hits.append(
                {
                    "chunk_id": item.get("chunk_id"),
                    "doc_id": item.get("doc_id"),
                    "page_id": item.get("page_id") or item.get("page_number"),
                    "score": item.get("score"),
                    "subquery_id": item.get("subquery_id"),
                    "rank": item.get("rank"),
                    "text": text,
                }
            )
        return json.dumps(
            {"hits": hits, "query": query, "subquery_id": sid},
            ensure_ascii=False,
        )

    return knowledge_search


def make_agent_lc_tools(
    search_fn: SearchFn,
    *,
    subquery_id: int = 0,
    top_k_default: int = 5,
) -> List[Any]:
    """Return langchain tools for the ReAct demo (narrow: knowledge_search only)."""
    return [
        make_knowledge_search_tool(
            search_fn,
            subquery_id=subquery_id,
            top_k_default=top_k_default,
        )
    ]


def parse_json_object(raw: str) -> Dict[str, Any]:
    """Parse a JSON object from model output (optional markdown fence).

    Same shape as ``self_rag._parse_verdict_json`` / CRAG ``_parse_json`` —
    copied here to avoid circular imports with generation/retrieval.
    """
    text = (raw or "").strip()
    if not text:
        raise ValueError("empty model response")
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", text, re.I)
    if fence:
        text = fence.group(1).strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            raise ValueError(f"no JSON object in response: {raw[:200]!r}")
        data = json.loads(text[start : end + 1])
    if not isinstance(data, dict):
        raise ValueError("JSON must be an object")
    return data


_SUBQUERY_DICT_KEYS = ("query", "subquery", "text", "q", "question")
_MAX_SUBQUERY_CHARS = 500


def normalize_subquery_text(item: Any, *, max_chars: int = _MAX_SUBQUERY_CHARS) -> Optional[str]:
    """Coerce one decompose/HITL subquery item into a plain search string.

    Models often emit objects (or stringified dicts) like
    ``{"query": "...", "source": "..."}`` instead of bare strings.  Those must
    not be passed to ``knowledge_search`` as ``str(dict)``.
    """
    if item is None:
        return None

    if isinstance(item, dict):
        for k in _SUBQUERY_DICT_KEYS:
            v = item.get(k)
            if v is None:
                continue
            # Nested dict rare; flatten once
            if isinstance(v, dict):
                inner = normalize_subquery_text(v, max_chars=max_chars)
                if inner:
                    return inner
            t = str(v).strip()
            if t:
                return t[:max_chars]
        return None

    if isinstance(item, (list, tuple)):
        # Prefer first usable element
        for el in item:
            t = normalize_subquery_text(el, max_chars=max_chars)
            if t:
                return t
        return None

    s = str(item).strip()
    if not s:
        return None

    # Stringified JSON / Python dict from the model
    if (s.startswith("{") and s.endswith("}")) or (s.startswith("[") and s.endswith("]")):
        parsed: Any = None
        try:
            parsed = json.loads(s)
        except (json.JSONDecodeError, TypeError):
            try:
                import ast

                parsed = ast.literal_eval(s)
            except (ValueError, SyntaxError, TypeError):
                parsed = None
        if parsed is not None:
            t = normalize_subquery_text(parsed, max_chars=max_chars)
            if t:
                return t

    # Strip wrapping quotes leftover from loose model formatting
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        s = s[1:-1].strip()
    if not s:
        return None
    return s[:max_chars]


def normalize_subquery_list(
    items: Any,
    *,
    max_n: int = 3,
    max_chars: int = _MAX_SUBQUERY_CHARS,
) -> List[str]:
    """Normalize a list of subquery items; drop empties; de-dupe; clamp length."""
    if items is None:
        return []
    if not isinstance(items, (list, tuple)):
        one = normalize_subquery_text(items, max_chars=max_chars)
        return [one] if one else []
    out: List[str] = []
    seen: set[str] = set()
    for raw in items:
        t = normalize_subquery_text(raw, max_chars=max_chars)
        if not t:
            continue
        key = t.casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(t)
        if len(out) >= max(1, int(max_n)):
            break
    return out


def _format_evidence(evidence: Sequence[dict], *, max_chars: int = 600) -> str:
    parts: List[str] = []
    for i, d in enumerate(evidence, 1):
        cid = str(d.get("chunk_id") or f"idx-{i}")
        text = (d.get("text") or "").strip()
        if len(text) > max_chars:
            text = text[: max_chars - 1] + "…"
        sq = d.get("subquery_id")
        prefix = f"[{i}] chunk_id={cid}"
        if sq is not None:
            prefix += f" subquery_id={sq}"
        parts.append(f"{prefix}\n{text}")
    return "\n\n".join(parts) if parts else "(none)"


class AgentToolBox:
    """Narrow agent tools with injectable search / complete / generate deps."""

    def __init__(
        self,
        *,
        search_fn: SearchFn,
        complete_fn: CompleteFn,
        generate_fn: GenerateFn,
        cfg: Optional[Dict[str, Any]] = None,
        prompt_get_active: Optional[PromptGetActive] = None,
    ) -> None:
        self.search_fn = search_fn
        self.complete_fn = complete_fn
        self.generate_fn = generate_fn
        self.cfg: Dict[str, Any] = dict(cfg or {})
        self._prompt_get_active = prompt_get_active

    def _get_active(self, prompt_id: str) -> Any:
        if self._prompt_get_active is not None:
            return self._prompt_get_active(prompt_id)
        from src.prompts import get_active

        return get_active(prompt_id)

    def _max_subqueries(self) -> int:
        return int(self.cfg.get("max_subqueries", 3))

    def decompose_query(self, query: str) -> dict:
        """Split query into sub-queries; fallback to atomic on any failure."""
        q = (query or "").strip()
        fallback = {
            "subqueries": [q] if q else [""],
            "strategy": "atomic",
            "reason": "fallback",
            "fallback": True,
        }
        if not q:
            return fallback

        max_sq = self._max_subqueries()
        try:
            prompt_id = "agent_decompose"
            dec = self.cfg.get("decompose")
            if isinstance(dec, dict) and dec.get("prompt_id"):
                prompt_id = str(dec["prompt_id"])
            elif self.cfg.get("decompose_prompt_id"):
                prompt_id = str(self.cfg["decompose_prompt_id"])
            pv = self._get_active(prompt_id)
            prompt = pv.render("template", query=q, max_subqueries=max_sq)
            raw = self.complete_fn(prompt)
            data = parse_json_object(raw)
        except Exception as e:
            logger.warning("decompose_query failed (%s); using atomic fallback", e)
            return fallback

        subs_raw = data.get("subqueries")
        # Accept list, or single string / object under alternate keys
        if not isinstance(subs_raw, list):
            for alt in ("queries", "sub_queries", "subquery"):
                if isinstance(data.get(alt), list):
                    subs_raw = data[alt]
                    break
            else:
                # single object/string → treat as one-item list
                if subs_raw is not None:
                    subs_raw = [subs_raw]
                else:
                    return fallback

        subs = normalize_subquery_list(subs_raw, max_n=max_sq)
        if not subs:
            return fallback

        strategy = str(data.get("strategy") or "").strip().lower()
        if strategy not in ("atomic", "multi"):
            strategy = "atomic" if len(subs) <= 1 else "multi"
        if len(subs) > 1:
            strategy = "multi"
        elif strategy != "multi":
            strategy = "atomic"

        return {
            "subqueries": subs,
            "strategy": strategy,
            "reason": str(data.get("reason") or "")[:300],
            "fallback": False,
        }

    def knowledge_search(
        self, query: str, *, subquery_id: int, top_k: int = 5
    ) -> dict:
        """Retrieve hits and tag each with subquery_id + rank."""
        # Defense in depth: strip dict-shaped leftovers from bad decompose output
        q = normalize_subquery_text(query) or str(query or "").strip()
        try:
            hits_raw = self.search_fn(q, k=top_k)
        except TypeError:
            # Some injectors use search_fn(query) only
            hits_raw = self.search_fn(q)
        hits: List[dict] = []
        for i, h in enumerate(hits_raw or []):
            if not isinstance(h, dict):
                continue
            item = dict(h)
            item["subquery_id"] = subquery_id
            item["rank"] = i + 1
            hits.append(item)
        return {"hits": hits, "query": q, "subquery_id": subquery_id}

    def grade_evidence(self, query: str, evidence: list) -> dict:
        """LLM JSON grade: sufficient / missing / score.

        On parse or model error → sufficient=True pass-through (CRAG-like),
        so grading failures do not wrongly abstain.
        """
        ev = list(evidence or [])
        if not ev:
            return {
                "sufficient": False,
                "missing": "no evidence",
                "score": 0.0,
                "fallback": False,
            }

        try:
            prompt_id = "agent_grade_evidence"
            gcfg = self.cfg.get("grade")
            if isinstance(gcfg, dict) and gcfg.get("prompt_id"):
                prompt_id = str(gcfg["prompt_id"])
            pv = self._get_active(prompt_id)
            prompt = pv.render(
                "template",
                query=query or "",
                evidence=_format_evidence(ev),
            )
            raw = self.complete_fn(prompt)
            data = parse_json_object(raw)
        except Exception as e:
            logger.warning(
                "grade_evidence failed (%s); pass-through sufficient=True", e
            )
            return {
                "sufficient": True,
                "missing": "",
                "score": 1.0,
                "fallback": True,
                "error": str(e)[:200],
            }

        sufficient = data.get("sufficient")
        if sufficient is None:
            # Conservative: treat as sufficient on ambiguous JSON (anti-false-reject)
            sufficient = True
        missing = data.get("missing")
        if isinstance(missing, list):
            missing_out: Any = [str(m)[:200] for m in missing]
        else:
            missing_out = str(missing or "")[:300]

        score = data.get("score")
        try:
            score_f = float(score) if score is not None else (1.0 if sufficient else 0.0)
        except (TypeError, ValueError):
            score_f = 1.0 if sufficient else 0.0
        if score_f > 1.0 and score_f <= 100.0:
            score_f = score_f / 100.0
        score_f = max(0.0, min(1.0, score_f))

        return {
            "sufficient": bool(sufficient),
            "missing": missing_out,
            "score": score_f,
            "fallback": False,
        }

    def refine_subquery(self, query: str, subquery: str, missing: str) -> str:
        """Rewrite one subquery given missing-evidence feedback; fallback to original."""
        original = (subquery or "").strip() or (query or "").strip()
        try:
            prompt_id = "agent_refine_subquery"
            rcfg = self.cfg.get("refine")
            if isinstance(rcfg, dict) and rcfg.get("prompt_id"):
                prompt_id = str(rcfg["prompt_id"])
            pv = self._get_active(prompt_id)
            miss = missing
            if isinstance(missing, list):
                miss = "; ".join(str(m) for m in missing)
            prompt = pv.render(
                "template",
                query=query or "",
                subquery=original,
                missing=str(miss or ""),
            )
            raw = (self.complete_fn(prompt) or "").strip()
            # Prefer JSON {"query": "..."} if present; else first non-empty line
            rewritten = ""
            try:
                data = parse_json_object(raw)
                rewritten = normalize_subquery_text(data) or ""
                if not rewritten:
                    rewritten = normalize_subquery_text(
                        data.get("query")
                        or data.get("subquery")
                        or data.get("refined")
                        or data.get("text")
                    ) or ""
            except Exception:
                # strip fences / take first line; still normalize dict-shaped lines
                text = raw
                fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", text, re.I)
                if fence:
                    text = fence.group(1).strip()
                for line in text.splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    cand = normalize_subquery_text(line)
                    if cand:
                        rewritten = cand
                        break
                if not rewritten:
                    rewritten = normalize_subquery_text(text) or ""
            if not rewritten:
                return original
            return rewritten[:_MAX_SUBQUERY_CHARS]
        except Exception as e:
            logger.warning("refine_subquery failed (%s); keeping original", e)
            return original

    def synthesize_answer(self, query: str, evidence: list) -> dict:
        """Generate final answer from evidence; empty evidence → unified abstain."""
        ev = list(evidence or [])
        if not ev:
            return {
                "answer": abstain_message(),
                "citations": [],
                "rejected": True,
            }

        # Pass evidence as retrieved-style dicts for Generator.answer compatibility
        hits: List[dict] = [dict(h) if isinstance(h, dict) else {"text": str(h)} for h in ev]
        out = self.generate_fn(query, hits)
        if not isinstance(out, dict):
            return {
                "answer": abstain_message(),
                "citations": [],
                "rejected": True,
            }
        answer = out.get("answer", "")
        citations = list(out.get("citations") or [])
        if "rejected" in out:
            rejected = bool(out["rejected"])
        else:
            rejected = is_rejection(answer)
        return {
            "answer": answer,
            "citations": citations,
            "rejected": rejected,
            **(
                {"context": out["context"]}
                if "context" in out
                else {}
            ),
        }
