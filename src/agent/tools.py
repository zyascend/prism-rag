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
    "diversify_evidence_for_synthesis",
    "synthesis_k_context",
    "make_knowledge_search_tool",
    "make_agent_lc_tools",
]


def _hit_score(h: dict) -> float:
    """Prefer rerank_score, then score, for fill-up ranking."""
    for key in ("rerank_score", "score", "rrf_score"):
        v = h.get(key)
        if v is None:
            continue
        try:
            return float(v)
        except (TypeError, ValueError):
            continue
    return 0.0


def synthesis_k_context(
    n_subqueries: int,
    *,
    base_k: int = 5,
    per_subquery: int = 2,
    max_k: int = 12,
) -> int:
    """Target context size for multi-subquery synthesis.

    Multi-hop must not collapse to first-subquery-only top-5 (Phase2 root cause).
    """
    n = max(1, int(n_subqueries or 1))
    base = max(1, int(base_k or 5))
    per = max(1, int(per_subquery or 2))
    cap = max(base, int(max_k or 12))
    return min(cap, max(base, per * n))


def diversify_evidence_for_synthesis(
    evidence: Sequence[Any],
    *,
    k: int = 5,
) -> List[dict]:
    """Select up to ``k`` hits for generation with per-subquery fairness.

    Algorithm
    ---------
    1. Group by ``subquery_id`` (missing → single bucket ``-1``).
    2. Round-robin one hit from each subquery (preserving within-bucket order).
    3. Fill remaining slots by descending score across leftovers.
    4. Deduplicate by ``chunk_id`` (fallback: id(text)).

    This prevents ``Generator.answer``'s ``retrieved[:k_context]`` from seeing
    only the first subquery's hits when multi-search appends evidence in order.
    """
    if not evidence or k <= 0:
        return []

    buckets: Dict[Any, List[dict]] = {}
    order_keys: List[Any] = []
    for raw in evidence:
        h = dict(raw) if isinstance(raw, dict) else {"text": str(raw)}
        sid = h.get("subquery_id")
        if sid is None:
            sid = -1
        if sid not in buckets:
            buckets[sid] = []
            order_keys.append(sid)
        buckets[sid].append(h)

    # Round-robin
    selected: List[dict] = []
    seen: set = set()
    indices = {sid: 0 for sid in order_keys}

    def _key(h: dict) -> Any:
        cid = h.get("chunk_id")
        if cid is not None and cid != "":
            return ("c", cid)
        return ("t", (h.get("text") or "")[:120])

    def _take(h: dict) -> bool:
        key = _key(h)
        if key in seen:
            return False
        seen.add(key)
        selected.append(h)
        return True

    progressed = True
    while len(selected) < k and progressed:
        progressed = False
        for sid in order_keys:
            if len(selected) >= k:
                break
            i = indices[sid]
            bucket = buckets[sid]
            while i < len(bucket):
                hit = bucket[i]
                i += 1
                indices[sid] = i
                if _take(hit):
                    progressed = True
                    break

    if len(selected) < k:
        leftovers: List[dict] = []
        for sid in order_keys:
            i = indices[sid]
            leftovers.extend(buckets[sid][i:])
        leftovers.sort(key=_hit_score, reverse=True)
        for hit in leftovers:
            if len(selected) >= k:
                break
            _take(hit)

    return selected


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
        search_fns: Optional[Dict[str, SearchFn]] = None,
    ) -> None:
        """Narrow agent tools.

        ``search_fn`` — 兼容单检索注入（Phase 1 前形态，arms 全开时用）。
        ``search_fns`` — 多臂注入（supervisor 派单用）：{"bm25": fn, "dense": fn,
        "visual": fn}，``knowledge_search(arms=[...])`` 按臂只调对应 fn，结果合并。
        """
        self.search_fn = search_fn
        self.search_fns = search_fns or {}
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

        # 可选 arm_hints（supervisor 派单强先验）：{subquery: visual|bm25|dense|mixed}
        arm_hints: Dict[str, str] = {}
        raw_hints = data.get("arm_hints")
        if isinstance(raw_hints, dict):
            for k, v in raw_hints.items():
                sq = normalize_subquery_text(k)
                s = str(v or "").strip().lower()
                if sq and s in ("visual", "bm25", "dense", "mixed"):
                    arm_hints[sq] = s

        return {
            "subqueries": subs,
            "strategy": strategy,
            "reason": str(data.get("reason") or "")[:300],
            "arm_hints": arm_hints,
            "fallback": False,
        }

    def _search_arms(self, q: str, *, top_k: int, arms: Optional[List[str]]) -> List[dict]:
        """按 arms 检索，三种注入形态：

        1. ``search_fns`` 多臂注入：逐臂调 + 合并（最细粒度）。
        2. 仅 ``search_fn`` 且其接受 ``arms`` kwarg：把 arms 透传下去，
           ``PrismRAGRetriever.search`` 按 use_bm25/use_dense/use_visual 开臂
           —— 单臂注入下 supervise 选臂也能真实生效。
        3. 仅 ``search_fn`` 且不接受 arms：退化为全臂（arms 只作语义提示）。
        """
        if arms and self.search_fns:
            hits_raw: List[dict] = []
            for arm in arms:
                fn = self.search_fns.get(arm)
                if fn is None:
                    continue
                try:
                    hits_raw.extend(fn(q, k=top_k) or [])
                except TypeError:
                    hits_raw.extend(fn(q) or [])
            return hits_raw
        if arms:
            # 单臂注入但 search_fn 接受 arms → 透传（开臂）
            try:
                return self.search_fn(q, k=top_k, arms=list(arms))
            except TypeError:
                pass  # 不接受 arms kwarg → 退化全臂
        try:
            return self.search_fn(q, k=top_k)
        except TypeError:
            return self.search_fn(q)

    def knowledge_search(
        self,
        query: str,
        *,
        subquery_id: int,
        top_k: int = 5,
        arms: Optional[List[str]] = None,
    ) -> dict:
        """Retrieve hits and tag each with subquery_id + rank.

        ``arms`` 非空时只调指定检索臂（supervisor 派单）；否则走默认全臂 search_fn。
        """
        # Defense in depth: strip dict-shaped leftovers from bad decompose output
        q = normalize_subquery_text(query) or str(query or "").strip()
        hits_raw = self._search_arms(q, top_k=top_k, arms=arms)
        hits: List[dict] = []
        for i, h in enumerate(hits_raw or []):
            if not isinstance(h, dict):
                continue
            item = dict(h)
            item["subquery_id"] = subquery_id
            item["rank"] = i + 1
            if arms:
                item["arms"] = list(arms)
            hits.append(item)
        return {"hits": hits, "query": q, "subquery_id": subquery_id, "arms": arms}

    _ARMS = ("bm25", "dense", "visual")

    def _normalize_arms(self, arms: Any) -> List[str]:
        """Coerce arms to a validated subset of {bm25, dense, visual}; empty → all."""
        if not isinstance(arms, (list, tuple, set)):
            return list(self._ARMS)
        out: List[str] = []
        for a in arms:
            s = str(a).strip().lower()
            if s in self._ARMS and s not in out:
                out.append(s)
        return out if out else list(self._ARMS)

    def validate_dispatch_plan(self, data: Dict[str, Any], *, max_assignments: int) -> Dict[str, Any]:
        """Validate supervisor DispatchPlan JSON → normalized plan; any failure → fallback.

        Fallback plan = 规则行为（mode 用 decompose 的 strategy，assignments 空 → 调用方全臂均分配额）。
        """
        fallback = {"mode": "", "assignments": [], "fallback": True, "reason": ""}

        mode = str(data.get("mode") or "").strip().lower()
        if mode not in ("atomic", "multi"):
            mode = ""
        assignments_raw = data.get("assignments")
        if not isinstance(assignments_raw, list) or not assignments_raw:
            return {
                **fallback,
                "mode": mode,
                "reason": str(data.get("reason") or "")[:300],
            }

        assignments: List[dict] = []
        for a in assignments_raw[:max_assignments]:
            if not isinstance(a, dict):
                continue
            sq = normalize_subquery_text(a.get("subquery"))
            if not sq:
                continue
            arms = self._normalize_arms(a.get("arms"))
            try:
                n_searches = int(a.get("searches") or 1)
            except (TypeError, ValueError):
                n_searches = 1
            n_searches = max(1, min(n_searches, 3))
            assignments.append(
                {
                    "subquery": sq,
                    "arms": arms,
                    "searches": n_searches,
                }
            )
        if not assignments:
            return {**fallback, "mode": mode, "reason": str(data.get("reason") or "")[:300]}

        return {
            "mode": mode,
            "assignments": assignments,
            "fallback": False,
            "reason": str(data.get("reason") or "")[:300],
        }

    def supervise_dispatch(
        self,
        query: str,
        subqueries: List[str],
        strategy: str,
        *,
        budget: Dict[str, Any],
        arm_hints: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Supervisor: LLM 一次调用产 DispatchPlan；失败 → fallback 规则计划。

        fallback 计划 = 保持 decompose 的 strategy，assignments 为空（调用方全臂均分）。
        ``arm_hints``（decompose 产出）作为派单强先验拼进 prompt，但不强制（LLM 可覆盖）。
        """
        max_assignments = max(1, int(self.cfg.get("supervise_max_assignments") or 3))
        prompt_id = "agent_supervise"
        sup = self.cfg.get("supervise")
        if isinstance(sup, dict) and sup.get("prompt_id"):
            prompt_id = str(sup["prompt_id"])
        fallback = {
            "mode": strategy,
            "assignments": [],
            "fallback": True,
            "reason": "",
        }

        if int(budget.get("llm_calls_left") or 0) <= 0:
            # 预算不足 → 不调 LLM，走规则派单
            return fallback

        try:
            pv = self._get_active(prompt_id)
            hints_json = (
                json.dumps(dict(arm_hints or {}), ensure_ascii=False)
                if isinstance(arm_hints, dict) and arm_hints
                else "{}"
            )
            prompt = pv.render(
                "template",
                query=query,
                subqueries=json.dumps(subqueries, ensure_ascii=False),
                max_assignments=max_assignments,
                max_total_searches=int(self.cfg.get("max_total_searches") or 3),
                arm_hints=hints_json,
            )
            raw = self.complete_fn(prompt)
            data = parse_json_object(raw)
        except Exception as e:
            logger.warning("supervise_dispatch failed (%s); fallback to rules", e)
            return fallback

        plan = self.validate_dispatch_plan(data, max_assignments=max_assignments)
        if plan.get("fallback"):
            plan["mode"] = plan["mode"] or strategy
        return plan

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
        """Generate final answer from evidence; empty evidence → unified abstain.

        Multi-subquery evidence is **diversified** before generation so
        ``Generator.answer``'s ``[:k_context]`` slice is not first-subquery-only.
        """
        ev = list(evidence or [])
        if not ev:
            return {
                "answer": abstain_message(),
                "citations": [],
                "rejected": True,
            }

        hits: List[dict] = [
            dict(h) if isinstance(h, dict) else {"text": str(h)} for h in ev
        ]
        n_subs = len(
            {
                h.get("subquery_id")
                for h in hits
                if h.get("subquery_id") is not None
            }
        ) or 1
        base_k = int(
            self.cfg.get("synthesize_k_context")
            or self.cfg.get("search_top_k")
            or 5
        )
        per_sub = int(self.cfg.get("synthesize_per_subquery") or 2)
        max_k = int(self.cfg.get("synthesize_max_k") or 12)
        target_k = synthesis_k_context(
            n_subs, base_k=base_k, per_subquery=per_sub, max_k=max_k
        )
        selected = diversify_evidence_for_synthesis(hits, k=target_k)
        if not selected:
            selected = hits[:target_k]

        # Prefer generate_fn(..., k_context=) so curated list is not re-sliced short
        try:
            out = self.generate_fn(query, selected, k_context=len(selected))
        except TypeError:
            out = self.generate_fn(query, selected)

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
            "n_evidence_in": len(hits),
            "n_evidence_used": len(selected),
            "synthesize_k": target_k,
            **(
                {"context": out["context"]}
                if "context" in out
                else {}
            ),
        }
