"""Corrective RAG (CRAG) — 检索后文档打分 + 可选改写再检索。

对齐 Self-RAG 设计中的 Gate1 检索侧反馈（生成前纠错），**不做联网兜底**：
闭域工业 PDF 场景 web fallback 会伤害 Faithfulness。

流程::

    retrieved → grade_documents → filter irrelevant
        → if insufficient and reformulate → rewrite query → re-search once
        → return corrected results + crag trace

默认 ``retrieval.crag.enabled: false``。
"""
from __future__ import annotations

import json
import logging
import re
import time
from typing import Any, Callable, Dict, List, Optional, Sequence

from src.config import cfg
from src.observability import get_tracer
from src.prompts import get_active

logger = logging.getLogger(__name__)

CompleteFn = Callable[[str], str]
SearchFn = Callable[[str], List[dict]]

__all__ = [
    "crag_config",
    "crag_cache_salt",
    "CorrectiveRAG",
    "grade_documents",
    "reformulate_query",
    "apply_grades",
    "apply_crag_if_enabled",
]


def crag_config() -> Dict[str, Any]:
    """读取 ``retrieval.crag``（带默认值）。"""
    raw = cfg.get("retrieval.crag", {}) or {}
    return {
        "enabled": bool(raw.get("enabled", False)),
        "grade_top_n": int(raw.get("grade_top_n", 10)),
        "min_relevant": int(raw.get("min_relevant", 1)),
        "reformulate": bool(raw.get("reformulate", True)),
        "max_retrieve_attempts": int(raw.get("max_retrieve_attempts", 2)),
        "on_grade_error": str(raw.get("on_grade_error", "pass_through")),
        "judge_timeout_ms": int(raw.get("judge_timeout_ms", 8000)),
        "judge_model": raw.get("judge_model"),
        # 再检索后是否用二次 grade 过滤；默认 true
        "regrade_after_retrieve": bool(raw.get("regrade_after_retrieve", True)),
        # 文本截断：控制 judge prompt 体积
        "chunk_text_max_chars": int(raw.get("chunk_text_max_chars", 600)),
    }


def crag_cache_salt(cr: Optional[Dict[str, Any]] = None) -> str:
    """L4 Answer 缓存盐：CRAG 开关/参数变化不得串答案。"""
    cr = cr or crag_config()
    if not cr.get("enabled"):
        return "crag=off"
    return (
        f"crag=on"
        f"|n={cr.get('grade_top_n', 10)}"
        f"|min={cr.get('min_relevant', 1)}"
        f"|rf={int(bool(cr.get('reformulate', True)))}"
        f"|att={cr.get('max_retrieve_attempts', 2)}"
    )


def _parse_json(raw: str) -> Dict[str, Any]:
    text = (raw or "").strip()
    if not text:
        raise ValueError("empty judge response")
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
        raise ValueError("judge JSON must be an object")
    return data


def _format_documents(
    docs: Sequence[dict], *, max_chars: int
) -> str:
    parts: List[str] = []
    for i, d in enumerate(docs, 1):
        cid = str(d.get("chunk_id") or f"idx-{i}")
        text = (d.get("text") or "").strip()
        if len(text) > max_chars:
            text = text[: max_chars - 1] + "…"
        ctype = d.get("chunk_type") or "text"
        parts.append(
            f"[{i}] chunk_id={cid} type={ctype}\n{text}"
        )
    return "\n\n".join(parts) if parts else "(no documents)"


def grade_documents(
    query: str,
    documents: Sequence[dict],
    *,
    complete_fn: CompleteFn,
    chunk_text_max_chars: int = 600,
) -> Dict[str, Any]:
    """对一批 chunk 做相关/充分判定。

    返回::

        {
          "grades": [{"chunk_id", "relevant", "reason"}, ...],
          "sufficient": bool,
          "missing": str,
          "latency_ms": float,
        }
    """
    if not documents:
        return {
            "grades": [],
            "sufficient": False,
            "missing": "no documents retrieved",
            "latency_ms": 0.0,
        }

    pv = get_active("crag_grade_documents")
    prompt = pv.render(
        "template",
        query=query,
        documents=_format_documents(documents, max_chars=chunk_text_max_chars),
    )
    t0 = time.perf_counter()
    raw = complete_fn(prompt)
    latency_ms = (time.perf_counter() - t0) * 1000.0
    data = _parse_json(raw)

    grades_raw = data.get("grades") or []
    if not isinstance(grades_raw, list):
        raise ValueError(f"grades must be a list: {data!r}")

    # 建立 chunk_id → doc 索引，兼容模型漏返回某些 id
    id_order = [str(d.get("chunk_id") or f"idx-{i+1}") for i, d in enumerate(documents)]
    by_id = {cid: i for i, cid in enumerate(id_order)}

    grades: List[Dict[str, Any]] = []
    seen: set = set()
    for g in grades_raw:
        if not isinstance(g, dict):
            continue
        cid = str(g.get("chunk_id") or "").strip()
        if not cid or cid not in by_id or cid in seen:
            continue
        seen.add(cid)
        grades.append(
            {
                "chunk_id": cid,
                "relevant": bool(g.get("relevant")),
                "reason": str(g.get("reason") or "")[:200],
            }
        )

    # 模型漏标的 chunk：保守视为 relevant=False
    for cid in id_order:
        if cid not in seen:
            grades.append(
                {
                    "chunk_id": cid,
                    "relevant": False,
                    "reason": "missing from judge output",
                }
            )

    # 保持与 documents 相同顺序
    grades.sort(key=lambda x: by_id.get(x["chunk_id"], 999))

    sufficient = data.get("sufficient")
    if sufficient is None:
        # 兜底：至少 1 个 relevant 视为可能充分（后续由 min_relevant 再卡）
        sufficient = any(g["relevant"] for g in grades)
    missing = str(data.get("missing") or "")

    return {
        "grades": grades,
        "sufficient": bool(sufficient),
        "missing": missing[:300],
        "latency_ms": latency_ms,
        "raw": (raw or "")[:400],
    }


def apply_grades(
    documents: Sequence[dict],
    grades: Sequence[dict],
) -> List[dict]:
    """按 grade 过滤；保留 relevant=true 的 chunk（稳定顺序）。"""
    rel = {
        str(g["chunk_id"]): bool(g.get("relevant"))
        for g in grades
        if g.get("chunk_id") is not None
    }
    kept: List[dict] = []
    for i, d in enumerate(documents):
        cid = str(d.get("chunk_id") or f"idx-{i+1}")
        if rel.get(cid, False):
            kept.append(d)
    return kept


def reformulate_query(
    query: str,
    *,
    complete_fn: CompleteFn,
    feedback: str = "",
) -> Dict[str, Any]:
    """轻量改写检索 query（非 HyDE）。"""
    pv = get_active("crag_reformulate_query")
    prompt = pv.render(
        "template",
        query=query,
        feedback=feedback or "previous retrieval insufficient or low relevance",
    )
    t0 = time.perf_counter()
    raw = complete_fn(prompt)
    latency_ms = (time.perf_counter() - t0) * 1000.0
    data = _parse_json(raw)
    new_q = str(data.get("query") or "").strip()
    if not new_q:
        raise ValueError(f"reformulate returned empty query: {data!r}")
    # 防止模型复读过长答案
    if len(new_q) > 500:
        new_q = new_q[:500]
    return {
        "query": new_q,
        "rationale": str(data.get("rationale") or "")[:300],
        "latency_ms": latency_ms,
        "raw": (raw or "")[:300],
    }


def _openai_complete(client: Any, model: str, *, timeout_s: float) -> CompleteFn:
    def complete(prompt: str) -> str:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            timeout=timeout_s,
        )
        return resp.choices[0].message.content or ""

    return complete


def apply_crag_if_enabled(
    query: str,
    retrieved: Optional[List[dict]],
    *,
    retriever: Any = None,
    k: int = 5,
    use_rerank: bool = True,
    use_visual: bool = True,
    client: Any = None,
    model: Optional[str] = None,
    complete_fn: Optional[CompleteFn] = None,
    config: Optional[Dict[str, Any]] = None,
) -> tuple:
    """评测 /ask 共用：若 CRAG 开启则纠错，否则原样返回。

    Returns:
        (results, crag_meta)
    """
    docs = list(retrieved or [])
    cr = config if config is not None else crag_config()
    if not cr.get("enabled"):
        return docs, {
            "enabled": False,
            "applied": False,
            "skip_reason": "disabled",
        }

    search_fn: Optional[SearchFn] = None
    if retriever is not None:

        def _search(q: str) -> List[dict]:
            return (
                retriever.search(
                    q, k=k, use_rerank=use_rerank, use_visual=use_visual
                )
                or []
            )

        search_fn = _search

    orch = CorrectiveRAG(
        search_fn=search_fn,
        complete_fn=complete_fn,
        client=client,
        model=model,
        config=cr,
    )
    out = orch.correct(query, docs, k=k)
    return out.get("results") or docs, out.get("crag") or {
        "enabled": True,
        "applied": False,
    }


class CorrectiveRAG:
    """检索后纠错编排：grade → filter → (optional) reformulate + re-search。"""

    def __init__(
        self,
        *,
        search_fn: Optional[SearchFn] = None,
        complete_fn: Optional[CompleteFn] = None,
        config: Optional[Dict[str, Any]] = None,
        client: Any = None,
        model: Optional[str] = None,
    ):
        self._search_fn = search_fn
        self._complete_fn = complete_fn
        self._cfg = config
        self._client = client
        self._model = model

    def _cfg_now(self) -> Dict[str, Any]:
        return self._cfg if self._cfg is not None else crag_config()

    def _judge_fn(self, cr: Dict[str, Any]) -> CompleteFn:
        if self._complete_fn is not None:
            return self._complete_fn
        model = (
            cr.get("judge_model")
            or self._model
            or cfg.get("llm.model", "gpt-4o-mini")
        )
        client = self._client
        if client is None:
            raise RuntimeError(
                "CRAG needs complete_fn or client for LLM grade/reformulate"
            )
        timeout_s = max(0.5, float(cr.get("judge_timeout_ms", 8000)) / 1000.0)
        return _openai_complete(client, model, timeout_s=timeout_s)

    def correct(
        self,
        query: str,
        retrieved: List[dict],
        *,
        k: Optional[int] = None,
    ) -> Dict[str, Any]:
        """对一次检索结果做纠错。

        Returns::

            {
              "query": original,
              "query_used": final query for generation-side awareness,
              "results": [...],
              "crag": {enabled, applied, ... trace ...},
            }
        """
        cr = self._cfg_now()
        if not cr.get("enabled"):
            return {
                "query": query,
                "query_used": query,
                "results": list(retrieved),
                "crag": {
                    "enabled": False,
                    "applied": False,
                    "skip_reason": "disabled",
                },
            }

        tracer = get_tracer()
        with tracer.start_span(
            "crag.correct",
            metadata={
                "enabled": True,
                "grade_top_n": cr.get("grade_top_n"),
                "min_relevant": cr.get("min_relevant"),
                "reformulate": cr.get("reformulate"),
            },
        ) as span:
            try:
                out = self._run(query, retrieved, cr=cr, k=k)
            except Exception as e:
                logger.warning("CRAG failed, pass-through: %s", e)
                meta = {
                    "enabled": True,
                    "applied": False,
                    "skip_reason": "error",
                    "error": str(e)[:300],
                    "final_action": "pass_through_on_error",
                }
                span.set_metadata(meta)
                return {
                    "query": query,
                    "query_used": query,
                    "results": list(retrieved),
                    "crag": meta,
                }
            span.set_metadata(out.get("crag") or {})
            return out

    def _run(
        self,
        query: str,
        retrieved: List[dict],
        *,
        cr: Dict[str, Any],
        k: Optional[int],
    ) -> Dict[str, Any]:
        judge = self._judge_fn(cr)
        top_n = max(1, int(cr.get("grade_top_n", 10)))
        min_relevant = max(0, int(cr.get("min_relevant", 1)))
        max_chars = int(cr.get("chunk_text_max_chars", 600))
        max_attempts = max(1, int(cr.get("max_retrieve_attempts", 2)))
        allow_reform = bool(cr.get("reformulate", True)) and max_attempts > 1

        attempts: List[Dict[str, Any]] = []
        current_query = query
        current_docs = list(retrieved[:top_n]) if retrieved else []
        # 完整列表保留用于：grade 失败 pass_through 时回退
        full_pool = list(retrieved)

        final_action = "use_filtered"
        query_used = query

        for attempt in range(1, max_attempts + 1):
            grade_input = current_docs[:top_n]
            try:
                grade = grade_documents(
                    current_query,
                    grade_input,
                    complete_fn=judge,
                    chunk_text_max_chars=max_chars,
                )
            except Exception as e:
                logger.warning("CRAG grade error attempt=%s: %s", attempt, e)
                on_err = str(cr.get("on_grade_error", "pass_through"))
                attempts.append(
                    {
                        "attempt": attempt,
                        "query": current_query,
                        "action": "grade_error",
                        "error": str(e)[:200],
                    }
                )
                if on_err == "pass_through":
                    final_action = "pass_through_on_grade_error"
                    return {
                        "query": query,
                        "query_used": query,
                        "results": full_pool,
                        "crag": {
                            "enabled": True,
                            "applied": True,
                            "final_action": final_action,
                            "attempts": attempts,
                            "grade_degraded": True,
                        },
                    }
                # filter_none：当作全部不相关，走 reformulate
                grade = {
                    "grades": [
                        {
                            "chunk_id": str(d.get("chunk_id") or f"idx-{i+1}"),
                            "relevant": False,
                            "reason": "grade_error",
                        }
                        for i, d in enumerate(grade_input)
                    ],
                    "sufficient": False,
                    "missing": f"grade error: {e}",
                    "latency_ms": 0.0,
                }

            kept = apply_grades(grade_input, grade["grades"])
            n_rel = len(kept)
            sufficient = bool(grade.get("sufficient")) and n_rel >= min_relevant
            # 即使模型说 sufficient，相关数不足仍视为不够
            if n_rel < min_relevant:
                sufficient = False

            att_rec: Dict[str, Any] = {
                "attempt": attempt,
                "query": current_query,
                "num_graded": len(grade_input),
                "num_relevant": n_rel,
                "sufficient": sufficient,
                "missing": grade.get("missing"),
                "grade_latency_ms": grade.get("latency_ms"),
                "grades": grade.get("grades"),
            }

            if sufficient or not allow_reform or attempt >= max_attempts:
                att_rec["action"] = "accept" if sufficient else "accept_best_effort"
                attempts.append(att_rec)
                final_action = att_rec["action"]
                query_used = current_query
                # 相关集优先；若为空则回退原始检索（避免全灭）
                results = kept if kept else full_pool
                # 若 kept 非空但短于 k，不强制补噪声；调用方 k_context 再切
                return {
                    "query": query,
                    "query_used": query_used,
                    "results": results,
                    "crag": {
                        "enabled": True,
                        "applied": True,
                        "final_action": final_action,
                        "query_original": query,
                        "query_used": query_used,
                        "num_relevant": n_rel,
                        "sufficient": sufficient,
                        "missing": grade.get("missing"),
                        "attempts": attempts,
                        "grade_degraded": False,
                    },
                }

            # 需要改写再检索
            feedback = grade.get("missing") or (
                f"only {n_rel} relevant chunks; need better coverage"
            )
            try:
                rw = reformulate_query(
                    query,  # 始终基于原问题改写，避免漂移
                    complete_fn=judge,
                    feedback=str(feedback),
                )
            except Exception as e:
                logger.warning("CRAG reformulate error: %s", e)
                att_rec["action"] = "reformulate_error"
                att_rec["error"] = str(e)[:200]
                attempts.append(att_rec)
                results = kept if kept else full_pool
                return {
                    "query": query,
                    "query_used": current_query,
                    "results": results,
                    "crag": {
                        "enabled": True,
                        "applied": True,
                        "final_action": "reformulate_error_use_filtered",
                        "query_original": query,
                        "query_used": current_query,
                        "num_relevant": n_rel,
                        "sufficient": False,
                        "attempts": attempts,
                        "grade_degraded": True,
                    },
                }

            new_q = rw["query"]
            att_rec["action"] = "reformulate_and_research"
            att_rec["reformulated_query"] = new_q
            att_rec["reformulate_latency_ms"] = rw.get("latency_ms")
            att_rec["reformulate_rationale"] = rw.get("rationale")
            attempts.append(att_rec)

            if self._search_fn is None:
                logger.warning("CRAG reformulate ok but no search_fn; stop")
                results = kept if kept else full_pool
                return {
                    "query": query,
                    "query_used": new_q,
                    "results": results,
                    "crag": {
                        "enabled": True,
                        "applied": True,
                        "final_action": "no_search_fn_use_filtered",
                        "query_original": query,
                        "query_used": new_q,
                        "num_relevant": n_rel,
                        "sufficient": False,
                        "attempts": attempts,
                    },
                }

            # 再检索
            try:
                second = self._search_fn(new_q) or []
            except Exception as e:
                logger.warning("CRAG re-search failed: %s", e)
                results = kept if kept else full_pool
                return {
                    "query": query,
                    "query_used": new_q,
                    "results": results,
                    "crag": {
                        "enabled": True,
                        "applied": True,
                        "final_action": "research_error_use_filtered",
                        "query_original": query,
                        "query_used": new_q,
                        "error": str(e)[:200],
                        "attempts": attempts,
                        "grade_degraded": True,
                    },
                }

            current_query = new_q
            current_docs = list(second[:top_n]) if second else []
            full_pool = list(second) if second else full_pool
            # 若配置不 regrade，直接用第二轮结果
            if not cr.get("regrade_after_retrieve", True):
                return {
                    "query": query,
                    "query_used": current_query,
                    "results": full_pool,
                    "crag": {
                        "enabled": True,
                        "applied": True,
                        "final_action": "use_research_no_regrade",
                        "query_original": query,
                        "query_used": current_query,
                        "num_relevant": None,
                        "sufficient": None,
                        "attempts": attempts,
                    },
                }

        # 理论上不会落到这里
        return {
            "query": query,
            "query_used": query_used,
            "results": full_pool,
            "crag": {
                "enabled": True,
                "applied": True,
                "final_action": "fallback",
                "attempts": attempts,
            },
        }
