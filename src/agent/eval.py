"""Agent eval entry — dual-arm helpers for smoke / Phase2 scripts.

Mirrors ``answer_for_eval`` shape: returns answer/citations/context plus an
``agent`` metadata block (status, subqueries, trajectory, counts).
HITL is forced off for batch eval.
"""
from __future__ import annotations

import json
import statistics
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

from src.agent.config import agent_config
from src.agent.runner import merge_agent_cfg, run_agent
from src.rejection import is_rejection

__all__ = [
    "DEFAULT_AGENT_EVAL_QA",
    "agent_answer_for_eval",
    "load_agent_eval_qa",
    "score_answer",
    "run_dual_arm_item",
    "summarize_dual_arm",
    "go_nogo_draft",
]

DEFAULT_AGENT_EVAL_QA = Path("data/agent_eval_qa.json")


def agent_answer_for_eval(
    query: str,
    *,
    retriever,
    generator,
    k_context: int = 5,
    use_rerank: bool = True,
    use_visual: bool = True,
    cfg: Optional[Dict[str, Any]] = None,
    pipeline_fallback_fn: Optional[Callable[[], Dict[str, Any]]] = None,
    **search_kwargs,
) -> dict:
    """Run the agent graph for one query; return dict for E2E / dual-arm scripts.

    Returns
    -------
    dict
        ``{answer, citations, context, agent: {status, subqueries, trajectory,
        counts, error, thread_id}}`` — same top-level keys as ``Generator.answer``
        / ``answer_for_eval`` so callers can swap modes.

    Notes
    -----
    - Forces ``hitl.review_subqueries=False`` for unattended batch runs.
    - Does **not** require ``agent.enabled`` in yaml (caller opts into this path).
    - Extra ``search_kwargs`` are forwarded to ``retriever.search`` (e.g. flags).
    """

    def search_fn(q, k=None, arms=None):
        kk = int(k) if k is not None else k_context
        # supervise 派单的 arms → 只开对应检索臂；无 arms → 三路全开
        use_b = use_d = use_v = True
        if arms:
            use_b = "bm25" in arms
            use_d = "dense" in arms
            use_v = "visual" in arms
        return retriever.search(
            q,
            k=kk,
            use_rerank=use_rerank,
            use_bm25=use_b,
            use_dense=use_d,
            use_visual=use_v,
            **search_kwargs,
        )

    def complete_fn(prompt: str) -> str:
        if hasattr(generator, "complete") and callable(generator.complete):
            return generator.complete(prompt) or ""
        # Fallback: injected private complete (tests / older Generator)
        injected = getattr(generator, "_complete_fn", None)
        if callable(injected):
            return injected(prompt) or ""
        raise AttributeError(
            "generator has no complete(); add Generator.complete or inject _complete_fn"
        )

    default_k_context = int(k_context)

    def generate_fn(q, hits, k_context=None):
        # synthesize may pass a diversified list + explicit k so we do not
        # re-slice multi-subquery evidence back to the first 5 hits only.
        if k_context is not None:
            kk = int(k_context)
        else:
            kk = default_k_context
            if hits:
                kk = max(kk, min(len(hits), 12))
        return generator.answer(q, hits, k_context=kk)

    agent_cfg = merge_agent_cfg(
        agent_config(),
        {
            "hitl": {"review_subqueries": False},
            # Batch eval never resumes HITL; keep MemorySaver off so long dual-arm
            # runs do not retain per-thread state in the process singleton.
            "checkpoint": {"enabled": False},
            **(cfg or {}),
        },
    )
    # ensure nested hitl/checkpoint stay batch-safe even if cfg overwrote partially
    hitl = dict(agent_cfg.get("hitl") or {})
    hitl["review_subqueries"] = False
    agent_cfg["hitl"] = hitl
    # Allow explicit checkpoint override only when caller sets hitl (not batch).
    if not hitl.get("review_subqueries"):
        ckpt = dict(agent_cfg.get("checkpoint") or {})
        ckpt["enabled"] = False
        agent_cfg["checkpoint"] = ckpt

    res = run_agent(
        query,
        search_fn=search_fn,
        complete_fn=complete_fn,
        generate_fn=generate_fn,
        cfg=agent_cfg,
        pipeline_fallback_fn=pipeline_fallback_fn,
        # Unique per call (run_agent also generates if omitted); explicit for clarity.
        trace_id=None,
    )
    return {
        "answer": res.answer,
        "citations": res.citations,
        "context": res.context,
        "agent": {
            "status": res.status,
            "subqueries": res.subqueries,
            "trajectory": res.trajectory,
            "counts": res.counts,
            "error": res.error,
            "thread_id": res.thread_id,
        },
        "evidence": res.evidence,
    }


def load_agent_eval_qa(
    path: Optional[Union[str, Path]] = None,
    *,
    tags: Optional[List[str]] = None,
    max_items: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Load ``data/agent_eval_qa.json`` items; optional tag filter and max_items.

    Accepts either ``{"version": N, "items": [...]}`` or a bare list.
    """
    p = Path(path) if path else DEFAULT_AGENT_EVAL_QA
    raw = json.loads(p.read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        items = list(raw.get("items") or [])
    elif isinstance(raw, list):
        items = list(raw)
    else:
        raise ValueError(f"unsupported agent_eval_qa format in {p}")

    if tags:
        tag_set = {t.strip().lower() for t in tags if t and str(t).strip()}
        items = [
            it
            for it in items
            if str(it.get("tag") or "").strip().lower() in tag_set
        ]
    if max_items is not None:
        items = items[: max(0, int(max_items))]
    return items


def _heuristic_correct(
    answer: str,
    gold: Optional[str],
    expect_reject: bool,
) -> Optional[bool]:
    """Loose token overlap / reject-phrase check. For smoke & offline fallback."""
    if expect_reject:
        return bool(is_rejection(answer or ""))
    if not gold:
        return None
    a = (answer or "").lower()
    tokens = [t for t in gold.lower().replace(",", " ").split() if len(t) > 4][:8]
    if not tokens:
        return None
    hits = sum(1 for t in tokens if t in a)
    return hits >= max(1, len(tokens) // 3)


def score_answer(
    *,
    question: str,
    answer: str,
    gold: Optional[str],
    expect_reject: bool,
    judge: str = "heuristic",
) -> Dict[str, Any]:
    """Score one answer for dual-arm eval.

    Parameters
    ----------
    judge
        ``heuristic`` — token overlap / reject phrases (no LLM).
        ``llm`` — reuse E2E ``compute_answer_correctness`` (Ollama/local judge).

    Returns
    -------
    dict with keys: correct (bool|None), is_rejected, false_reject, judge, reasoning
    """
    ans = answer or ""
    rejected = bool(is_rejection(ans))
    out: Dict[str, Any] = {
        "correct": None,
        "is_rejected": rejected,
        "false_reject": False,
        "judge": judge,
        "reasoning": "",
    }

    if expect_reject:
        out["correct"] = rejected
        out["reasoning"] = "reject phrase match" if rejected else "expected reject, got answer"
        return out

    if rejected:
        out["correct"] = False
        out["false_reject"] = True
        out["reasoning"] = "answerable question rejected"
        return out

    mode = (judge or "heuristic").strip().lower()
    if mode == "llm":
        try:
            from src.evaluation.e2e_qa import compute_answer_correctness

            r = compute_answer_correctness(
                question=question,
                expected_answer=gold or "",
                generated_answer=ans,
            )
            out["correct"] = bool(r.is_correct)
            out["reasoning"] = (r.judge_reasoning or "")[:500]
            return out
        except Exception as e:  # pragma: no cover — network/Ollama failures
            out["reasoning"] = f"llm judge failed ({e}); fallback heuristic"
            mode = "heuristic"
            out["judge"] = "heuristic_fallback"

    h = _heuristic_correct(ans, gold, expect_reject=False)
    out["correct"] = h
    out["reasoning"] = out.get("reasoning") or "heuristic token overlap"
    return out


def run_dual_arm_item(
    item: Dict[str, Any],
    *,
    retriever,
    generator,
    k: int = 5,
    use_visual: bool = True,
    use_rerank: bool = True,
    judge: str = "heuristic",
    agent_cfg: Optional[Dict[str, Any]] = None,
    run_pipeline: bool = True,
    run_agent_arm: bool = True,
) -> Dict[str, Any]:
    """Run pipeline and/or agent arms for one QA item; attach scores."""
    from src.generation.self_rag import answer_for_eval

    q = item.get("query") or item.get("question") or ""
    gold = item.get("gold_answer")
    expect_reject = bool(item.get("expect_reject"))
    row: Dict[str, Any] = {
        "id": item.get("id"),
        "tag": item.get("tag"),
        "query": q,
        "expect_reject": expect_reject,
        "source_e2e_id": item.get("source_e2e_id"),
    }

    if run_pipeline:
        t0 = time.perf_counter()
        try:
            hits = retriever.search(
                q, k=k, use_visual=use_visual, use_rerank=use_rerank
            )
            pipe = answer_for_eval(
                q,
                hits,
                k_context=k,
                generator=generator,
                retriever=retriever,
                use_rerank=use_rerank,
                use_visual=use_visual,
            )
            ans = pipe.get("answer") or ""
            scored = score_answer(
                question=q,
                answer=ans,
                gold=gold,
                expect_reject=expect_reject,
                judge=judge,
            )
            row["pipeline"] = {
                "answer": ans,
                "n_citations": len(pipe.get("citations") or []),
                "latency_ms": int((time.perf_counter() - t0) * 1000),
                "error": None,
                "correct": scored["correct"],
                "is_rejected": scored["is_rejected"],
                "false_reject": scored["false_reject"],
                "judge": scored["judge"],
                "judge_reasoning": scored["reasoning"],
                # backward-compat alias for smoke consumers
                "correct_placeholder": scored["correct"],
            }
        except Exception as e:
            row["pipeline"] = {
                "answer": "",
                "n_citations": 0,
                "latency_ms": int((time.perf_counter() - t0) * 1000),
                "error": str(e)[:400],
                "correct": False,
                "is_rejected": True,
                "false_reject": False,
                "judge": judge,
                "judge_reasoning": f"arm error: {e}",
                "correct_placeholder": False,
            }

    if run_agent_arm:
        t1 = time.perf_counter()
        try:
            ag = agent_answer_for_eval(
                q,
                retriever=retriever,
                generator=generator,
                k_context=k,
                use_rerank=use_rerank,
                use_visual=use_visual,
                cfg={
                    "return_trajectory": True,
                    "hitl": {"review_subqueries": False},
                    **(agent_cfg or {}),
                },
            )
            agent_meta = ag.get("agent") or {}
            ans = ag.get("answer") or ""
            scored = score_answer(
                question=q,
                answer=ans,
                gold=gold,
                expect_reject=expect_reject,
                judge=judge,
            )
            counts = agent_meta.get("counts") or {}
            status = agent_meta.get("status")
            row["agent"] = {
                "answer": ans,
                "n_citations": len(ag.get("citations") or []),
                "latency_ms": int((time.perf_counter() - t1) * 1000),
                "status": status,
                "subqueries": agent_meta.get("subqueries") or [],
                "counts": counts,
                "trajectory_summary": [
                    {
                        "step": t.get("step"),
                        "node": t.get("node"),
                        "tool": t.get("tool"),
                    }
                    for t in (agent_meta.get("trajectory") or [])[:16]
                    if isinstance(t, dict)
                ],
                "error": agent_meta.get("error"),
                "degraded": status == "degraded",
                "correct": scored["correct"],
                "is_rejected": scored["is_rejected"],
                "false_reject": scored["false_reject"],
                "judge": scored["judge"],
                "judge_reasoning": scored["reasoning"],
                "correct_placeholder": scored["correct"],
            }
        except Exception as e:
            row["agent"] = {
                "answer": "",
                "n_citations": 0,
                "latency_ms": int((time.perf_counter() - t1) * 1000),
                "status": "error",
                "subqueries": [],
                "counts": {},
                "trajectory_summary": [],
                "error": str(e)[:400],
                "degraded": False,
                "correct": False,
                "is_rejected": True,
                "false_reject": False,
                "judge": judge,
                "judge_reasoning": f"arm error: {e}",
                "correct_placeholder": False,
            }

    return row


def _pct(vals: Sequence[Optional[bool]]) -> Optional[float]:
    known = [bool(v) for v in vals if v is not None]
    if not known:
        return None
    return sum(1 for v in known if v) / len(known)


def _latency_stats(lats: List[float]) -> Dict[str, Optional[float]]:
    if not lats:
        return {"mean": None, "p50": None, "p95": None}
    ordered = sorted(lats)
    n = len(ordered)

    def _pctile(p: float) -> float:
        if n == 1:
            return float(ordered[0])
        idx = min(n - 1, max(0, int(round((p / 100.0) * (n - 1)))))
        return float(ordered[idx])

    return {
        "mean": float(statistics.fmean(ordered)),
        "p50": _pctile(50),
        "p95": _pctile(95),
    }


def _arm_summary(rows: List[Dict[str, Any]], arm: str) -> Dict[str, Any]:
    blocks = [r.get(arm) for r in rows if isinstance(r.get(arm), dict)]
    corrects_all = [b.get("correct") for b in blocks]
    rejects_expected = [
        b.get("correct")
        for r, b in zip(rows, [r.get(arm) for r in rows])
        if isinstance(b, dict) and r.get("expect_reject")
    ]
    answerable = [
        (r, b)
        for r in rows
        if isinstance(r.get(arm), dict) and not r.get("expect_reject")
        for b in [r.get(arm)]
    ]
    false_rejects = sum(1 for _, b in answerable if b.get("false_reject"))
    lats = [float(b["latency_ms"]) for b in blocks if b.get("latency_ms") is not None]
    errs = sum(1 for b in blocks if b.get("error"))
    degraded = sum(1 for b in blocks if b.get("degraded"))

    by_tag: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        b = r.get(arm)
        if not isinstance(b, dict):
            continue
        tag = str(r.get("tag") or "untagged")
        bucket = by_tag.setdefault(tag, {"n": 0, "corrects": [], "lats": []})
        bucket["n"] += 1
        if b.get("correct") is not None:
            bucket["corrects"].append(bool(b["correct"]))
        if b.get("latency_ms") is not None:
            bucket["lats"].append(float(b["latency_ms"]))

    tag_out = {}
    for tag, bucket in by_tag.items():
        tag_out[tag] = {
            "n": bucket["n"],
            "correct_rate": (
                sum(bucket["corrects"]) / len(bucket["corrects"])
                if bucket["corrects"]
                else None
            ),
            "latency_ms_mean": (
                statistics.fmean(bucket["lats"]) if bucket["lats"] else None
            ),
        }

    out: Dict[str, Any] = {
        "n": len(blocks),
        "errors": errs,
        "error_rate": (errs / len(blocks)) if blocks else 0.0,
        "correct_rate": _pct(corrects_all),
        "reject_accuracy": _pct(rejects_expected),
        "false_reject_count": false_rejects,
        "degrade_count": degraded,
        "latency_ms": _latency_stats(lats),
        "by_tag": tag_out,
    }

    if arm == "agent" or arm.startswith("agent"):
        searches = []
        llm_calls = []
        for b in blocks:
            c = b.get("counts") or {}
            if "searches" in c:
                searches.append(float(c["searches"]))
            if "llm_calls" in c:
                llm_calls.append(float(c["llm_calls"]))
        out["avg_searches"] = statistics.fmean(searches) if searches else None
        out["avg_llm_calls"] = statistics.fmean(llm_calls) if llm_calls else None
        out["max_searches"] = max(searches) if searches else None
    return out


def summarize_dual_arm(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate pipeline/agent metrics for Phase2 README."""
    arms = []
    for key in ("pipeline", "agent", "agent_grade_off"):
        if any(isinstance(r.get(key), dict) for r in rows):
            arms.append(key)

    summary: Dict[str, Any] = {
        "n_items": len(rows),
        "tag_counts": {},
        "arms": {},
    }
    for r in rows:
        t = str(r.get("tag") or "untagged")
        summary["tag_counts"][t] = summary["tag_counts"].get(t, 0) + 1

    for arm in arms:
        summary["arms"][arm] = _arm_summary(rows, arm)

    if "pipeline" in summary["arms"] and "agent" in summary["arms"]:
        summary["delta"] = _delta_vs_pipeline(
            summary["arms"]["pipeline"], summary["arms"]["agent"]
        )
        summary["go_nogo"] = go_nogo_draft(
            summary["arms"]["pipeline"], summary["arms"]["agent"]
        )
    return summary


def _delta_vs_pipeline(pipe: Dict[str, Any], agent: Dict[str, Any]) -> Dict[str, Any]:
    def _d(a, b):
        if a is None or b is None:
            return None
        return float(b) - float(a)

    out: Dict[str, Any] = {
        "correct_rate": _d(pipe.get("correct_rate"), agent.get("correct_rate")),
        "reject_accuracy": _d(pipe.get("reject_accuracy"), agent.get("reject_accuracy")),
        "false_reject_count": None,
        "latency_ms_mean": _d(
            (pipe.get("latency_ms") or {}).get("mean"),
            (agent.get("latency_ms") or {}).get("mean"),
        ),
        "by_tag": {},
    }
    if pipe.get("false_reject_count") is not None and agent.get(
        "false_reject_count"
    ) is not None:
        out["false_reject_count"] = int(agent["false_reject_count"]) - int(
            pipe["false_reject_count"]
        )
    p_tags = pipe.get("by_tag") or {}
    a_tags = agent.get("by_tag") or {}
    for tag in sorted(set(p_tags) | set(a_tags)):
        out["by_tag"][tag] = {
            "correct_rate": _d(
                (p_tags.get(tag) or {}).get("correct_rate"),
                (a_tags.get(tag) or {}).get("correct_rate"),
            )
        }
    return out


def go_nogo_draft(
    pipe: Dict[str, Any],
    agent: Dict[str, Any],
    *,
    max_total_searches: float = 3.0,
    atomic_delta_min: float = -0.02,
) -> Dict[str, Any]:
    """Apply Phase2 draft gates (not final product decision without human review).

    Pass draft when:
      1) atomic Correct Δ ≥ -2pt (or either missing)
      2) multi_hop Correct ≥ pipeline
      3) avg searches ≤ budget
      4) no mass degrade (degrade_count ≤ 10% of n)
    """
    checks: List[Dict[str, Any]] = []

    p_tags = pipe.get("by_tag") or {}
    a_tags = agent.get("by_tag") or {}

    p_at = (p_tags.get("atomic") or {}).get("correct_rate")
    a_at = (a_tags.get("atomic") or {}).get("correct_rate")
    if p_at is not None and a_at is not None:
        d_at = a_at - p_at
        ok = d_at >= atomic_delta_min
        checks.append(
            {
                "id": "atomic_not_hurt",
                "ok": ok,
                "detail": f"atomic Δ={d_at:+.4f} (need ≥ {atomic_delta_min})",
            }
        )
    else:
        checks.append(
            {
                "id": "atomic_not_hurt",
                "ok": None,
                "detail": "missing atomic rates",
            }
        )

    p_mh = (p_tags.get("multi_hop") or {}).get("correct_rate")
    a_mh = (a_tags.get("multi_hop") or {}).get("correct_rate")
    if p_mh is not None and a_mh is not None:
        ok = a_mh >= p_mh
        checks.append(
            {
                "id": "multi_hop_ge_pipeline",
                "ok": ok,
                "detail": f"multi_hop agent={a_mh:.4f} pipeline={p_mh:.4f}",
            }
        )
    else:
        checks.append(
            {
                "id": "multi_hop_ge_pipeline",
                "ok": None,
                "detail": "missing multi_hop rates",
            }
        )

    avg_s = agent.get("avg_searches")
    if avg_s is not None:
        ok = avg_s <= max_total_searches + 1e-9
        checks.append(
            {
                "id": "searches_within_budget",
                "ok": ok,
                "detail": f"avg_searches={avg_s:.3f} budget={max_total_searches}",
            }
        )
    else:
        checks.append(
            {
                "id": "searches_within_budget",
                "ok": None,
                "detail": "no search counts",
            }
        )

    n = int(agent.get("n") or 0)
    deg = int(agent.get("degrade_count") or 0)
    if n > 0:
        ratio = deg / n
        ok = ratio <= 0.10
        checks.append(
            {
                "id": "no_mass_degrade",
                "ok": ok,
                "detail": f"degrade={deg}/{n} ({ratio:.1%})",
            }
        )
    else:
        checks.append({"id": "no_mass_degrade", "ok": None, "detail": "n=0"})

    known = [c for c in checks if c["ok"] is not None]
    if not known:
        verdict = "INCONCLUSIVE"
    elif all(c["ok"] for c in known):
        verdict = "GO_DRAFT"
    else:
        verdict = "NO_GO_DRAFT"

    return {
        "verdict": verdict,
        "checks": checks,
        "note": (
            "Draft only — human must confirm before any default enabled change. "
            "Even GO_DRAFT keeps agent.enabled=false unless separate config PR."
        ),
    }
