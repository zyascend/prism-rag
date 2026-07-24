"""Self-RAG Gate2 — 生成后忠实性门（工程闭环 MVP）。

设计见 docs/self-rag-closed-loop-design-2026-07-09.md（v2）。
默认关闭；不微调模型，用独立 LLM judge 对「答案是否被入模 context 支撑」打分，
不通过则重生（可选）或强制拒答。
"""
from __future__ import annotations

import json
import logging
import re
import time
from typing import Any, Callable, Dict, List, Optional

from src.config import cfg
from src.observability import get_tracer
from src.prompts import get_active
from src.rejection import ABSTAIN_ANSWER, abstain_message

logger = logging.getLogger(__name__)

# 兼容：历史 import from self_rag import ABSTAIN_ANSWER
__all__ = ["ABSTAIN_ANSWER", "SelfRAGOrchestrator", "self_rag_config", "answer_for_eval"]

# Trace / self_rag.attempts_detail 中答案截断长度（控制 api_traces.jsonl 体积）
_ANSWER_TRACE_MAX = 500

CompleteFn = Callable[[str], str]


def _truncate_answer(text: str, max_len: int = _ANSWER_TRACE_MAX) -> str:
    if text is None:
        return ""
    s = str(text)
    if len(s) <= max_len:
        return s
    return s[: max_len - 1] + "…"


def _attempt_record(
    *,
    attempt: int,
    prompt_id: str,
    answer: str,
    verdict: Optional[Dict[str, Any]],
    action: str,
) -> Dict[str, Any]:
    """单轮生成+Gate2 的可序列化记录，供 Trace / 事后分析。"""
    v = verdict or {}
    rec: Dict[str, Any] = {
        "attempt": attempt,
        "prompt_id": prompt_id,
        "answer": _truncate_answer(answer),
        "action": action,  # continue_regen | return | abstain | degrade_pass | empty_retrieval | ...
        "passed": v.get("passed"),
        "score": v.get("score"),
        "unsupported": list(v.get("unsupported") or []),
        "latency_ms": v.get("latency_ms"),
        "gate_degraded": bool(v.get("gate_degraded", False)),
    }
    if v.get("error"):
        rec["error"] = str(v["error"])[:300]
    return rec


def self_rag_config() -> Dict[str, Any]:
    """读取 generation.self_rag 配置（带默认值）。"""
    raw = cfg.get("generation.self_rag", {}) or {}
    return {
        "enabled": bool(raw.get("enabled", False)),
        "mode": str(raw.get("mode", "gate2_only")),
        # always: 开启即每条过 Gate2；low_rerank: 仅 max(rerank_score) < 阈值时过门
        "trigger": str(raw.get("trigger", "low_rerank")),
        "low_rerank_threshold": float(raw.get("low_rerank_threshold", 0.35)),
        "judge_model": raw.get("judge_model"),
        "faith_threshold": float(raw.get("faith_threshold", 0.8)),
        "max_generate_attempts": int(raw.get("max_generate_attempts", 2)),
        "on_fail": str(raw.get("on_fail", "regenerate_then_abstain")),
        "gate_timeout_ms": int(raw.get("gate_timeout_ms", 8000)),
        "on_judge_error": str(raw.get("on_judge_error", "degrade_pass")),
        "verdict_mode": str(raw.get("verdict_mode", "whole_answer")),
    }


def should_apply_gate2(
    retrieved: Optional[List[dict]],
    sr: Optional[Dict[str, Any]] = None,
) -> bool:
    """是否对本条请求跑 Gate2（enabled + trigger）。"""
    sr = sr or self_rag_config()
    if not sr.get("enabled"):
        return False
    trigger = str(sr.get("trigger", "low_rerank")).lower()
    if trigger in ("always", "all", "on"):
        return True
    if trigger in ("low_rerank", "low_confidence", "rerank"):
        scores = [
            float(r["rerank_score"])
            for r in (retrieved or [])
            if r.get("rerank_score") is not None
        ]
        if not scores:
            # 无 rerank 分：保守过门（避免漏拦）
            return True
        thr = float(sr.get("low_rerank_threshold", 0.35))
        return max(scores) < thr
    if trigger in ("never", "off", "none"):
        return False
    logger.warning("unknown self_rag.trigger=%s; treating as always", trigger)
    return True


def self_rag_cache_salt(sr: Optional[Dict[str, Any]] = None) -> str:
    """L4 Answer 缓存 key 盐：开/关与门参数变化不得串答案。"""
    sr = sr or self_rag_config()
    if not sr.get("enabled"):
        return "sr=off"
    return (
        f"sr=on"
        f"|mode={sr.get('mode', 'gate2_only')}"
        f"|trg={sr.get('trigger', 'low_rerank')}"
        f"|lr={sr.get('low_rerank_threshold', 0.35)}"
        f"|th={sr.get('faith_threshold', 0.8)}"
        f"|vm={sr.get('verdict_mode', 'whole_answer')}"
        f"|of={sr.get('on_fail', 'regenerate_then_abstain')}"
    )


def _parse_verdict_json(raw: str) -> Dict[str, Any]:
    """从 judge 输出解析 JSON（支持可选 markdown fence）。"""
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
            raise ValueError(f"no JSON object in judge response: {raw[:200]!r}")
        data = json.loads(text[start : end + 1])
    if not isinstance(data, dict):
        raise ValueError("judge JSON must be an object")
    return data


def evaluate_gate2_whole_answer(
    query: str,
    answer: str,
    context: str,
    *,
    complete_fn: CompleteFn,
    faith_threshold: float = 0.8,
) -> Dict[str, Any]:
    """整答一次判定。返回 passed/score/detail；异常向上抛由调用方降级。"""
    pv = get_active("self_rag_gate2_verdict")
    prompt = pv.render("template", query=query, answer=answer, context=context)
    t0 = time.perf_counter()
    raw = complete_fn(prompt)
    latency_ms = (time.perf_counter() - t0) * 1000.0
    data = _parse_verdict_json(raw)

    score = data.get("score")
    grounded = data.get("grounded")
    if score is None and grounded is not None:
        score = 1.0 if grounded else 0.0
    if score is None:
        raise ValueError(f"judge JSON missing score/grounded: {data!r}")
    score_f = float(score)
    if not (0.0 <= score_f <= 1.0):
        if 1.0 < score_f <= 100.0:
            score_f = score_f / 100.0
        else:
            raise ValueError(f"score out of range: {score_f}")

    if grounded is None:
        passed = score_f >= faith_threshold
    else:
        passed = bool(grounded) and score_f >= faith_threshold

    unsupported = data.get("unsupported") or []
    if not isinstance(unsupported, list):
        unsupported = [str(unsupported)]

    return {
        "passed": passed,
        "score": score_f,
        "grounded": bool(grounded) if grounded is not None else passed,
        "unsupported": [str(u) for u in unsupported],
        "latency_ms": latency_ms,
        "gate_degraded": False,
        "raw": raw[:500],
    }


def _openai_judge_complete(
    client: Any,
    model: str,
    *,
    timeout_s: float,
) -> CompleteFn:
    def complete(prompt: str) -> str:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            timeout=timeout_s,
        )
        return resp.choices[0].message.content or ""

    return complete


def eval_via_generator() -> bool:
    """评测是否走生产 Generator 路径（与 /ask 对齐）。

    True 条件：显式 ``generation.eval_via_generator`` 或 Self-RAG 已开启。
    云上 A/B 脚本对两臂都开 eval_via_generator，仅切换 self_rag.enabled。
    """
    if bool(cfg.get("generation.eval_via_generator", False)):
        return True
    return bool(self_rag_config().get("enabled"))


def answer_for_eval(
    query: str,
    retrieved: List[dict],
    *,
    k_context: int = 5,
    bge_embedder=None,
    client=None,
    generator=None,
    retriever=None,
    use_rerank: bool = True,
    use_visual: bool = True,
) -> dict:
    """评测/对照用统一生成入口：可选 CRAG → Generator ± Self-RAG Gate2。

    返回 ``{answer, citations, context, self_rag, crag}``，与 ``Generator.answer`` /
    ``SelfRAGOrchestrator.answer`` 一致。空检索返回拒答文案。

    传入 ``retriever`` 时，若 ``retrieval.crag.enabled`` 可改写后再检索（与 /ask 对齐）。
    """
    from src.generation.generator import Generator
    from src.retrieval.crag import apply_crag_if_enabled

    if generator is None:
        generator = Generator(client=client, bge_embedder=bge_embedder)

    retrieved, crag_meta = apply_crag_if_enabled(
        query,
        retrieved,
        retriever=retriever,
        k=k_context,
        use_rerank=use_rerank,
        use_visual=use_visual,
        client=getattr(generator, "client", None),
        model=getattr(generator, "model", None),
    )

    orch = SelfRAGOrchestrator(generator)
    out = orch.answer(query, retrieved, k_context=k_context)
    out["crag"] = crag_meta
    return out


class SelfRAGOrchestrator:
    """Gate2-only 编排：生成 → 忠实性门 → 重生/拒答。"""

    def __init__(
        self,
        generator,
        *,
        config: Optional[Dict[str, Any]] = None,
        judge_complete_fn: Optional[CompleteFn] = None,
    ):
        self.generator = generator
        self._cfg = config  # None → 每次读 live config
        self._judge_complete_fn = judge_complete_fn

    def _cfg_now(self) -> Dict[str, Any]:
        return self._cfg if self._cfg is not None else self_rag_config()

    def answer(
        self,
        query: str,
        retrieved: List[dict],
        k_context: int = 5,
    ) -> dict:
        sr = self._cfg_now()
        apply_gate = should_apply_gate2(retrieved, sr)
        if not apply_gate:
            out = self.generator.answer(query, retrieved, k_context=k_context)
            out["self_rag"] = {
                "enabled": bool(sr.get("enabled")),
                "applied": False,
                "trigger": sr.get("trigger"),
                "skip_reason": (
                    "disabled" if not sr.get("enabled") else "trigger_not_met"
                ),
            }
            return out

        tracer = get_tracer()
        with tracer.start_span(
            "self_rag.gate2",
            metadata={
                "enabled": True,
                "applied": True,
                "trigger": sr.get("trigger"),
                "faith_threshold": sr["faith_threshold"],
                "verdict_mode": sr.get("verdict_mode", "whole_answer"),
                "on_fail": sr.get("on_fail"),
            },
        ) as span:
            max_attempts = max(1, int(sr.get("max_generate_attempts", 2)))
            on_fail = str(sr.get("on_fail", "regenerate_then_abstain"))
            allow_regen = on_fail == "regenerate_then_abstain" and max_attempts > 1

            last: Optional[dict] = None
            last_verdict: Optional[dict] = None
            attempts_done = 0
            final_action = "abstain"
            # 每轮 append：fail→regen 事后可回放 score / answer / unsupported
            attempts_detail: List[Dict[str, Any]] = []

            def _finish_meta(**kwargs: Any) -> Dict[str, Any]:
                meta = {
                    "enabled": True,
                    "applied": True,
                    "trigger": sr.get("trigger"),
                    "attempts_detail": list(attempts_detail),
                    **kwargs,
                }
                span.set_metadata(meta)
                return meta

            for attempt in range(1, max_attempts + 1):
                attempts_done = attempt
                prompt_id = (
                    "answer_generation" if attempt == 1 else "self_rag_regenerate"
                )
                pre_ctx = (
                    last["context"] if last is not None and attempt > 1 else None
                )

                # 子 span：按 attempt 过滤日志 / GET /trace
                with tracer.start_span(
                    f"self_rag.gate2.attempt.{attempt}",
                    metadata={"attempt": attempt, "prompt_id": prompt_id},
                ) as attempt_span:
                    last = self.generator.answer(
                        query,
                        retrieved,
                        k_context=k_context,
                        prompt_id=prompt_id,
                        precomputed_context=pre_ctx,
                    )

                    # 空检索拒答：无需 gate
                    if not last.get("context") and not last.get("citations"):
                        rec = _attempt_record(
                            attempt=attempt,
                            prompt_id=prompt_id,
                            answer=last.get("answer") or "",
                            verdict={"passed": True, "score": None},
                            action="empty_retrieval",
                        )
                        attempts_detail.append(rec)
                        attempt_span.set_metadata(rec)
                        meta = _finish_meta(
                            passed=True,
                            score=None,
                            attempts=attempt,
                            final_action="empty_retrieval",
                            gate_degraded=False,
                        )
                        last["self_rag"] = meta
                        return last

                    try:
                        last_verdict = self._run_gate2(
                            query,
                            last["answer"],
                            last.get("context") or "",
                            sr,
                        )
                    except Exception as e:
                        logger.warning("Gate2 judge failed: %s", e)
                        last_verdict = self._handle_judge_error(e, sr)
                        if sr.get("on_judge_error", "degrade_pass") == "degrade_pass":
                            rec = _attempt_record(
                                attempt=attempt,
                                prompt_id=prompt_id,
                                answer=last.get("answer") or "",
                                verdict=last_verdict,
                                action="degrade_pass",
                            )
                            attempts_detail.append(rec)
                            attempt_span.set_metadata(rec)
                            meta = _finish_meta(
                                passed=True,
                                score=last_verdict.get("score"),
                                attempts=attempt,
                                final_action="degrade_pass",
                                gate_degraded=True,
                                error=last_verdict.get("error"),
                            )
                            last["self_rag"] = meta
                            return last
                        rec = _attempt_record(
                            attempt=attempt,
                            prompt_id=prompt_id,
                            answer=last.get("answer") or "",
                            verdict=last_verdict,
                            action="abstain_on_judge_error",
                        )
                        attempts_detail.append(rec)
                        attempt_span.set_metadata(rec)
                        final_action = "abstain_on_judge_error"
                        break

                    if last_verdict.get("passed"):
                        action = (
                            "return" if attempt == 1 else "return_after_regen"
                        )
                        rec = _attempt_record(
                            attempt=attempt,
                            prompt_id=prompt_id,
                            answer=last.get("answer") or "",
                            verdict=last_verdict,
                            action=action,
                        )
                        attempts_detail.append(rec)
                        attempt_span.set_metadata(rec)
                        meta = _finish_meta(
                            passed=True,
                            score=last_verdict.get("score"),
                            attempts=attempt,
                            final_action=action,
                            gate_degraded=False,
                            unsupported=last_verdict.get("unsupported", []),
                            latency_ms=last_verdict.get("latency_ms"),
                        )
                        last["self_rag"] = meta
                        return last

                    # Gate fail
                    if allow_regen and attempt < max_attempts:
                        rec = _attempt_record(
                            attempt=attempt,
                            prompt_id=prompt_id,
                            answer=last.get("answer") or "",
                            verdict=last_verdict,
                            action="continue_regen",
                        )
                        attempts_detail.append(rec)
                        attempt_span.set_metadata(rec)
                        continue

                    rec = _attempt_record(
                        attempt=attempt,
                        prompt_id=prompt_id,
                        answer=last.get("answer") or "",
                        verdict=last_verdict,
                        action="abstain",
                    )
                    attempts_detail.append(rec)
                    attempt_span.set_metadata(rec)
                    final_action = "abstain"
                    break

            meta = _finish_meta(
                passed=False,
                score=(last_verdict or {}).get("score"),
                attempts=attempts_done,
                final_action=final_action,
                gate_degraded=bool((last_verdict or {}).get("gate_degraded")),
                unsupported=(last_verdict or {}).get("unsupported", []),
            )
            return {
                "answer": abstain_message(),
                "citations": [],
                "context": (last or {}).get("context", ""),
                "self_rag": meta,
            }

    def _run_gate2(
        self, query: str, answer: str, context: str, sr: Dict[str, Any]
    ) -> Dict[str, Any]:
        mode = str(sr.get("verdict_mode", "whole_answer"))
        if mode != "whole_answer":
            logger.info(
                "verdict_mode=%s not implemented; using whole_answer", mode
            )

        complete_fn = self._judge_complete_fn
        if complete_fn is None:
            complete_fn = self._default_judge_fn(sr)

        return evaluate_gate2_whole_answer(
            query,
            answer,
            context,
            complete_fn=complete_fn,
            faith_threshold=float(sr.get("faith_threshold", 0.8)),
        )

    def _default_judge_fn(self, sr: Dict[str, Any]) -> CompleteFn:
        model = sr.get("judge_model") or getattr(self.generator, "model", None)
        if not model:
            model = cfg.get("llm.model", "gpt-4o-mini")
        client = getattr(self.generator, "client", None)
        if client is None:
            raise RuntimeError(
                "Self-RAG judge needs generator.client or judge_complete_fn"
            )
        timeout_s = max(0.5, float(sr.get("gate_timeout_ms", 8000)) / 1000.0)
        return _openai_judge_complete(client, model, timeout_s=timeout_s)

    def _handle_judge_error(
        self, err: Exception, sr: Dict[str, Any]
    ) -> Dict[str, Any]:
        return {
            "passed": False,
            "score": None,
            "gate_degraded": True,
            "error": str(err),
            "unsupported": [],
        }
