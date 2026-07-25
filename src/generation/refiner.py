"""生成前 Refiner：对检索 hits 做 query-focused 上下文加工。

模式（config refiner.mode / 兼容 context_filter.mode）:
  off              — 原文拼接
  bge              — 硬 top-ratio 句过滤（兼容现状 compress_context）
  soft_rank        — 句级 sim 赋权；默认不硬删（prune_below=null）
  llm              — LLM keep 列表
  bge_then_llm     — 先 bge 再 llm
  soft_rank_then_llm — 先 soft_rank 再 llm

表 chunk（chunk_type=table）默认全文保护，不进句压缩。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from src.config import cfg
from src.evaluation.ragas_metrics import compress_context, split_context_to_sentences

logger = logging.getLogger(__name__)

CompleteFn = Callable[[str], str]


def _filter_sentences_llm(*args, **kwargs):
    """Lazy import 避免与 context_filter 循环依赖。"""
    from src.generation.context_filter import filter_sentences_llm

    return filter_sentences_llm(*args, **kwargs)

_LEGACY_MODES = frozenset({"off", "bge", "llm", "bge_then_llm"})
_ALL_MODES = _LEGACY_MODES | frozenset({"soft_rank", "soft_rank_then_llm"})


@dataclass
class RefineResult:
    context: str
    trace: Dict[str, Any] = field(default_factory=dict)


def refiner_config() -> Dict[str, Any]:
    """读取 refiner 配置；缺失时用兼容默认。"""
    try:
        raw = cfg.get("refiner", {}) or {}
    except Exception:
        raw = {}
    if not isinstance(raw, dict):
        raw = {}
    soft = raw.get("soft_rank") if isinstance(raw.get("soft_rank"), dict) else {}
    adaptive = raw.get("adaptive_ratio") if isinstance(raw.get("adaptive_ratio"), dict) else {}
    return {
        "enabled": bool(raw.get("enabled", True)),
        "mode": str(raw.get("mode") or ""),
        "protect_table_chunks": bool(raw.get("protect_table_chunks", True)),
        "emit_trace": bool(raw.get("emit_trace", True)),
        "soft_rank": {
            "min_sim": float(soft.get("min_sim", 0.25)),
            "prune_below": soft.get("prune_below", None),
            "temperature": float(soft.get("temperature", 1.0)),
            "keep_ratio": float(
                soft.get("keep_ratio")
                if soft.get("keep_ratio") is not None
                else cfg.get("retrieval.context_compression_ratio", 0.4)
            ),
        },
        "adaptive_ratio": {
            "enabled": bool(adaptive.get("enabled", False)),
            "low_rerank_threshold": float(adaptive.get("low_rerank_threshold", 0.35)),
            "ratio_high_conf": float(adaptive.get("ratio_high_conf", 0.4)),
            "ratio_low_conf": float(adaptive.get("ratio_low_conf", 0.7)),
        },
    }


def refiner_cache_salt(rc: Optional[Dict[str, Any]] = None) -> str:
    """L4 Answer 缓存盐：refiner 开关/模式变化不得串答案。"""
    rc = rc or refiner_config()
    if not rc.get("enabled"):
        # 仍可能走 context_filter.mode
        mode = str(cfg.get("context_filter.mode", "bge"))
        return f"refiner=legacy:{mode}"
    mode = resolve_mode(rc)
    soft = rc.get("soft_rank") or {}
    prune = soft.get("prune_below")
    return (
        f"refiner=on:{mode}:"
        f"kr={soft.get('keep_ratio', 0.4)}:"
        f"ms={soft.get('min_sim', 0.25)}:"
        f"pb={prune if prune is not None else 'null'}"
    )


def resolve_mode(rc: Optional[Dict[str, Any]] = None) -> str:
    """refiner.mode 优先；空则回退 context_filter.mode。"""
    rc = rc or refiner_config()
    if not rc.get("enabled"):
        return str(cfg.get("context_filter.mode", "bge"))
    mode = str(rc.get("mode") or "").strip()
    if not mode:
        mode = str(cfg.get("context_filter.mode", "bge"))
    if mode not in _ALL_MODES:
        logger.warning("Unknown refiner.mode=%r; using bge", mode)
        return "bge"
    return mode


def score_sentences(
    query: str,
    sentences: Sequence[str],
    bge_embedder,
) -> List[float]:
    """BGE cosine sim（L2-normalized → dot）。返回与 sentences 等长的 score 列表。"""
    import torch as _torch

    if not sentences:
        return []
    if bge_embedder is None:
        return [0.0] * len(sentences)
    query_emb = bge_embedder.encode([query])
    sent_embs = bge_embedder.encode(list(sentences))
    query_vec = query_emb[0]
    return [
        float(_torch.dot(query_vec, sent_embs[i]))
        for i in range(len(sentences))
    ]


def soft_rank_sentences(
    query: str,
    sentences: Sequence[str],
    bge_embedder,
    *,
    min_sim: float = 0.25,
    prune_below: Optional[float] = None,
    keep_ratio: float = 0.5,
    preserve_order: bool = True,
) -> Tuple[List[str], Dict[str, Any]]:
    """对句子 soft_rank：赋权 + 可选硬 prune + keep_ratio 截断低权尾。

    默认 preserve_order=True：按原文顺序输出保留句。
    """
    n = len(sentences)
    if n == 0:
        return [], {
            "num_sentences_in": 0,
            "num_sentences_out": 0,
            "mean_sim": 0.0,
            "pruned": 0,
            "mode": "soft_rank",
        }
    # keep_ratio>=1 且无 prune 时才可跳过；有 prune_below 仍需打分过滤
    if n <= 2 or bge_embedder is None or (keep_ratio >= 1.0 and prune_below is None):
        return list(sentences), {
            "num_sentences_in": n,
            "num_sentences_out": n,
            "mean_sim": 0.0,
            "pruned": 0,
            "skipped": True,
            "mode": "soft_rank",
        }

    sims = score_sentences(query, sentences, bge_embedder)
    weights: List[float] = []
    kept_mask = [True] * n
    pruned = 0
    denom = max(1e-6, 1.0 - min_sim)
    for i, s in enumerate(sims):
        if prune_below is not None and s < float(prune_below):
            kept_mask[i] = False
            pruned += 1
            weights.append(0.0)
            continue
        w = max(0.0, (s - min_sim) / denom)
        weights.append(w)

    candidates = [i for i in range(n) if kept_mask[i]]
    if not candidates:
        # 全 prune 不安全：回退 top-3 by sim
        ranked = sorted(range(n), key=lambda i: sims[i], reverse=True)
        candidates = ranked[: max(1, min(3, n))]
        pruned = n - len(candidates)
        for i in range(n):
            kept_mask[i] = i in set(candidates)

    # keep_ratio：在 candidates 中按 weight 保留 top
    num_keep = max(1, int(len(candidates) * keep_ratio)) if keep_ratio < 1.0 else len(candidates)
    num_keep = max(1, min(len(candidates), num_keep))
    by_weight = sorted(candidates, key=lambda i: (weights[i], sims[i]), reverse=True)
    final_set = set(by_weight[:num_keep])

    if preserve_order:
        out = [sentences[i] for i in range(n) if i in final_set]
    else:
        out = [sentences[i] for i in by_weight if i in final_set]

    mean_sim = sum(sims) / n if n else 0.0
    kept_sims = [sims[i] for i in final_set] or [0.0]
    trace = {
        "mode": "soft_rank",
        "num_sentences_in": n,
        "num_sentences_out": len(out),
        "mean_sim": round(mean_sim, 4),
        "min_sim_kept": round(min(kept_sims), 4),
        "pruned": pruned,
        "keep_ratio": keep_ratio,
        "min_sim": min_sim,
        "prune_below": prune_below,
    }
    return out, trace


def _effective_keep_ratio(rc: Dict[str, Any], hits: Sequence[dict]) -> float:
    soft = rc.get("soft_rank") or {}
    ratio = float(soft.get("keep_ratio", 0.5))
    adaptive = rc.get("adaptive_ratio") or {}
    if not adaptive.get("enabled"):
        return ratio
    scores = [
        float(h.get("rerank_score") or h.get("score") or 0.0)
        for h in hits
    ]
    if not scores:
        return ratio
    mx = max(scores)
    thr = float(adaptive.get("low_rerank_threshold", 0.35))
    if mx < thr:
        return float(adaptive.get("ratio_low_conf", 0.7))
    return float(adaptive.get("ratio_high_conf", ratio))


def _split_hits(
    hits: Sequence[dict], protect_table: bool
) -> Tuple[Dict[int, str], List[int], List[str]]:
    """返回 table_parts[orig_i]=text, text_idx, text_texts。"""
    table_parts: Dict[int, str] = {}
    text_idx: List[int] = []
    text_texts: List[str] = []
    for i, r in enumerate(hits):
        text = (r.get("text") or "").strip()
        if not text:
            continue
        is_table = protect_table and (r.get("chunk_type") or "").lower() == "table"
        if is_table:
            table_parts[i] = text
        else:
            text_idx.append(i)
            text_texts.append(text)
    return table_parts, text_idx, text_texts


def _merge_parts(table_parts: Dict[int, str], text_idx: List[int], compressed: str) -> str:
    if text_idx and compressed:
        mi = min(text_idx)
        prev = table_parts.get(mi, "")
        table_parts[mi] = (prev + "\n\n" + compressed).strip() if prev else compressed
    if not table_parts:
        return compressed
    return "\n\n".join(table_parts[i] for i in sorted(table_parts))


def refine_context(
    query: str,
    hits: Sequence[dict],
    bge_embedder=None,
    *,
    mode: Optional[str] = None,
    complete_fn: Optional[CompleteFn] = None,
    ratio: Optional[float] = None,
    rc: Optional[Dict[str, Any]] = None,
) -> RefineResult:
    """统一 Refiner 入口。hits 为检索结果 dict 列表（含 text / chunk_type）。"""
    rc = rc or refiner_config()
    mode = mode or resolve_mode(rc)
    protect = bool(rc.get("protect_table_chunks", True))
    soft_cfg = rc.get("soft_rank") or {}
    keep_ratio = (
        float(ratio)
        if ratio is not None
        else _effective_keep_ratio(rc, hits)
    )
    if ratio is None and mode in ("bge", "bge_then_llm"):
        keep_ratio = float(cfg.get("retrieval.context_compression_ratio", 0.4))

    top = [h for h in hits if (h.get("text") or "").strip()]
    if not top:
        return RefineResult(context="", trace={"mode": mode, "empty": True})

    table_parts, text_idx, text_texts = _split_hits(top, protect)
    base_trace: Dict[str, Any] = {
        "mode": mode,
        "table_chunks": len(table_parts),
        "text_chunks": len(text_texts),
        "ratio_effective": keep_ratio,
        "adaptive": bool((rc.get("adaptive_ratio") or {}).get("enabled")),
    }

    if not text_texts:
        ctx = "\n\n".join(table_parts[i] for i in sorted(table_parts))
        base_trace["num_sentences_in"] = 0
        base_trace["num_sentences_out"] = 0
        return RefineResult(context=ctx, trace=base_trace)

    def _bge_hard(texts: List[str]) -> str:
        if bge_embedder is None or keep_ratio >= 1.0:
            return "\n\n".join(texts)
        return compress_context(query, texts, bge_embedder, ratio=keep_ratio)

    def _soft(texts: List[str]) -> Tuple[str, Dict[str, Any]]:
        sentences = split_context_to_sentences(texts)
        kept, tr = soft_rank_sentences(
            query,
            sentences,
            bge_embedder,
            min_sim=float(soft_cfg.get("min_sim", 0.25)),
            prune_below=(
                float(soft_cfg["prune_below"])
                if soft_cfg.get("prune_below") is not None
                else None
            ),
            keep_ratio=keep_ratio,
            preserve_order=True,
        )
        return "\n".join(kept), tr

    if mode == "off":
        compressed = "\n\n".join(text_texts)
        tr = {"skipped": True}
    elif mode == "bge":
        compressed = _bge_hard(text_texts)
        tr = {"mode": "bge"}
    elif mode == "soft_rank":
        compressed, tr = _soft(text_texts)
    elif mode == "llm":
        joined = "\n\n".join(text_texts)
        if complete_fn is None:
            logger.warning("refiner mode=llm but no complete_fn; fallback bge")
            compressed = _bge_hard(text_texts)
            tr = {"mode": "llm", "fallback": "bge"}
        else:
            compressed = _filter_sentences_llm(
                joined,
                query,
                complete_fn=complete_fn,
                fallback=lambda t, q: _bge_hard(text_texts),
            )
            tr = {"mode": "llm"}
    elif mode == "bge_then_llm":
        mid = _bge_hard(text_texts)
        if complete_fn is None:
            compressed = mid
            tr = {"mode": "bge_then_llm", "fallback": "bge_only"}
        else:
            compressed = _filter_sentences_llm(
                mid, query, complete_fn=complete_fn, fallback=lambda t, q: mid
            )
            tr = {"mode": "bge_then_llm"}
    elif mode == "soft_rank_then_llm":
        mid, tr = _soft(text_texts)
        if complete_fn is None:
            compressed = mid
            tr["fallback"] = "soft_only"
        else:
            compressed = _filter_sentences_llm(
                mid, query, complete_fn=complete_fn, fallback=lambda t, q: mid
            )
            tr["mode"] = "soft_rank_then_llm"
    else:
        compressed = _bge_hard(text_texts)
        tr = {"mode": "bge", "unknown_fallback": True}

    context = _merge_parts(table_parts, text_idx, compressed)
    base_trace.update(tr)
    if not rc.get("emit_trace", True):
        base_trace = {"mode": mode}
    return RefineResult(context=context, trace=base_trace)


def prepare_context_via_refiner(
    query: str,
    chunks: Sequence[str],
    bge_embedder=None,
    *,
    mode: Optional[str] = None,
    ratio: Optional[float] = None,
    complete_fn: Optional[CompleteFn] = None,
) -> str:
    """兼容 prepare_context(chunks: List[str]) 的字符串入口。"""
    hits = [{"text": c, "chunk_type": "text"} for c in chunks if c]
    return refine_context(
        query,
        hits,
        bge_embedder,
        mode=mode,
        ratio=ratio,
        complete_fn=complete_fn,
    ).context
