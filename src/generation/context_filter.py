"""生成侧上下文过滤：BGE 句级压缩 + 可选 LLM 句过滤。

mode（config context_filter.mode）:
  off           — 原文拼接
  bge           — 仅 BGE compress_context（默认，兼容现状）
  llm           — 仅 LLM 句过滤
  bge_then_llm  — 先 BGE 再 LLM

LLM 与 CtxRel 评分使用不同 prompt（context_sentence_filter），避免自循环。
失败时 fallback 到 BGE 或原文。
"""

from __future__ import annotations

import json
import logging
import re
from typing import Callable, List, Optional, Sequence

from src.config import cfg
from src.evaluation.ragas_metrics import compress_context, split_context_to_sentences

logger = logging.getLogger(__name__)

CompleteFn = Callable[[str], str]


def _split_to_sentences_from_text(text: str) -> List[str]:
    """将已拼接文本再拆句（LLM 过滤用）。"""
    return split_context_to_sentences([text]) if text.strip() else []


def filter_sentences_llm(
    text: str,
    query: str,
    complete_fn: CompleteFn,
    fallback: Optional[Callable[[str, str], str]] = None,
) -> str:
    """按 LLM 返回的 keep 下标过滤句子。

    complete_fn: 接收完整 user+system 拼好的 prompt 字符串，返回模型原文。
    fallback: 解析失败时 (text, query) -> str；默认返回原 text。
    """
    sentences = _split_to_sentences_from_text(text)
    if len(sentences) <= 1:
        return text

    numbered = "\n".join(f"{i}: {s}" for i, s in enumerate(sentences))
    # 轻量内联模板（避免强制依赖 prompt registry 在单测中加载）
    prompt = (
        "You select which context sentences are needed to answer the user question.\n"
        'Return ONLY valid JSON: {"keep": [0-based indices]}.\n\n'
        f"Question:\n{query}\n\n"
        f"Sentences:\n{numbered}\n"
    )
    raw = complete_fn(prompt)
    keep = _parse_keep_indices(raw, n=len(sentences))
    if keep is None:
        if fallback is not None:
            return fallback(text, query)
        logger.warning("LLM context filter parse failed; keeping original text")
        return text
    if not keep:
        # 空 keep 不安全：回退
        if fallback is not None:
            return fallback(text, query)
        return text
    ordered = sorted(set(keep))
    return "\n\n".join(sentences[i] for i in ordered if 0 <= i < len(sentences))


def _parse_keep_indices(raw: str, n: int) -> Optional[List[int]]:
    if not raw or not str(raw).strip():
        return None
    text = str(raw).strip()
    # 允许 markdown fence
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    if fence:
        text = fence.group(1)
    try:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end < start:
            return None
        obj = json.loads(text[start : end + 1])
        keep = obj.get("keep")
        if not isinstance(keep, list):
            return None
        out = []
        for x in keep:
            if isinstance(x, bool):
                continue
            if isinstance(x, (int, float)):
                i = int(x)
                if 0 <= i < n:
                    out.append(i)
        return out
    except (json.JSONDecodeError, TypeError, ValueError):
        return None


def openai_complete_fn(client, model: str) -> CompleteFn:
    def _complete(prompt: str) -> str:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        return resp.choices[0].message.content or ""

    return _complete


def prepare_context(
    query: str,
    chunks: Sequence[str],
    bge_embedder=None,
    *,
    mode: Optional[str] = None,
    ratio: Optional[float] = None,
    complete_fn: Optional[CompleteFn] = None,
) -> str:
    """统一上下文管线，供 Generator / RAGAS 共用。"""
    mode = mode or str(cfg.get("context_filter.mode", "bge"))
    ratio = (
        ratio
        if ratio is not None
        else float(cfg.get("retrieval.context_compression_ratio", 0.4))
    )
    chunks_list = [c for c in chunks if c]
    if not chunks_list:
        return ""

    joined = "\n\n".join(chunks_list)

    def _bge(text_chunks: List[str]) -> str:
        if bge_embedder is None or ratio >= 1.0:
            return "\n\n".join(text_chunks)
        return compress_context(query, text_chunks, bge_embedder, ratio=ratio)

    if mode == "off":
        return joined

    if mode == "bge":
        return _bge(chunks_list)

    if mode == "llm":
        if complete_fn is None:
            logger.warning("context_filter.mode=llm but no complete_fn; fallback to bge/join")
            return _bge(chunks_list)
        return filter_sentences_llm(
            joined,
            query,
            complete_fn=complete_fn,
            fallback=lambda t, q: _bge(chunks_list),
        )

    if mode == "bge_then_llm":
        mid = _bge(chunks_list)
        if complete_fn is None:
            return mid
        return filter_sentences_llm(
            mid,
            query,
            complete_fn=complete_fn,
            fallback=lambda t, q: mid,
        )

    logger.warning("Unknown context_filter.mode=%r; using bge", mode)
    return _bge(chunks_list)
