"""表格自然语言摘要生成器

为 markdown 表格生成 1-3 句自然语言摘要，存入向量库：
- 检索时用语义摘要定位表格（Dense 向量对摘要编码，精度更高、context 更省）
- 生成时仍用完整 markdown 表格（见 generator.py 的表格保护逻辑）

可选上下文（Phase A1）：同页邻段 / 标题 / caption，用于消歧；默认关。
复用 ragas_metrics.call_llm（与评测同套 LLM 调用）。
"""
from __future__ import annotations

import logging
from functools import lru_cache
from typing import Iterable, Optional, Sequence

from src.evaluation.ragas_metrics import call_llm
from src.prompts import get_active, get_prompt

logger = logging.getLogger(__name__)

# v1 生效模板（无上下文）；import 期解析，与历史单测字节一致。
_TABLE_SUMMARY_PROMPT = get_active("table_summary").template

# 有上下文时使用的版本号（YAML 中 active: false，由本模块显式选取）
_CONTEXT_PROMPT_VERSION = 2


class TableSummarizer:
    """为 markdown 表格生成 NL 摘要。

    带进程内缓存（相同表格+上下文不重复调 LLM）与失败降级（异常/空响应返回 ""）。
    """

    def __init__(
        self,
        enabled: bool = True,
        context_enabled: bool = False,
        context_max_chars: int = 1500,
    ):
        self.enabled = enabled
        self.context_enabled = context_enabled
        self.context_max_chars = max(0, int(context_max_chars))
        # 缓存键为 (table, context)，避免同一张表在不同语境下串缓存
        self._summarize_cached = lru_cache(maxsize=2048)(self._summarize_uncached)

    def summarize(self, table_md: str, *, context: str = "") -> str:
        if not self.enabled or not table_md or not table_md.strip():
            return ""
        ctx = self._normalize_context(context)
        try:
            return self._summarize_cached(table_md.strip(), ctx)
        except Exception as e:  # 任意异常都降级，不阻塞入库
            logger.warning(f"表格摘要生成失败，降级为空: {e}")
            return ""

    def _normalize_context(self, context: str) -> str:
        if not self.context_enabled:
            return ""
        if not context or not str(context).strip():
            return ""
        text = str(context).strip()
        if self.context_max_chars and len(text) > self.context_max_chars:
            text = text[: self.context_max_chars].rstrip() + "..."
        return text

    def build_page_context(
        self,
        chunks: Sequence,
        *,
        exclude_chunk_ids: Optional[Iterable[str]] = None,
    ) -> str:
        """从同页 chunk 装配摘要用上下文（排除表自身，优先非 table 文本）。

        chunks 元素需有 ``text``、``chunk_type``，可选 ``chunk_id``。
        context_enabled=false 时返回空串（调用方可跳过）。
        """
        if not self.context_enabled or self.context_max_chars <= 0:
            return ""
        exclude = set(exclude_chunk_ids or ())
        parts: list[str] = []
        for c in chunks:
            cid = getattr(c, "chunk_id", None) or (c.get("chunk_id") if isinstance(c, dict) else None)
            if cid is not None and cid in exclude:
                continue
            ctype = getattr(c, "chunk_type", None) or (c.get("chunk_type") if isinstance(c, dict) else "text")
            if ctype == "table":
                continue
            text = getattr(c, "text", None) or (c.get("text") if isinstance(c, dict) else "") or ""
            text = text.strip()
            if text:
                parts.append(text)
        if not parts:
            return ""
        return self._normalize_context("\n".join(parts))

    def _summarize_uncached(self, table_md: str, context: str = "") -> str:
        if context:
            prompt = self._context_prompt().format(table=table_md, context=context)
        else:
            prompt = _TABLE_SUMMARY_PROMPT.format(table=table_md)
        resp = call_llm(prompt, max_retries=2)
        summary = (resp or "").strip()
        # 去掉模型可能加的 ```markdown 包裹
        if summary.startswith("```"):
            summary = summary.strip("`")
            if summary.lower().startswith("markdown"):
                summary = summary[len("markdown"):].strip()
        return summary

    @staticmethod
    def _context_prompt() -> str:
        prompt = get_prompt("table_summary")
        for v in prompt.versions:
            if v.version == _CONTEXT_PROMPT_VERSION:
                return v.template
        raise RuntimeError(
            f"table_summary prompt version {_CONTEXT_PROMPT_VERSION} not found"
        )
