"""表格自然语言摘要生成器

为 markdown 表格生成 1-3 句自然语言摘要，存入向量库：
- 检索时用语义摘要定位表格（Dense 向量对摘要编码，精度更高、context 更省）
- 生成时仍用完整 markdown 表格（见 generator.py 的表格保护逻辑）

复用 ragas_metrics.call_llm（与评测同套 LLM 调用）。
"""
from __future__ import annotations

import logging
from functools import lru_cache

from src.evaluation.ragas_metrics import call_llm
from src.prompts import get_active

logger = logging.getLogger(__name__)

# 模板已外置到 src/prompts/prompts/table_summary.yaml，import 期解析为生效版本文本。
_TABLE_SUMMARY_PROMPT = get_active("table_summary").template


class TableSummarizer:
    """为 markdown 表格生成 NL 摘要。

    带进程内缓存（相同表格不重复调 LLM）与失败降级（异常/空响应返回 ""）。
    """

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        # 缓存键为表格文本，避免同一张表在文档/页面间重复生成
        self._summarize_cached = lru_cache(maxsize=2048)(self._summarize_uncached)

    def summarize(self, table_md: str) -> str:
        if not self.enabled or not table_md or not table_md.strip():
            return ""
        try:
            return self._summarize_cached(table_md.strip())
        except Exception as e:  # 任意异常都降级，不阻塞入库
            logger.warning(f"表格摘要生成失败，降级为空: {e}")
            return ""

    def _summarize_uncached(self, table_md: str) -> str:
        prompt = _TABLE_SUMMARY_PROMPT.format(table=table_md)
        resp = call_llm(prompt, max_retries=2)
        summary = (resp or "").strip()
        # 去掉模型可能加的 ```markdown 包裹
        if summary.startswith("```"):
            summary = summary.strip("`")
            if summary.lower().startswith("markdown"):
                summary = summary[len("markdown"):].strip()
        return summary
