"""HyDE (Hypothetical Document Embeddings) 查询改写

使用 LLM 为每个 query 生成假设性答案文档，作为额外的检索查询增强召回。
"""

from __future__ import annotations

import logging

import requests

from src.config import cfg
from src.observability import get_tracer

logger = logging.getLogger(__name__)

OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"

HYDE_PROMPT = (
    "Write a technical passage that answers the following question about "
    "industrial documents. Be specific and factual.\n\n"
    "Question: {query}\n\n"
    "Passage:"
)


class HyDEGenerator:
    """使用 Ollama 生成假设性文档用于查询增强。

    Args:
        model: Ollama 模型名，默认取 config 中 llm 配置项。
        timeout: API 超时（秒），默认 60。
    """

    def __init__(self, model: str | None = None, timeout: int = 60):
        self.model = model or cfg.llm_model_id
        self.timeout = timeout
        self._cache: dict[str, str] = {}

    def precompute(self, queries: list[str]) -> dict[str, str]:
        """批量预计算 HyDE 答案，存入缓存。

        在加载大模型前调用，可独占 GPU 加速 Ollama 推理。
        预计算完成后可关闭 Ollama 释放显存，eval 阶段直接从缓存读取。

        Args:
            queries: 查询文本列表

        Returns:
            {query: hyde_answer} 字典
        """
        from tqdm import tqdm

        for q in tqdm(queries, desc="HyDE precompute"):
            if q not in self._cache:
                self._cache[q] = self._generate_impl(q)
        return dict(self._cache)

    def generate(self, query: str) -> str:
        """为给定 query 生成假设性答案文档。

        优先从缓存读取，缓存未命中时实时调用 Ollama。

        Args:
            query: 原始查询文本

        Returns:
            生成的假设文档文本。失败时返回空字符串。
        """
        tracer = get_tracer()
        with tracer.start_span("hyde_generate") as span:
            if query in self._cache:
                span.set_metadata({"cache_hit": True})
                return self._cache[query]
            span.set_metadata({"cache_hit": False})
            result = self._generate_impl(query)
            return result

    def _generate_impl(self, query: str) -> str:
        """实际调用 Ollama 生成 HyDE 答案。"""
        prompt = HYDE_PROMPT.format(query=query)

        try:
            resp = requests.post(
                OLLAMA_CHAT_URL,
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False,
                },
                timeout=self.timeout,
            )
            resp.raise_for_status()
            answer = resp.json()["message"]["content"].strip()
            if answer:
                logger.debug(f"HyDE generated {len(answer)} chars for query: {query[:80]}...")
            else:
                logger.warning("HyDE returned empty response")
            return answer
        except requests.exceptions.Timeout:
            logger.warning(f"HyDE timed out after {self.timeout}s for query: {query[:80]}...")
        except requests.exceptions.ConnectionError:
            logger.warning("HyDE: Ollama not reachable at localhost:11434")
        except Exception as e:
            logger.warning(f"HyDE generation failed: {e}")

        return ""
