"""LLM 生成（OpenAI SDK）+ 引用回链。引用以检索 chunk 为准，不依赖模型自报。"""
from __future__ import annotations
import logging
import os
from typing import List

import openai

from src.config import cfg
from src.evaluation.ragas_metrics import compress_context
from src.observability import get_tracer
from src.prompts import get_active

logger = logging.getLogger(__name__)


class GenerationError(RuntimeError):
    pass


class Generator:
    def __init__(self, client=None, bge_embedder=None):
        if client is None:
            from openai import OpenAI
            client = OpenAI(
                base_url=cfg.get("llm.base_url", "https://api.openai.com/v1"),
                api_key=cfg.get("llm.api_key", "") or os.environ.get("OPENAI_API_KEY", ""),
            )
        self.client = client
        self.model = cfg.get("llm.model", "gpt-4o-mini")
        self.bge = bge_embedder
        # 生成温度：硬编码 0.0 保证答案确定性，从而可安全缓存（L4 Answer 缓存守卫）。
        self.temperature = 0.0

    def answer(self, query: str, retrieved: List[dict], k_context: int = 5) -> dict:
        tracer = get_tracer()
        with tracer.start_span(
            "generation", metadata={"model": self.model, "k_context": k_context}
        ) as gen_span:
            top = retrieved[:k_context]
            if not top:
                gen_span.set_metadata({
                    "num_retrieved": 0, "num_citations": 0,
                    "citations": [], "context": "",
                })
                return {"answer": "I don't have enough information to answer that question.",
                        "citations": [], "context": ""}

        # 表格 chunk 检索时按摘要定位，但生成时必须展开完整 Markdown 表格喂给 LLM，
        # 因此整表跳过 compress_context（句级压缩会把表格行删掉，破坏结构）。
        # 非表格 chunk 仍走 BGE 句级压缩，按原排序拼接上下文。
        table_parts: dict = {}        # orig_index -> 完整表格 markdown
        text_idx: list = []           # 非表格 chunk 的原排序下标
        text_texts: list = []         # 非表格 chunk 的文本
        for i, r in enumerate(top):
            if r.get("chunk_type") == "table":
                table_parts[i] = r["text"]
            else:
                text_idx.append(i)
                text_texts.append(r["text"])

        if text_texts:
            if self.bge is not None:
                compressed_text = compress_context(
                    query, text_texts, self.bge,
                    ratio=cfg.get("retrieval.context_compression_ratio", 0.4),
                )
            else:
                compressed_text = "\n\n".join(text_texts)
            # 压缩结果是一个整体块，挂在非表格 chunk 的最小下标处，保持原有相对顺序
            table_parts[min(text_idx)] = (
                table_parts.get(min(text_idx), "") + "\n\n" + compressed_text
            ).strip()

        context = "\n\n".join(table_parts[i] for i in sorted(table_parts))

        pv = get_active("answer_generation")
        prompt = [
            {"role": "system", "content": pv.system},
            {"role": "user", "content": pv.render("user", context=context, query=query)},
        ]
        try:
            resp = self.client.chat.completions.create(
                model=self.model, messages=prompt, temperature=self.temperature,
            )
        except (openai.APIError, openai.APIConnectionError, openai.APITimeoutError) as e:
            raise GenerationError(f"LLM call failed: {e}") from e

        answer_text = resp.choices[0].message.content
        citations = [
            {"chunk_id": r["chunk_id"], "page_id": r["page_id"],
             "doc_id": r.get("doc_id"), "page_number": r.get("page_number"),
             "snippet": (r.get("text") or "")[:200]}
            for r in top
        ]
        # 完整 context 写入 span metadata —— 排查"context 里有没有答案"的关键
        gen_span.set_metadata({
            "num_retrieved": len(top),
            "num_citations": len(citations),
            "citations": citations,
            "context": context,
        })
        return {"answer": answer_text, "citations": citations, "context": context}

    @property
    def cacheable(self) -> bool:
        """生成结果是否可安全缓存：仅当温度确定性（temperature==0）时为 True。

        非确定性生成（temperature>0）不可缓存，否则会返回不稳定的旧答案。
        L4 Answer 缓存据此守卫：非 cacheable 时既不读也不写答案缓存。
        """
        return self.temperature == 0.0
