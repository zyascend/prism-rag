"""LLM 生成（OpenAI SDK）+ 引用回链。引用以检索 chunk 为准，不依赖模型自报。"""
from __future__ import annotations
import logging
import os
from typing import List

import openai

from src.config import cfg
from src.evaluation.ragas_metrics import compress_context

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

    def answer(self, query: str, retrieved: List[dict], k_context: int = 5) -> dict:
        top = retrieved[:k_context]
        if not top:
            return {"answer": "I don't have enough information to answer that question.",
                    "citations": [], "context": ""}
        contexts = [r["text"] for r in top]
        if self.bge is not None:
            context = compress_context(
                query, contexts, self.bge,
                ratio=cfg.get("retrieval.context_compression_ratio", 0.4),
            )
        else:
            context = "\n\n".join(contexts)

        prompt = [
            {"role": "system", "content":
             "You are a precise assistant. Answer ONLY from the provided context. "
             "If the context lacks the answer, say you don't know."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
        ]
        try:
            resp = self.client.chat.completions.create(
                model=self.model, messages=prompt, temperature=0.0,
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
        return {"answer": answer_text, "citations": citations, "context": context}
