"""统一拒答文案与检测（生成 / RAGAS / E2E 共用）。

背景：Self-RAG Gate2 拒答句与 RAGAS 短语表不一致时，拒答会被记 Faith=0 并污染均值。
见 runs/20260721-self-rag-gate2/badcase_analysis.md。
"""
from __future__ import annotations

from typing import Optional

# 全系统规范拒答句（Self-RAG / 空检索 / 文档建议对齐此句）
ABSTAIN_ANSWER = "I don't have enough information to answer that question."

# 子串匹配（answer.lower()）；覆盖硬拒答 + 常见软拒答
REJECTION_PHRASES = (
    "cannot answer",
    "not enough information",
    "don't have enough information",
    "do not have enough information",
    "don't have enough",
    "do not have enough",
    "based on the available",
    "cannot provide",
    "i don't have",
    "i do not have",
    "i don't know",
    "i do not know",
    "no information",
    "not covered",
    "out of scope",
    "beyond the scope",
    "the context does not contain",
    "the provided context does not",
    "context provided does not",
    "context does not provide",
    "does not contain information",
    "does not include information",
    "not provided in the context",
    "not specified in the context",
    "not in the provided context",
)


def is_rejection(answer: Optional[str]) -> bool:
    """判断回答是否为拒绝回答（含空答案）。"""
    if not answer:
        return True
    lower = answer.lower()
    return any(phrase in lower for phrase in REJECTION_PHRASES)


# 兼容旧名
is_answer_rejected = is_rejection
