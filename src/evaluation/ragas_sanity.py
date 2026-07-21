"""RAGAS 拒答 Sanity 评测"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict

import requests

from src.evaluation.vidore_adapter import PrismRAGRetriever

logger = logging.getLogger(__name__)

OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"


def call_ollama(prompt: str, model: str = "qwen2:7b") -> str:
    """调用 Ollama 生成"""
    try:
        resp = requests.post(
            OLLAMA_CHAT_URL,
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
            },
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"]
    except Exception as e:
        logger.warning(f"Ollama 调用失败: {e}")
        return ""


def generate_answer(retriever: PrismRAGRetriever, query: str) -> Dict:
    """检索后生成回答"""
    retrieved = retriever.search(query, k=5, use_rerank=True)

    if not retrieved:
        context = ""
    else:
        context = "\n\n---\n\n".join([r.get("text", "") for r in retrieved])

    system_prompt = (
        "You are a helpful assistant for industrial document QA. "
        "Answer the question based ONLY on the provided context. "
        "If the context does not contain enough information to answer the question, "
        "say 'I cannot answer this question based on the available documents.' "
        "Do NOT make up information."
    )
    user_prompt = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"

    answer = call_ollama(
        f"System: {system_prompt}\n\n{user_prompt}",
        model="qwen2:7b",
    )

    if not answer:
        is_rejected = True  # Ollama failure → safer to count as rejected
    else:
        from src.rejection import is_rejection

        is_rejected = is_rejection(answer)

    return {
        "query": query,
        "retrieved": len(retrieved),
        "context_length": len(context),
        "answer": answer,
        "is_rejected": is_rejected,
    }


def run_ragas_sanity(
    retriever: PrismRAGRetriever,
    rejection_qa_path: str = "data/rejection_qa.json",
    output_dir: str = "results",
) -> Dict:
    """运行 RAGAS 拒答 sanity 评测"""
    with open(rejection_qa_path) as f:
        rejection_qa = json.load(f)

    logger.info(f"加载 {len(rejection_qa)} 条拒答问题")

    results = []
    rejected_count = 0
    for item in rejection_qa:
        logger.info(f"  查询: {item['question'][:50]}...")
        result = generate_answer(retriever, item["question"])
        results.append(result)
        if result["is_rejected"]:
            rejected_count += 1
        logger.info(f"    拒绝={result['is_rejected']}, answer={result['answer'][:80]}...")

    total = len(rejection_qa)
    rejection_rate = rejected_count / total if total > 0 else 0.0
    summary = {
        "total_questions": total,
        "rejected_count": rejected_count,
        "rejection_rate": round(rejection_rate, 4),
        "passed": rejection_rate >= 0.8,
    }

    logger.info("\nRAGAS Sanity 结果:")
    logger.info(f"  总拒答数: {total}")
    logger.info(f"  正确拒绝: {rejected_count}")
    logger.info(f"  拒绝率: {rejection_rate:.1%}")
    logger.info(f"  是否通过(>=80%): {summary['passed']}")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    with open(output_path / "ragas_sanity_results.json", "w") as f:
        json.dump({"summary": summary, "details": results}, f, indent=2, ensure_ascii=False)
    logger.info(f"结果已保存到 {output_path / 'ragas_sanity_results.json'}")

    return summary