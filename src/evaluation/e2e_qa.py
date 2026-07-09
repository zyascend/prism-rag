"""端到端 QA 评测 — Answer Correctness + Rejection Accuracy

评测三层中的第三层（端到端层）：
  - 可回答问题：答案是否正确（LLM-as-judge 判断）
  - 拒答问题：系统是否正确拒绝回答问题

评测流程：
  1. 加载 QA 数据集（可回答 + 拒答混合）
  2. 对每条 query：检索 → 生成
  3. 可回答：LLM-as-judge 比较生成答案与预期答案
  4. 拒答：检查是否包含拒绝短语
  5. 输出汇总指标
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np
import requests

from src.observability import get_tracer

logger = logging.getLogger(__name__)

# ─── 常量 ─────────────────────────────────────────────────────
OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"
DEFAULT_LLM_MODEL = "qwen2:7b"

# ─── 数据结构 ──────────────────────────────────────────────────


@dataclass
class AnswerCorrectnessResult:
    """单条可回答问题的答案正确性结果"""
    question: str
    expected_answer: str
    generated_answer: str
    context_length: int
    is_correct: bool = False
    correctness_score: float = 0.0
    judge_reasoning: str = ""

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "expected_answer": self.expected_answer,
            "generated_answer": self.generated_answer,
            "context_length": self.context_length,
            "is_correct": self.is_correct,
            "correctness_score": round(self.correctness_score, 4),
            "judge_reasoning": self.judge_reasoning,
        }


@dataclass
class RejectionResult:
    """单条拒答问题结果"""
    question: str
    expected_rejection: bool
    generated_answer: str
    is_rejected: bool = False
    rejection_correct: bool = False

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "expected_rejection": self.expected_rejection,
            "generated_answer": self.generated_answer,
            "is_rejected": self.is_rejected,
            "rejection_correct": self.rejection_correct,
        }


@dataclass
class E2EQAEvalResult:
    """端到端 QA 评测完整结果"""
    correctness_results: List[AnswerCorrectnessResult] = field(default_factory=list)
    rejection_results: List[RejectionResult] = field(default_factory=list)
    avg_correctness: float = 0.0
    rejection_accuracy: float = 0.0
    combined_score: float = 0.0
    num_answerable: int = 0
    num_rejection: int = 0
    latency_total: float = 0.0
    rejected_count_answerable: int = 0  # 可回答问题中被拒绝的（不合理拒答）

    def _compute_combined_score(self) -> float:
        """计算综合分数：正确率 70% + 拒答准确率 30%"""
        total_weight = 0.7 if self.num_answerable > 0 else 0.0
        rejection_weight = 0.3 if self.num_rejection > 0 else 0.0
        total_weight_sum = total_weight + rejection_weight
        if total_weight_sum > 0:
            return (self.avg_correctness * total_weight + self.rejection_accuracy * rejection_weight) / total_weight_sum
        return 0.0

    def summary_dict(self) -> dict:
        return {
            "num_answerable": self.num_answerable,
            "num_rejection": self.num_rejection,
            "avg_correctness": round(self.avg_correctness, 4),
            "rejection_accuracy": round(self.rejection_accuracy, 4),
            "combined_score": round(self._compute_combined_score(), 4),
            "avg_latency_seconds": round(
                self.latency_total / (self.num_answerable + self.num_rejection)
                if (self.num_answerable + self.num_rejection) > 0 else 0.0, 2
            ),
            "rejected_count_answerable": self.rejected_count_answerable,
        }


# ─── Ollama 调用 ──────────────────────────────────────────────


def call_llm(prompt: str, model: str = DEFAULT_LLM_MODEL, max_retries: int = 2) -> str:
    """调用 Ollama 生成文本，带重试"""
    for attempt in range(max_retries + 1):
        try:
            resp = requests.post(
                OLLAMA_CHAT_URL,
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False,
                    "options": {"num_predict": 1024, "temperature": 0.1},
                },
                timeout=120,
            )
            resp.raise_for_status()
            return resp.json()["message"]["content"]
        except Exception as e:
            logger.warning(f"Ollama 调用失败 (attempt {attempt+1}): {e}")
            if attempt < max_retries:
                time.sleep(3)
    return ""


# ─── Prompt 模板 ──────────────────────────────────────────────

ANSWER_CORRECTNESS_JUDGE_PROMPT = """\
You are a strict judge evaluating answer correctness for industrial document QA.
Compare the GENERATED ANSWER against the EXPECTED ANSWER for the given QUESTION.

Rules:
- Return YES if the generated answer is semantically equivalent to the expected answer.
- Return NO if the generated answer contradicts the expected answer or misses key information.
- Accept minor phrasing differences, but not missing facts.
- If the generated answer says "I cannot answer" but the question IS answerable, return NO.

Format your response as:
JUDGMENT: YES|NO
REASONING: <brief explanation>

Question: {question}

Expected Answer: {expected_answer}

Generated Answer: {generated_answer}

JUDGMENT:"""

GENERATION_PROMPT = """\
You are a helpful assistant for industrial document QA.
Answer the question based ONLY on the provided context.
If the context does not contain enough information, say "I cannot answer this question based on the available documents."
Do NOT make up information.

Context:
{context}

Question: {question}

Answer:"""

REJECTION_PHRASES = [
    "cannot answer", "not enough information",
    "based on the available", "cannot provide",
    "i don't have", "i do not have",
    "no information", "not covered",
    "out of scope", "beyond the scope",
    "the context does not contain",
    "the provided context does not",
]

# ─── 核心评测函数 ──────────────────────────────────────────────


def is_answer_rejected(answer: str) -> bool:
    """判断回答是否为拒绝回答"""
    if not answer:
        return True
    return any(phrase in answer.lower() for phrase in REJECTION_PHRASES)


def compute_answer_correctness(
    question: str,
    expected_answer: str,
    generated_answer: str,
) -> AnswerCorrectnessResult:
    """用 LLM-as-judge 判断生成答案是否与预期答案含义一致

    Args:
        question: 原始问题
        expected_answer: 预期正确答案
        generated_answer: 系统生成的回答

    Returns:
        AnswerCorrectnessResult
    """
    result = AnswerCorrectnessResult(
        question=question,
        expected_answer=expected_answer,
        generated_answer=generated_answer,
        context_length=0,  # 调用方填充
    )

    # 如果生成答案为空，直接判错
    if not generated_answer:
        result.is_correct = False
        result.correctness_score = 0.0
        result.judge_reasoning = "Empty generated answer"
        return result

    # 如果生成答案是拒答（说明检索没找到相关内容），判错
    if is_answer_rejected(generated_answer):
        result.is_correct = False
        result.correctness_score = 0.0
        result.judge_reasoning = "System refused to answer (rejection detected)"
        return result

    prompt = ANSWER_CORRECTNESS_JUDGE_PROMPT.format(
        question=question,
        expected_answer=expected_answer,
        generated_answer=generated_answer,
    )
    judge_response = call_llm(prompt)

    if not judge_response:
        # Ollama 失败 → 保守判错
        result.is_correct = False
        result.correctness_score = 0.0
        result.judge_reasoning = "LLM judge call failed"
        return result

    # 解析 JUDGMENT 行
    lines = judge_response.strip().split("\n")
    judgment_line = ""
    reasoning_lines = []
    for line in lines:
        line_stripped = line.strip()
        if line_stripped.upper().startswith("JUDGMENT:"):
            judgment_line = line_stripped
        elif line_stripped.upper().startswith("REASONING:"):
            reasoning_lines.append(line_stripped)

    # 如果无 JUDGMENT: 前缀，模型可能直接输出 YES/NO
    # （prompt 末尾已是 JUDGMENT:，模型只填 YES/NO 不重复前缀）
    if not judgment_line:
        for line in lines:
            ls = line.strip().upper()
            if ls in ("YES", "NO") or ls.startswith("YES") or ls.startswith("NO"):
                judgment_line = f"JUDGMENT: {ls}"
                break

    judgment_text = judgment_line.replace("JUDGMENT:", "", 1).strip().upper()
    result.judge_reasoning = "\n".join(reasoning_lines) if reasoning_lines else judge_response

    if judgment_text.startswith("YES"):
        result.is_correct = True
        result.correctness_score = 1.0
    else:
        result.is_correct = False
        result.correctness_score = 0.0

    return result


def generate_answer(query: str, context: str) -> str:
    """基于检索上下文生成回答"""
    if not context:
        return "I cannot answer this question based on the available documents."
    prompt = GENERATION_PROMPT.format(context=context[:12000], question=query)
    answer = call_llm(prompt)
    return answer if answer else ""


# ─── 数据集加载 ──────────────────────────────────────────────


def load_qa_dataset(qa_path: str) -> tuple:
    """加载端到端 QA 数据集

    Args:
        qa_path: JSON 文件路径

    Returns:
        (answerable_questions, rejection_questions)
        answerable_questions: List[dict] - 可回答问题
        rejection_questions: List[dict] - 拒答问题
    """
    with open(qa_path) as f:
        data = json.load(f)

    answerable = []
    rejection = []
    for item in data:
        q_type = item.get("type", item.get("expected_rejection", False) and "rejection" or "answerable")
        if q_type == "rejection" or item.get("expected_rejection", False):
            rejection.append(item)
        else:
            answerable.append(item)

    logger.info(f"加载端到端 QA 数据集: {len(answerable)} 条可回答 + {len(rejection)} 条拒答")
    return answerable, rejection


# ─── 批量评测 ──────────────────────────────────────────────────


def evaluate_e2e_qa(
    retriever,
    qa_path: str = "data/e2e_qa.json",
    k: int = 5,
    use_rerank: bool = True,
    max_queries: Optional[int] = None,
    output_dir: str = "results",
) -> E2EQAEvalResult:
    """端到端 QA 批量评测

    流程：
      1. 加载 QA 数据集
      2. 对每条 question：检索 → 生成
      3. 可回答 → LLM-as-judge 判正确性
      4. 拒答 → 检查拒绝短语
      5. 输出汇总指标

    Args:
        retriever: PrismRAGRetriever 实例
        qa_path: QA 数据集 JSON 路径
        k: 检索 top-k
        use_rerank: 是否使用重排序
        max_queries: 最大总查询数（None=全部）
        output_dir: 输出目录

    Returns:
        E2EQAEvalResult 汇总
    """
    from tqdm import tqdm

    answerable_qs, rejection_qs = load_qa_dataset(qa_path)

    # 限制总数
    total_items = len(answerable_qs) + len(rejection_qs)
    if max_queries and max_queries < total_items:
        # 优先保留 rejections，再截取 answerable
        if max_queries < len(rejection_qs):
            rejection_qs = rejection_qs[:max_queries]
            answerable_qs = []
        else:
            max_answerable = max_queries - len(rejection_qs)
            answerable_qs = answerable_qs[:max_answerable]

    logger.info(
        f"开始端到端 QA 评测: {len(answerable_qs)} 条可回答 + {len(rejection_qs)} 条拒答, "
        f"top-k={k}, rerank={use_rerank}"
    )

    correctness_results = []
    rejection_results = []
    latency_total = 0.0
    rejected_answerable = 0

    # ── 处理可回答问题 ─────────────────────────────────────────
    for item in tqdm(answerable_qs, desc="Answerable QA"):
        question = item["question"]
        expected_answer = item["expected_answer"]

        tracer = get_tracer()
        with tracer.start_span("e2e_qa_answerable") as span:
            start = time.time()

            # 检索
            retrieved = retriever.search(question, k=k, use_rerank=use_rerank)
            context = "\n\n---\n\n".join(
                [r.get("text", "") for r in retrieved]
            ) if retrieved else ""

            # 生成
            generated = generate_answer(question, context)

            elapsed = time.time() - start
            latency_total += elapsed

            # 检查是否被不合理拒答
            if is_answer_rejected(generated):
                rejected_answerable += 1

            # LLM-as-judge 判正确性
            correctness_result = compute_answer_correctness(
                question=question,
                expected_answer=expected_answer,
                generated_answer=generated,
            )
            correctness_result.context_length = len(context)
            correctness_results.append(correctness_result)

            span.set_metadata({
                "type": "answerable",
                "correct": correctness_result.is_correct,
                "latency": round(elapsed, 2),
            })

            if not correctness_result.is_correct:
                logger.debug(
                    f"  ✗ {question[:60]}... → expected='{expected_answer[:60]}...' "
                    f"got='{generated[:60]}...'"
                )

    # ── 处理拒答问题 ───────────────────────────────────────────
    for item in tqdm(rejection_qs, desc="Rejection QA"):
        question = item["question"]
        expected_rejection = item.get("expected_rejection", True)

        tracer = get_tracer()
        with tracer.start_span("e2e_qa_rejection") as span:
            start = time.time()

            retrieved = retriever.search(question, k=k, use_rerank=use_rerank)
            context = "\n\n---\n\n".join(
                [r.get("text", "") for r in retrieved]
            ) if retrieved else ""

            generated = generate_answer(question, context)

            elapsed = time.time() - start
            latency_total += elapsed

            is_rejected = is_answer_rejected(generated)
            rejection_correct = is_rejected == expected_rejection

            result = RejectionResult(
                question=question,
                expected_rejection=expected_rejection,
                generated_answer=generated,
                is_rejected=is_rejected,
                rejection_correct=rejection_correct,
            )
            rejection_results.append(result)

            span.set_metadata({
                "type": "rejection",
                "expected": expected_rejection,
                "actual": is_rejected,
                "correct": rejection_correct,
                "latency": round(elapsed, 2),
            })

    # ── 汇总 ──────────────────────────────────────────────────
    num_answerable = len(correctness_results)
    num_rejection = len(rejection_results)

    correctness_scores = [r.correctness_score for r in correctness_results]
    avg_correctness = float(np.mean(correctness_scores)) if correctness_scores else 0.0

    rejection_accuracy_scores = [1.0 if r.rejection_correct else 0.0 for r in rejection_results]
    rejection_accuracy = float(np.mean(rejection_accuracy_scores)) if rejection_accuracy_scores else 0.0

    # 综合分数：正确率 70% + 拒答准确率 30%
    total_weight = 0.7 if num_answerable > 0 else 0.0
    rejection_weight = 0.3 if num_rejection > 0 else 0.0
    total_weight_sum = total_weight + rejection_weight
    combined_score = (
        (avg_correctness * total_weight + rejection_accuracy * rejection_weight) / total_weight_sum
        if total_weight_sum > 0 else 0.0
    )

    result = E2EQAEvalResult(
        correctness_results=correctness_results,
        rejection_results=rejection_results,
        avg_correctness=avg_correctness,
        rejection_accuracy=rejection_accuracy,
        combined_score=combined_score,
        num_answerable=num_answerable,
        num_rejection=num_rejection,
        latency_total=latency_total,
        rejected_count_answerable=rejected_answerable,
    )

    # ── 日志输出 ──
    avg_latency = latency_total / (num_answerable + num_rejection) if (num_answerable + num_rejection) > 0 else 0.0

    logger.info(f"\n{'='*60}")
    logger.info("端到端 QA 评测结果")
    logger.info(f"{'='*60}")
    logger.info(f"  可回答问题:     {num_answerable}")
    logger.info(f"  拒答问题:       {num_rejection}")
    logger.info(f"  平均延迟:       {avg_latency:.1f}s")
    logger.info("  ───────────────")
    logger.info(f"  答案正确率:     {avg_correctness:.2%} ({num_answerable} 条)")
    logger.info(f"  不合理拒答:     {rejected_answerable} ({rejected_answerable/num_answerable:.1%})" if num_answerable > 0 else "")
    logger.info(f"  拒答准确率:     {rejection_accuracy:.2%} ({num_rejection} 条)")
    logger.info(f"  综合分数:       {combined_score:.2%}")
    logger.info(f"{'='*60}")

    # ── 保存 ──
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    with open(output_path / "e2e_qa_results.json", "w") as f:
        json.dump({
            "summary": result.summary_dict(),
            "correctness": [r.to_dict() for r in correctness_results],
            "rejection": [r.to_dict() for r in rejection_results],
        }, f, indent=2, ensure_ascii=False)

    logger.info(f"  结果已保存: {output_path / 'e2e_qa_results.json'}")

    # ── Bad Case 报告 ──
    _write_badcase_report(correctness_results, output_path)

    return result


def _write_badcase_report(
    correctness_results: List[AnswerCorrectnessResult],
    output_path: Path,
) -> None:
    """生成 Bad Case 分析 Markdown 报告"""
    incorrect = [r for r in correctness_results if not r.is_correct]
    if not incorrect:
        return

    lines = [
        "# 端到端 QA Bad Case 分析\n",
        f"总可回答: {len(correctness_results)}, 错误: {len(incorrect)} ({len(incorrect)/len(correctness_results):.1%})\n",
    ]

    for i, r in enumerate(incorrect, 1):
        lines.append(f"## Bad Case #{i}: {r.question}\n")
        lines.append(f"- **预期答案**: {r.expected_answer}")
        lines.append(f"- **生成答案**: {r.generated_answer}")
        lines.append(f"- **Judge 推理**: {r.judge_reasoning}")
        lines.append("")

    report_path = output_path / "badcase_e2e_qa_analysis.md"
    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    logger.info(f"  Bad Case 报告: {report_path}")