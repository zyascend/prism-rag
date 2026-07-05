"""RAGAS 指标自实现 — Faithfulness + Answer Relevancy

使用 Ollama qwen2:7b 作为 Judge LLM，BGE 作为嵌入模型。
不依赖 ragas 库（0.4.x 依赖链断裂），按照 RAGAS 论文算法逻辑自实现。

指标说明：
  - Faithfulness（忠实度）：答案中的每个原子声明是否被检索上下文支持。
    分解答案为原子声明 → 逐个问 LLM "上下文是否支持该声明" → 支持率。
  - Answer Relevancy（答案相关性）：答案与问题的相关程度。
    从答案反向生成 N 个推测性问题 → BGE 嵌入对比原问题 → 余弦相似度均值。
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import requests

from src.config import cfg
from src.observability import get_tracer, get_collector

logger = logging.getLogger(__name__)

# ─── 常量 ─────────────────────────────────────────────────────
OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"
OLLAMA_EMBED_URL = "http://localhost:11434/api/embed"
DEFAULT_LLM_MODEL = "qwen2:7b"
DEFAULT_N_REVERSE_QUESTIONS = 3  # Answer Relevancy 生成的反向问题数量


# ─── 数据结构 ──────────────────────────────────────────────────

@dataclass
class FaithfulnessResult:
    """单条查询的 Faithfulness 结果"""
    query: str
    answer: str
    context_length: int
    claims: List[str] = field(default_factory=list)
    supported: List[bool] = field(default_factory=list)
    scores: List[float] = field(default_factory=list)
    faithfulness_score: float = 0.0

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "answer": self.answer,
            "context_length": self.context_length,
            "claims": self.claims,
            "supported": self.supported,
            "faithfulness_score": round(self.faithfulness_score, 4),
        }


@dataclass
class AnswerRelevancyResult:
    """单条查询的 Answer Relevancy 结果"""
    query: str
    answer: str
    generated_questions: List[str] = field(default_factory=list)
    similarities: List[float] = field(default_factory=list)
    relevancy_score: float = 0.0

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "answer": self.answer,
            "generated_questions": self.generated_questions,
            "similarities": [round(s, 4) for s in self.similarities],
            "relevancy_score": round(self.relevancy_score, 4),
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


def call_ollama_embed(texts: List[str], model: str = "nomic-embed-text") -> Optional[np.ndarray]:
    """调用 Ollama 批量获取文本嵌入

    使用 nomic-embed-text（轻量，~137M 参数），
    如有 bge-large-en-v1.5 也可使用。

    Returns:
        shape (n_texts, dim) 的 numpy 数组，失败时返回 None
    """
    try:
        resp = requests.post(
            OLLAMA_EMBED_URL,
            json={"model": model, "input": texts},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        embeddings = data.get("embeddings")
        if embeddings:
            return np.array(embeddings, dtype=np.float32)
        return None
    except Exception as e:
        logger.warning(f"Ollama embed 调用失败: {e}")
        return None


# ─── Prompt 模板 ──────────────────────────────────────────────

CLAIM_DECOMPOSITION_PROMPT = """\
Break down the following answer into individual factual claims.
Each claim must be a single, atomic, verifiable statement.
Format: one claim per line, numbered.

Answer: {answer}

Claims:"""

CLAIM_VERIFICATION_PROMPT = """\
Determine whether the following claim is DIRECTLY SUPPORTED by the given context.
Answer ONLY with YES or NO.

Context:
{context}

Claim: {claim}

Is this claim directly supported by the context? (YES/NO)"""

REVERSE_QUESTION_PROMPT = """\
Given the following answer, generate {n} different questions that this answer could be answering.
Each question should be phrased as a natural question someone might ask.
Format: one question per line, numbered.

Answer: {answer}

Questions:"""

GENERATION_PROMPT = """\
You are a helpful assistant for industrial document QA.
Answer the question based ONLY on the provided context.
If the context does not contain enough information, say "I cannot answer this question based on the available documents."
Do NOT make up information.

Context:
{context}

Question: {question}

Answer:"""


# ─── Faithfulness ─────────────────────────────────────────────

def decompose_claims(answer: str) -> List[str]:
    """将答案分解为原子声明列表"""
    if not answer or len(answer.strip()) < 5:
        return []

    prompt = CLAIM_DECOMPOSITION_PROMPT.format(answer=answer)
    response = call_llm(prompt)

    if not response:
        return []

    claims = []
    for line in response.strip().split("\n"):
        line = line.strip()
        # 去掉编号前缀 "1.", "2.", "- ", "* "
        line = re.sub(r"^\s*(?:\d+[\.\)]|[-*])\s*", "", line)
        if line and len(line) > 5:
            claims.append(line)

    return claims if claims else [answer.strip()]


def verify_claim(claim: str, context: str) -> float:
    """验证单个声明是否被上下文支持，返回置信度分数"""
    if not context or len(context.strip()) < 10:
        return 0.0

    prompt = CLAIM_VERIFICATION_PROMPT.format(
        context=context[:8000],  # 限制上下文长度
        claim=claim,
    )
    response = call_llm(prompt)

    if not response:
        return 0.0

    response_clean = response.strip().upper()
    if response_clean.startswith("YES"):
        return 1.0
    elif response_clean.startswith("NO"):
        return 0.0
    else:
        # 如果没有明确 YES/NO，尝试从文本推断
        if "yes" in response_clean:
            return 0.5
        return 0.0


def compute_faithfulness(answer: str, context: str) -> FaithfulnessResult:
    """计算单条 (answer, context) 的 Faithfulness 分数

    Faithfulness = supported_claims / total_claims

    算法：
      1. 将 answer 分解为原子声明
      2. 对每个声明，询问 LLM "是否被 context 支持"
      3. Faithfulness = 支持的声明数 / 总声明数
    """
    result = FaithfulnessResult(
        query="",
        answer=answer,
        context_length=len(context),
    )

    # Step 1: 分解声明
    claims = decompose_claims(answer)
    result.claims = claims

    if not claims:
        # 无声明可验证 → 无法判断，返回 0
        result.faithfulness_score = 0.0
        return result

    # Step 2: 逐条验证
    for claim in claims:
        score = verify_claim(claim, context)
        result.scores.append(score)
        result.supported.append(score >= 0.5)

    # Step 3: 计算分数
    supported_count = sum(result.supported)
    result.faithfulness_score = supported_count / len(claims)

    return result


# ─── Answer Relevancy ─────────────────────────────────────────

def generate_reverse_questions(answer: str, n: int = DEFAULT_N_REVERSE_QUESTIONS) -> List[str]:
    """从答案反向生成推测性问题"""
    if not answer or len(answer.strip()) < 5:
        return []

    prompt = REVERSE_QUESTION_PROMPT.format(answer=answer, n=n)
    response = call_llm(prompt)

    if not response:
        return []

    questions = []
    for line in response.strip().split("\n"):
        line = line.strip()
        line = re.sub(r"^\s*(?:\d+[\.\)]|[-*])\s*", "", line)
        if line and line.endswith("?") and len(line) > 5:
            questions.append(line)

    return questions[:n]


def compute_answer_relevancy(
    question: str,
    answer: str,
    embed_fn: Optional[Callable] = None,
) -> AnswerRelevancyResult:
    """计算单条 (question, answer) 的 Answer Relevancy 分数

    Answer Relevancy = mean(cos_sim(orig_q_emb, gen_q_emb_i))

    算法：
      1. 从 answer 反向生成 N 个推测性问题
      2. 用 BGE（或 nomic-embed-text）嵌入原文和生成问题
      3. 计算原问题嵌入与每个生成问题嵌入的余弦相似度
      4. 取均值
    """
    result = AnswerRelevancyResult(query=question, answer=answer)

    # Step 1: 生成反向问题
    gen_questions = generate_reverse_questions(answer)
    result.generated_questions = gen_questions

    if not gen_questions:
        result.relevancy_score = 0.0
        return result

    # Step 2: 嵌入
    texts_to_embed = [question] + gen_questions
    embeddings = call_ollama_embed(texts_to_embed)

    if embeddings is None or len(embeddings) < len(texts_to_embed):
        logger.warning("嵌入失败，使用 LLM 直接评分 fallback")
        result.relevancy_score = _llm_relevancy_fallback(question, answer)
        return result

    orig_emb = embeddings[0]
    gen_embs = embeddings[1:]

    # Step 3: 余弦相似度
    for gen_emb in gen_embs:
        sim = float(np.dot(orig_emb, gen_emb) / (
            np.linalg.norm(orig_emb) * np.linalg.norm(gen_emb) + 1e-10
        ))
        # 裁剪到 [0, 1]
        sim = max(0.0, min(1.0, sim))
        result.similarities.append(sim)

    result.relevancy_score = float(np.mean(result.similarities)) if result.similarities else 0.0
    return result


_RELEVANCY_FALLBACK_PROMPT = """\
On a scale of 0.0 to 1.0, how relevant is the following ANSWER to the QUESTION?
Consider: does the answer directly address the question, or is it tangential?
Reply with ONLY a number between 0.0 and 1.0.

Question: {question}
Answer: {answer}

Relevance score:"""


def _llm_relevancy_fallback(question: str, answer: str) -> float:
    """LLM 直接评分 fallback（当嵌入不可用时）"""
    prompt = _RELEVANCY_FALLBACK_PROMPT.format(question=question, answer=answer)
    response = call_llm(prompt)
    if not response:
        return 0.0

    try:
        # 提取数字
        score = float(re.search(r"(\d+\.?\d*)", response.strip()).group(1))
        return max(0.0, min(1.0, score))
    except (ValueError, AttributeError):
        return 0.5  # 无法解析时给中间值


# ─── 问答生成 ──────────────────────────────────────────────────

def generate_answer(query: str, context: str) -> str:
    """基于检索上下文生成回答"""
    if not context:
        return "I cannot answer this question based on the available documents."

    tracer = get_tracer()
    with tracer.start_span("llm_generate") as span:
        prompt = GENERATION_PROMPT.format(context=context[:12000], question=query)
        answer = call_llm(prompt)
        span.set_metadata({
            "context_chars": len(context[:12000]),
            "answer_length": len(answer) if answer else 0,
        })
    return answer if answer else ""


def retrieve_and_generate(
    retriever,
    query: str,
    k: int = 5,
    use_rerank: bool = True,
) -> tuple:
    """检索 → 生成，返回 (retrieved_chunks, context, answer)"""
    retrieved = retriever.search(query, k=k, use_rerank=use_rerank)

    if not retrieved:
        return [], "", generate_answer(query, "")

    context = "\n\n---\n\n".join(
        [r.get("text", "") for r in retrieved]
    )
    answer = generate_answer(query, context)

    return retrieved, context, answer


# ─── 批量评测 ──────────────────────────────────────────────────

@dataclass
class RagasGenerationEvalResult:
    """生成层评测完整结果"""
    faithfulness_results: List[FaithfulnessResult] = field(default_factory=list)
    relevancy_results: List[AnswerRelevancyResult] = field(default_factory=list)
    avg_faithfulness: float = 0.0
    avg_relevancy: float = 0.0
    num_queries: int = 0
    generated_count: int = 0
    rejected_count: int = 0
    
    def summary_dict(self) -> dict:
        return {
            "num_queries": self.num_queries,
            "generated_count": self.generated_count,
            "rejected_count": self.rejected_count,
            "avg_faithfulness": round(self.avg_faithfulness, 4),
            "avg_relevancy": round(self.avg_relevancy, 4),
        }


def evaluate_generation(
    retriever,
    queries_ds,
    k: int = 5,
    use_rerank: bool = True,
    max_queries: Optional[int] = None,
    output_dir: str = "results",
    label: str = "",
) -> RagasGenerationEvalResult:
    """批量评测：检索 → 生成 → Faithfulness + Answer Relevancy

    Args:
        retriever: PrismRAGRetriever 实例
        queries_ds: HuggingFace Dataset，包含 "query" 列
        k: 检索 top-k
        use_rerank: 是否使用重排序
        max_queries: 最大查询数
        output_dir: 输出目录
        label: 结果文件名标签

    Returns:
        RagasGenerationEvalResult 汇总
    """
    import torch
    from tqdm import tqdm
    
    logger.info(f"开始生成层评测 ({len(queries_ds)} 条查询, top-k={k}, rerank={use_rerank})...")
    
    faithfulness_results = []
    relevancy_results = []
    rejected_count = 0
    generated_count = 0
    latency_total = 0.0
    
    num_queries = min(max_queries, len(queries_ds)) if max_queries else len(queries_ds)
    
    for q_idx in tqdm(range(num_queries), desc="Retrieve→Generate→Eval"):
        q = queries_ds[q_idx]
        query_text = str(q["query"])
        
        start = time.time()
        
        # Step 1: 检索
        retrieved = retriever.search(query_text, k=k, use_rerank=use_rerank)
        
        if not retrieved:
            context = ""
        else:
            context = "\n\n---\n\n".join(
                [r.get("text", "") for r in retrieved]
            )
        
        # Step 2: 生成
        answer = generate_answer(query_text, context)
        latency_total += time.time() - start
        
        # 检查是否拒答
        rejection_phrases = [
            "cannot answer", "not enough information",
            "based on the available", "cannot provide",
        ]
        is_rejected = any(phrase in answer.lower() for phrase in rejection_phrases)
        
        if is_rejected:
            rejected_count += 1
        else:
            generated_count += 1
        
        # Step 3: Faithfulness（只在非拒答时算，否则无意义）
        if not is_rejected and answer:
            f_result = compute_faithfulness(answer, context)
            f_result.query = query_text
            faithfulness_results.append(f_result)
        else:
            f_result = FaithfulnessResult(
                query=query_text,
                answer=answer,
                context_length=len(context),
                faithfulness_score=0.0,
            )
            faithfulness_results.append(f_result)
        
        # Step 4: Answer Relevancy（对拒答也有意义：拒答本身是否相关）
        r_result = compute_answer_relevancy(query_text, answer)
        relevancy_results.append(r_result)

        # Record observability metrics
        collector = get_collector()
        collector.record_ragas_score(
            config_label=label or "default",
            query_id=query_text,
            faithfulness=f_result.faithfulness_score,
            answer_relevancy=r_result.relevancy_score,
        )
    
    # ── 汇总 ──
    faithfulness_scores = [r.faithfulness_score for r in faithfulness_results if r.claims]
    avg_faithfulness = float(np.mean(faithfulness_scores)) if faithfulness_scores else 0.0
    relevancy_scores = [r.relevancy_score for r in relevancy_results]
    avg_relevancy = float(np.mean(relevancy_scores)) if relevancy_scores else 0.0
    
    result = RagasGenerationEvalResult(
        faithfulness_results=faithfulness_results,
        relevancy_results=relevancy_results,
        avg_faithfulness=avg_faithfulness,
        avg_relevancy=avg_relevancy,
        num_queries=num_queries,
        generated_count=generated_count,
        rejected_count=rejected_count,
    )
    
    # ── 日志输出 ──
    rejection_rate = rejected_count / num_queries if num_queries > 0 else 0.0
    avg_latency = latency_total / num_queries if num_queries > 0 else 0.0
    
    logger.info(f"\n{'='*60}")
    logger.info(f"生成层评测结果 [{label or 'default'}]")
    logger.info(f"{'='*60}")
    logger.info(f"  查询总数:     {num_queries}")
    logger.info(f"  生成回答:     {generated_count}")
    logger.info(f"  拒绝回答:     {rejected_count} ({rejection_rate:.1%})")
    logger.info(f"  平均延迟:     {avg_latency:.1f}s")
    logger.info(f"  Faithfulness: {avg_faithfulness:.4f}  ({len(faithfulness_scores)} 条有声明)")
    logger.info(f"  Relevancy:    {avg_relevancy:.4f}")
    
    # ── 保存 ──
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    label_suffix = f"_{label}" if label else ""
    with open(output_path / f"ragas_metrics{label_suffix}.json", "w") as f:
        json.dump({
            "summary": result.summary_dict(),
            "faithfulness": [r.to_dict() for r in faithfulness_results],
            "relevancy": [r.to_dict() for r in relevancy_results],
        }, f, indent=2, ensure_ascii=False)
    
    logger.info(f"  结果已保存: {output_path / f'ragas_metrics{label_suffix}.json'}")
    
    return result


# ─── 配置消融对比评测 ──────────────────────────────────────────

def evaluate_generation_configs(
    retriever,
    queries_ds,
    configs: List[dict],
    max_queries: Optional[int] = None,
    output_dir: str = "results",
) -> Dict[str, RagasGenerationEvalResult]:
    """在多种检索配置下运行生成层评测

    Args:
        retriever: PrismRAGRetriever 实例
        queries_ds: queries dataset
        configs: 配置列表，每项为 {
            "name": str,
            "k": int,
            "use_rerank": bool,
        }
    
    Returns:
        {config_name: RagasGenerationEvalResult}
    """
    results = {}
    for config in configs:
        logger.info(f"\n{'='*60}")
        logger.info(f"生成层消融: {config['name']}")
        logger.info(f"{'='*60}")
        
        result = evaluate_generation(
            retriever=retriever,
            queries_ds=queries_ds,
            k=config.get("k", 5),
            use_rerank=config.get("use_rerank", True),
            max_queries=max_queries,
            output_dir=output_dir,
            label=config["name"],
        )
        results[config["name"]] = result
    
    # 消融对比表格
    logger.info(f"\n{'='*80}")
    logger.info("生成层消融对比")
    header = f"{'Config':<25} {'Queries':<8} {'Gen':<6} {'Rej':<6} {'Faith':<10} {'Relev':<10}"
    logger.info(header)
    logger.info("-" * 80)
    for name, r in results.items():
        logger.info(
            f"{name:<25} {r.num_queries:<8} {r.generated_count:<6} {r.rejected_count:<6} "
            f"{r.avg_faithfulness:<10.4f} {r.avg_relevancy:<10.4f}"
        )
    
    # 保存消融对比
    comparison = [
        {"config": name, **r.summary_dict()}
        for name, r in results.items()
    ]
    with open(Path(output_dir) / "ragas_metrics_ablation_comparison.json", "w") as f:
        json.dump(comparison, f, indent=2)
    
    return results