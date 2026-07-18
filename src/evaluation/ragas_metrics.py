"""RAGAS 指标自实现 — Faithfulness + Answer Relevancy

使用 Ollama qwen2:7b 作为 Judge LLM，BGE 作为嵌入模型。
不依赖 ragas 库（0.4.x 依赖链断裂），按照 RAGAS 论文算法逻辑自实现。

指标说明：
  - Faithfulness（忠实度）：答案中的每个原子声明是否被检索上下文支持。
    分解答案为原子声明 → 逐个问 LLM "上下文是否支持该声明" → 支持率。
  - Answer Relevancy（答案相关性）：答案与问题的相关程度。
    从答案反向生成 N 个推测性问题 → BGE 嵌入对比原问题 → 余弦相似度均值。
  - Context Relevance（上下文相关性）：检索回的上下文中有多少内容真正与问题相关。
    拆分上下文为句子 → 逐句问 LLM "这句话跟问题有关吗？" → 相关句占比。
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
from src.prompts import get_active

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


@dataclass
class ContextRelevancyResult:
    """单条查询的 Context Relevance 结果"""
    query: str
    context_chunks: List[str] = field(default_factory=list)
    num_sentences: int = 0
    num_relevant: int = 0
    relevance_score: float = 0.0
    per_sentence: List[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "context_chunks": self.context_chunks,
            "num_sentences": self.num_sentences,
            "num_relevant": self.num_relevant,
            "relevance_score": round(self.relevance_score, 4),
            "per_sentence": self.per_sentence,
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


# ─── Prompt 模板（集中管理，从 src/prompts registry 加载生效版本） ────
# 模板正文已外置到 src/prompts/prompts/*.yaml（带 version/changelog，进 git 可追溯）。
# 下方模块级常量在 import 期解析为生效版本文本，保持原有 .format() 调用不变。

CLAIM_DECOMPOSITION_PROMPT = get_active("claim_decomposition").template

CLAIM_VERIFICATION_PROMPT = get_active("claim_verification").template

REVERSE_QUESTION_PROMPT = get_active("reverse_question").template

GENERATION_PROMPT = get_active("ragas_generation").template


# ─── Faithfulness ─────────────────────────────────────────────

def decompose_claims(answer: str) -> List[str]:
    """将答案分解为原子声明列表"""
    if not answer or len(answer.strip()) < 5:
        return []

    tracer = get_tracer()
    with tracer.start_span("decompose_claims") as span:
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

        result = claims if claims else [answer.strip()]
        span.set_metadata({
            "num_claims": len(result),
            "answer_length": len(answer),
        })
        return result


def verify_claim(claim: str, context: str) -> float:
    """验证单个声明是否被上下文支持，返回置信度分数"""
    if not context or len(context.strip()) < 10:
        return 0.0

    tracer = get_tracer()
    with tracer.start_span("verify_claim") as span:
        prompt = CLAIM_VERIFICATION_PROMPT.format(
            context=context[:8000],  # 限制上下文长度
            claim=claim,
        )
        response = call_llm(prompt)

        if not response:
            return 0.0

        response_clean = response.strip().upper()
        if response_clean.startswith("YES"):
            verdict = 1.0
        elif response_clean.startswith("NO"):
            verdict = 0.0
        else:
            # 如果没有明确 YES/NO，尝试从文本推断
            if "yes" in response_clean:
                verdict = 0.5
            else:
                verdict = 0.0

        span.set_metadata({
            "claim_length": len(claim),
            "verdict": verdict,
        })
        return verdict


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

    tracer = get_tracer()
    with tracer.start_span("generate_reverse_questions") as span:
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

        result = questions[:n]
        span.set_metadata({
            "n_requested": n,
            "n_generated": len(result),
        })
        return result


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
    tracer = get_tracer()
    with tracer.start_span("ollama_embed") as span:
        embeddings = call_ollama_embed(texts_to_embed)
        span.set_metadata({"num_texts": len(texts_to_embed)})

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


_RELEVANCY_FALLBACK_PROMPT = get_active("relevancy_fallback").template


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


# ─── Context Compression ────────────────────────────────────────

def compress_context(
    query: str,
    chunks: List[str],
    bge_embedder,
    ratio: float = 0.4,
) -> str:
    """BGE 句级 cosine 过滤：保留与 query 最相关的句子，压缩上下文。

    Args:
        query: 查询文本
        chunks: 检索回的 chunk 文本列表
        bge_embedder: BGEEmbedder 实例（需要有 encode() 方法，返回 L2-normalized 嵌入）
        ratio: 保留比例，0.4 表示保留 top 40% 最相关句子

    Returns:
        压缩后的上下文字符串（句子按原文顺序拼接）
    """
    import time as _time
    import torch as _torch

    tracer = get_tracer()
    with tracer.start_span("context_compression") as span:
        t0 = _time.time()

        # Step 1: 拆句
        sentences = split_context_to_sentences(chunks)
        num_before = len(sentences)

        # 太少不值得压缩
        if num_before <= 5:
            span.set_metadata({
                "num_before": num_before, "num_after": num_before,
                "ratio": 1.0, "skipped": True, "reason": "too_few_sentences",
            })
            return "\n\n".join(chunks)

        # Step 2: BGE 编码
        query_emb = bge_embedder.encode([query])  # [1, dim]
        sent_embs = bge_embedder.encode(sentences)  # [N, dim]

        # Step 3: Cosine = dot product（BGE 已 L2-normalize）
        query_vec = query_emb[0]  # [dim]
        similarities = [
            float(_torch.dot(query_vec, sent_embs[i]))
            for i in range(num_before)
        ]

        # Step 4: 按相似度排序，保留 top ratio（至少 3 句）
        num_keep = max(3, int(num_before * ratio))
        ranked = sorted(
            enumerate(similarities), key=lambda x: x[1], reverse=True
        )
        keep_indices = {idx for idx, _ in ranked[:num_keep]}

        # Step 5: 按原文顺序拼接
        kept = [sentences[i] for i in range(num_before) if i in keep_indices]
        compressed = "\n".join(kept)

        elapsed = (_time.time() - t0) * 1000
        span.set_metadata({
            "num_before": num_before,
            "num_after": len(kept),
            "ratio": ratio,
            "compression_ms": round(elapsed, 2),
            "skipped": False,
        })
        return compressed


# ─── Context Relevance ─────────────────────────────────────────

CONTEXT_RELEVANCE_PROMPT = get_active("context_relevance").template


def split_context_to_sentences(context_chunks: List[str]) -> List[str]:
    """将检索上下文拆分为句子列表（纯函数，无 LLM 调用）

    按句号、换行分割，过滤过短的碎片（<3 词），保证每句至少可评估。
    """
    sentences = []
    for chunk in context_chunks:
        if not chunk or not chunk.strip():
            continue
        # 按句号和换行分割
        parts = re.split(r'\.|\n', chunk)
        for part in parts:
            cleaned = part.strip()
            # 过滤：至少 3 个词，长度 >= 10 字符
            if len(cleaned.split()) >= 3 and len(cleaned) >= 10:
                # 去掉列表编号前缀
                cleaned = re.sub(r"^\s*(?:\d+[\.\)]|[-*])\s*", "", cleaned)
                if len(cleaned.split()) >= 3:
                    sentences.append(cleaned)
    return sentences


def parse_relevance_response(response_text: str, num_sentences: int) -> List[bool]:
    """解析 LLM 返回的相关性判断结果（纯函数，无 LLM 调用）

    支持三种格式：
      1. JSON: [{"id": 0, "relevant": true}, ...]
      2. Markdown JSON block: ```json [...] ```
      3. Text: [0] YES / 0: yes

    返回: 长度为 num_sentences 的 bool 列表，未覆盖的默认 False
    """
    result = [False] * num_sentences
    if not response_text or not response_text.strip():
        return result

    text = response_text.strip()

    # Attempt 1: JSON format (possibly wrapped in markdown)
    json_str = None
    # Try extracting from markdown code block
    md_match = re.search(r'```(?:json)?\s*(\[.*?\])\s*```', text, re.DOTALL)
    if md_match:
        json_str = md_match.group(1)
    elif text.startswith('['):
        json_str = text

    if json_str:
        try:
            parsed = json.loads(json_str)
            if isinstance(parsed, list):
                for item in parsed:
                    idx = item.get("id", -1)
                    if 0 <= idx < num_sentences:
                        result[idx] = bool(item.get("relevant", False))
            return result
        except (json.JSONDecodeError, TypeError):
            pass

    # Attempt 2: Text format — [0] YES / 0: yes
    for i in range(num_sentences):
        patterns = [
            rf'\[{i}\]\s*(YES|yes)',
            rf'{i}\s*[:\)]\s*(YES|yes)',
            rf'\[{i}\]\s*:\s*(True|true)',
            rf'{i}\s*[:\)]\s*(True|true)',
        ]
        for pat in patterns:
            if re.search(pat, text):
                result[i] = True
                break

    return result


_CONTEXT_RELEVANCE_BATCH_SIZE = 20  # 每批最多 20 句


def compute_context_relevancy(query: str, context_chunks: List[str]) -> ContextRelevancyResult:
    """计算单条 query 的 Context Relevance 分数

    Context Relevance = relevant_sentences / total_sentences

    算法：
      1. 将 context_chunks 拆为句子列表
      2. 分批（每批 ≤20 句），逐批问 LLM "这句话跟问题有关吗？"
      3. 汇总：相关句数 / 总句数
    """
    result = ContextRelevancyResult(
        query=query,
        context_chunks=list(context_chunks),
    )

    # Step 1: 分句
    sentences = split_context_to_sentences(context_chunks)
    result.num_sentences = len(sentences)

    if not sentences:
        result.relevance_score = 0.0
        return result

    # Step 2: 分批问 LLM
    all_relevant = []
    for batch_start in range(0, len(sentences), _CONTEXT_RELEVANCE_BATCH_SIZE):
        batch = sentences[batch_start:batch_start + _CONTEXT_RELEVANCE_BATCH_SIZE]

        # 构建带编号的句子列表
        sentences_block = "\n".join(
            f"[{batch_start + i}] {s}" for i, s in enumerate(batch)
        )
        prompt = CONTEXT_RELEVANCE_PROMPT.format(query=query, sentences=sentences_block)
        tracer = get_tracer()
        with tracer.start_span("context_relevance_llm") as span:
            response = call_llm(prompt)
            span.set_metadata({
                "batch_start": batch_start,
                "batch_size": len(batch),
            })

        # 解析响应（传入总数以避免 LLM 漏句）
        batch_relevant = parse_relevance_response(response, len(batch))
        all_relevant.extend(batch_relevant)

    # Step 3: 汇总
    result.num_relevant = sum(all_relevant)
    result.relevance_score = result.num_relevant / len(sentences) if sentences else 0.0

    # 构建逐句明细
    result.per_sentence = [
        {"id": i, "text": sentences[i], "relevant": all_relevant[i] if i < len(all_relevant) else False}
        for i in range(len(sentences))
    ]

    return result


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
    bge_embedder=None,
) -> tuple:
    """检索 → 生成，返回 (retrieved_chunks, context, answer)

    Args:
        retriever: PrismRAGRetriever 实例
        query: 查询文本
        k: 检索 top-k
        use_rerank: 是否重排
        bge_embedder: 可选，BGEEmbedder 实例。传入时自动压缩上下文。
    """
    retrieved = retriever.search(query, k=k, use_rerank=use_rerank)

    if not retrieved:
        return [], "", generate_answer(query, "")

    chunks = [r.get("text", "") for r in retrieved]
    compress_ratio = cfg.get("retrieval.context_compression_ratio", 1.0)
    if bge_embedder is not None and compress_ratio < 1.0:
        context = compress_context(query, chunks, bge_embedder, ratio=compress_ratio)
    else:
        context = "\n\n---\n\n".join(chunks)

    answer = generate_answer(query, context)
    return retrieved, context, answer


# ─── 批量评测 ──────────────────────────────────────────────────

@dataclass
class RagasGenerationEvalResult:
    """生成层评测完整结果"""
    faithfulness_results: List[FaithfulnessResult] = field(default_factory=list)
    relevancy_results: List[AnswerRelevancyResult] = field(default_factory=list)
    context_relevancy_results: List[ContextRelevancyResult] = field(default_factory=list)
    avg_faithfulness: float = 0.0
    avg_relevancy: float = 0.0
    avg_context_relevancy: float = 0.0
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
            "avg_context_relevancy": round(self.avg_context_relevancy, 4),
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
    from tqdm import tqdm
    
    logger.info(f"开始生成层评测 ({len(queries_ds)} 条查询, top-k={k}, rerank={use_rerank})...")
    
    faithfulness_results = []
    relevancy_results = []
    context_relevancy_results = []
    rejected_count = 0
    generated_count = 0
    latency_total = 0.0
    
    num_queries = min(max_queries, len(queries_ds)) if max_queries else len(queries_ds)
    
    for q_idx in tqdm(range(num_queries), desc="Retrieve→Generate→Eval"):
        q = queries_ds[q_idx]
        query_text = str(q["query"])
        
        # ── 创建端到端 Trace ───────────────────────────────
        tracer = get_tracer()
        config_label = label or "default"
        tracer.start_trace(query=query_text, config_label=config_label)
        
        start = time.time()
        
        # Step 1: 检索（span 由 search_with_trace 内部创建，自动挂载到父 Trace）
        with tracer.start_span("retrieval") as retrieval_span:
            retrieved = retriever.search(query_text, k=k, use_rerank=use_rerank)
            retrieval_span.set_metadata({
                "num_retrieved": len(retrieved),
                "k": k,
            })
        
        if not retrieved:
            context = ""
            context_chunks = []
            ctx_for_eval = ""
        else:
            context_chunks = [r.get("text", "") for r in retrieved]

            # ── Step 1.5a: 置信度阈值检测 ──────────────────
            rerank_scores = [r.get("rerank_score", 0.0) for r in retrieved]
            max_rerank = max(rerank_scores) if rerank_scores else 0.0
            reject_threshold = cfg.get("retrieval.rerank_score_reject_threshold", 0.0)

            with tracer.start_span("rerank_threshold_check") as thresh_span:
                threshold_rejected = (
                    reject_threshold > 0
                    and max_rerank < reject_threshold
                    and len(retrieved) > 0
                )
                thresh_span.set_metadata({
                    "max_rerank_score": round(max_rerank, 4),
                    "threshold": reject_threshold,
                    "rejected": threshold_rejected,
                })

            # ── Step 1.5b: 上下文压缩（非拒答时）─────────────
            compress_ratio = cfg.get("retrieval.context_compression_ratio", 1.0)
            if not threshold_rejected and compress_ratio < 1.0 and context_chunks:
                context = compress_context(
                    query_text, context_chunks, retriever.bge, ratio=compress_ratio,
                )
            else:
                context = "\n\n---\n\n".join(context_chunks)

            # CtxRel 评估的是 LLM 实际看到的检索上下文（压缩后），
            # 与 generate_answer / compute_faithfulness 使用同一份 context；
            # doc_ref 前缀是 grounding 元数据，不计入相关性评估
            ctx_for_eval = context

            # ── 注入 doc_ref 作为 grounding 元数据 ────────────
            doc_refs = list(dict.fromkeys(
                r.get("doc_ref", "") for r in retrieved if r.get("doc_ref")
            ))  # 去重，保序
            if doc_refs and context:
                ref_prefix = "Source documents: " + "; ".join(doc_refs)
                context = ref_prefix + "\n\n" + context

        # ── 拒答判断（短语拒答 + 阈值拒答）─────────────────
        if threshold_rejected:
            answer = "I cannot answer this question based on the available documents."
        else:
            # Step 2: 生成（llm_generate span 由 generate_answer 内部创建，自动挂载）
            answer = generate_answer(query_text, context)
        latency_total += time.time() - start

        # 检查是否短语拒答
        rejection_phrases = [
            "cannot answer", "not enough information",
            "based on the available", "cannot provide",
        ]
        phrase_rejected = any(phrase in answer.lower() for phrase in rejection_phrases)
        is_rejected = phrase_rejected or threshold_rejected

        if is_rejected:
            rejected_count += 1
        else:
            generated_count += 1
        
        # Step 3: Faithfulness（只在非拒答时算，否则无意义）
        with tracer.start_span("ragas_faithfulness") as faith_span:
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
            faith_span.set_metadata({
                "is_rejected": is_rejected,
                "num_claims": len(f_result.claims),
                "score": f_result.faithfulness_score,
            })
        
        # Step 4: Answer Relevancy
        with tracer.start_span("ragas_answer_relevancy") as ar_span:
            r_result = compute_answer_relevancy(query_text, answer)
            relevancy_results.append(r_result)
            ar_span.set_metadata({
                "score": r_result.relevancy_score,
                "num_gen_questions": len(r_result.generated_questions),
            })

        # Step 5: Context Relevance — 评估 LLM 实际看到的上下文（压缩后），
        # 而非原始检索 chunk，确保与 Faithfulness/生成口径一致
        with tracer.start_span("ragas_context_relevancy") as cr_span:
            if retrieved and ctx_for_eval:
                c_result = compute_context_relevancy(query_text, [ctx_for_eval])
            else:
                c_result = ContextRelevancyResult(query=query_text, context_chunks=[])
            context_relevancy_results.append(c_result)
            cr_span.set_metadata({
                "score": c_result.relevance_score,
                "num_sentences": c_result.num_sentences,
                "num_relevant": c_result.num_relevant,
            })

        # Record observability metrics (with per-sentence detail)
        collector = get_collector()
        collector.record_ragas_score(
            config_label=config_label,
            query_id=query_text,
            faithfulness=f_result.faithfulness_score,
            answer_relevancy=r_result.relevancy_score,
            context_relevancy=c_result.relevance_score,
            context_relevancy_details={
                "num_sentences": c_result.num_sentences,
                "num_relevant": c_result.num_relevant,
                "per_sentence": c_result.per_sentence,
            },
            faithfulness_details={
                "num_claims": len(f_result.claims),
                "num_supported": sum(f_result.supported),
                "claims": f_result.claims,
                "supported": f_result.supported,
            },
        )

        # ── 完成 Trace，ingest 到 collector ─────────────────
        trace = tracer.finish_trace()
        if trace:
            collector.ingest_trace(trace)
    
    # ── 汇总 ──
    faithfulness_scores = [r.faithfulness_score for r in faithfulness_results if r.claims]
    avg_faithfulness = float(np.mean(faithfulness_scores)) if faithfulness_scores else 0.0
    relevancy_scores = [r.relevancy_score for r in relevancy_results]
    avg_relevancy = float(np.mean(relevancy_scores)) if relevancy_scores else 0.0
    context_relevancy_scores = [r.relevance_score for r in context_relevancy_results if r.num_sentences > 0]
    avg_context_relevancy = float(np.mean(context_relevancy_scores)) if context_relevancy_scores else 0.0

    result = RagasGenerationEvalResult(
        faithfulness_results=faithfulness_results,
        relevancy_results=relevancy_results,
        context_relevancy_results=context_relevancy_results,
        avg_faithfulness=avg_faithfulness,
        avg_relevancy=avg_relevancy,
        avg_context_relevancy=avg_context_relevancy,
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
    logger.info(f"  查询总数:         {num_queries}")
    logger.info(f"  生成回答:         {generated_count}")
    logger.info(f"  拒绝回答:         {rejected_count} ({rejection_rate:.1%})")
    logger.info(f"  平均延迟:         {avg_latency:.1f}s")
    logger.info(f"  Faithfulness:     {avg_faithfulness:.4f}  ({len(faithfulness_scores)} 条有声明)")
    logger.info(f"  Answer Relevancy: {avg_relevancy:.4f}")
    logger.info(f"  Context Relevance:{avg_context_relevancy:.4f}")
    
    # ── 保存 ──
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    label_suffix = f"_{label}" if label else ""
    with open(output_path / f"ragas_metrics{label_suffix}.json", "w") as f:
        json.dump({
            "summary": result.summary_dict(),
            "faithfulness": [r.to_dict() for r in faithfulness_results],
            "relevancy": [r.to_dict() for r in relevancy_results],
            "context_relevancy": [r.to_dict() for r in context_relevancy_results],
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