#!/usr/bin/env python
"""端到端 QA 数据集生成器 — 从 ViDoRe queries 半自动生成 QA 对

流程：
  1. 从 HuggingFace 加载 ViDoRe v3 Industrial 数据集的 English queries
  2. 对每条 query，从 corpus 获取 ground-truth 页面内容
  3. 用 Ollama qwen2:7b 基于 ground-truth 页面生成预期答案
  4. 保存为 data/e2e_qa.json（可回答 + 拒答混合格式）

使用方式：
  # 全量生成（50 条）
  python scripts/generate_e2e_qa.py --output data/e2e_qa.json

  # 指定数据集路径
  python scripts/generate_e2e_qa.py --dataset vidore/vidore_v3_industrial

  # 合并已有拒答数据
  python scripts/generate_e2e_qa.py --rejection-qa data/rejection_qa.json

注意事项：
  - 需要 HuggingFace 数据集访问权限（datasets 库）
  - 需要本地 Ollama 服务运行（qwen2:7b）
  - 生成 50 条约需 10-15 分钟（每条含 LLM 调用）
  - 建议审核生成的预期答案后再用于评测
"""

import argparse
import json
import logging
import os
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets import load_dataset as hf_load_dataset
from src.evaluation.e2e_qa import call_llm
from src.evaluation.ablation import load_eval_data, _LANGUAGE_MAP

logger = logging.getLogger(__name__)

# ─── Prompt ────────────────────────────────────────────────────

EXPECTED_ANSWER_PROMPT = """\
You are an expert in industrial document QA.

Given a question and the relevant context from an industrial document (PDF page), generate a concise, factual expected answer.

Rules:
- Answer based ONLY on the provided context.
- Be precise and specific — include numbers, model numbers, specifications, etc.
- If the context doesn't contain enough information to answer, say "Cannot answer from context"
- Keep the answer to 1-3 sentences.
- Do NOT reference the page/context in your answer.

Context:
{context}

Question: {question}

Expected Answer:"""


# ─── 核心逻辑 ──────────────────────────────────────────────────


def generate_expected_answer(query: str, context: str) -> str:
    """用 LLM 从 ground-truth 页面生成预期答案"""
    if not context or len(context.strip()) < 20:
        return "Cannot answer from context"

    prompt = EXPECTED_ANSWER_PROMPT.format(
        context=context[:8000],
        question=query,
    )
    answer = call_llm(prompt)

    if not answer:
        return "Cannot answer from context"

    # 清理多余换行
    answer = answer.strip().replace("\n", " ").strip()
    return answer


def sample_queries(queries_ds, qrel_map, corpus_text: dict, n: int = 50) -> list:
    """从查询数据集中精选 n 条

    策略：
      - 优先选有 ground-truth pages 的 query
      - 按文档类型/页面数均匀采样
      - 避免太相似的 query

    Returns:
        List[dict] - 精选后的 QA 条目
    """
    # 构建候选：有 ground-truth 且页面有内容的 queries
    candidates = []
    for q_idx in range(len(queries_ds)):
        q = queries_ds[q_idx]
        qid = q.get("query_id", q_idx)
        query_text = str(q["query"])

        # 检查是否有 ground-truth pages
        relevant_pages = qrel_map.get(qid, set())
        if not relevant_pages or not query_text.strip():
            continue

        # 检查 ground-truth pages 是否有文本内容
        has_content = any(int(pid) in corpus_text for pid in relevant_pages)
        if not has_content:
            continue

        candidates.append({
            "query_id": qid,
            "query": query_text,
            "relevant_page_ids": list(relevant_pages),
        })

    logger.info(f"有 ground-truth 的候选 queries: {len(candidates)}")

    if len(candidates) <= n:
        # 不够 n 条，全部使用
        selected = candidates
    else:
        # 按文档类型均匀采样（如果没有 doc_id，随机采）
        random.shuffle(candidates)
        selected = candidates[:n]

    logger.info(f"精选后: {len(selected)} 条")
    return selected


def fetch_context(corpus_text: dict, page_ids: list) -> str:
    """从 corpus 中获取 ground-truth 页面的文本"""
    texts = []
    for pg_id in page_ids:
        md = corpus_text.get(int(pg_id))
        if md and len(md) > 20:
            texts.append(md)

    if not texts:
        return ""

    # 合并，限制总长度
    combined = "\n\n---\n\n".join(texts)
    return combined[:12000]


def generate_qa_dataset(
    queries_ds,
    qrel_map,
    corpus_text: dict,
    n: int = 50,
    rejection_qa_path: Optional[str] = None,
) -> list:
    """生成端到端 QA 数据集

    Returns:
        List[dict] - [{type, question, expected_answer, ...}, ...]
    """
    # 1. 精选 queries
    selected = sample_queries(queries_ds, qrel_map, corpus_text, n=n)

    # 2. 逐条生成预期答案
    output = []
    for i, item in enumerate(selected):
        query = item["query"]
        relevant_pages = item["relevant_page_ids"]

        logger.info(f"  [{i+1}/{len(selected)}] {query[:60]}...")

        context = fetch_context(corpus_text, relevant_pages)
        expected_answer = generate_expected_answer(query, context)

        entry = {
            "id": f"e2e_{i+1:03d}",
            "type": "answerable",
            "question": query,
            "expected_answer": expected_answer,
            "relevant_page_ids": relevant_pages,
            "difficulty": "medium",
        }
        output.append(entry)

        logger.info(f"    → {expected_answer[:80]}...")
        time.sleep(0.5)  # 避免 Ollama 过载

    # 3. 合并拒答数据
    if rejection_qa_path and os.path.exists(rejection_qa_path):
        with open(rejection_qa_path) as f:
            rejection_items = json.load(f)
        for item in rejection_items:
            item["type"] = "rejection"
            item["id"] = f"rej_{1 + rejection_items.index(item):03d}"
        output.extend(rejection_items)
        logger.info(f"合并 {len(rejection_items)} 条拒答问题")

    return output


# ─── CLI ──────────────────────────────────────────────────────


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="生成端到端 QA 评测数据集")
    parser.add_argument("--dataset", default="vidore/vidore_v3_industrial",
                        help="HF 数据集路径")
    parser.add_argument("--output", default="data/e2e_qa.json",
                        help="输出文件路径")
    parser.add_argument("--num-qa", type=int, default=50,
                        help="可回答 QA 数量")
    parser.add_argument("--rejection-qa", default="data/rejection_qa.json",
                        help="拒答数据集路径（合并用）")
    parser.add_argument("--language", default="en",
                        choices=["en", "all"],
                        help="查询语言")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子")
    args = parser.parse_args()

    random.seed(args.seed)

    # 加载数据集
    logger.info(f"加载数据集: {args.dataset} (language={args.language})...")
    queries_ds, qrel_map = load_eval_data(
        dataset_path=args.dataset,
        max_queries=None,
        language=args.language,
    )
    logger.info(f"加载 {len(queries_ds)} 条 queries, {len(qrel_map)} 个 query 有 ground-truth pages")

    # 单独加载 corpus（load_eval_data 不返回 corpus）
    corpus_ds = hf_load_dataset(args.dataset, "corpus", split="test")
    corpus_text = {}
    for pg in corpus_ds:
        cid = pg.get("corpus_id")
        if cid is not None:
            md = pg.get("markdown", "")
            if md and len(md.strip()) > 50:
                corpus_text[int(cid)] = md.strip()
    logger.info(f"加载 {len(corpus_ds)} 条 corpus 页面, {len(corpus_text)} 条有文本内容")

    # 生成 QA 数据集
    output = generate_qa_dataset(
        queries_ds=queries_ds,
        qrel_map=qrel_map,
        corpus_text=corpus_text,
        n=args.num_qa,
        rejection_qa_path=args.rejection_qa,
    )

    # 保存
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    logger.info(f"QA 数据集已保存: {output_path}")
    logger.info(f"总计: {len(output)} 条 (可回答={sum(1 for i in output if i.get('type') != 'rejection')}, 拒答={sum(1 for i in output if i.get('type') == 'rejection')})")


if __name__ == "__main__":
    main()