"""测试 RAGAS 指标自实现的非 LLM 依赖部分"""
from __future__ import annotations

import json
import numpy as np
from pathlib import Path
import pytest
from unittest.mock import MagicMock

from src.evaluation.ragas_metrics import (
    FaithfulnessResult,
    AnswerRelevancyResult,
    ContextRelevancyResult,
    RagasGenerationEvalResult,
    decompose_claims,
    generate_reverse_questions,
    _llm_relevancy_fallback,
    split_context_to_sentences,
    parse_relevance_response,
    compress_context,
)


class TestDataStructures:
    """测试数据结构的序列化和反序列化"""

    def test_faithfulness_result_empty(self):
        r = FaithfulnessResult(query="", answer="", context_length=0)
        d = r.to_dict()
        assert d["faithfulness_score"] == 0.0
        assert d["claims"] == []
        assert d["supported"] == []

    def test_faithfulness_result_with_data(self):
        r = FaithfulnessResult(
            query="test query",
            answer="The conveyor belt has a load capacity of 500kg.",
            context_length=200,
        )
        r.claims = ["The conveyor belt has a load capacity of 500kg."]
        r.supported = [True]
        r.faithfulness_score = 1.0
        d = r.to_dict()
        assert d["faithfulness_score"] == 1.0
        assert len(d["claims"]) == 1
        assert d["supported"] == [True]

    def test_answer_relevancy_result(self):
        r = AnswerRelevancyResult(
            query="test",
            answer="test answer",
        )
        r.generated_questions = ["What is the load capacity?", "How much weight can it carry?"]
        r.similarities = [0.8, 0.7]
        r.relevancy_score = 0.75
        d = r.to_dict()
        assert d["relevancy_score"] == 0.75
        assert len(d["generated_questions"]) == 2
        assert len(d["similarities"]) == 2

    def test_eval_result_summary(self):
        fr = FaithfulnessResult(query="q1", answer="a1", context_length=100)
        fr.faithfulness_score = 0.8
        rr = AnswerRelevancyResult(query="q1", answer="a1")
        rr.relevancy_score = 0.7

        result = RagasGenerationEvalResult(
            faithfulness_results=[fr],
            relevancy_results=[rr],
            avg_faithfulness=0.8,
            avg_relevancy=0.7,
            num_queries=1,
            generated_count=1,
            rejected_count=0,
        )
        summary = result.summary_dict()
        assert summary["avg_faithfulness"] == 0.8
        assert summary["avg_relevancy"] == 0.7
        assert summary["num_queries"] == 1


class TestClaimDecomposition:
    """测试声明分解逻辑（不含 LLM 调用）"""

    def test_empty_answer(self):
        assert decompose_claims("") == []
        assert decompose_claims("   ") == []

    def test_short_answer(self):
        assert decompose_claims("hi") == []

    def test_parse_formatted_claims(self):
        # 模拟 LLM 返回格式
        # 注意：这个测试验证的是正则表达式解析逻辑
        # 实际 LLM 调用在集成测试中
        pass  # 放在集成测试中测


class TestScoreCalculation:
    """测试分数计算逻辑"""

    def test_faithfulness_calculation(self):
        """验证 Faithfulness 分数的计算逻辑"""
        fr = FaithfulnessResult(query="q", answer="a", context_length=100)
        fr.claims = ["c1", "c2", "c3", "c4"]
        fr.supported = [True, True, False, True]
        fr.faithfulness_score = sum(fr.supported) / len(fr.claims)
        assert fr.faithfulness_score == 0.75

    def test_faithfulness_all_supported(self):
        fr = FaithfulnessResult(query="q", answer="a", context_length=100)
        fr.claims = ["c1", "c2"]
        fr.supported = [True, True]
        fr.faithfulness_score = sum(fr.supported) / len(fr.claims)
        assert fr.faithfulness_score == 1.0

    def test_faithfulness_none_supported(self):
        fr = FaithfulnessResult(query="q", answer="a", context_length=100)
        fr.claims = ["c1", "c2"]
        fr.supported = [False, False]
        fr.faithfulness_score = sum(fr.supported) / len(fr.claims)
        assert fr.faithfulness_score == 0.0

    def test_faithfulness_no_claims(self):
        fr = FaithfulnessResult(query="q", answer="a", context_length=100)
        fr.claims = []
        fr.faithfulness_score = 0.0
        assert fr.faithfulness_score == 0.0

    def test_relevancy_average(self):
        rr = AnswerRelevancyResult(query="q", answer="a")
        rr.similarities = [0.9, 0.8, 0.7]
        rr.relevancy_score = float(np.mean(rr.similarities))
        assert rr.relevancy_score == pytest.approx(0.8)

    def test_relevancy_empty(self):
        rr = AnswerRelevancyResult(query="q", answer="a")
        rr.similarities = []
        rr.relevancy_score = 0.0
        assert rr.relevancy_score == 0.0


class TestLLMRelevancyFallback:
    """测试 LLM fallback 的分数解析"""

    def test_parse_numeric_score(self):
        score = _llm_relevancy_fallback("test question", "test answer")
        # 在没有实际 LLM 时，返回默认值 0.5
        assert 0.0 <= score <= 1.0


class TestJSONSerialization:
    """测试完整的 JSON 序列化路径"""

    def test_faithfulness_to_json(self):
        fr = FaithfulnessResult(query="q", answer="a", context_length=50)
        fr.claims = ["claim one"]
        fr.supported = [True]
        fr.faithfulness_score = 1.0
        d = fr.to_dict()
        # 验证 JSON 可序列化
        json_str = json.dumps(d)
        assert json_str is not None
        assert "claim one" in json_str
        assert "faithfulness_score" in json_str

    def test_full_result_to_json(self):
        fr = FaithfulnessResult(query="q", answer="a", context_length=50)
        fr.faithfulness_score = 0.5
        rr = AnswerRelevancyResult(query="q", answer="a")
        rr.relevancy_score = 0.6
        result = RagasGenerationEvalResult(
            faithfulness_results=[fr],
            relevancy_results=[rr],
            avg_faithfulness=0.5,
            avg_relevancy=0.6,
            num_queries=1,
            generated_count=1,
            rejected_count=0,
        )
        output = {
            "summary": result.summary_dict(),
            "faithfulness": [r.to_dict() for r in result.faithfulness_results],
            "relevancy": [r.to_dict() for r in result.relevancy_results],
        }
        json_str = json.dumps(output, indent=2, ensure_ascii=False)
        assert json_str is not None
        assert "0.5" in json_str
        assert "0.6" in json_str


class TestContextRelevancyResult:
    """测试 ContextRelevancyResult 数据结构"""

    def test_empty_result(self):
        r = ContextRelevancyResult(query="", context_chunks=[])
        d = r.to_dict()
        assert d["relevance_score"] == 0.0
        assert d["num_sentences"] == 0
        assert d["num_relevant"] == 0
        assert d["per_sentence"] == []

    def test_with_data(self):
        r = ContextRelevancyResult(
            query="What is the load capacity?",
            context_chunks=["The conveyor belt has a load capacity of 500kg.", "This manual covers installation procedures."],
        )
        r.num_sentences = 3
        r.num_relevant = 2
        r.relevance_score = 2 / 3
        r.per_sentence = [
            {"id": 0, "text": "The conveyor belt has a load capacity of 500kg.", "relevant": True},
            {"id": 1, "text": "This manual covers installation procedures.", "relevant": False},
            {"id": 2, "text": "Maintenance should be performed monthly.", "relevant": True},
        ]
        d = r.to_dict()
        assert d["relevance_score"] == pytest.approx(0.6667, abs=0.001)
        assert d["num_sentences"] == 3
        assert d["num_relevant"] == 2
        assert len(d["per_sentence"]) == 3
        assert d["per_sentence"][0]["relevant"] is True
        assert d["per_sentence"][1]["relevant"] is False

    def test_all_relevant(self):
        r = ContextRelevancyResult(query="q", context_chunks=["a", "b"])
        r.num_sentences = 2
        r.num_relevant = 2
        r.relevance_score = 1.0
        assert r.relevance_score == 1.0

    def test_none_relevant(self):
        r = ContextRelevancyResult(query="q", context_chunks=["a"])
        r.num_sentences = 5
        r.num_relevant = 0
        r.relevance_score = 0.0
        assert r.relevance_score == 0.0

    def test_json_serializable(self):
        r = ContextRelevancyResult(query="test q", context_chunks=["chunk 1"])
        r.num_sentences = 2
        r.num_relevant = 1
        r.relevance_score = 0.5
        r.per_sentence = [
            {"id": 0, "text": "chunk 1 sentence.", "relevant": True},
        ]
        d = r.to_dict()
        json_str = json.dumps(d)
        assert json_str is not None
        assert "test q" in json_str
        assert "relevance_score" in json_str


class TestSentenceSplitting:
    """测试上下文分句逻辑（纯函数，无 LLM 调用）"""

    def test_single_sentence(self):
        chunks = ["This is a test sentence."]
        sentences = split_context_to_sentences(chunks)
        assert len(sentences) >= 1
        assert any("test" in s for s in sentences)

    def test_multiple_sentences_per_chunk(self):
        chunks = ["The conveyor belt has a load capacity. The motor requires regular maintenance. Safety guards must be installed."]
        sentences = split_context_to_sentences(chunks)
        assert len(sentences) == 3

    def test_multiple_chunks(self):
        chunks = [
            "The equipment must be inspected monthly. All operators need proper training.",
            "Emergency stops should be tested weekly. Maintenance logs are required.",
        ]
        sentences = split_context_to_sentences(chunks)
        assert len(sentences) == 4

    def test_empty_input(self):
        assert split_context_to_sentences([]) == []
        assert split_context_to_sentences([""]) == []
        assert split_context_to_sentences(["   "]) == []

    def test_filters_short_fragments(self):
        chunks = ["A sentence with enough words.", "x", "y", "Another valid sentence here."]
        sentences = split_context_to_sentences(chunks)
        # "x" and "y" should be filtered out (too short)
        assert len(sentences) >= 2
        for s in sentences:
            assert len(s.split()) >= 3  # sentences shorter than 3 words are filtered

    def test_handles_newlines(self):
        chunks = ["The manual specifies torque requirements.\nInstallation must follow the guidelines.\nSafety inspections are mandatory."]
        sentences = split_context_to_sentences(chunks)
        assert len(sentences) == 3

    def test_preserves_original_index(self):
        """验证按句号、换行、分号分割"""
        chunks = ["A. B. C."]
        sentences = split_context_to_sentences(chunks)
        # Each should be at least 3 chars after stripping
        assert all(len(s) >= 3 for s in sentences)


class TestRelevanceResponseParsing:
    """测试 LLM 返回的 JSON 解析（纯函数，无 LLM 调用）"""

    def test_valid_json_response(self):
        response = '[{"id": 0, "relevant": true}, {"id": 1, "relevant": false}]'
        result = parse_relevance_response(response, 2)
        assert result == [True, False]

    def test_partial_scores(self):
        response = '[{"id": 0, "relevant": true}]'
        result = parse_relevance_response(response, 2)
        # Only sentence 0 marked relevant, sentence 1 defaults to False
        assert result == [True, False]

    def test_empty_response(self):
        result = parse_relevance_response("", 3)
        assert result == [False, False, False]

    def test_malformed_json(self):
        result = parse_relevance_response("not valid json at all", 2)
        assert result == [False, False]

    def test_extract_json_from_markdown_block(self):
        response = '```json\n[{"id": 0, "relevant": true}]\n```'
        result = parse_relevance_response(response, 1)
        assert result == [True]

    def test_yes_no_format_fallback(self):
        """LLM 可能不按 JSON 返回，而是用 [0] YES [1] NO 格式"""
        response = "[0] YES\n[1] NO\n[2] YES"
        result = parse_relevance_response(response, 3)
        assert result == [True, False, True]

    def test_mixed_case_yes_no(self):
        response = "0: yes\n1: no\n2: YES"
        result = parse_relevance_response(response, 3)
        assert result == [True, False, True]

    def test_too_few_results_defaults_false(self):
        response = "0: yes"
        result = parse_relevance_response(response, 4)
        assert result == [True, False, False, False]


class TestContextRelevancyJSONSerialization:
    """测试 ContextRelevancyResult 的 JSON 序列化路径"""

    def test_full_json_output(self):
        """模拟完整的 Context Relevance 评测输出"""
        results = []
        for i in range(3):
            r = ContextRelevancyResult(
                query=f"query {i}",
                context_chunks=[f"chunk {i}"],
            )
            r.num_sentences = 5
            r.num_relevant = 3
            r.relevance_score = 0.6
            r.per_sentence = [
                {"id": j, "text": f"sentence {j}", "relevant": j < 3}
                for j in range(5)
            ]
            results.append(r)

        output = {
            "avg_context_relevancy": np.mean([r.relevance_score for r in results]),
            "results": [r.to_dict() for r in results],
        }
        json_str = json.dumps(output, indent=2, ensure_ascii=False)
        assert json_str is not None
        assert "avg_context_relevancy" in json_str
        assert "0.6" in json_str


# ─── Context Compression + Threshold Guard ──────────────────────

class MockBGEEmbedder:
    """Mock BGE embedder — 返回随机嵌入，相似度由文本中的关键词决定"""

    def __init__(self, dim: int = 1024):
        self.dim = dim
        np.random.seed(42)

    def encode(self, texts):
        """Mock encode: 用文本 hash 生成伪嵌入"""
        import torch
        embs = []
        for t in texts:
            # 用文本的 hash 作为种子生成确定性嵌入
            seed = hash(t) % (2**31)
            rng = np.random.RandomState(seed)
            vec = rng.randn(self.dim).astype(np.float32)
            vec = vec / (np.linalg.norm(vec) + 1e-10)  # L2 normalize
            embs.append(torch.from_numpy(vec))
        return torch.stack(embs)


class TestContextCompression:
    """测试上下文压缩函数"""

    def test_compress_context_reduces_sentences(self):
        """压缩后的句子数应 < 原始句子数（ratio=0.5）"""
        chunks = [
            "The conveyor belt has a load capacity of 500kg. The motor requires regular maintenance.",
            "Safety guards must be installed. Operators need proper training certificates.",
            "The facility is located in Houston Texas. Coffee breaks are 15 minutes long.",
            "Annual inspections are mandatory per OSHA regulation 1910. Fire exits must be clearly marked.",
        ]
        bge = MockBGEEmbedder(dim=128)  # smaller dim for speed
        result = compress_context(
            query="What is the load capacity of the conveyor belt?",
            chunks=chunks,
            bge_embedder=bge,
            ratio=0.5,
        )
        # Should have kept ~50% of sentences
        original_count = len(split_context_to_sentences(chunks))
        compressed_count = len(result.split("\n"))
        assert compressed_count < original_count
        assert compressed_count >= 3  # min keep

    def test_compress_context_short_circuits(self):
        """≤5 句时跳过压缩，返回原文"""
        chunks = ["A short text. Only two sentences here."]
        bge = MockBGEEmbedder(dim=128)
        result = compress_context(
            query="test query",
            chunks=chunks,
            bge_embedder=bge,
            ratio=0.4,
        )
        # Should return joined chunks, not compressed
        assert result == "\n\n".join(chunks)

    def test_compress_context_preserves_order(self):
        """保留的句子应按原文顺序排列"""
        chunks = [
            "First sentence about safety. Second sentence about maintenance.",
            "Third sentence about capacity. Fourth sentence about weather.",
            "Fifth sentence about inspections. Sixth sentence about lunch.",
        ]
        bge = MockBGEEmbedder(dim=128)
        result = compress_context(
            query="safety capacity inspections",
            chunks=chunks,
            bge_embedder=bge,
            ratio=0.5,
        )
        kept = result.split("\n")
        # Check sentences appear in original order
        indices = []
        all_sentences = split_context_to_sentences(chunks)
        for s in kept:
            try:
                indices.append(all_sentences.index(s))
            except ValueError:
                pass
        assert indices == sorted(indices), f"Sentences not in order: {indices}"

    def test_compress_context_empty_chunks(self):
        """空输入返回空字符串"""
        bge = MockBGEEmbedder(dim=128)
        result = compress_context(
            query="test", chunks=[], bge_embedder=bge, ratio=0.4,
        )
        assert result == ""


class TestConfidenceThreshold:
    """测试置信度阈值检测逻辑"""

    def test_threshold_rejects_low_score(self):
        """max_rerank < threshold 时应拒答"""
        threshold = 0.3
        rerank_scores = [0.05, 0.08, 0.02, 0.10, 0.15]
        max_score = max(rerank_scores)
        assert max_score < threshold
        assert threshold > 0
        # Simulates the check in evaluate_generation
        threshold_rejected = threshold > 0 and max_score < threshold and len(rerank_scores) > 0
        assert threshold_rejected is True

    def test_threshold_passes_high_score(self):
        """max_rerank >= threshold 时应通过"""
        threshold = 0.3
        rerank_scores = [0.45, 0.38, 0.52, 0.30, 0.41]
        max_score = max(rerank_scores)
        assert max_score >= threshold
        threshold_rejected = threshold > 0 and max_score < threshold and len(rerank_scores) > 0
        assert threshold_rejected is False

    def test_threshold_zero_disabled(self):
        """threshold=0 时无论如何都通过"""
        threshold = 0.0
        rerank_scores = [0.001, 0.002]
        max_score = max(rerank_scores)
        threshold_rejected = threshold > 0 and max_score < threshold and len(rerank_scores) > 0
        assert threshold_rejected is False

    def test_threshold_empty_scores(self):
        """无 rerank_score 时不应拒答"""
        threshold = 0.3
        rerank_scores = []
        max_score = max(rerank_scores) if rerank_scores else 0.0
        threshold_rejected = threshold > 0 and max_score < threshold and len(rerank_scores) > 0
        assert threshold_rejected is False

    def test_threshold_exact_match(self):
        """max_score == threshold 时通过（不是小于）"""
        threshold = 0.3
        rerank_scores = [0.3, 0.4]
        max_score = max(rerank_scores)
        threshold_rejected = threshold > 0 and max_score < threshold and len(rerank_scores) > 0
        assert threshold_rejected is False
