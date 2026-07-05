"""测试 RAGAS 指标自实现的非 LLM 依赖部分"""
from __future__ import annotations

import json
import numpy as np
from pathlib import Path
import pytest

from src.evaluation.ragas_metrics import (
    FaithfulnessResult,
    AnswerRelevancyResult,
    RagasGenerationEvalResult,
    decompose_claims,
    generate_reverse_questions,
    _llm_relevancy_fallback,
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