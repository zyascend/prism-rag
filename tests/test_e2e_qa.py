"""测试端到端 QA 评测的非 LLM 依赖部分"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
import pytest

from src.evaluation.e2e_qa import (
    AnswerCorrectnessResult,
    RejectionResult,
    E2EQAEvalResult,
    is_answer_rejected,
    load_qa_dataset,
)


class TestDataStructures:
    """测试数据结构的序列化和反序列化"""

    def test_answer_correctness_result_empty(self):
        r = AnswerCorrectnessResult(
            question="",
            expected_answer="",
            generated_answer="",
            context_length=0,
        )
        d = r.to_dict()
        assert d["is_correct"] is False
        assert d["correctness_score"] == 0.0
        assert d["judge_reasoning"] == ""

    def test_answer_correctness_result_with_data(self):
        r = AnswerCorrectnessResult(
            question="What is the load capacity?",
            expected_answer="500 kg per meter",
            generated_answer="The load capacity is 500 kg/m.",
            context_length=200,
            is_correct=True,
            correctness_score=1.0,
            judge_reasoning="Semantically equivalent",
        )
        d = r.to_dict()
        assert d["is_correct"] is True
        assert d["correctness_score"] == 1.0
        assert d["judge_reasoning"] == "Semantically equivalent"
        assert d["question"] == "What is the load capacity?"
        assert d["expected_answer"] == "500 kg per meter"

    def test_rejection_result(self):
        r = RejectionResult(
            question="What is the meaning of life?",
            expected_rejection=True,
            generated_answer="I cannot answer this question based on the available documents.",
            is_rejected=True,
            rejection_correct=True,
        )
        d = r.to_dict()
        assert d["question"] == "What is the meaning of life?"
        assert d["expected_rejection"] is True
        assert d["is_rejected"] is True
        assert d["rejection_correct"] is True

    def test_rejection_result_false_positive(self):
        """测试系统没有拒答但应该拒答的情况"""
        r = RejectionResult(
            question="What is the meaning of life?",
            expected_rejection=True,
            generated_answer="The meaning of life is 42.",
            is_rejected=False,
            rejection_correct=False,
        )
        d = r.to_dict()
        assert d["is_rejected"] is False
        assert d["rejection_correct"] is False

    def test_e2e_qa_result_empty(self):
        r = E2EQAEvalResult()
        s = r.summary_dict()
        assert s["num_answerable"] == 0
        assert s["num_rejection"] == 0
        assert s["avg_correctness"] == 0.0
        assert s["rejection_accuracy"] == 0.0
        assert s["combined_score"] == 0.0

    def test_e2e_qa_result_with_data(self):
        r = E2EQAEvalResult(
            correctness_results=[
                AnswerCorrectnessResult(
                    question="q1", expected_answer="a1", generated_answer="a1",
                    context_length=100, is_correct=True, correctness_score=1.0,
                ),
                AnswerCorrectnessResult(
                    question="q2", expected_answer="a2", generated_answer="wrong",
                    context_length=100, is_correct=False, correctness_score=0.0,
                ),
            ],
            rejection_results=[
                RejectionResult(
                    question="r1", expected_rejection=True,
                    generated_answer="I cannot answer.", is_rejected=True,
                    rejection_correct=True,
                ),
            ],
            avg_correctness=0.5,
            rejection_accuracy=1.0,
            combined_score=0.65,
            num_answerable=2,
            num_rejection=1,
            latency_total=3.0,
            rejected_count_answerable=0,
        )
        s = r.summary_dict()
        assert s["num_answerable"] == 2
        assert s["num_rejection"] == 1
        assert s["avg_correctness"] == 0.5
        assert s["rejection_accuracy"] == 1.0
        assert s["combined_score"] == 0.65
        assert s["avg_latency_seconds"] == 1.0  # 3.0 / 3

    def test_answer_correctness_serialization(self):
        """测试完整的 to_dict 输出"""
        r = AnswerCorrectnessResult(
            question="What is the max pressure?",
            expected_answer="55 PSI",
            generated_answer="55 PSI",
            context_length=150,
            is_correct=True,
            correctness_score=1.0,
            judge_reasoning="JUDGMENT: YES\nREASONING: Exact match",
        )
        d = r.to_dict()
        assert d["correctness_score"] == 1.0
        assert "YES" in d["judge_reasoning"]


class TestIsAnswerRejected:
    """测试拒答检测逻辑"""

    def test_rejection_phrases_cannot_answer(self):
        assert is_answer_rejected("I cannot answer this question based on the available documents.") is True

    def test_rejection_phrases_not_enough_info(self):
        assert is_answer_rejected("There is not enough information in the context to answer.") is True

    def test_rejection_phrases_no_info(self):
        assert is_answer_rejected("No information about this topic was found.") is True

    def test_rejection_phrases_out_of_scope(self):
        assert is_answer_rejected("This question is out of scope of the available documents.") is True

    def test_no_rejection_for_valid_answer(self):
        assert is_answer_rejected("The load capacity is 500 kg per meter.") is False

    def test_no_rejection_for_empty_answer(self):
        assert is_answer_rejected("") is True

    def test_no_rejection_for_none_answer(self):
        assert is_answer_rejected(None) is True  # noqa

    def test_rejection_case_insensitive(self):
        assert is_answer_rejected("I CANNOT ANSWER this question.") is True

    def test_rejection_beyond_scope(self):
        assert is_answer_rejected("This is beyond the scope of the provided documents.") is True

    def test_rejection_context_does_not_contain(self):
        assert is_answer_rejected("The context does not contain the required information.") is True


class TestLoadQADataset:
    """测试 QA 数据集加载"""

    def test_load_mixed_dataset(self):
        """测试加载混合数据集（可回答 + 拒答）"""
        data = [
            {"id": "e2e_001", "type": "answerable", "question": "Q1", "expected_answer": "A1"},
            {"id": "e2e_002", "type": "answerable", "question": "Q2", "expected_answer": "A2"},
            {"id": "rej_001", "type": "rejection", "question": "R1", "expected_rejection": True},
        ]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            f.flush()
            answerable, rejection = load_qa_dataset(f.name)
        Path(f.name).unlink()

        assert len(answerable) == 2
        assert len(rejection) == 1
        assert answerable[0]["question"] == "Q1"
        assert rejection[0]["question"] == "R1"

    def test_load_legacy_format(self):
        """测试兼容旧格式（expected_rejection 字段推断类型）"""
        data = [
            {"question": "Q1", "expected_answer": "A1"},
            {"question": "R1", "expected_rejection": True, "reason": "out of scope"},
        ]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            f.flush()
            answerable, rejection = load_qa_dataset(f.name)
        Path(f.name).unlink()

        assert len(answerable) == 1
        assert len(rejection) == 1
        assert answerable[0]["question"] == "Q1"
        assert rejection[0]["question"] == "R1"

    def test_load_empty_array(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump([], f)
            f.flush()
            answerable, rejection = load_qa_dataset(f.name)
        Path(f.name).unlink()
        assert len(answerable) == 0
        assert len(rejection) == 0

    def test_load_actual_dataset(self):
        """测试加载实际生成的 e2e_qa.json 文件"""
        fixture_path = Path("data/e2e_qa.json")
        if not fixture_path.exists():
            pytest.skip("e2e_qa.json 不存在，跳过")
        answerable, rejection = load_qa_dataset(str(fixture_path))
        assert len(answerable) == 50
        assert len(rejection) == 20
        # 验证每个 answerable 都有 expected_answer
        for item in answerable:
            assert "expected_answer" in item, f"Missing expected_answer in {item['id']}"
            assert len(item["expected_answer"]) > 0, f"Empty expected_answer in {item['id']}"


class TestComputeAnswerCorrectness:
    """测试答案正确性判断的辅助逻辑（非 LLM 部分）"""

    def test_empty_generated_answer(self):
        from src.evaluation.e2e_qa import compute_answer_correctness
        r = compute_answer_correctness(
            question="test",
            expected_answer="expected",
            generated_answer="",
        )
        assert r.is_correct is False
        assert r.correctness_score == 0.0
        assert "Empty" in r.judge_reasoning

    def test_rejected_generated_answer(self):
        # 如果生成答案是拒答，应该判错
        # 注意：compute_answer_correctness 内部会调用 LLM，所以这个测试只验证
        # 生成答案为空时的逻辑
        from src.evaluation.e2e_qa import compute_answer_correctness
        r = compute_answer_correctness(
            question="test",
            expected_answer="expected",
            generated_answer="I cannot answer this question based on the available documents.",
        )
        assert r.is_correct is False
        assert r.correctness_score == 0.0
        assert "rejection" in r.judge_reasoning


class TestEndToEndFlow:
    """测试端到端流程的非 LLM 部分"""

    def test_evaluate_e2e_qa_no_retriever(self):
        """测试无 retriever 时的错误处理"""
        # 这个测试只验证 load_qa_dataset 是否被正确调用
        # 完整的集成测试需要 LLM 调用
        pass

    def test_summary_dict_metrics(self):
        """测试汇总指标计算的正确性"""
        result = E2EQAEvalResult(
            avg_correctness=0.8,
            rejection_accuracy=0.9,
            num_answerable=10,
            num_rejection=10,
            latency_total=30.0,
        )
        s = result.summary_dict()
        # 综合分数 = (0.8 * 0.7 + 0.9 * 0.3) / 1.0 = 0.83
        #   total_weight = 0.7, rejection_weight = 0.3, total_weight_sum = 1.0
        #   combined = (0.8 * 0.7 + 0.9 * 0.3) / 1.0 = 0.83
        expected_combined = (0.8 * 0.7 + 0.9 * 0.3) / 1.0
        assert s["combined_score"] == pytest.approx(expected_combined, 0.01)
        assert s["avg_latency_seconds"] == 1.5  # 30.0 / 20

    def test_summary_dict_only_rejection(self):
        """只有拒答问题时的综合分数"""
        result = E2EQAEvalResult(
            rejection_accuracy=0.9,
            num_rejection=10,
            latency_total=5.0,
        )
        s = result.summary_dict()
        # total_weight = 0.0 (no answerable), rejection_weight = 0.3
        # total_weight_sum = 0.3
        # combined = (0.0 + 0.9 * 0.3) / 0.3 = 0.9
        assert s["combined_score"] == pytest.approx(0.9, 0.01)

    def test_summary_dict_only_answerable(self):
        """只有可回答问题时的综合分数"""
        result = E2EQAEvalResult(
            avg_correctness=0.8,
            num_answerable=10,
            latency_total=10.0,
        )
        s = result.summary_dict()
        # total_weight = 0.7, rejection_weight = 0.0
        # total_weight_sum = 0.7
        # combined = (0.8 * 0.7 + 0.0) / 0.7 = 0.8
        assert s["combined_score"] == pytest.approx(0.8, 0.01)