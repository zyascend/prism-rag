"""Self-RAG Gate2 单测（mock judge / generator，无真模型）。"""
from __future__ import annotations

import json

from src.generation.self_rag import (
    ABSTAIN_ANSWER,
    SelfRAGOrchestrator,
    _parse_verdict_json,
    evaluate_gate2_whole_answer,
    self_rag_cache_salt,
    should_apply_gate2,
)
from src.rejection import is_rejection


class _FakeCompletions:
    def __init__(self, answers):
        self._answers = list(answers)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        content = self._answers.pop(0) if self._answers else "fallback"
        return type(
            "R",
            (),
            {
                "choices": [
                    type(
                        "C",
                        (),
                        {"message": type("M", (), {"content": content})()},
                    )()
                ]
            },
        )()


class _FakeClient:
    def __init__(self, answers):
        self.chat = type(
            "Chat", (), {"completions": _FakeCompletions(answers)}
        )()


class _FakeGenerator:
    """按调用顺序返回预设答案；记录 prompt_id / precomputed_context。"""

    def __init__(self, answers):
        self._answers = list(answers)
        self.calls = []
        self.model = "fake-model"
        self.client = None
        self.temperature = 0.0

    def answer(
        self,
        query,
        retrieved,
        k_context=5,
        *,
        prompt_id="answer_generation",
        precomputed_context=None,
    ):
        self.calls.append(
            {
                "query": query,
                "prompt_id": prompt_id,
                "precomputed_context": precomputed_context,
                "k_context": k_context,
            }
        )
        text = self._answers.pop(0) if self._answers else "out of answers"
        ctx = precomputed_context
        if ctx is None and retrieved:
            ctx = retrieved[0].get("text", "")
        citations = [
            {
                "chunk_id": r["chunk_id"],
                "page_id": r["page_id"],
                "doc_id": r.get("doc_id"),
                "page_number": r.get("page_number"),
                "snippet": (r.get("text") or "")[:200],
            }
            for r in retrieved[:k_context]
        ]
        return {"answer": text, "citations": citations, "context": ctx or ""}


def _retrieved():
    return [
        {
            "chunk_id": "c1",
            "page_id": 1,
            "doc_id": "d1",
            "page_number": 1,
            "text": "The hydraulic pump service interval is 500 hours.",
        }
    ]


def test_parse_verdict_json_fence():
    raw = '```json\n{"grounded": true, "score": 0.9, "unsupported": []}\n```'
    data = _parse_verdict_json(raw)
    assert data["grounded"] is True
    assert data["score"] == 0.9


def test_evaluate_gate2_pass():
    def complete(_prompt: str) -> str:
        return json.dumps(
            {"grounded": True, "score": 0.95, "unsupported": []}
        )

    v = evaluate_gate2_whole_answer(
        "interval?",
        "500 hours",
        "pump interval is 500 hours",
        complete_fn=complete,
        faith_threshold=0.8,
    )
    assert v["passed"] is True
    assert v["score"] == 0.95


def test_evaluate_gate2_fail_low_score():
    def complete(_prompt: str) -> str:
        return json.dumps(
            {
                "grounded": False,
                "score": 0.2,
                "unsupported": ["made-up torque"],
            }
        )

    v = evaluate_gate2_whole_answer(
        "torque?",
        "max torque is 999 Nm",
        "no torque listed",
        complete_fn=complete,
        faith_threshold=0.8,
    )
    assert v["passed"] is False
    assert "torque" in v["unsupported"][0].lower() or v["unsupported"]


def test_orchestrator_disabled_passthrough():
    gen = _FakeGenerator(["plain answer"])
    orch = SelfRAGOrchestrator(
        gen, config={"enabled": False}, judge_complete_fn=lambda p: "{}"
    )
    out = orch.answer("q", _retrieved(), k_context=5)
    assert out["answer"] == "plain answer"
    assert out["self_rag"]["enabled"] is False
    assert out["self_rag"].get("applied") is False
    assert len(gen.calls) == 1


def test_orchestrator_low_rerank_skips_when_confident():
    """高 rerank 分时 trigger=low_rerank 跳过 Gate2。"""
    gen = _FakeGenerator(["confident answer"])
    retrieved = _retrieved()
    retrieved[0]["rerank_score"] = 0.9
    orch = SelfRAGOrchestrator(
        gen,
        config={
            "enabled": True,
            "trigger": "low_rerank",
            "low_rerank_threshold": 0.35,
            "faith_threshold": 0.8,
            "max_generate_attempts": 2,
            "on_fail": "abstain",
            "on_judge_error": "degrade_pass",
            "verdict_mode": "whole_answer",
        },
        judge_complete_fn=lambda p: (_ for _ in ()).throw(AssertionError("gate should not run")),
    )
    out = orch.answer("q", retrieved)
    assert out["answer"] == "confident answer"
    assert out["self_rag"]["applied"] is False
    assert out["self_rag"]["skip_reason"] == "trigger_not_met"
    assert len(gen.calls) == 1


def test_should_apply_gate2_low_rerank():
    assert should_apply_gate2(
        [{"rerank_score": 0.1}],
        {"enabled": True, "trigger": "low_rerank", "low_rerank_threshold": 0.35},
    )
    assert not should_apply_gate2(
        [{"rerank_score": 0.9}],
        {"enabled": True, "trigger": "low_rerank", "low_rerank_threshold": 0.35},
    )
    assert should_apply_gate2(
        [{"rerank_score": 0.9}],
        {"enabled": True, "trigger": "always"},
    )


def test_gate2_abstain_is_rejection():
    assert is_rejection(ABSTAIN_ANSWER) is True
    assert is_rejection("I don't know.") is True
    assert is_rejection("The torque is 50 Nm.") is False


def test_orchestrator_pass_first_try():
    gen = _FakeGenerator(["500 hours"])
    orch = SelfRAGOrchestrator(
        gen,
        config={
            "enabled": True,
            "trigger": "always",
            "faith_threshold": 0.8,
            "max_generate_attempts": 2,
            "on_fail": "regenerate_then_abstain",
            "on_judge_error": "degrade_pass",
            "verdict_mode": "whole_answer",
        },
        judge_complete_fn=lambda p: json.dumps(
            {"grounded": True, "score": 1.0, "unsupported": []}
        ),
    )
    out = orch.answer("interval?", _retrieved())
    assert out["answer"] == "500 hours"
    assert out["self_rag"]["passed"] is True
    assert out["self_rag"]["applied"] is True
    assert out["self_rag"]["attempts"] == 1
    assert out["self_rag"]["final_action"] == "return"
    assert len(gen.calls) == 1


def test_orchestrator_fail_then_regen_pass():
    gen = _FakeGenerator(["hallucinated 999", "500 hours"])
    scores = iter(
        [
            json.dumps(
                {"grounded": False, "score": 0.1, "unsupported": ["999"]}
            ),
            json.dumps(
                {"grounded": True, "score": 0.95, "unsupported": []}
            ),
        ]
    )
    orch = SelfRAGOrchestrator(
        gen,
        config={
            "enabled": True,
            "trigger": "always",
            "faith_threshold": 0.8,
            "max_generate_attempts": 2,
            "on_fail": "regenerate_then_abstain",
            "on_judge_error": "degrade_pass",
            "verdict_mode": "whole_answer",
        },
        judge_complete_fn=lambda p: next(scores),
    )
    out = orch.answer("interval?", _retrieved())
    assert out["answer"] == "500 hours"
    assert out["self_rag"]["final_action"] == "return_after_regen"
    assert out["self_rag"]["attempts"] == 2
    assert gen.calls[0]["prompt_id"] == "answer_generation"
    assert gen.calls[1]["prompt_id"] == "self_rag_regenerate"
    assert gen.calls[1]["precomputed_context"] == gen.calls[0][
        "precomputed_context"
    ] or gen.calls[1]["precomputed_context"]

    # attempts_detail：可回放 fail → regen
    detail = out["self_rag"]["attempts_detail"]
    assert len(detail) == 2
    assert detail[0]["attempt"] == 1
    assert detail[0]["action"] == "continue_regen"
    assert detail[0]["passed"] is False
    assert detail[0]["score"] == 0.1
    assert "999" in detail[0]["answer"]
    assert detail[0]["unsupported"] == ["999"]
    assert detail[0]["prompt_id"] == "answer_generation"
    assert detail[1]["attempt"] == 2
    assert detail[1]["action"] == "return_after_regen"
    assert detail[1]["passed"] is True
    assert detail[1]["score"] == 0.95
    assert "500" in detail[1]["answer"]
    assert detail[1]["prompt_id"] == "self_rag_regenerate"


def test_attempts_detail_on_double_fail_abstain():
    gen = _FakeGenerator(["bad1", "bad2 still wrong"])
    orch = SelfRAGOrchestrator(
        gen,
        config={
            "enabled": True,
            "trigger": "always",
            "faith_threshold": 0.8,
            "max_generate_attempts": 2,
            "on_fail": "regenerate_then_abstain",
            "on_judge_error": "degrade_pass",
            "verdict_mode": "whole_answer",
        },
        judge_complete_fn=lambda p: json.dumps(
            {"grounded": False, "score": 0.0, "unsupported": ["x"]}
        ),
    )
    out = orch.answer("q", _retrieved())
    detail = out["self_rag"]["attempts_detail"]
    assert len(detail) == 2
    assert detail[0]["action"] == "continue_regen"
    assert detail[1]["action"] == "abstain"
    assert detail[1]["answer"] == "bad2 still wrong"


def test_orchestrator_fail_abstain():
    gen = _FakeGenerator(["bad1", "bad2"])
    orch = SelfRAGOrchestrator(
        gen,
        config={
            "enabled": True,
            "trigger": "always",
            "faith_threshold": 0.8,
            "max_generate_attempts": 2,
            "on_fail": "regenerate_then_abstain",
            "on_judge_error": "degrade_pass",
            "verdict_mode": "whole_answer",
        },
        judge_complete_fn=lambda p: json.dumps(
            {"grounded": False, "score": 0.0, "unsupported": ["x"]}
        ),
    )
    out = orch.answer("q", _retrieved())
    assert out["answer"] == ABSTAIN_ANSWER
    assert out["citations"] == []
    assert out["self_rag"]["passed"] is False
    assert out["self_rag"]["final_action"] == "abstain"
    assert out["self_rag"]["attempts"] == 2


def test_orchestrator_on_fail_abstain_no_regen():
    gen = _FakeGenerator(["only once"])
    orch = SelfRAGOrchestrator(
        gen,
        config={
            "enabled": True,
            "trigger": "always",
            "faith_threshold": 0.8,
            "max_generate_attempts": 2,
            "on_fail": "abstain",
            "on_judge_error": "degrade_pass",
            "verdict_mode": "whole_answer",
        },
        judge_complete_fn=lambda p: json.dumps(
            {"grounded": False, "score": 0.0, "unsupported": []}
        ),
    )
    out = orch.answer("q", _retrieved())
    assert out["answer"] == ABSTAIN_ANSWER
    assert len(gen.calls) == 1
    assert out["self_rag"]["attempts"] == 1


def test_orchestrator_judge_error_degrade_pass():
    gen = _FakeGenerator(["maybe ok"])
    orch = SelfRAGOrchestrator(
        gen,
        config={
            "enabled": True,
            "trigger": "always",
            "faith_threshold": 0.8,
            "max_generate_attempts": 2,
            "on_fail": "regenerate_then_abstain",
            "on_judge_error": "degrade_pass",
            "verdict_mode": "whole_answer",
        },
        judge_complete_fn=lambda p: "not-json-at-all",
    )
    out = orch.answer("q", _retrieved())
    assert out["answer"] == "maybe ok"
    assert out["self_rag"]["gate_degraded"] is True
    assert out["self_rag"]["final_action"] == "degrade_pass"


def test_orchestrator_judge_error_abstain():
    gen = _FakeGenerator(["maybe ok"])
    orch = SelfRAGOrchestrator(
        gen,
        config={
            "enabled": True,
            "trigger": "always",
            "faith_threshold": 0.8,
            "max_generate_attempts": 2,
            "on_fail": "regenerate_then_abstain",
            "on_judge_error": "abstain",
            "verdict_mode": "whole_answer",
        },
        judge_complete_fn=lambda p: "not-json",
    )
    out = orch.answer("q", _retrieved())
    assert out["answer"] == ABSTAIN_ANSWER
    assert out["self_rag"]["final_action"] == "abstain_on_judge_error"


def test_self_rag_cache_salt_on_off():
    off = self_rag_cache_salt({"enabled": False})
    on = self_rag_cache_salt(
        {
            "enabled": True,
            "mode": "gate2_only",
            "faith_threshold": 0.8,
            "verdict_mode": "whole_answer",
            "on_fail": "regenerate_then_abstain",
        }
    )
    assert off == "sr=off"
    assert "sr=on" in on
    assert off != on


def test_answer_for_eval_disabled_uses_generator(monkeypatch):
    from src.generation import self_rag as sr_mod

    monkeypatch.setattr(
        sr_mod,
        "self_rag_config",
        lambda: {"enabled": False},
    )
    gen = _FakeGenerator(["via gen"])
    out = sr_mod.answer_for_eval("q", _retrieved(), generator=gen)
    assert out["answer"] == "via gen"
    assert out["self_rag"]["enabled"] is False
