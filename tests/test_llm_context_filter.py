"""LLM 句级上下文过滤单测（mock complete_fn，无真模型）。"""

from src.generation.context_filter import (
    filter_sentences_llm,
    prepare_context,
    _parse_keep_indices,
)


def test_filter_keeps_marked_sentences():
    def fake_complete(prompt: str) -> str:
        return '{"keep": [0, 2]}'

    text = (
        "Relevant torque is fifty Nm for this unit.\n"
        "Noise line about company history and founders only.\n"
        "See table note on page three for limits."
    )
    out = filter_sentences_llm(text, query="max torque?", complete_fn=fake_complete)
    assert "fifty Nm" in out or "torque" in out.lower()
    assert "company history" not in out


def test_filter_fallback_on_bad_json():
    def fake_complete(prompt: str) -> str:
        return "not-json"

    text = (
        "First sentence with enough words here.\n"
        "Second sentence with enough words too.\n"
        "Third sentence also long enough now."
    )
    out = filter_sentences_llm(
        text, query="q", complete_fn=fake_complete, fallback=lambda t, q: t
    )
    assert out == text


def test_parse_keep_indices_fence():
    raw = '```json\n{"keep": [1, 0]}\n```'
    assert _parse_keep_indices(raw, n=3) == [1, 0]


def test_prepare_context_off():
    chunks = ["Sentence one has enough words here.", "Sentence two has enough words too."]
    out = prepare_context("q", chunks, bge_embedder=None, mode="off")
    assert "Sentence one" in out and "Sentence two" in out


def test_prepare_context_llm_with_mock():
    chunks = [
        "The max torque rating is fifty newton meters exactly.",
        "The company was founded in nineteen eighty by engineers.",
        "Ambient temperature limits are listed in section four.",
    ]

    def fake_complete(prompt: str) -> str:
        return '{"keep": [0]}'

    out = prepare_context(
        "What is max torque?",
        chunks,
        bge_embedder=None,
        mode="llm",
        complete_fn=fake_complete,
    )
    assert "torque" in out.lower()
    assert "nineteen eighty" not in out
