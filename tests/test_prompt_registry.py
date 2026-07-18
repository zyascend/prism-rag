"""src/prompts 注册表单测 + prompt 迁移回归校验。

覆盖：
- loader 正常加载 / 各类非法配置抛 PromptConfigError
- registry get_active / list_prompts / PromptNotFound
- 迁移回归：9 个内置 prompt 与迁移前硬编码文本字节一致
"""
from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from src.prompts import (
    PromptConfigError,
    PromptNotFound,
    get_active,
    list_prompts,
)
from src.prompts.loader import load_prompt_file
from src.prompts.registry import PromptRegistry


# ─── 测试夹具：写临时 yaml ─────────────────────────────────────
def _write(tmp_path: Path, name: str, content: str) -> Path:
    p = tmp_path / name
    p.write_text(textwrap.dedent(content), encoding="utf-8")
    return p


# ─── loader 正常路径 ───────────────────────────────────────────
def test_load_valid_single_field(tmp_path):
    f = _write(tmp_path, "demo.yaml", """
        id: demo
        description: d
        versions:
          - version: 1
            created_at: "2026-07-18"
            author: yang
            changelog: init
            active: true
            template: |-
              Hello {name}
    """)
    p = load_prompt_file(f)
    assert p.id == "demo"
    assert p.active_version.version == 1
    assert p.active_version.template == "Hello {name}"
    assert p.active_version.render(name="X") == "Hello X"


def test_load_multi_version_active_selection(tmp_path):
    f = _write(tmp_path, "demo.yaml", """
        id: demo
        versions:
          - version: 1
            active: false
            changelog: old
            template: v1
          - version: 2
            active: true
            changelog: new
            template: v2
    """)
    p = load_prompt_file(f)
    assert p.active_version.version == 2
    assert p.active_version.template == "v2"
    assert [v.version for v in p.versions] == [1, 2]


def test_system_user_fields(tmp_path):
    f = _write(tmp_path, "demo.yaml", """
        id: demo
        versions:
          - version: 1
            active: true
            system: sys
            user: |-
              U {q}
    """)
    pv = load_prompt_file(f).active_version
    assert pv.system == "sys"
    assert pv.render("user", q="hi") == "U hi"


# ─── loader 异常路径 ───────────────────────────────────────────
def test_missing_id(tmp_path):
    f = _write(tmp_path, "bad.yaml", """
        versions:
          - version: 1
            active: true
            template: x
    """)
    with pytest.raises(PromptConfigError):
        load_prompt_file(f)


def test_no_active(tmp_path):
    f = _write(tmp_path, "bad.yaml", """
        id: bad
        versions:
          - version: 1
            active: false
            template: x
    """)
    with pytest.raises(PromptConfigError):
        load_prompt_file(f)


def test_multiple_active(tmp_path):
    f = _write(tmp_path, "bad.yaml", """
        id: bad
        versions:
          - version: 1
            active: true
            template: x
          - version: 2
            active: true
            template: y
    """)
    with pytest.raises(PromptConfigError):
        load_prompt_file(f)


def test_duplicate_version(tmp_path):
    f = _write(tmp_path, "bad.yaml", """
        id: bad
        versions:
          - version: 1
            active: true
            template: x
          - version: 1
            active: false
            template: y
    """)
    with pytest.raises(PromptConfigError):
        load_prompt_file(f)


def test_empty_versions(tmp_path):
    f = _write(tmp_path, "bad.yaml", """
        id: bad
        versions: []
    """)
    with pytest.raises(PromptConfigError):
        load_prompt_file(f)


def test_no_body_fields(tmp_path):
    f = _write(tmp_path, "bad.yaml", """
        id: bad
        versions:
          - version: 1
            active: true
    """)
    with pytest.raises(PromptConfigError):
        load_prompt_file(f)


# ─── registry ─────────────────────────────────────────────────
def test_registry_get_active_and_list(tmp_path):
    _write(tmp_path, "a.yaml", """
        id: a
        versions:
          - version: 1
            active: true
            template: ta
    """)
    _write(tmp_path, "b.yaml", """
        id: b
        description: bd
        versions:
          - version: 1
            active: false
            template: b1
          - version: 2
            active: true
            template: b2
    """)
    reg = PromptRegistry()
    reg.init(prompts_dir=str(tmp_path))
    assert reg.get_active("a").template == "ta"
    assert reg.get_active("b").template == "b2"
    listing = reg.list_prompts()
    assert listing["b"]["active_version"] == 2
    assert listing["b"]["versions"] == [1, 2]


def test_registry_not_found(tmp_path):
    _write(tmp_path, "a.yaml", """
        id: a
        versions:
          - version: 1
            active: true
            template: ta
    """)
    reg = PromptRegistry()
    reg.init(prompts_dir=str(tmp_path))
    with pytest.raises(PromptNotFound):
        reg.get_active("nope")


def test_registry_duplicate_id(tmp_path):
    _write(tmp_path, "a.yaml", """
        id: dup
        versions:
          - version: 1
            active: true
            template: x
    """)
    _write(tmp_path, "b.yaml", """
        id: dup
        versions:
          - version: 1
            active: true
            template: y
    """)
    reg = PromptRegistry()
    with pytest.raises(PromptConfigError):
        reg.init(prompts_dir=str(tmp_path))


def test_registry_missing_dir():
    reg = PromptRegistry()
    with pytest.raises(PromptConfigError):
        reg.init(prompts_dir="/nonexistent/path/xyz")


# ─── 迁移回归：内置 prompt 与迁移前硬编码文本字节一致 ──────────
# 下方期望值直接复制自迁移前的源码常量，任何后续改动若破坏 v1 文本会被此测试拦截。
_EXPECTED = {
    ("answer_generation", "system"):
        "You are a precise assistant. Answer ONLY from the provided context. "
        "If the context lacks the answer, say you don't know.",
    ("answer_generation", "user"):
        "Context:\n{context}\n\nQuestion: {query}",
    ("claim_decomposition", "template"):
        "Break down the following answer into individual factual claims.\n"
        "Each claim must be a single, atomic, verifiable statement.\n"
        "Format: one claim per line, numbered.\n\n"
        "Answer: {answer}\n\nClaims:",
    ("claim_verification", "template"):
        "Determine whether the following claim is DIRECTLY SUPPORTED by the given context.\n"
        "Answer ONLY with YES or NO.\n\nContext:\n{context}\n\nClaim: {claim}\n\n"
        "Is this claim directly supported by the context? (YES/NO)",
    ("reverse_question", "template"):
        "Given the following answer, generate {n} different questions that this answer could be answering.\n"
        "Each question should be phrased as a natural question someone might ask.\n"
        "Format: one question per line, numbered.\n\nAnswer: {answer}\n\nQuestions:",
    ("ragas_generation", "template"):
        "You are a helpful assistant for industrial document QA.\n"
        "Answer the question based ONLY on the provided context.\n"
        'If the context does not contain enough information, say "I cannot answer this question based on the available documents."\n'
        "Do NOT make up information.\n\nContext:\n{context}\n\nQuestion: {question}\n\nAnswer:",
    ("relevancy_fallback", "template"):
        "On a scale of 0.0 to 1.0, how relevant is the following ANSWER to the QUESTION?\n"
        "Consider: does the answer directly address the question, or is it tangential?\n"
        "Reply with ONLY a number between 0.0 and 1.0.\n\n"
        "Question: {question}\nAnswer: {answer}\n\nRelevance score:",
    ("context_relevance", "template"):
        "Please evaluate whether each sentence below is relevant to answering the given question.\n"
        "A sentence is relevant if it contains information that could help answer the question,\n"
        'even indirectly. Reply with a JSON array: [{{"id": <number>, "relevant": true/false}}, ...]\n\n'
        "Question: {query}\n\nSentences:\n{sentences}",
    ("hyde", "template"):
        "Write a technical passage that answers the following question about "
        "industrial documents. Be specific and factual.\n\n"
        "Question: {query}\n\nPassage:",
    ("table_summary", "template"):
        "You are a precise technical writer. Summarize the following markdown table into "
        "1-3 factual sentences that describe: (1) what the table is about, (2) its columns, "
        "and (3) any notable rows or extreme values. Do NOT invent data not present in the table. "
        "Output only the summary, no preamble.\n\nTable:\n{table}\n",
}


@pytest.mark.parametrize("key,expected", list(_EXPECTED.items()))
def test_builtin_prompt_migration_byte_identical(key, expected):
    prompt_id, field = key
    assert get_active(prompt_id).text(field) == expected


def test_context_relevance_format_escaping():
    """context_relevance 经 .format() 后 {{ }} 应还原为字面 { }。"""
    tmpl = get_active("context_relevance").template
    out = tmpl.format(query="Q", sentences="S")
    assert '[{"id": <number>, "relevant": true/false}, ...]' in out
    assert "Question: Q" in out


def test_list_builtin_prompts_present():
    listing = list_prompts()
    for pid in [
        "answer_generation", "claim_decomposition", "claim_verification",
        "reverse_question", "ragas_generation", "relevancy_fallback",
        "context_relevance", "hyde", "table_summary",
    ]:
        assert pid in listing
        assert listing[pid]["active_version"] == 1


def test_prompts_endpoint():
    """GET /prompts 只读端点返回全部生效版本摘要。"""
    from fastapi.testclient import TestClient
    from src.api.routes import app

    client = TestClient(app)
    resp = client.get("/prompts")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert "answer_generation" in body["prompts"]
    assert body["prompts"]["answer_generation"]["active_version"] == 1
