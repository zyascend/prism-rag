"""Failure Clinic unit tests."""
from __future__ import annotations

import json

from src.diagnostics import (
    diagnose_failure,
    format_diagnosis_markdown,
    get_pattern,
    list_patterns,
    parse_diagnosis_json,
)
from src.diagnostics.failure_clinic import heuristic_diagnose


def test_list_patterns_has_12():
    pats = list_patterns()
    assert len(pats) == 12
    assert pats[0]["id"] == "P01"
    assert get_pattern("p04").id == "P04"


def test_heuristic_stale_index():
    bug = "After delete_document ghost chunks still recalled; tombstone not applied"
    d = heuristic_diagnose(bug)
    assert d["primary_pattern"] == "P04"
    assert d["mode"] == "heuristic"


def test_heuristic_hallucination():
    bug = "Model hallucinated Bitcoin payment; answer unsupported by FAQ chunks"
    d = heuristic_diagnose(bug)
    assert d["primary_pattern"] == "P01"


def test_parse_diagnosis_json():
    raw = json.dumps(
        {
            "primary_pattern": "P09",
            "secondary_patterns": ["P01"],
            "reasoning": ["sample size"],
            "minimal_structural_fix": "run full 283q protocol",
            "confidence": 0.8,
        }
    )
    p = parse_diagnosis_json(raw)
    assert p["primary_pattern"] == "P09"
    assert p["secondary_patterns"] == ["P01"]


def test_diagnose_with_llm_mock():
    def complete(_prompt: str) -> str:
        return json.dumps(
            {
                "primary_pattern": "P05",
                "secondary_patterns": ["P03"],
                "reasoning": ["HyDE rewrite hurt nDCG"],
                "minimal_structural_fix": "disable HyDE by default",
                "confidence": 0.9,
            }
        )

    d = diagnose_failure("HyDE rewrite sent queries to wrong path", complete_fn=complete)
    assert d["primary_pattern"] == "P05"
    assert d["mode"] == "llm"
    assert "HyDE" in d["minimal_structural_fix"] or "disable" in d["minimal_structural_fix"]


def test_diagnose_llm_fallback_to_heuristic():
    d = diagnose_failure(
        "embedding MaxSim visual_only is noise",
        complete_fn=lambda p: "broken",
        use_heuristic_fallback=True,
    )
    assert d["mode"] == "heuristic_fallback"
    assert d["primary_pattern"] in {"P03", "P01"}


def test_format_markdown():
    d = diagnose_failure("config secret missing only in production env")
    md = format_diagnosis_markdown(d)
    assert "Failure Clinic" in md
    assert "P11" in md or d["primary_pattern"] in md
