"""Failure Clinic — 将 badcase / 事故描述映射到 P01–P12。

可在无 LLM 时做规则兜底；有 complete_fn 时走结构化 triage。
"""
from __future__ import annotations

import json
import re
import time
from typing import Any, Callable, Dict, List, Optional

from src.diagnostics.failure_patterns import (
    FAILURE_PATTERNS,
    get_pattern,
    patterns_prompt_block,
)

CompleteFn = Callable[[str], str]

_VALID_IDS = {p.id for p in FAILURE_PATTERNS}


def _parse_json(raw: str) -> Dict[str, Any]:
    text = (raw or "").strip()
    if not text:
        raise ValueError("empty diagnosis response")
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", text, re.I)
    if fence:
        text = fence.group(1).strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            raise ValueError(f"no JSON in diagnosis: {raw[:200]!r}")
        data = json.loads(text[start : end + 1])
    if not isinstance(data, dict):
        raise ValueError("diagnosis must be a JSON object")
    return data


def parse_diagnosis_json(raw: str) -> Dict[str, Any]:
    """解析并规范化 clinic 输出。"""
    data = _parse_json(raw)
    primary = str(data.get("primary_pattern") or data.get("primary") or "").upper()
    if primary not in _VALID_IDS:
        raise ValueError(f"invalid primary_pattern: {primary!r}")
    secondary_raw = data.get("secondary_patterns") or data.get("secondary") or []
    if isinstance(secondary_raw, str):
        secondary_raw = [secondary_raw]
    secondary: List[str] = []
    for s in secondary_raw:
        sid = str(s).upper().strip()
        if sid in _VALID_IDS and sid != primary and sid not in secondary:
            secondary.append(sid)
        if len(secondary) >= 2:
            break
    reasoning = data.get("reasoning") or []
    if isinstance(reasoning, str):
        reasoning = [reasoning]
    if not isinstance(reasoning, list):
        reasoning = [str(reasoning)]
    fix = str(data.get("minimal_structural_fix") or data.get("fix") or "").strip()
    return {
        "primary_pattern": primary,
        "secondary_patterns": secondary,
        "reasoning": [str(r)[:300] for r in reasoning][:8],
        "minimal_structural_fix": fix[:800],
        "confidence": float(data["confidence"])
        if data.get("confidence") is not None
        else None,
    }


def build_clinic_prompt(bug_description: str) -> str:
    return f"""You are triaging failures in PrismRAG (industrial multimodal PDF RAG:
BM25 + Dense + Visual ColPali, RRF, rerank, optional CRAG / Self-RAG Gate2).

Map the bug to exactly ONE primary pattern id from P01–P12, optionally up to TWO
secondary candidates. Propose a MINIMAL structural fix (retrieval, indexing,
routing, eval, config, infra) — not generic "use a better model".

Patterns:
{patterns_prompt_block()}

Return ONLY valid JSON (no markdown fence):
{{"primary_pattern": "P0X", "secondary_patterns": ["P0Y"], "reasoning": ["..."], "minimal_structural_fix": "...", "confidence": 0.0_to_1.0}}

BUG DESCRIPTION:
{bug_description.strip()}
"""


def heuristic_diagnose(bug_description: str) -> Dict[str, Any]:
    """无 LLM 时的关键词启发式（可测、可离线）。"""
    text = (bug_description or "").lower()
    scores: Dict[str, int] = {p.id: 0 for p in FAILURE_PATTERNS}

    rules = [
        ("P01", ["hallucin", "grounding", "unsupported", "contradict", "invent", "faith"]),
        ("P02", ["chunk", "segment", "truncat", "table split", "boundary", "cut off"]),
        ("P03", ["embedding", "maxsim", "cosine", "visual_only", "dense rank", "vector"]),
        ("P04", ["stale", "deleted", "ghost", "tombstone", "index_version", "still recall"]),
        ("P05", ["router", "hyde", "rewrite", "reformulat", "wrong path", "visual routing"]),
        ("P06", ["multi-step", "multi hop", "reasoning drift", "regenerate changed"]),
        ("P07", ["tool call", "tool-call", "function call", "wrong argument"]),
        ("P08", ["session", "multi-turn", "forgot", "previous turn", "memory"]),
        ("P09", ["eval", "sample size", "offline", "ragas", "blind spot", "50q", "283"]),
        ("P10", ["startup", "not ready", "cold start", "connection timeout", "5xx first"]),
        ("P11", ["config", "secret", "env", "local-dev", "hf_hub_offline", "api key"]),
        ("P12", ["oom", "concurrent", "race", "multi-tenant", "gpu contention"]),
    ]
    for pid, kws in rules:
        for kw in kws:
            if kw in text:
                scores[pid] += 1

    ranked = sorted(scores.items(), key=lambda x: (-x[1], x[0]))
    primary = ranked[0][0] if ranked[0][1] > 0 else "P01"
    secondary = [pid for pid, sc in ranked[1:3] if sc > 0]
    pat = get_pattern(primary)
    return {
        "primary_pattern": primary,
        "secondary_patterns": secondary,
        "reasoning": [
            "heuristic keyword triage (no LLM)",
            f"matched score={ranked[0][1]} for {primary}",
        ],
        "minimal_structural_fix": (pat.minimal_fix_hints if pat else ""),
        "confidence": 0.35 if ranked[0][1] > 0 else 0.15,
        "mode": "heuristic",
    }


def diagnose_failure(
    bug_description: str,
    *,
    complete_fn: Optional[CompleteFn] = None,
    use_heuristic_fallback: bool = True,
) -> Dict[str, Any]:
    """诊断一条 bug 描述。

    Returns 含 primary_pattern / secondary_patterns / reasoning / fix / latency_ms。
    """
    if not (bug_description or "").strip():
        raise ValueError("bug_description is empty")

    if complete_fn is None:
        out = heuristic_diagnose(bug_description)
        out["latency_ms"] = 0.0
        out["pattern"] = get_pattern(out["primary_pattern"]).to_dict() if get_pattern(out["primary_pattern"]) else None
        return out

    prompt = build_clinic_prompt(bug_description)
    t0 = time.perf_counter()
    try:
        raw = complete_fn(prompt)
        parsed = parse_diagnosis_json(raw)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        parsed["latency_ms"] = latency_ms
        parsed["mode"] = "llm"
        parsed["raw"] = (raw or "")[:500]
        pat = get_pattern(parsed["primary_pattern"])
        parsed["pattern"] = pat.to_dict() if pat else None
        return parsed
    except Exception as e:
        if not use_heuristic_fallback:
            raise
        out = heuristic_diagnose(bug_description)
        out["latency_ms"] = (time.perf_counter() - t0) * 1000.0
        out["mode"] = "heuristic_fallback"
        out["error"] = str(e)[:300]
        out["pattern"] = get_pattern(out["primary_pattern"]).to_dict() if get_pattern(out["primary_pattern"]) else None
        return out


def format_diagnosis_markdown(diagnosis: Dict[str, Any]) -> str:
    """可读 Markdown，便于写入 runs/*/badcase_analysis。"""
    primary = diagnosis.get("primary_pattern", "?")
    pat = get_pattern(str(primary))
    lines = [
        f"## Failure Clinic Diagnosis",
        "",
        f"- **Primary**: `{primary}` — {pat.name if pat else ''}",
        f"- **Secondary**: {', '.join(f'`{s}`' for s in diagnosis.get('secondary_patterns') or []) or '—'}",
        f"- **Mode**: {diagnosis.get('mode', '?')}",
        f"- **Confidence**: {diagnosis.get('confidence')}",
        "",
        "### Reasoning",
    ]
    for r in diagnosis.get("reasoning") or []:
        lines.append(f"- {r}")
    lines.extend(
        [
            "",
            "### Minimal structural fix",
            diagnosis.get("minimal_structural_fix") or "—",
            "",
        ]
    )
    if pat:
        lines.extend(
            [
                "### Catalog hints (PrismRAG)",
                f"- Symptoms: {pat.typical_symptoms}",
                f"- Where to look: {pat.prismrag_hints}",
                "",
            ]
        )
    return "\n".join(lines)
