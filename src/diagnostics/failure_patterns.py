"""RAG / LLM 故障模式目录（P01–P12），对齐 awesome-llm-apps Failure Clinic，
并补充 PrismRAG 工业 PDF 场景说明。

用于 badcase 标注、离线 triage、事后分析模板；**不是**在线拦截逻辑。
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Optional


@dataclass(frozen=True)
class FailurePattern:
    id: str
    name: str
    summary: str
    typical_symptoms: str
    prismrag_hints: str
    minimal_fix_hints: str

    def to_dict(self) -> dict:
        return asdict(self)


FAILURE_PATTERNS: List[FailurePattern] = [
    FailurePattern(
        id="P01",
        name="Retrieval hallucination / grounding drift",
        summary="Answer confidently contradicts or ignores retrieved evidence.",
        typical_symptoms="Faith low; citations present but claims not in chunks; Gate2 would fail.",
        prismrag_hints="Check Gate2 logs, Faithfulness badcases, rejection false-negatives.",
        minimal_fix_hints="Tighten answer prompt; enable Gate2; add citation verbatim check.",
    ),
    FailurePattern(
        id="P02",
        name="Chunk boundary or segmentation bug",
        summary="Relevant facts split/truncated or mis-grouped across chunks.",
        typical_symptoms="Partial numbers; broken tables; answer misses half a procedure.",
        prismrag_hints="TextChunker / table split; large-table summary path; MinerU md quality.",
        minimal_fix_hints="Adjust chunk size/overlap; keep table header; protect table chunks from sentence compress.",
    ),
    FailurePattern(
        id="P03",
        name="Embedding mismatch / semantic vs vector distance",
        summary="Vector similarity does not match true relevance.",
        typical_symptoms="Dense/Visual ranks wrong pages; BM25 alone much better; Visual_only weak.",
        prismrag_hints="BGE space drift; ColPali query/page encoding mismatch; MaxSim scoring bugs.",
        minimal_fix_hints="Verify encode path parity; rebuild index with same model versions; recheck fusion weights.",
    ),
    FailurePattern(
        id="P04",
        name="Index skew or staleness",
        summary="Index returns old/missing data vs source of truth.",
        typical_symptoms="Deleted doc still recalled; update not visible; ghost pages in FAISS.",
        prismrag_hints="delete_document tombstone/compact; BM25 remove; cache index_version salt.",
        minimal_fix_hints="Invalidate cache; verify three-way delete; re-ingest page-diff path.",
    ),
    FailurePattern(
        id="P05",
        name="Query rewriting or router misalignment",
        summary="Router/rewriter sends query to wrong tool or path.",
        typical_symptoms="HyDE hurts; Visual always/never wrong; CRAG rewrite drifts.",
        prismrag_hints="VisualRouter mode; HyDE ablations; CRAG reformulate traces.",
        minimal_fix_hints="Disable bad rewrite; log route decision; constrain reformulate prompt.",
    ),
    FailurePattern(
        id="P06",
        name="Long-chain reasoning drift",
        summary="Multi-step tasks lose earlier constraints or goals.",
        typical_symptoms="Multi-hop questions fail; Self-RAG regen changes topic.",
        prismrag_hints="E2E multi-step numerical errors; Gate2 regenerate prompt too loose.",
        minimal_fix_hints="Stronger regenerate constraints; claim-level check; split multi-hop plan.",
    ),
    FailurePattern(
        id="P07",
        name="Tool-call misuse or ungrounded tools",
        summary="Tools called with wrong args or without grounding.",
        typical_symptoms="Wrong API params; tool output ignored; agent invents tool results.",
        prismrag_hints="Less common in pure RAG path; relevant if agent/tools added later.",
        minimal_fix_hints="Schema-validate tool args; require evidence before tool use.",
    ),
    FailurePattern(
        id="P08",
        name="Session memory leak / missing context",
        summary="Conversation loses important facts across turns/sessions.",
        typical_symptoms="Follow-ups forget doc_id; chat loses prior constraints.",
        prismrag_hints="Stateless /ask today; multi-turn demo must pass history explicitly.",
        minimal_fix_hints="Thread-bound doc filter; store session constraints; avoid silent drops.",
    ),
    FailurePattern(
        id="P09",
        name="Evaluation blind spots",
        summary="System passes tests but fails on real incidents.",
        typical_symptoms="50q Faith high, 283q drops; offline ≠ online path.",
        prismrag_hints="Sample size bias; eval_via_generator off; rejection polluting Faith mean.",
        minimal_fix_hints="Use full protocol; exclude rejections from Faith/Rel; align eval with /ask.",
    ),
    FailurePattern(
        id="P10",
        name="Startup ordering / dependency not ready",
        summary="Services 5xx until dependencies warm up.",
        typical_symptoms="Ollama/pg/FAISS not ready; first requests timeout then recover.",
        prismrag_hints="cloud_boot scripts; ollama serve; postgresql start order.",
        minimal_fix_hints="Healthchecks + retries; readiness probe; warm encode once.",
    ),
    FailurePattern(
        id="P11",
        name="Config or secrets drift across environments",
        summary="Works locally, breaks only in staging/prod settings.",
        typical_symptoms="Wrong HF offline flags; missing API keys; parser=simple in prod.",
        prismrag_hints="CONFIG_PROFILE; models.local-dev vs models.yaml; cloud_env.sh.",
        minimal_fix_hints="Config checklist; fail-fast on missing secrets; env snapshot in runs/.",
    ),
    FailurePattern(
        id="P12",
        name="Multi-tenant or multi-agent interference",
        summary="Requests/agents overwrite each other's state or resources.",
        typical_symptoms="Shared FAISS/BM25 mutation races; GPU OOM from concurrent jobs.",
        prismrag_hints="Single-process demo; cloud GPU contention with ColPali + LLM.",
        minimal_fix_hints="Serialize GPU jobs; isolate indexes per tenant; queue heavy encode.",
    ),
]

_BY_ID: Dict[str, FailurePattern] = {p.id: p for p in FAILURE_PATTERNS}


def get_pattern(pattern_id: str) -> Optional[FailurePattern]:
    return _BY_ID.get(str(pattern_id).strip().upper())


def list_patterns() -> List[dict]:
    return [p.to_dict() for p in FAILURE_PATTERNS]


def patterns_prompt_block() -> str:
    """给 judge 用的紧凑模式列表。"""
    lines = []
    for p in FAILURE_PATTERNS:
        lines.append(f"{p.id}: {p.name} — {p.summary}")
    return "\n".join(lines)
