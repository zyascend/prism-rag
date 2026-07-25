"""P0-B Refiner 单测（mock embedder，无真模型）。"""
from __future__ import annotations

import torch

from src.generation.refiner import (
    refine_context,
    score_sentences,
    soft_rank_sentences,
    refiner_cache_salt,
)
from src.evaluation.ragas_metrics import compress_context


class FakeBGE:
    """可预测 embedder：句中含 query 关键词 → 高 sim。"""

    def __init__(self, query_token: str = "torque"):
        self.query_token = query_token

    def encode(self, texts):
        vecs = []
        for t in texts:
            # 1-d unit vector on axis 0 if keyword match else axis 1
            v = torch.zeros(4)
            if self.query_token.lower() in (t or "").lower():
                v[0] = 1.0
            else:
                v[1] = 1.0
            vecs.append(v)
        return torch.stack(vecs)


def test_score_sentences_prefers_keyword():
    emb = FakeBGE("torque")
    sents = [
        "The max torque is fifty Nm for this unit.",
        "The company was founded in nineteen eighty by engineers.",
    ]
    scores = score_sentences("what is max torque?", sents, emb)
    assert scores[0] > scores[1]


def test_soft_rank_keeps_relevant_drops_low_when_pruned():
    emb = FakeBGE("torque")
    sents = [
        "The max torque is fifty Nm for this unit exactly here.",
        "Company history and founders story about the plant only.",
        "Torque limits also appear in appendix notes for operators.",
        "Unrelated weather forecast for the region next week.",
    ]
    kept, tr = soft_rank_sentences(
        "max torque?",
        sents,
        emb,
        min_sim=0.0,
        prune_below=0.5,  # FakeBGE: match ~1.0, non-match ~0.0
        keep_ratio=1.0,
    )
    assert any("torque" in s.lower() for s in kept)
    assert not any("weather" in s.lower() for s in kept)
    assert tr["pruned"] >= 1


def test_soft_rank_without_prune_may_keep_low_weight_via_ratio():
    emb = FakeBGE("torque")
    sents = [
        "The max torque is fifty Nm for this unit exactly here.",
        "Company history and founders story about the plant only.",
        "Another torque related note for installation staff here.",
        "Unrelated weather forecast for the region next week now.",
    ]
    kept, tr = soft_rank_sentences(
        "max torque?",
        sents,
        emb,
        min_sim=0.0,
        prune_below=None,
        keep_ratio=0.5,
    )
    assert tr["num_sentences_out"] <= tr["num_sentences_in"]
    assert any("torque" in s.lower() for s in kept)


def test_refine_protects_table_chunks():
    emb = FakeBGE("pressure")
    hits = [
        {
            "chunk_id": "t1",
            "chunk_type": "table",
            "text": "| Param | Val |\n| --- | --- |\n| Pressure | 100 psi |",
        },
        {
            "chunk_id": "x1",
            "chunk_type": "text",
            "text": "Noise about company history and random founders only here. Pressure rating is one hundred psi in the manual.",
        },
    ]
    out = refine_context("pressure rating?", hits, emb, mode="soft_rank", ratio=0.5)
    assert "100 psi" in out.context or "Pressure" in out.context
    assert out.trace.get("table_chunks") == 1


def test_refine_bge_matches_compress_context_on_plain_text():
    emb = FakeBGE("torque")
    chunks = [
        "The max torque is fifty newton meters for this pump unit.",
        "The company was founded in nineteen eighty by senior engineers.",
        "Ambient temperature limits are listed in section four of the book.",
        "Torque calibration procedures appear again in the appendix notes.",
        "Unrelated shipping delays affect warehouse logistics next month.",
        "More filler text about cafeteria hours and parking lot access rules.",
    ]
    legacy = compress_context("max torque?", chunks, emb, ratio=0.4)
    hits = [{"text": c, "chunk_type": "text"} for c in chunks]
    refined = refine_context("max torque?", hits, emb, mode="bge", ratio=0.4)
    # 两者都保序相关句；允许空白差异但核心重叠
    assert "torque" in refined.context.lower()
    # legacy 与 refined 对同一 encode 逻辑应高度一致
    assert set(legacy.split()) & set(refined.context.split())


def test_refine_off_joins():
    hits = [
        {"text": "Sentence one has enough words here.", "chunk_type": "text"},
        {"text": "Sentence two has enough words too.", "chunk_type": "text"},
    ]
    out = refine_context("q", hits, None, mode="off")
    assert "Sentence one" in out.context and "Sentence two" in out.context


def test_refiner_cache_salt_changes_with_mode(monkeypatch):
    from src.generation import refiner as R

    def fake_cfg_bge(path, default=None):
        if path == "refiner":
            return {"enabled": True, "mode": "bge", "soft_rank": {}}
        if path == "context_filter.mode":
            return "bge"
        return default

    def fake_cfg_soft(path, default=None):
        if path == "refiner":
            return {
                "enabled": True,
                "mode": "soft_rank",
                "soft_rank": {"keep_ratio": 0.5, "min_sim": 0.25, "prune_below": None},
            }
        return default

    monkeypatch.setattr(R.cfg, "get", fake_cfg_bge)
    s1 = refiner_cache_salt(R.refiner_config())
    monkeypatch.setattr(R.cfg, "get", fake_cfg_soft)
    s2 = refiner_cache_salt(R.refiner_config())
    assert s1 != s2
