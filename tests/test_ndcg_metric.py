"""锁定 NDCG 标准 log2 折扣与 page 去重（eval protocol v1）。"""

from src.evaluation.ablation import (
    GOLDEN_NO_HYDE_NAMES,
    ABLATION_CONFIGS,
    compute_ndcg,
    compute_mrr,
    compute_recall,
)


def test_ndcg_log2_first_rank_perfect():
    assert abs(compute_ndcg({"p1"}, ["p1", "p2"], k=10) - 1.0) < 1e-9


def test_ndcg_second_rank():
    # 相关在 rank1（0-based pos=1）→ gain 1/log2(3)
    expected = 1.0 / __import__("math").log2(3)
    idcg = 1.0  # 一个相关文档的 ideal
    score = compute_ndcg({"p1"}, ["p0", "p1"], k=10)
    assert abs(score - expected / idcg) < 1e-9


def test_ndcg_dedupes_repeated_pages():
    s1 = compute_ndcg({"p1"}, ["p1", "p1", "p2"], k=10)
    s2 = compute_ndcg({"p1"}, ["p1", "p2"], k=10)
    assert abs(s1 - s2) < 1e-9


def test_recall_dedupes():
    assert compute_recall({"a", "b"}, ["a", "a", "x"], k=5) == 0.5


def test_mrr_first_hit():
    assert compute_mrr({"t"}, ["x", "t"]) == 0.5


def test_golden_no_hyde_excludes_hyde_configs():
    names = {c.name for c in ABLATION_CONFIGS}
    assert "Full_zerank2_HyDE" in names
    assert "Full_BGE_HyDE" in names
    assert "Full_zerank2_HyDE" not in GOLDEN_NO_HYDE_NAMES
    assert "Full_BGE_HyDE" not in GOLDEN_NO_HYDE_NAMES
    assert "Full_zerank2" in GOLDEN_NO_HYDE_NAMES
    assert "Full_no_rerank" in GOLDEN_NO_HYDE_NAMES
    assert len(GOLDEN_NO_HYDE_NAMES) == 8
