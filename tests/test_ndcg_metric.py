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


def test_recall_unique_k_not_list_prefix():
    """同页多 chunk 时 Recall@k 应按唯一 page 计，而非 list 前 k 条。"""
    # list 前 2 条都是 a → 旧实现只看到 {a}；新实现唯一页 a,b
    assert compute_recall({"a", "b"}, ["a", "a", "b"], k=2) == 1.0


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
    assert "Visual_only_pages" in GOLDEN_NO_HYDE_NAMES
    assert len(GOLDEN_NO_HYDE_NAMES) == 9
    page_cfg = next(c for c in ABLATION_CONFIGS if c.name == "Visual_only_pages")
    assert page_cfg.visual_page_level is True
