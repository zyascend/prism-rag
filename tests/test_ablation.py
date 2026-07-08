"""消融实验模块测试（mock datasets）"""

from unittest.mock import MagicMock, patch

import torch

from src.evaluation.ablation import load_eval_data, run_ablation


def _make_mock_dataset(num_queries: int = 5):
    """创建 mock 的 ViDoRe 格式 dataset"""
    queries_ds = MagicMock()
    queries_ds.__len__.return_value = num_queries
    queries_ds.__getitem__ = MagicMock()
    queries_ds.select = MagicMock(return_value=queries_ds)
    queries_ds.filter = MagicMock(return_value=queries_ds)

    def getitem_side_effect(idx):
        return {
            "query_id": idx,
            "query": f"test query {idx}",
            "language": "english",
        }
    queries_ds.__getitem__.side_effect = getitem_side_effect

    qrels_ds = MagicMock()
    qrels_ds.__iter__.return_value = [
        {"query_id": 0, "corpus_id": 101},
        {"query_id": 0, "corpus_id": 102},
        {"query_id": 1, "corpus_id": 103},
    ]

    return queries_ds, qrels_ds


@patch("src.evaluation.ablation.hf_load_dataset")
def test_load_eval_data_filters_language(mock_load_dataset):
    """load_eval_data(language='en') 应调用 filter 并按 query_lang 过滤"""
    queries_ds, qrels_ds = _make_mock_dataset()
    mock_load_dataset.side_effect = lambda path, name, *a, **kw: {
        "queries": queries_ds, "qrels": qrels_ds
    }[name]

    queries_out, qrel_map = load_eval_data(
        dataset_path="vidore/vidore_v3_industrial",
        max_queries=None,
        language="en",
    )

    queries_ds.filter.assert_called_once()
    assert isinstance(qrel_map, dict)
    if qrel_map:
        assert isinstance(next(iter(qrel_map.values())), set)


@patch("src.evaluation.ablation.hf_load_dataset")
def test_load_eval_data_applies_max_queries(mock_load_dataset):
    """load_eval_data 应用 max_queries 限制"""
    queries_ds, qrels_ds = _make_mock_dataset(num_queries=10)
    mock_load_dataset.side_effect = lambda path, name, *a, **kw: {
        "queries": queries_ds, "qrels": qrels_ds
    }[name]

    queries_out, qrel_map = load_eval_data(
        dataset_path="vidore/vidore_v3_industrial",
        max_queries=3,
        language="all",
    )

    queries_ds.select.assert_called_once_with(range(3))


@patch("src.evaluation.ablation.hf_load_dataset")
def test_run_ablation_passes_pre_encoded_visual(mock_load_dataset):
    """run_ablation() 在 visual 配置下会把 pre_encoded_visual 透传给 retriever.search"""
    queries_ds, qrels_ds = _make_mock_dataset(num_queries=2)
    mock_load_dataset.side_effect = lambda path, split, *a, **kw: {
        queries_ds if split == "queries" else qrels_ds
    }.get(split, qrels_ds)

    mock_retriever = MagicMock()
    mock_retriever.search.return_value = []

    pre_encoded = {
        0: torch.randn(1, 10, 128),
        1: torch.randn(1, 10, 128),
    }

    run_ablation(
        retriever=mock_retriever,
        queries_ds=queries_ds,
        qrel_map={0: {101, 102}, 1: {103}},
        output_dir="/tmp/test_ablation_results",
        pre_encoded_visual=pre_encoded,
        language="en",
    )

    # 验证对 Visual_only 配置（use_visual=True）传入了 visual_query_embedding
    visual_config_calls = [
        call for call in mock_retriever.search.call_args_list
        if call.kwargs.get("use_visual") is True
    ]
    assert len(visual_config_calls) > 0
    for call in visual_config_calls:
        assert "visual_query_embedding" in call.kwargs


def test_expected_query_count_validation():
    """语言过滤后，可以验证 query 数量是否与预期一致"""
    queries_ds, qrels_ds = _make_mock_dataset(num_queries=3)
    with patch("src.evaluation.ablation.hf_load_dataset") as mock_load:
        mock_load.side_effect = lambda path, name, *a, **kw: {
            "queries": queries_ds, "qrels": qrels_ds
        }[name]

        queries_out, qrel_map = load_eval_data(
            dataset_path="vidore/vidore_v3_industrial",
            max_queries=None,
            language="en",
        )

        queries_ds.select.assert_not_called()
        queries_ds.filter.assert_called_once()
