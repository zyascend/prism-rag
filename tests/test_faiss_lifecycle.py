"""P1: FAISS 逻辑删除（墓碑）+ 压缩 + 持久化一致性测试

纯 CPU / flat 索引，不依赖 GPU 或 PostgreSQL，可在本地直接跑。
"""
import torch

from src.store.faiss_store import FaissColPaliStore


def _make_embeddings():
    """3 个页面，各 1 个 patch，分别与标准基向量对齐，便于断言 MaxSim 排名。

    注意：ColPali 页面嵌入是 [n_patches, 128]（2D），与查询 [1, n_q, 128]（3D）不同。
    """
    e1 = torch.tensor([[1.0] + [0.0] * 127], dtype=torch.float32)  # page 1
    e2 = torch.tensor([[0.0, 1.0] + [0.0] * 126], dtype=torch.float32)  # page 2
    e3 = torch.tensor([[0.0, 0.0, 1.0] + [0.0] * 125], dtype=torch.float32)  # page 3
    return {1: e1, 2: e2, 3: e3}


def _query_for_page(page_idx: int) -> torch.Tensor:
    """构造与 page_idx 对齐的查询（query patch = 标准基向量）。

    注意：页面嵌入中 page pid 的 1.0 位于 index pid-1，故查询对齐到 page_idx-1。
    """
    vec = [0.0] * 128
    vec[page_idx - 1] = 1.0
    return torch.tensor([[vec]], dtype=torch.float32)  # [1, 1, 128]


def _build_store(tmp_path, page_doc_ids):
    store = FaissColPaliStore(
        index_path=str(tmp_path / "idx.faiss"),
        id_map_path=str(tmp_path / "ids.npy"),
    )
    store.build_index(_make_embeddings(), index_type="flat", page_doc_ids=page_doc_ids)
    return store


def test_build_records_page_doc_mapping(tmp_path):
    store = _build_store(tmp_path, {1: "docA", 2: "docA", 3: "docB"})
    assert store._page_doc_ids == {1: "docA", 2: "docA", 3: "docB"}


def test_search_ranks_matching_page_first(tmp_path):
    store = _build_store(tmp_path, {1: "docA", 2: "docA", 3: "docB"})
    res = store.maxsim_search(_query_for_page(2), k=3)
    assert res[0]["page_id"] == 2


def test_delete_by_page_ids_tombstones_and_excludes(tmp_path):
    store = _build_store(tmp_path, {1: "docA", 2: "docA", 3: "docB"})
    removed = store.delete_by_page_ids([2])
    assert removed == 1
    # 墓碑后该页不应再出现在任何查询的 top 结果
    res = store.maxsim_search(_query_for_page(2), k=3)
    assert all(r["page_id"] != 2 for r in res)
    # 其他页不受影响
    res1 = store.maxsim_search(_query_for_page(1), k=3)
    assert res1[0]["page_id"] == 1


def test_delete_by_doc_id_removes_all_pages_of_doc(tmp_path):
    store = _build_store(tmp_path, {1: "docA", 2: "docA", 3: "docB"})
    removed = store.delete_by_doc_id("docA")
    assert removed == 2
    res = store.maxsim_search(_query_for_page(1), k=3)
    assert all(r["page_id"] != 1 for r in res)
    res2 = store.maxsim_search(_query_for_page(2), k=3)
    assert all(r["page_id"] != 2 for r in res2)
    # docB 的 page 3 仍可用
    res3 = store.maxsim_search(_query_for_page(3), k=3)
    assert res3[0]["page_id"] == 3


def test_compact_reclaims_tombstones_and_remaps(tmp_path):
    store = _build_store(tmp_path, {1: "docA", 2: "docA", 3: "docB"})
    store.delete_by_doc_id("docA")  # 墓碑 2/3 页
    compacted = store.maybe_compact(threshold=0.2)  # 比例 2/3 > 0.2
    assert compacted is True
    assert store._deleted_page_ids == set()
    assert store.num_pages == 1
    assert store._page_doc_ids == {3: "docB"}
    # 压缩后仍可正常检索，且不含被删页
    res = store.maxsim_search(_query_for_page(3), k=3)
    assert res[0]["page_id"] == 3


def test_save_load_roundtrip_preserves_tombstone_and_mapping(tmp_path):
    store = _build_store(tmp_path, {1: "docA", 2: "docA", 3: "docB"})
    store.delete_by_page_ids([2])
    store.save()

    # 新实例加载
    store2 = FaissColPaliStore(
        index_path=str(tmp_path / "idx.faiss"),
        id_map_path=str(tmp_path / "ids.npy"),
    )
    assert store2.load() is True
    # 墓碑与映射持久化正确
    assert 2 in store2._deleted_page_ids
    assert store2._page_doc_ids == {1: "docA", 2: "docA", 3: "docB"}
    # 加载后删除一致性仍然生效
    res = store2.maxsim_search(_query_for_page(2), k=3)
    assert all(r["page_id"] != 2 for r in res)


def test_delete_by_doc_id_missing_mapping_returns_zero(tmp_path):
    """旧索引（_page_doc_ids 为空）删除时安全返回 0，不崩溃"""
    store = FaissColPaliStore(
        index_path=str(tmp_path / "idx.faiss"),
        id_map_path=str(tmp_path / "ids.npy"),
    )
    store.build_index(_make_embeddings(), index_type="flat")  # 故意不传 page_doc_ids
    assert store.delete_by_doc_id("docX") == 0
