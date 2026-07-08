# tests/test_faiss_add_pages.py
import torch
from src.store.faiss_store import FaissColPaliStore


def _emb(n_patches=10):
    return torch.rand(n_patches, 128)


def test_add_pages_incremental():
    store = FaissColPaliStore(index_path="indexes/_test.faiss",
                              id_map_path="indexes/_test-ids.npy")
    store.add_pages({1: _emb(10), 2: _emb(12)})
    assert store.num_pages == 2
    assert store._vectors.shape[0] == 22
    store.add_pages({3: _emb(8)})
    assert store.num_pages == 3
    assert store._vectors.shape[0] == 30
    # MaxSim 仍可跑
    q = torch.rand(1, 5, 128)
    res = store.maxsim_search(q, k=3)
    assert len(res) == 3
