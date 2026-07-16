"""增量更新/删除生命周期测试（P0：修复 D2；P1：FAISS 墓碑已接入编排）

验证：
- BM25Retriever.remove_chunks：已删 chunk 不再被 search 返回
- PrismRAGRetriever.delete_document：先取 id → 删 pg → 清 bm25 → 清 FAISS，
  且删除后各路都不再返回已删内容
"""
import torch

from src.retrieval.bm25_retriever import BM25Retriever
from src.evaluation.vidore_adapter import PrismRAGRetriever
from src.store.faiss_store import FaissColPaliStore


def _build_faiss(tmp_path, page_doc_ids):
    """构造一个真实 FAISS 存储（flat，CPU），每页 1 个与基向量对齐的 patch"""
    store = FaissColPaliStore(
        index_path=str(tmp_path / "idx.faiss"),
        id_map_path=str(tmp_path / "ids.npy"),
    )
    embs = {
        pid: torch.tensor([[1.0 if i == (pid - 1) else 0.0 for i in range(128)]], dtype=torch.float32)
        for pid in page_doc_ids
    }
    store.build_index(embs, index_type="flat", page_doc_ids=dict(page_doc_ids))
    return store


class _FakePg:
    def __init__(self, chunks_by_doc):
        # chunks_by_doc: doc_id -> list of (chunk_id, page_id)
        self._data = chunks_by_doc
        self.deleted = []
        self.fetched_chunk_ids = []
        self.fetched_page_ids = []

    def get_chunk_ids_by_doc_id(self, doc_id):
        ids = [c[0] for c in self._data.get(doc_id, [])]
        self.fetched_chunk_ids = ids
        return ids

    def get_page_ids_by_doc_id(self, doc_id):
        ids = sorted({c[1] for c in self._data.get(doc_id, [])})
        self.fetched_page_ids = ids
        return ids

    def delete_by_doc_id(self, doc_id):
        n = len(self._data.get(doc_id, []))
        self.deleted.append(doc_id)
        self._data.pop(doc_id, None)
        return n


def _make_chunks(doc_id, texts):
    return [
        {
            "chunk_id": f"{doc_id}_c{i}",
            "page_id": i + 1,
            "doc_id": doc_id,
            "page_number": i + 1,
            "chunk_type": "text",
            "text": t,
        }
        for i, t in enumerate(texts)
    ]


def test_bm25_remove_chunks_excludes_deleted():
    # 主文档
    chunks = _make_chunks(
        "docA",
        [
            "Load capacity is 500 kg.",
            "Motor speed is 1500 RPM.",
            "Safety guidelines for operation.",
        ],
    )
    # 加入不相关 distractor，使 idf 为正（贴近生产大语料）。
    # rank_bm25 的 idf 在小语料下可为负，会导致 score>0 过滤误删匹配项；
    # 生产语料上千 chunk 时 idf 恒正，原 search 过滤正确。
    chunks += [
        {
            "chunk_id": f"dist_{i}",
            "page_id": 100 + i,
            "doc_id": "docX",
            "page_number": 100 + i,
            "chunk_type": "text",
            "text": t,
        }
        for i, t in enumerate(
            [
                "The weather is sunny today.",
                "Apple pie recipe with cinnamon.",
                "Quantum computing uses qubits.",
                "The train arrives at noon.",
            ]
        )
    ]
    bm25 = BM25Retriever()
    bm25.fit(chunks)

    before = bm25.search("load capacity", k=5)
    assert any(r["chunk_id"] == "docA_c0" for r in before)

    removed = bm25.remove_chunks({"docA_c0"})
    assert removed == 1

    after = bm25.search("load capacity", k=5)
    assert all(r["chunk_id"] != "docA_c0" for r in after)
    # 剩余 chunk 仍可被匹配它的查询检索到（未受影响）
    motor = bm25.search("motor speed", k=5)
    assert any(r["chunk_id"] == "docA_c1" for r in motor)


def test_bm25_remove_chunks_empty_is_noop():
    chunks = _make_chunks("docA", ["Load capacity is 500 kg."])
    bm25 = BM25Retriever()
    bm25.fit(chunks)
    assert bm25.remove_chunks(set()) == 0
    assert bm25.remove_chunks({"nonexistent"}) == 0


def _build_retriever(pg, faiss, bm25):
    return PrismRAGRetriever(
        pg_store=pg,
        faiss_store=faiss,
        bge=None,
        colpali=None,
        chunker=None,
        bm25=bm25,
        dense=None,
        visual=None,
        fusion=None,
        reranker=None,
    )


def test_delete_document_end_to_end_pg_bm25_faiss(tmp_path):
    # docA 占 page 1/2，docB 占 page 3
    pg = _FakePg({
        "docA": [("docA_c0", 1), ("docA_c1", 2)],
        "docB": [("docB_c0", 3)],
    })
    bm25 = BM25Retriever()
    bm25.fit(
        _make_chunks("docA", ["Load capacity is 500 kg.", "Motor speed is 1500 RPM."])
        + _make_chunks("docB", ["Safety guidelines for operation."])
    )
    faiss = _build_faiss(tmp_path, {1: "docA", 2: "docA", 3: "docB"})
    retr = _build_retriever(pg, faiss, bm25)

    out = retr.delete_document("docA")

    # pg 行已删，且删前已取 id（D4 顺序）
    assert out["pg_deleted_rows"] == 2
    assert pg.deleted == ["docA"]
    assert out["bm25_removed"] == 2
    # P1：FAISS 删除被自动调用（墓碑），返回该 doc 的 page 数
    assert out["faiss_removed"] == 2

    # 删除后 BM25 不再返回 docA 任何内容（修复 D2）
    assert bm25.search("load capacity", k=5) == []
    # FAISS MaxSim 不再返回 docA 的 page（修复 D1）
    q = torch.tensor([[[1.0] + [0.0] * 127]], dtype=torch.float32)
    vis_res = faiss.maxsim_search(q, k=3)
    assert all(r["page_id"] != 1 for r in vis_res)
    # docB 的 page 3 仍可用
    q3 = torch.tensor([[[0.0, 0.0, 1.0] + [0.0] * 125]], dtype=torch.float32)
    vis_res3 = faiss.maxsim_search(q3, k=3)
    assert vis_res3[0]["page_id"] == 3


def test_delete_document_faiss_compacts_when_threshold_exceeded(tmp_path):
    pg = _FakePg({"docA": [("docA_c0", 1), ("docA_c1", 2)], "docB": [("docB_c0", 3)]})
    bm25 = BM25Retriever()
    bm25.fit(
        _make_chunks("docA", ["Load capacity is 500 kg.", "Motor speed is 1500 RPM."])
        + _make_chunks("docB", ["Safety guidelines for operation."])
    )
    faiss = _build_faiss(tmp_path, {1: "docA", 2: "docA", 3: "docB"})
    retr = _build_retriever(pg, faiss, bm25)

    out = retr.delete_document("docA")  # 墓碑 2/3 页，比例 > 0.2 → 触发 compact
    assert out["faiss_compacted"] is True
    assert faiss._deleted_page_ids == set()
    assert faiss.num_pages == 1
