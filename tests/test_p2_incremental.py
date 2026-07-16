"""P2 测试：BM25 增量（§4.3）/ page diff（§4.2）/ 原子快照（§4.6）

全本地、无 PG / 模型 / 网络依赖。
"""
from pathlib import Path

import torch
from PIL import Image
from rank_bm25 import BM25Okapi

from src.ingestion.pdf_ingestor import PDFIngestor
from src.retrieval.bm25_retriever import BM25Retriever
from src.store.faiss_store import FaissColPaliStore
from src.store.snapshot import atomic_replace, build_chunk_swap_sql


# ── 通用 fixture ────────────────────────────────────────────

def _distractors(n=6, base_id="z"):
    """构造与查询词无关的中性文档，使查询词 idf 为正（避免小语料 idf<=0 被 score>0 过滤）。

    生产环境上千 chunk 时 idf 恒正，此仅为直测小语料兜底（与 P0 测试策略一致）。
    """
    words = ["zebra", "qux", "lambda", "omega", "theta", "sigma", "kappa", "rho",
             "nu", "xi", "psi", "tau"]
    out = []
    for i in range(n):
        w = words[i % len(words)]
        out.append({
            "chunk_id": f"{base_id}{i}", "page_id": 1000 + i, "doc_id": "d",
            "page_number": 1000 + i, "chunk_type": "text",
            "text": f"{w} {w} neutral filler {i}",
        })
    return out


def _emb(page_idx, n=1):
    vec = [0.0] * 128
    vec[page_idx - 1] = 1.0
    return torch.tensor([vec] * n, dtype=torch.float32)  # [n, 128]


def _make_embeddings():
    return {1: _emb(1), 2: _emb(2), 3: _emb(3)}


def _img(tag):
    """用 tag 派生确定性图像（不同尺寸 → 不同 tobytes → 不同 page_hash）"""
    n = (abs(hash(tag)) % 40) + 8
    return Image.new("RGB", (n, n), (120, 80, 200))


class _Page:
    def __init__(self, page_number, markdown, image):
        self.page_number = page_number
        self.markdown = markdown
        self.image = image


class _FakeParser:
    def __init__(self, pages):
        self._pages = pages
    def parse(self, pdf_path):
        return self._pages


class _FakeChunker:
    def chunk_page(self, page_id, doc_id, page_number, markdown_text):
        text = (markdown_text or "").strip()
        if not text:
            return []
        return [type("C", (), {
            "chunk_id": f"pg{page_id:05d}_ch001", "page_id": page_id,
            "doc_id": doc_id, "page_number": page_number, "text": text,
            "chunk_type": "text", "doc_ref": "",
        })()]


class _FakeBGE:
    def encode(self, texts, **_):
        return torch.zeros((len(texts), 1024))


class _FakeColPali:
    def __init__(self):
        self.calls = []          # 每次 encode_pages 的入参图像列表
        self.total_images = 0
    def encode_pages(self, images, **_):
        self.calls.append(list(images))
        self.total_images += len(images)
        return [torch.rand(10, 128) for _ in images]


class _FakeFAISS:
    def __init__(self):
        self.added_pages = 0
        self.deleted = []
        self.compacted = False
        self.saved = 0
    def add_pages(self, embs, **_):
        self.added_pages += len(embs)
    def delete_by_page_ids(self, page_ids):
        self.deleted.extend(page_ids)
        return len(page_ids)
    def maybe_compact(self, threshold=0.2):
        self.compacted = True
        return False
    def save(self):
        self.saved += 1


class _FakePG:
    def __init__(self):
        self.rows = []
        self.docs = {}
        self._hash_to_doc = {}
        self.page_hashes = {}
    def create_schema(self):
        pass
    def insert_chunks(self, chunks):
        for r in chunks:
            self.rows.append(r)
            self.page_hashes[(r[2], r[3])] = r[9]
    def count(self):
        return len(self.rows)
    def get_doc_id_by_content_hash(self, content_hash):
        return self._hash_to_doc.get(content_hash)
    def document_exists(self, doc_id):
        return doc_id in self.docs
    def upsert_document(self, doc_id, content_hash, source_path=""):
        self.docs[doc_id] = content_hash
        self._hash_to_doc[content_hash] = doc_id
    def update_document(self, doc_id, content_hash, source_path=""):
        self.docs[doc_id] = content_hash
        self._hash_to_doc[content_hash] = doc_id
    def get_pages_by_doc_id(self, doc_id):
        seen = {}
        for r in self.rows:
            if r[2] == doc_id:
                seen[r[3]] = r[1]
        return [(pid, pn) for pn, pid in sorted(seen.items())]
    def get_page_hashes_by_doc_id(self, doc_id):
        out = {}
        for (d, pn), h in self.page_hashes.items():
            if d == doc_id:
                out.setdefault(pn, h)
        return out
    def get_chunk_ids_by_page_ids(self, page_ids):
        return [r[0] for r in self.rows if r[1] in page_ids]
    def delete_chunks_by_page_ids(self, page_ids):
        before = len(self.rows)
        self.rows = [r for r in self.rows if r[1] not in page_ids]
        return before - len(self.rows)


# ── P2-A: BM25 增量 ────────────────────────────────────────

def test_bm25_scores_match_rank_bm25(tmp_path):
    """手动 Okapi BM25 打分须与 rank_bm25.BM25Okapi 数值一致"""
    chunks = [
        {"chunk_id": "c0", "page_id": 1, "doc_id": "d", "page_number": 1, "chunk_type": "text", "text": "load capacity is 500 kg"},
        {"chunk_id": "c1", "page_id": 2, "doc_id": "d", "page_number": 2, "chunk_type": "text", "text": "motor speed is 1500 rpm"},
        {"chunk_id": "c2", "page_id": 3, "doc_id": "d", "page_number": 3, "chunk_type": "text", "text": "safety guidelines for operation"},
    ]
    bm = BM25Retriever(index_path=str(tmp_path / "bm25.pkl"))
    bm.fit(chunks)
    corpus = [bm._tokenize(c["text"]) for c in chunks]
    ref = BM25Okapi(corpus)
    for q in ["motor speed", "load capacity safety", "operation guidelines"]:
        qt = bm._tokenize(q)
        ref_scores = ref.get_scores(qt)
        got = {r["chunk_id"]: r["score"] for r in bm.search(q, k=10)}
        for i, c in enumerate(chunks):
            s = got.get(c["chunk_id"], 0.0)
            assert abs(s - ref_scores[i]) < 1e-4, f"q={q} chunk={c['chunk_id']} {s} vs {ref_scores[i]}"


def test_bm25_fit_incremental_and_remove_od(tmp_path):
    """fit_incremental 追加、remove_chunks O(Δ) 删除，打分与全量 build 一致"""
    base = _distractors(6) + [
        {"chunk_id": "a0", "page_id": 1, "doc_id": "d", "page_number": 1, "chunk_type": "text", "text": "apple banana cherry"},
        {"chunk_id": "a1", "page_id": 2, "doc_id": "d", "page_number": 2, "chunk_type": "text", "text": "dog elephant fox"},
    ]
    bm = BM25Retriever(index_path=str(tmp_path / "inc.pkl"))
    bm.fit(base)
    # 增量追加
    added = bm.fit_incremental([
        {"chunk_id": "a2", "page_id": 3, "doc_id": "d", "page_number": 3, "chunk_type": "text", "text": "grape apple horse"},
    ])
    assert added == 1
    assert bm.search("apple", k=5)[0]["chunk_id"] in ("a0", "a2")
    # 删除一个旧 chunk（O(Δ)，非 O(N) 重建）
    removed = bm.remove_chunks({"a0"})
    assert removed == 1
    after = bm.search("apple", k=5)
    assert all(r["chunk_id"] != "a0" for r in after)
    assert any(r["chunk_id"] == "a2" for r in after)
    # 与一次性全量 build 的打分一致
    full = BM25Retriever(index_path=str(tmp_path / "full.pkl"))
    full.fit(_distractors(6) + [
        base[7],  # a1（原 base[1]，因前面插入了 6 个 distractor）
        {"chunk_id": "a2", "page_id": 3, "doc_id": "d", "page_number": 3, "chunk_type": "text", "text": "grape apple horse"},
    ])
    for q in ["apple", "dog", "grape"]:
        s1 = {r["chunk_id"]: r["score"] for r in bm.search(q, k=5)}
        s2 = {r["chunk_id"]: r["score"] for r in full.search(q, k=5)}
        assert set(s1) == set(s2)


def test_bm25_reconcile_from_pgvector(tmp_path):
    """启动对账：pg 多则增量 fit，pg 少则 remove（以 pg 真相源为准）"""
    class _FakeReconPg:
        def __init__(self, ids):
            self._ids = ids
        def get_all_chunk_ids(self):
            return list(self._ids)
        def get_chunks_by_ids(self, ids):
            return [{"chunk_id": i, "page_id": int(i[1:]), "doc_id": "d",
                     "page_number": int(i[1:]), "chunk_type": "text", "text": f"text {i}"}
                    for i in ids]

    bm = BM25Retriever(index_path=str(tmp_path / "recon.pkl"))
    bm.fit([{"chunk_id": "p1", "page_id": 1, "doc_id": "d", "page_number": 1, "chunk_type": "text", "text": "one"},
            {"chunk_id": "p2", "page_id": 2, "doc_id": "d", "page_number": 2, "chunk_type": "text", "text": "two"}])
    # pg 少了 p1、多了 p3
    bm.reconcile_from_pgvector(_FakeReconPg({"p2", "p3"}))
    assert set(bm._chunk_id_order) == {"p2", "p3"}


# ── P2-B: page diff（省 GPU）──────────────────────────────

def _make_ingestor(tmp_path, pg, faiss, colpali, parser):
    bm25 = BM25Retriever(index_path=str(tmp_path / "bm25.pkl"))
    ing = PDFIngestor(pg, faiss, _FakeBGE(), colpali, _FakeChunker(), parser=parser, bm25=bm25)
    return ing, bm25


def _pages_v1():
    return [
        _Page(1, "page one alpha", _img("p1-v1")),
        _Page(2, "page two beta", _img("p2-v1")),
        _Page(3, "page three gamma", _img("p3-v1")),
    ]


def _pages_v2_changed_page2():
    # 仅 page2 内容变化（page1/page3 不变）；v2 用 DELTA/updated 替换原 BETA，
    # 使旧内容（含 BETA）被真正移除、新内容（含 DELTA）可被检索。
    return [
        _Page(1, "page one alpha", _img("p1-v1")),
        _Page(2, "page two DELTA updated", _img("p2-v2")),
        _Page(3, "page three gamma", _img("p3-v1")),
    ]


def test_page_diff_only_reencodes_changed_page(tmp_path):
    pg = _FakePG()
    faiss = _FakeFAISS()
    colpali = _FakeColPali()
    parser = _FakeParser(_pages_v1())
    ing, bm25 = _make_ingestor(tmp_path, pg, faiss, colpali, parser)

    f1 = tmp_path / "v1.pdf"
    f1.write_bytes(b"version-one")
    r1 = ing.ingest(f1, doc_id="d1")
    assert r1["status"] == "inserted"
    assert colpali.total_images == 3          # 首次全量编码 3 页

    # 修改 page2 后同 doc_id 重入库
    parser._pages = _pages_v2_changed_page2()
    f2 = tmp_path / "v2.pdf"
    f2.write_bytes(b"version-two")
    r2 = ing.ingest(f2, doc_id="d1")
    assert r2["status"] == "updated"
    assert r2["unchanged"] == 2               # page1/page3 复用
    assert r2["changed"] == 1                 # page2 重编码
    assert r2["new"] == 0 and r2["deleted"] == 0
    # 关键：第二次仅编码 1 页（省 ColQwen2 GPU）
    assert colpali.total_images == 4
    assert len(colpali.calls[-1]) == 1
    # FAISS：旧 page2 墓碑 + 新 page2 加入
    assert len(faiss.deleted) == 1
    assert faiss.added_pages == 4             # 首次3 + 变更1
    # BM25：旧 page2 chunk（BETA）被移除，新 page2 chunk（DELTA）被加入
    assert bm25.search("DELTA updated", k=5)[0]["chunk_id"].startswith("pg")
    assert bm25.search("BETA", k=5) == []  # 旧 page2 文本已不可检索


def test_identical_content_is_noop(tmp_path):
    pg = _FakePG()
    faiss = _FakeFAISS()
    colpali = _FakeColPali()
    parser = _FakeParser(_pages_v1())
    ing, _ = _make_ingestor(tmp_path, pg, faiss, colpali, parser)
    f1 = tmp_path / "same.pdf"
    f1.write_bytes(b"identical-bytes")
    ing.ingest(f1, doc_id="d1")
    calls_before = colpali.total_images
    # 同内容再次入库 → 幂等 no-op（不重编码）
    r = ing.ingest(f1, doc_id="d1")
    assert r["status"] == "noop_identical"
    assert colpali.total_images == calls_before


# ── P2-C: 原子快照切换 ─────────────────────────────────────

def test_faiss_apply_snapshot(tmp_path):
    live_idx = str(tmp_path / "live.faiss")
    live_ids = str(tmp_path / "live-ids.npy")
    snap_idx = str(tmp_path / "snap.faiss")
    snap_ids = str(tmp_path / "snap-ids.npy")

    live = FaissColPaliStore(index_path=live_idx, id_map_path=live_ids)
    live.build_index({1: _emb(1), 2: _emb(2)}, index_type="flat", page_doc_ids={1: "a", 2: "b"})
    live.save()

    snap = FaissColPaliStore(index_path=snap_idx, id_map_path=snap_ids)
    snap.build_index({10: _emb(1), 20: _emb(2), 30: _emb(3)}, index_type="flat",
                     page_doc_ids={10: "x", 20: "y", 30: "z"})
    snap.save()

    live.apply_snapshot(snap_idx, snap_ids)

    live2 = FaissColPaliStore(index_path=live_idx, id_map_path=live_ids)
    assert live2.load() is True
    assert live2.num_pages == 3
    assert set(int(x) for x in live2._page_ids.tolist()) == {10, 20, 30}


def test_bm25_save_atomic_roundtrip(tmp_path):
    p = str(tmp_path / "bm25.pkl")
    bm = BM25Retriever(index_path=p)
    bm.fit(_distractors(4) + [
        {"chunk_id": "x1", "page_id": 1, "doc_id": "d", "page_number": 1, "chunk_type": "text", "text": "hello world"},
        {"chunk_id": "x2", "page_id": 2, "doc_id": "d", "page_number": 2, "chunk_type": "text", "text": "foo bar"},
    ])
    bm.save_atomic()
    assert Path(p).exists()
    bm2 = BM25Retriever(index_path=p)
    assert bm2.load() is True
    assert bm2.search("hello", k=5)[0]["chunk_id"] == "x1"


def test_build_chunk_swap_sql():
    sql = build_chunk_swap_sql()
    joined = " ".join(sql)
    assert "CREATE TABLE chunks_staging" in joined
    assert "chunks RENAME TO chunks_old" in joined
    assert "chunks_staging RENAME TO chunks" in joined
    assert "DROP TABLE chunks_old" in joined


def test_atomic_replace(tmp_path):
    src = tmp_path / "a.txt"
    dst = tmp_path / "b.txt"
    dst.write_text("old")
    src.write_text("new")
    atomic_replace(str(src), str(dst))
    assert dst.read_text() == "new"
    assert not src.exists()
