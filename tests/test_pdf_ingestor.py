# tests/test_pdf_ingestor.py
from pathlib import Path
from src.ingestion.pdf_ingestor import PDFIngestor

FIXTURE = Path(__file__).parent / "fixtures" / "sample.pdf"


class _FakeChunker:
    def chunk_page(self, page_id, doc_id, page_number, markdown_text):
        text = (markdown_text or "").strip()
        if not text:
            return []
        return [type("C", (), {"chunk_id": f"pg{page_id:05d}_ch001",
                               "page_id": page_id, "doc_id": doc_id,
                               "page_number": page_number, "text": text,
                               "chunk_type": "text", "doc_ref": ""})()]


class _FakeBGE:
    def encode(self, texts, **_):
        import torch
        return torch.zeros((len(texts), 1024))


class _FakeColPali:
    def encode_pages(self, images, **_):
        import torch
        return [torch.rand(10, 128) for _ in images]


class _FakePG:
    def __init__(self):
        self.rows = []
        self.docs = {}
        self.page_hashes = {}  # (doc_id, page_number) -> page_hash
    def create_schema(self): pass
    def insert_chunks(self, chunks):
        for r in chunks:
            self.rows.append(r)
            self.page_hashes[(r[2], r[3])] = r[9]
    def count(self): return len(self.rows)
    def get_doc_id_by_content_hash(self, content_hash): return None
    def document_exists(self, doc_id): return doc_id in self.docs
    def upsert_document(self, doc_id, content_hash, source_path=""): self.docs[doc_id] = content_hash
    def update_document(self, doc_id, content_hash, source_path=""): self.docs[doc_id] = content_hash
    def get_pages_by_doc_id(self, doc_id):
        pns = sorted({r[3] for r in self.rows if r[2] == doc_id})
        return [(r[1], r[3]) for r in self.rows if r[2] == doc_id and r[3] in pns][:len(pns)] if False else [(self._pid_for(doc_id, pn), pn) for pn in pns]
    def _pid_for(self, doc_id, pn):
        for r in self.rows:
            if r[2] == doc_id and r[3] == pn:
                return r[1]
        return None
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


class _FakeFAISS:
    def __init__(self):
        self.added = 0
        self.deleted = []
        self.compacted = False
        self.saved = 0
    def add_pages(self, embs, **_):
        self.added += len(embs)
    def delete_by_page_ids(self, page_ids):
        self.deleted.extend(page_ids)
        return len(page_ids)
    def maybe_compact(self, threshold=0.2):
        self.compacted = True
        return False
    def save(self):
        self.saved += 1


def test_ingest_builds_chunks_and_index():
    ing = PDFIngestor(_FakePG(), _FakeFAISS(), _FakeBGE(), _FakeColPali(), _FakeChunker())
    res = ing.ingest(FIXTURE, doc_id="docX")
    assert res["doc_id"] == "docX"
    assert res["num_pages"] == 1
    assert res["num_chunks"] >= 1
