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
    def __init__(self): self.rows = []
    def create_schema(self): pass
    def insert_chunks(self, chunks): self.rows.extend(chunks)
    def count(self): return len(self.rows)


class _FakeFAISS:
    def __init__(self): self.added = 0
    def add_pages(self, embs): self.added += len(embs)
    def save(self): pass


def test_ingest_builds_chunks_and_index():
    ing = PDFIngestor(_FakePG(), _FakeFAISS(), _FakeBGE(), _FakeColPali(), _FakeChunker())
    res = ing.ingest(FIXTURE, doc_id="docX")
    assert res["doc_id"] == "docX"
    assert res["num_pages"] == 1
    assert res["num_chunks"] >= 1
