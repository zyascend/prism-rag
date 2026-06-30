"""BM25 检索器测试"""

from src.retrieval.bm25_retriever import BM25Retriever


def test_bm25_tokenize():
    tokens = BM25Retriever._tokenize("Load Capacity: 500 kg")
    assert "load" in tokens
    assert "capacity" in tokens
    assert "500" in tokens
    assert "kg" in tokens


def test_bm25_simple_search():
    chunks = [
        {"chunk_id": "ch1", "page_id": 1, "doc_id": "doc1", "page_number": 1, "chunk_type": "text", "text": "The conveyor belt has a load capacity of 500 kg."},
        {"chunk_id": "ch2", "page_id": 1, "doc_id": "doc1", "page_number": 1, "chunk_type": "text", "text": "Motor speed is 1500 RPM."},
        {"chunk_id": "ch3", "page_id": 2, "doc_id": "doc1", "page_number": 2, "chunk_type": "text", "text": "Safety guidelines for operation."},
    ]
    retriever = BM25Retriever()
    retriever.fit(chunks)

    results = retriever.search("conveyor load capacity", k=2)
    assert len(results) == 1
    assert results[0]["chunk_id"] == "ch1"


def test_bm25_no_match():
    chunks = [
        {"chunk_id": "ch1", "page_id": 1, "doc_id": "doc1", "page_number": 1, "chunk_type": "text", "text": "Safety guidelines."},
    ]
    retriever = BM25Retriever()
    retriever.fit(chunks)
    results = retriever.search("unrelated query about nothing", k=5)
    assert len(results) == 0


def test_bm25_retrieval_type_tag():
    chunks = [
        {"chunk_id": "ch1", "page_id": 1, "doc_id": "doc1", "page_number": 1, "chunk_type": "text", "text": "Load capacity is 500 kg."},
        {"chunk_id": "ch2", "page_id": 2, "doc_id": "doc1", "page_number": 2, "chunk_type": "text", "text": "Safety guidelines for operation."},
        {"chunk_id": "ch3", "page_id": 3, "doc_id": "doc1", "page_number": 3, "chunk_type": "text", "text": "Motor speed is 1500 RPM."},
    ]
    retriever = BM25Retriever()
    retriever.fit(chunks)
    results = retriever.search("load capacity", k=5)
    assert len(results) >= 1
    assert results[0]["retrieval_type"] == "bm25"