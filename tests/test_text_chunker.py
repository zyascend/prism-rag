"""文本分块器测试"""

from src.ingestion.text_chunker import TextChunker


def test_empty_text():
    chunker = TextChunker()
    chunks = chunker.chunk_page(page_id=1, doc_id="doc1", page_number=1, markdown_text=None)
    assert chunks == []

    chunks = chunker.chunk_page(page_id=1, doc_id="doc1", page_number=1, markdown_text="")
    assert chunks == []


def test_single_paragraph():
    chunker = TextChunker()
    text = "This is a single paragraph with a reasonable length."
    chunks = chunker.chunk_page(page_id=1, doc_id="doc1", page_number=1, markdown_text=text)
    assert len(chunks) == 1
    assert chunks[0].text == text
    assert chunks[0].chunk_id == "pg00001_ch001"


def test_multiple_paragraphs():
    chunker = TextChunker()
    text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
    chunks = chunker.chunk_page(page_id=1, doc_id="doc1", page_number=1, markdown_text=text)
    assert len(chunks) == 3


def test_long_paragraph_splits():
    chunker = TextChunker(max_tokens=10)  # 40 chars
    text = "This is a very long paragraph that should be split into multiple chunks because it exceeds the maximum token limit."
    chunks = chunker.chunk_page(page_id=1, doc_id="doc1", page_number=1, markdown_text=text)
    assert len(chunks) >= 2


def test_table_detection():
    chunker = TextChunker()
    text = "| Col1 | Col2 | Col3 |\n|------|------|------|\n| A    | B    | C    |"
    chunks = chunker.chunk_page(page_id=1, doc_id="doc1", page_number=1, markdown_text=text)
    assert len(chunks) == 1
    assert chunks[0].chunk_type == "table"
