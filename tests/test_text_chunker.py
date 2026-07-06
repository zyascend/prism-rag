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


# ─── TO 手册清洗测试 ────────────────────────────────────────────

class TestCleanMarkdown:
    """测试 clean_to_markdown() 对 TO 军事手册噪音的清洗"""

    def test_strips_empty_table_rows(self):
        """仅移除含空单元格的表格碎片，保留普通表格"""
        # Empty-cell rows should be removed
        text = "Content.\n|  | TO WP 011 | 35E1-2-13-1 00 |\nMore content."
        result = TextChunker.clean_to_markdown(text)
        assert "Content" in result
        assert "More content" in result
        assert "TO WP 011" not in result

        # Normal table rows should be preserved
        text2 = "| Col1 | Col2 | Col3 |\n| A | B | C |"
        result2 = TextChunker.clean_to_markdown(text2)
        assert "Col1" in result2
        assert "Col2" in result2

    def test_strips_to_reference(self):
        text = "Some text.\nTO 35E1-2-13-1, WP 004 00, Page 3 and 4\nMore text."
        result = TextChunker.clean_to_markdown(text)
        assert "Some text" in result
        assert "More text" in result
        assert "TO 35E1-2-13-1" not in result

    def test_strips_doc_id(self):
        text = "Content.\n35E1-2-13-1 00\nMore content."
        result = TextChunker.clean_to_markdown(text)
        assert "Content" in result
        assert "More content" in result
        assert "35E1-2-13-1" not in result

    def test_strips_allcaps_header(self):
        text = "PERFORM EMERGENCY SHUTDOWN PROCEDURES\nActual step: push the button."
        result = TextChunker.clean_to_markdown(text)
        assert "Actual step" in result
        assert "PERFORM EMERGENCY" not in result

    def test_preserves_allcaps_warnings(self):
        """Short all-caps like WARNING/DANGER should be preserved (they're < 15 chars)"""
        text = "WARNING: High voltage.\nDANGER: Do not enter."
        result = TextChunker.clean_to_markdown(text)
        assert "WARNING" in result
        assert "DANGER" in result

    def test_fixes_hyphen_breaks(self):
        text = "The equipment requires Protec-\ntive Equipment (PPE)."
        result = TextChunker.clean_to_markdown(text)
        assert "Protective Equipment" in result
        assert "Protec-" not in result

    def test_fixes_hyphen_breaks_multiple(self):
        text = "In accor-\ndance with regula-\ntions."
        result = TextChunker.clean_to_markdown(text)
        assert "accordance" in result
        assert "regulations" in result

    def test_compresses_blank_lines(self):
        text = "Line 1.\n\n\n\nLine 2.\n\n\nLine 3."
        result = TextChunker.clean_to_markdown(text)
        # Should have at most double newlines
        assert "\n\n\n" not in result

    def test_empty_input(self):
        assert TextChunker.clean_to_markdown("") == ""
        assert TextChunker.clean_to_markdown("   ") == ""
        assert TextChunker.clean_to_markdown(None) == ""

    def test_chunk_page_with_noisy_to_text(self):
        """端到端：带噪音的 TO 文本 → chunk，验证清洗生效"""
        chunker = TextChunker()
        text = (
            "TO 35E1-2-13-1 WP 004 00\n"
            "PERFORM EMERGENCY SHUTDOWN\n"
            "| Step | Action |\n"
            "| 1 | Push button |\n"
            "The operator must push the emergency\n"
            "shutdown button located on the control\n"
            "panel. This will immediately stop fuel flow.\n\n"
            "After shutdown, verify that all burner\n"
            "valves are in the closed position."
        )
        chunks = chunker.chunk_page(page_id=1, doc_id="d1", page_number=1, markdown_text=text)
        # Should have at least one chunk with the real content
        all_text = " ".join(c.text for c in chunks)
        assert "operator must push" in all_text
        assert "fuel flow" in all_text
        assert "TO 35E1-2-13-1" not in all_text
        assert "PERFORM EMERGENCY SHUTDOWN" not in all_text

    def test_preserves_real_content(self):
        """清洗后不应丢失正文内容"""
        text = (
            "The design incorporates a lined burn area and\n"
            "conservation pond for water recycling.\n"
            "A 100-foot burn area with an aircraft mock-up\n"
            "and 10,000 gallon liquid propane fuel tank."
        )
        result = TextChunker.clean_to_markdown(text)
        assert "lined burn area" in result
        assert "conservation pond" in result
        assert "aircraft mock-up" in result
