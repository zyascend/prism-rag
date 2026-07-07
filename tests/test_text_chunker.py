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

    @staticmethod
    def _clean(text):
        """Helper: call clean_to_markdown and return just the text"""
        result, _ = TextChunker.clean_to_markdown(text)
        return result

    def test_strips_empty_table_rows(self):
        """仅移除含空单元格的表格碎片，保留普通表格"""
        result = self._clean("Content.\n|  | TO WP 011 | 35E1-2-13-1 00 |\nMore content.")
        assert "Content" in result
        assert "More content" in result
        assert "TO WP 011" not in result

        result2 = self._clean("| Col1 | Col2 | Col3 |\n| A | B | C |")
        assert "Col1" in result2
        assert "Col2" in result2

    def test_strips_to_reference(self):
        result = self._clean("Some text.\nTO 35E1-2-13-1, WP 004 00, Page 3 and 4\nMore text.")
        assert "Some text" in result
        assert "More text" in result
        assert "TO 35E1-2-13-1" not in result

    def test_strips_doc_id(self):
        result = self._clean("Content.\n35E1-2-13-1 00\nMore content.")
        assert "Content" in result
        assert "More content" in result
        assert "35E1-2-13-1" not in result

    def test_strips_allcaps_header(self):
        result = self._clean("PERFORM EMERGENCY SHUTDOWN PROCEDURES\nActual step: push the button.")
        assert "Actual step" in result
        assert "PERFORM EMERGENCY" not in result

    def test_preserves_allcaps_warnings(self):
        result = self._clean("WARNING: High voltage.\nDANGER: Do not enter.")
        assert "WARNING" in result
        assert "DANGER" in result

    def test_fixes_hyphen_breaks(self):
        result = self._clean("The equipment requires Protec-\ntive Equipment (PPE).")
        assert "Protective Equipment" in result
        assert "Protec-" not in result

    def test_fixes_hyphen_breaks_multiple(self):
        result = self._clean("In accor-\ndance with regula-\ntions.")
        assert "accordance" in result
        assert "regulations" in result

    def test_compresses_blank_lines(self):
        result = self._clean("Line 1.\n\n\n\nLine 2.\n\n\nLine 3.")
        assert "\n\n\n" not in result

    def test_empty_input(self):
        result, ref = TextChunker.clean_to_markdown("")
        assert result == ""
        assert ref == ""
        result, ref = TextChunker.clean_to_markdown("   ")
        assert result == ""
        result, ref = TextChunker.clean_to_markdown(None)
        assert result == ""

    def test_extracts_doc_ref(self):
        """验证 TO 引用被提取为 doc_ref"""
        text = "TO 35E1-2-13-1, WP 004 00, Page 3 and 4\nSome content here."
        result, ref = TextChunker.clean_to_markdown(text)
        assert "TO 35E1-2-13-1" in ref
        assert "Some content" in result
        assert "TO 35E1-2-13-1" not in result  # removed from text

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
        result = self._clean(text)
        assert "lined burn area" in result
        assert "conservation pond" in result
        assert "aircraft mock-up" in result

    def test_chunk_page_attaches_doc_ref(self):
        """端到端：doc_ref 被提取并附加到每个 chunk"""
        chunker = TextChunker()
        text = (
            "TO 35E1-2-13-1 WP 004 00\n"
            "The operator must push the emergency shutdown button.\n\n"
            "After shutdown, verify all burner valves are closed."
        )
        chunks = chunker.chunk_page(page_id=1, doc_id="d1", page_number=1, markdown_text=text)
        assert len(chunks) > 0
        for c in chunks:
            assert "TO 35E1-2-13-1" in c.doc_ref
            assert "TO 35E1-2-13-1" not in c.text  # stripped from text
