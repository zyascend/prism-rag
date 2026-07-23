"""Phase A3: section_path / prev-next neighbor 元数据."""
from __future__ import annotations

from src.ingestion.parser import ContentBlock, pages_from_content_list
from src.ingestion.text_chunker import TextChunker
from src.store.pgvector_store import PgVectorStore


def test_heading_stack_section_path():
    chunker = TextChunker()
    chunker.reset_headings()
    md = """# Cooling System

## Pressure Limits

Do not exceed 40 PSI during rinse.

## Temperature

Keep below 80C.
"""
    chunks = chunker.chunk_page(page_id=1, doc_id="d", page_number=1, markdown_text=md)
    assert len(chunks) >= 3
    # 正文块应带上章节路径
    pressure_body = next(c for c in chunks if "40 PSI" in c.text)
    assert "Cooling" in pressure_body.section_path
    assert "Pressure" in pressure_body.section_path


def test_link_neighbors_chain():
    chunker = TextChunker()
    chunks = chunker.chunk_page(
        page_id=7,
        doc_id="d",
        page_number=1,
        markdown_text="First paragraph here.\n\nSecond paragraph here.\n\nThird paragraph here.",
    )
    assert len(chunks) == 3
    assert chunks[0].prev_chunk_id == ""
    assert chunks[0].next_chunk_id == chunks[1].chunk_id
    assert chunks[1].prev_chunk_id == chunks[0].chunk_id
    assert chunks[1].next_chunk_id == chunks[2].chunk_id
    assert chunks[2].next_chunk_id == ""
    assert chunks[2].prev_chunk_id == chunks[1].chunk_id


def test_chunk_blocks_preserves_caption_and_neighbors():
    content_list = [
        {"type": "text", "text": "Limits", "text_level": 1, "page_idx": 0},
        {
            "type": "table",
            "table_caption": ["Table A"],
            "table_body": "| X | Y |\n| --- | --- |\n| 1 | 2 |\n",
            "page_idx": 0,
        },
        {"type": "text", "text": "See table above for values.", "page_idx": 0},
    ]
    pages = pages_from_content_list(content_list)
    chunker = TextChunker()
    chunker.reset_headings()
    chunks = chunker.chunk_blocks(
        page_id=1, doc_id="d", page_number=1, blocks=pages[0].blocks
    )
    assert len(chunks) >= 2
    table = next(c for c in chunks if c.chunk_type == "table")
    assert table.caption.startswith("Table A")
    assert any(c.next_chunk_id for c in chunks[:-1])


def test_normalize_chunk_row_pads_short_tuples():
    row9 = ("id", 1, "d", 1, "text", "hello", [0.0] * 3, "ref", "sum")
    n = PgVectorStore._normalize_chunk_row(row9)
    assert len(n) == 14
    assert n[9] == ""  # page_hash
    assert n[10] == ""  # section_path
