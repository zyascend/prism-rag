"""Phase A2: MinerU content_list → typed ContentBlock → chunk_blocks."""
from __future__ import annotations

import json
from pathlib import Path

from src.ingestion.parser import (
    ContentBlock,
    content_block_from_mineru_item,
    pages_from_content_list,
)
from src.ingestion.text_chunker import TextChunker

FIXTURE_DIR = Path(__file__).parent / "fixtures"


def _sample_content_list() -> list[dict]:
    return [
        {
            "type": "text",
            "text": "Pressure limits for rinse operations.",
            "text_level": 1,
            "page_idx": 0,
        },
        {
            "type": "text",
            "text": "Use clean water only. Do not exceed rated pressure.",
            "page_idx": 0,
        },
        {
            "type": "table",
            "table_caption": ["Table 7. Max nozzle pressure"],
            "table_body": (
                "| System | Max PSI |\n"
                "| --- | --- |\n"
                "| Type I | 40 |\n"
                "| Type II | 25 |\n"
            ),
            "page_idx": 0,
        },
        {
            "type": "image",
            "img_caption": ["Figure 1. Clamp orientation diagram"],
            "page_idx": 0,
        },
        {
            "type": "text",
            "text": "Additional notes on page two.",
            "page_idx": 1,
        },
        {
            "type": "table",
            "table_body": (
                "<table><tr><th>A</th><th>B</th></tr>"
                "<tr><td>1</td><td>2</td></tr></table>"
            ),
            "table_caption": ["HTML table"],
            "page_idx": 1,
        },
    ]


def test_content_block_from_mineru_types():
    blocks = [content_block_from_mineru_item(x) for x in _sample_content_list()]
    types = [b.type for b in blocks]
    assert types.count("text") == 3
    assert types.count("table") == 2
    assert types.count("image") == 1
    assert blocks[0].text.startswith("#")  # text_level → heading
    assert "Max PSI" in blocks[2].text
    assert blocks[2].caption.startswith("Table 7")
    # HTML table converted to pipes
    assert "|" in blocks[5].text
    assert "A" in blocks[5].text


def test_pages_from_content_list_groups_by_page():
    pages = pages_from_content_list(_sample_content_list())
    assert len(pages) == 2
    assert pages[0].page_number == 1
    assert pages[0].blocks is not None
    assert len(pages[0].blocks) == 4
    assert pages[1].page_number == 2
    assert len(pages[1].blocks) == 2


def test_chunk_blocks_types_and_table():
    pages = pages_from_content_list(_sample_content_list())
    chunker = TextChunker(image_caption_chunks=False)
    chunks = chunker.chunk_blocks(
        page_id=1, doc_id="d1", page_number=1, blocks=pages[0].blocks
    )
    types = {c.chunk_type for c in chunks}
    assert "table" in types
    assert "text" in types
    assert "image" not in types  # caption chunks off
    table = next(c for c in chunks if c.chunk_type == "table")
    assert "40" in table.text
    assert table.caption.startswith("Table 7")


def test_chunk_blocks_image_caption_when_enabled():
    pages = pages_from_content_list(_sample_content_list())
    chunker = TextChunker(image_caption_chunks=True)
    chunks = chunker.chunk_blocks(
        page_id=1, doc_id="d1", page_number=1, blocks=pages[0].blocks
    )
    images = [c for c in chunks if c.chunk_type == "image"]
    assert len(images) == 1
    assert "Clamp orientation" in images[0].text
    assert images[0].caption


def test_empty_content_list_fallback_page():
    pages = pages_from_content_list([], fallback_markdown="hello")
    assert len(pages) == 1
    assert pages[0].blocks is None
    assert pages[0].markdown == "hello"


def test_chunk_page_still_works_without_blocks():
    """simple 路径不受 A2 影响。"""
    chunker = TextChunker()
    chunks = chunker.chunk_page(
        page_id=1,
        doc_id="d",
        page_number=1,
        markdown_text="Short paragraph about pumps.",
    )
    assert len(chunks) == 1
    assert chunks[0].chunk_type == "text"


def test_write_fixture_roundtrip(tmp_path: Path):
    """模拟从 content_list.json 文件加载。"""
    path = tmp_path / "doc_content_list.json"
    path.write_text(json.dumps(_sample_content_list()), encoding="utf-8")
    data = json.loads(path.read_text())
    pages = pages_from_content_list(data)
    assert len(pages[0].blocks) >= 3
