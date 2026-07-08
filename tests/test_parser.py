# tests/test_parser.py
from pathlib import Path
from src.ingestion.parser import SimplePDFParser, build_parser

FIXTURE = Path(__file__).parent / "fixtures" / "sample.pdf"

def test_simple_parser_returns_pages():
    pages = SimplePDFParser().parse(FIXTURE)
    assert len(pages) == 1
    assert "hydraulic pump" in pages[0].markdown
    assert pages[0].image is not None
    assert pages[0].page_number == 1

def test_build_parser_default_simple():
    p = build_parser()
    assert isinstance(p, SimplePDFParser)
