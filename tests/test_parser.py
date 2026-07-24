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


def test_build_parser_mineru_falls_back_without_cli(monkeypatch):
    """配置 mineru 但 CLI 不在 PATH 时，应降级 Simple 而非硬崩。"""
    from src.ingestion import parser as parser_mod

    monkeypatch.setattr(parser_mod.shutil, "which", lambda _: None)
    p = build_parser("mineru")
    assert isinstance(p, SimplePDFParser)
