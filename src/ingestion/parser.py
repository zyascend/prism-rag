"""PDF 解析抽象：生产用 MinerU，本地兜底用 PyMuPDF"""
from __future__ import annotations
import io
import re
import shutil
import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import List

from PIL import Image
import fitz  # PyMuPDF


@dataclass
class Page:
    page_number: int
    markdown: str
    image: Image.Image


class Parser(ABC):
    @abstractmethod
    def parse(self, pdf_path: Path) -> List[Page]:
        ...


class SimplePDFParser(Parser):
    """本地兜底：PyMuPDF 抽文本 + 渲染页面图。零外部依赖。"""

    def parse(self, pdf_path: Path) -> List[Page]:
        doc = fitz.open(pdf_path)
        pages: List[Page] = []
        try:
            for i, page in enumerate(doc):
                markdown = page.get_text("text") or ""
                pix = page.get_pixmap(dpi=150)
                image = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
                pages.append(Page(page_number=i + 1, markdown=markdown, image=image))
        finally:
            doc.close()
        return pages


class MinerUParser(Parser):
    """生产用：MinerU CLI 解析，质量最高。best-effort 逐页切分。"""

    def __init__(self, output_dir: Path | None = None):
        self.output_dir = output_dir or Path("data/mineru_output")

    def parse(self, pdf_path: Path) -> List[Page]:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if shutil.which("mineru") is None:
            raise RuntimeError(
                "mineru CLI 未安装；本地 dev 请改用 SimplePDFParser（parser=simple）"
            )
        subprocess.run(
            ["mineru", "-p", str(pdf_path), "-o", str(self.output_dir), "--device", "cpu"],
            check=True,
        )
        stem = pdf_path.stem
        base = self.output_dir / stem / stem
        md_path = base / f"{stem}.md"
        images_dir = base / "images"
        markdown = md_path.read_text() if md_path.exists() else ""
        parts = re.split(r"(!\[[^\]]*\]\(images/[^)]+\))", markdown)
        pages: List[Page] = []
        img_files = sorted(images_dir.glob("*.png")) if images_dir.exists() else []
        text_acc, idx = "", 0
        for part in parts:
            if re.match(r"!\[[^\]]*\]\(images/[^)]+\)", part or ""):
                image = (
                    Image.open(img_files[idx])
                    if idx < len(img_files)
                    else Image.new("RGB", (1000, 1600), 255)
                )
                pages.append(
                    Page(page_number=idx + 1, markdown=text_acc.strip(), image=image)
                )
                idx += 1
                text_acc = ""
            else:
                text_acc += part or ""
        if idx == 0:
            pages.append(
                Page(
                    page_number=1,
                    markdown=markdown.strip(),
                    image=Image.new("RGB", (1000, 1600), 255),
                )
            )
        return pages


def build_parser(name: str | None = None) -> Parser:
    from src.config import cfg

    name = name or cfg.get("ingestion.parser", "simple")
    if name == "simple":
        return SimplePDFParser()
    return MinerUParser()
