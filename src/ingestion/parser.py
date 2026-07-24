"""PDF 解析抽象：生产用 MinerU，本地兜底用 PyMuPDF

Phase A2：MinerU 优先消费 content_list 得到 typed ContentBlock；
缺失时降级为 markdown + 启发式分块（与历史行为一致）。
"""
from __future__ import annotations

import io
import json
import logging
import re
import shutil
import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from PIL import Image
import fitz  # PyMuPDF

logger = logging.getLogger(__name__)


@dataclass
class ContentBlock:
    """MinerU（或等价）解析出的类型化内容块。"""

    type: str  # text | table | image | equation
    text: str = ""
    caption: str = ""
    page_idx: int = 0  # 0-based（与 MinerU 一致）
    text_level: int = 0  # 标题层级，0=正文
    raw: Dict[str, Any] = field(default_factory=dict, repr=False)


@dataclass
class Page:
    page_number: int  # 1-based
    markdown: str
    image: Image.Image
    blocks: Optional[List[ContentBlock]] = None  # None = 走 markdown 启发式分块


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
                pages.append(
                    Page(
                        page_number=i + 1,
                        markdown=markdown,
                        image=image,
                        blocks=None,
                    )
                )
        finally:
            doc.close()
        return pages


class MinerUParser(Parser):
    """生产用：MinerU CLI 解析，质量最高。优先 content_list，否则 markdown 切页。"""

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
        return self._load_pages_from_output(pdf_path)

    def _load_pages_from_output(self, pdf_path: Path) -> List[Page]:
        """从已有 MinerU 输出目录构建 Page（可供测试注入）。"""
        stem = pdf_path.stem
        search_root = self.output_dir / stem
        if not search_root.exists():
            search_root = self.output_dir

        content_list = self._find_and_load_content_list(search_root, stem)
        md_path = self._find_first(search_root, f"{stem}.md") or self._find_first(
            search_root, "*.md"
        )
        markdown = md_path.read_text() if md_path and md_path.exists() else ""
        images_dir = self._find_images_dir(search_root)
        img_files = sorted(images_dir.glob("*.png")) if images_dir else []

        if content_list:
            logger.info(
                "MinerU content_list 命中 %d blocks，走 typed 路径", len(content_list)
            )
            return pages_from_content_list(
                content_list,
                images=img_files,
                fallback_markdown=markdown,
            )

        logger.warning(
            "MinerU content_list 未找到，降级 markdown 切页（stem=%s root=%s）",
            stem,
            search_root,
        )
        return self._pages_from_markdown(markdown, img_files)

    @staticmethod
    def _find_and_load_content_list(root: Path, stem: str) -> Optional[List[dict]]:
        if not root.exists():
            return None
        # 优先精确名，再通配
        candidates = list(root.rglob(f"{stem}_content_list.json"))
        if not candidates:
            candidates = list(root.rglob("*_content_list.json"))
        if not candidates:
            return None
        path = candidates[0]
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as e:
            logger.warning("content_list 读取失败 %s: %s", path, e)
            return None
        if not isinstance(data, list):
            logger.warning("content_list 非 list: %s", path)
            return None
        return data

    @staticmethod
    def _find_first(root: Path, pattern: str) -> Optional[Path]:
        if not root.exists():
            return None
        hits = sorted(root.rglob(pattern))
        return hits[0] if hits else None

    @staticmethod
    def _find_images_dir(root: Path) -> Optional[Path]:
        if not root.exists():
            return None
        for d in root.rglob("images"):
            if d.is_dir():
                return d
        return None

    @staticmethod
    def _pages_from_markdown(markdown: str, img_files: List[Path]) -> List[Page]:
        parts = re.split(r"(!\[[^\]]*\]\(images/[^)]+\))", markdown)
        pages: List[Page] = []
        text_acc, idx = "", 0
        for part in parts:
            if re.match(r"!\[[^\]]*\]\(images/[^)]+\)", part or ""):
                image = (
                    Image.open(img_files[idx])
                    if idx < len(img_files)
                    else Image.new("RGB", (1000, 1600), 255)
                )
                pages.append(
                    Page(
                        page_number=idx + 1,
                        markdown=text_acc.strip(),
                        image=image,
                        blocks=None,
                    )
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
                    blocks=None,
                )
            )
        return pages


def pages_from_content_list(
    content_list: List[dict],
    images: Optional[List[Path]] = None,
    fallback_markdown: str = "",
    blank_image_size: tuple[int, int] = (1000, 1600),
) -> List[Page]:
    """将 MinerU 风格 content_list 转为带 blocks 的 Page 列表（纯函数，便于单测）。"""
    images = images or []
    by_page: Dict[int, List[ContentBlock]] = {}
    for item in content_list:
        if not isinstance(item, dict):
            continue
        block = content_block_from_mineru_item(item)
        by_page.setdefault(block.page_idx, []).append(block)

    if not by_page:
        return [
            Page(
                page_number=1,
                markdown=fallback_markdown.strip(),
                image=Image.new("RGB", blank_image_size, 255),
                blocks=None,
            )
        ]

    max_idx = max(by_page.keys())
    pages: List[Page] = []
    for page_idx in range(max_idx + 1):
        blocks = by_page.get(page_idx, [])
        md_parts = []
        for b in blocks:
            if b.type == "table":
                if b.caption:
                    md_parts.append(b.caption)
                if b.text:
                    md_parts.append(b.text)
            elif b.type == "image":
                if b.caption:
                    md_parts.append(f"[Image: {b.caption}]")
            elif b.text:
                md_parts.append(b.text)
        markdown = "\n\n".join(md_parts).strip() or (
            fallback_markdown if page_idx == 0 else ""
        )
        image = (
            Image.open(images[page_idx]).convert("RGB")
            if page_idx < len(images)
            else Image.new("RGB", blank_image_size, 255)
        )
        pages.append(
            Page(
                page_number=page_idx + 1,
                markdown=markdown,
                image=image,
                blocks=blocks or None,
            )
        )
    return pages


def content_block_from_mineru_item(item: dict) -> ContentBlock:
    """单条 MinerU content_list 元素 → ContentBlock。"""
    raw_type = str(item.get("type") or "text").lower()
    if "page_idx" in item:
        page_idx = int(item.get("page_idx") or 0)
    elif "page_no" in item:
        page_idx = max(0, int(item["page_no"]) - 1)  # 1-based → 0-based
    else:
        page_idx = 0

    caption = _join_caption(
        item.get("table_caption")
        or item.get("img_caption")
        or item.get("image_caption")
        or item.get("caption")
        or []
    )
    text_level = int(item.get("text_level") or 0)

    if raw_type in ("table",):
        body = (
            item.get("table_body")
            or item.get("table_html")
            or item.get("html")
            or item.get("text")
            or ""
        )
        body = _maybe_html_table_to_text(str(body))
        return ContentBlock(
            type="table",
            text=body.strip(),
            caption=caption,
            page_idx=page_idx,
            text_level=0,
            raw=item,
        )

    if raw_type in ("image", "figure", "img"):
        return ContentBlock(
            type="image",
            text="",
            caption=caption,
            page_idx=page_idx,
            text_level=0,
            raw=item,
        )

    if raw_type in ("equation", "formula"):
        text = (
            item.get("text")
            or item.get("latex")
            or item.get("equation")
            or ""
        )
        return ContentBlock(
            type="equation",
            text=str(text).strip(),
            caption=caption,
            page_idx=page_idx,
            text_level=0,
            raw=item,
        )

    # text / title / other
    text = str(item.get("text") or item.get("content") or "").strip()
    if text_level > 0 and text and not text.startswith("#"):
        text = f"{'#' * min(text_level, 6)} {text}"
    return ContentBlock(
        type="text",
        text=text,
        caption=caption,
        page_idx=page_idx,
        text_level=text_level,
        raw=item,
    )


def _join_caption(cap: Any) -> str:
    if cap is None:
        return ""
    if isinstance(cap, str):
        return cap.strip()
    if isinstance(cap, (list, tuple)):
        return " ".join(str(x).strip() for x in cap if x).strip()
    return str(cap).strip()


def _maybe_html_table_to_text(body: str) -> str:
    """若是 HTML 表则粗转管道文本；已是 markdown 则原样。"""
    if not body:
        return body
    if "<table" not in body.lower() and "<tr" not in body.lower():
        return body
    # 极简 HTML → 行文本，避免引入额外依赖
    rows = re.findall(r"<tr[^>]*>(.*?)</tr>", body, flags=re.I | re.S)
    if not rows:
        # 剥标签兜底
        return re.sub(r"<[^>]+>", " ", body)
    out_lines: List[str] = []
    for i, row in enumerate(rows):
        cells = re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", row, flags=re.I | re.S)
        cells = [re.sub(r"<[^>]+>", "", c).strip() for c in cells]
        if not cells:
            continue
        line = "| " + " | ".join(cells) + " |"
        out_lines.append(line)
        if i == 0:
            out_lines.append("| " + " | ".join("---" for _ in cells) + " |")
    return "\n".join(out_lines) if out_lines else re.sub(r"<[^>]+>", " ", body)


def build_parser(name: str | None = None) -> Parser:
    from src.config import cfg

    name = name or cfg.get("ingestion.parser", "simple")
    if name == "simple":
        return SimplePDFParser()
    if name == "mineru":
        if shutil.which("mineru") is None:
            # 本地 demo 可配置 mineru；CLI 未装时降级，避免 /ingest 直接 500
            logger.warning(
                "ingestion.parser=mineru 但 PATH 中无 mineru CLI，"
                "降级为 SimplePDFParser。安装后重启服务即可："
                "pip/uv 装 mineru 或见 https://github.com/opendatalab/MinerU"
            )
            return SimplePDFParser()
        return MinerUParser()
    raise ValueError(f"unknown ingestion.parser: {name!r} (expected simple|mineru)")
