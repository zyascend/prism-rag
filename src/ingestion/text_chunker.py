"""ViDoRe 语料的文本分块策略

策略：
0. 预处理：正则清洗 TO 军事手册特有的噪音（表格碎片、文档编号、断词等）
1. 按双换行切段落
2. 段落 ≤ 512 tokens → 直接作为一块
3. 段落 > 512 tokens → 按句号/换行边界切到 ≤ 512
4. 空段落跳过

Phase A2：``chunk_blocks`` 消费 MinerU typed ContentBlock；无 blocks 时走 ``chunk_page``。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Sequence

if TYPE_CHECKING:
    from src.ingestion.parser import ContentBlock


@dataclass
class Chunk:
    chunk_id: str
    page_id: int
    doc_id: str
    page_number: int
    text: str
    chunk_type: str = "text"  # text | table | image
    doc_ref: str = ""         # TO 文档编号，用于 LLM grounding（不进 CtxRel 评估）
    table_summary: str = ""   # 表格自然语言摘要（仅 chunk_type=table 时由 TableSummarizer 填充）
    caption: str = ""         # 图/表 caption（A2/A3；可空）
    section_path: str = ""    # 章节路径 e.g. "3.2 Cooling > Limits"
    prev_chunk_id: str = ""
    next_chunk_id: str = ""

    def __repr__(self) -> str:
        return f"Chunk(id={self.chunk_id}, page={self.page_id}, type={self.chunk_type})"


class TextChunker:
    """ViDoRe 文本分块器"""

    MAX_TOKENS = 512
    # 简单 token 估算：英文约 4 chars/token
    TOKEN_EST_RATIO = 4

    # ── TO 手册清洗正则（编译一次，复用）─────────────────────
    # 表格行：仅移除含空单元格的碎片行（如 "|  | TO WP 011 |"），
    # 保留正常的 markdown 表格（如 "| Col1 | Col2 |"）
    _RE_EMPTY_TABLE_ROW = re.compile(
        r"^\s*\|\s*\|.+\|\s*$", re.MULTILINE
    )
    _RE_TO_REF = re.compile(
        r"^\s*TO\s+[\dA-Z][\dA-Z\-/]+\s*(?:,?\s*(?:WP|Page|para|and)[\s\d\-A-Za-z]+)*\s*$",
        re.MULTILINE,
    )
    _RE_DOC_ID = re.compile(
        r"^\s*[\d]{1,2}[A-Z]\d[\dA-Z\-/]{3,}\s*(?:\d{2})?\s*$",
        re.MULTILINE,
    )
    _RE_ALLCAPS_LINE = re.compile(
        r"^\s*[A-Z][A-Z\s/\-–]{15,60}\s*$", re.MULTILINE,
    )
    _RE_HYPHEN_BREAK = re.compile(r"(\w)-\n(\w)")
    _RE_MULTI_BLANK = re.compile(r"\n{3,}")
    # 表格分隔行（形如 |---|---| 或 :--: | :--:）
    _SEP_RE = re.compile(r"^\s*\|?[\s:|-]*-[\s:|-]*\|?\s*$")

    def __init__(
        self,
        max_tokens: int = MAX_TOKENS,
        image_caption_chunks: bool = False,
    ):
        self.max_tokens = max_tokens
        self.max_chars = max_tokens * self.TOKEN_EST_RATIO
        self.image_caption_chunks = image_caption_chunks

        self._heading_stack: list = []

    def reset_headings(self) -> None:
        """跨文档入库前清空标题栈。"""
        self._heading_stack: list = []

    def current_section_path(self) -> str:
        stack = getattr(self, "_heading_stack", None) or []
        return " > ".join(title for _, title in stack)

    def _note_heading(self, text: str, text_level: int = 0) -> str:
        """根据 markdown 标题或 text_level 更新栈，返回当前 section_path。"""
        if not hasattr(self, "_heading_stack"):
            self._heading_stack = []
        level = 0
        title = ""
        m = re.match(r"^(#{1,6})\s+(.+)$", (text or "").strip())
        if m:
            level = len(m.group(1))
            title = m.group(2).strip()
        elif text_level and text_level > 0:
            level = int(text_level)
            title = re.sub(r"^#{1,6}\s*", "", (text or "").strip())
        if level and title:
            while self._heading_stack and self._heading_stack[-1][0] >= level:
                self._heading_stack.pop()
            self._heading_stack.append((level, title))
        return self.current_section_path()

    @staticmethod
    def link_neighbors(chunks: List[Chunk]) -> None:
        """按列表顺序写 prev/next_chunk_id（同页或同批）。"""
        for i, c in enumerate(chunks):
            c.prev_chunk_id = chunks[i - 1].chunk_id if i > 0 else ""
            c.next_chunk_id = chunks[i + 1].chunk_id if i + 1 < len(chunks) else ""


    def chunk_blocks(
        self,
        page_id: int,
        doc_id: str,
        page_number: int,
        blocks: Sequence["ContentBlock"],
        doc_ref: str = "",
    ) -> List[Chunk]:
        """按 typed ContentBlock 分块（MinerU content_list 路径）。

        - table：caption 前缀 + table body，走大表按行切分
        - text / equation：清洗后按段落/长度切
        - image：仅当 ``image_caption_chunks`` 且有 caption 时生成可检索锚点 chunk
        """
        if not blocks:
            return []

        # 从块中尽量补 doc_ref
        if not doc_ref:
            for b in blocks:
                if b.type == "text" and b.text:
                    _, ref = self.clean_to_markdown(b.text)
                    if ref:
                        doc_ref = ref
                        break

        chunks: List[Chunk] = []
        chunk_idx = 0

        def _push(c: Chunk) -> None:
            nonlocal chunk_idx
            chunk_idx += 1
            c.chunk_id = f"pg{page_id:05d}_ch{chunk_idx:03d}"
            if not getattr(c, "section_path", ""):
                c.section_path = self.current_section_path()
            chunks.append(c)

        for block in blocks:
            btype = (block.type or "text").lower()

            if btype == "table":
                body = (block.text or "").strip()
                cap = (block.caption or "").strip()
                if not body and not cap:
                    continue
                table_md = body
                if cap and cap not in table_md[: max(len(cap) + 20, 80)]:
                    table_md = f"{cap}\n{table_md}" if table_md else cap
                if not table_md.strip():
                    continue
                # 表体仍做轻量清洗（不去掉管道结构为主）
                cleaned, _ = self.clean_to_markdown(table_md)
                table_md = cleaned or table_md
                table_md = self._normalize_separator_row(table_md)
                for t in self._split_table(
                    table_md, page_id, doc_id, page_number, doc_ref
                ):
                    t.caption = cap
                    _push(t)
                continue

            if btype == "image":
                cap = (block.caption or "").strip()
                if self.image_caption_chunks and cap:
                    _push(
                        Chunk(
                            chunk_id="",
                            page_id=page_id,
                            doc_id=doc_id,
                            page_number=page_number,
                            text=f"[Image] {cap}",
                            chunk_type="image",
                            doc_ref=doc_ref,
                            caption=cap,
                        )
                    )
                continue

            # text / equation / other
            text = (block.text or "").strip()
            if not text:
                continue
            # 标题块先更新 section 栈（保留 heading 自身为 chunk）
            self._note_heading(text, getattr(block, "text_level", 0) or 0)
            cleaned, _ = self.clean_to_markdown(text)
            text = cleaned or text
            if not text.strip():
                continue
            for c in self._chunk_plain_text(
                text, page_id, doc_id, page_number, doc_ref
            ):
                _push(c)

        self.link_neighbors(chunks)
        return chunks

    def _chunk_plain_text(
        self,
        text: str,
        page_id: int,
        doc_id: str,
        page_number: int,
        doc_ref: str,
    ) -> List[Chunk]:
        """将已清洗的纯文本切成 text chunk（不赋 chunk_id）。"""
        out: List[Chunk] = []
        if len(text) <= self.max_chars:
            out.append(
                Chunk(
                    chunk_id="",
                    page_id=page_id,
                    doc_id=doc_id,
                    page_number=page_number,
                    text=text,
                    chunk_type="text",
                    doc_ref=doc_ref,
                )
            )
            return out

        sentences = re.split(r"(?<=[.?!])\s+", text)
        buffer = ""
        for sent in sentences:
            if len(sent) > self.max_chars:
                if buffer:
                    out.append(
                        Chunk(
                            chunk_id="",
                            page_id=page_id,
                            doc_id=doc_id,
                            page_number=page_number,
                            text=buffer,
                            chunk_type="text",
                            doc_ref=doc_ref,
                        )
                    )
                    buffer = ""
                words = sent.split()
                word_buffer = ""
                for word in words:
                    if len(word_buffer) + len(word) + 1 <= self.max_chars:
                        word_buffer = (word_buffer + " " + word).strip()
                    else:
                        if word_buffer:
                            out.append(
                                Chunk(
                                    chunk_id="",
                                    page_id=page_id,
                                    doc_id=doc_id,
                                    page_number=page_number,
                                    text=word_buffer,
                                    chunk_type="text",
                                    doc_ref=doc_ref,
                                )
                            )
                        word_buffer = word
                if word_buffer:
                    out.append(
                        Chunk(
                            chunk_id="",
                            page_id=page_id,
                            doc_id=doc_id,
                            page_number=page_number,
                            text=word_buffer,
                            chunk_type="text",
                            doc_ref=doc_ref,
                        )
                    )
            elif len(buffer) + len(sent) + 1 <= self.max_chars:
                buffer = (buffer + " " + sent).strip()
            else:
                if buffer:
                    out.append(
                        Chunk(
                            chunk_id="",
                            page_id=page_id,
                            doc_id=doc_id,
                            page_number=page_number,
                            text=buffer,
                            chunk_type="text",
                            doc_ref=doc_ref,
                        )
                    )
                buffer = sent
        if buffer:
            out.append(
                Chunk(
                    chunk_id="",
                    page_id=page_id,
                    doc_id=doc_id,
                    page_number=page_number,
                    text=buffer,
                    chunk_type="text",
                    doc_ref=doc_ref,
                )
            )
        return out

    @classmethod
    def clean_to_markdown(cls, text: str | None) -> tuple[str, str]:
        """清洗 TO 军事手册噪音，同时提取文档编号用于 grounding。

        Returns:
            (cleaned_text, doc_ref) — doc_ref 为第一个匹配到的 TO 编号，无则为 ""
        """
        if not text or not text.strip():
            return "", ""

        # 提取第一个 TO 引用作为 doc_ref（用于 LLM grounding）
        doc_ref = ""
        ref_match = cls._RE_TO_REF.search(text)
        if ref_match:
            doc_ref = ref_match.group(0).strip()
            # 规范化：去掉多余空格和逗号
            doc_ref = re.sub(r"\s+", " ", doc_ref)

        # 1. 修复 PDF 断词
        text = cls._RE_HYPHEN_BREAK.sub(r"\1\2", text)

        # 2. 去含空单元格的表格碎片行（"|  | TO WP 011 |"），保留正常表格
        text = cls._RE_EMPTY_TABLE_ROW.sub("", text)

        # 3. 去 TO 文档引用行（"TO 35E1-2-13-1, WP 004 00, Page 3 and 4"）
        text = cls._RE_TO_REF.sub("", text)

        # 4. 去纯文档编号行（"35E1-2-13-1 00"）
        text = cls._RE_DOC_ID.sub("", text)

        # 5. 去全大写短行（章节标题如 "PRINCIPLES OF OPERATION"）
        #    注意：必须足够短且不含小写字母，避免误删正文中的缩写
        text = cls._RE_ALLCAPS_LINE.sub("", text)

        # 6. 压缩多余空行（≥3个换行 → 2个）
        text = cls._RE_MULTI_BLANK.sub("\n\n", text)

        return text.strip(), doc_ref

    def chunk_page(
        self,
        page_id: int,
        doc_id: str,
        page_number: int,
        markdown_text: str | None,
    ) -> List[Chunk]:
        """将一页 markdown 文本切成 chunk 列表"""
        if not markdown_text or not markdown_text.strip():
            return []

        # ── 预处理：清洗 TO 手册噪音 + 提取 doc_ref ─────
        markdown_text, doc_ref = self.clean_to_markdown(markdown_text)
        if not markdown_text:
            return []

        paragraphs = re.split(r"\n\s*\n", markdown_text.strip())
        # 合并被空行打断的表格：相邻两段都是表格形态则拼回一体，
        # 否则按行切分时表头/分隔行会掉队，破坏 markdown 表格结构。
        paragraphs = self._merge_table_blocks(paragraphs)
        chunks: List[Chunk] = []
        chunk_idx = 0

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            # markdown 标题更新 section 栈（表格跳过）
            if not self._looks_like_table(para):
                self._note_heading(para)

            if self._looks_like_table(para):
                # 表格：先归一化分隔行（无 |---|---| 时注入），再按"行"切分
                para = self._normalize_separator_row(para)
                for t in self._split_table(para, page_id, doc_id, page_number, doc_ref):
                    chunk_idx += 1
                    t.chunk_id = f"pg{page_id:05d}_ch{chunk_idx:03d}"
                    t.section_path = self.current_section_path()
                    chunks.append(t)
                continue

            if len(para) <= self.max_chars:
                # 短段落直接作为一块
                chunk_idx += 1
                chunks.append(Chunk(
                    chunk_id=f"pg{page_id:05d}_ch{chunk_idx:03d}",
                    page_id=page_id,
                    doc_id=doc_id,
                    page_number=page_number,
                    text=para,
                    chunk_type="text",
                    doc_ref=doc_ref,
                section_path=self.current_section_path(),
                ))
            else:
                # 长段落：按句子边界切
                sentences = re.split(r"(?<=[.?!])\s+", para)
                buffer = ""
                for sent in sentences:
                    # 如果单个句子本身就超过限制，按词切分
                    if len(sent) > self.max_chars:
                        if buffer:
                            chunk_idx += 1
                            chunks.append(Chunk(
                                chunk_id=f"pg{page_id:05d}_ch{chunk_idx:03d}",
                                page_id=page_id,
                                doc_id=doc_id,
                                page_number=page_number,
                                text=buffer,
                                chunk_type="text",
                            doc_ref=doc_ref,
                            section_path=self.current_section_path(),
                            ))
                            buffer = ""
                        words = sent.split()
                        word_buffer = ""
                        for word in words:
                            if len(word_buffer) + len(word) + 1 <= self.max_chars:
                                word_buffer = (word_buffer + " " + word).strip()
                            else:
                                if word_buffer:
                                    chunk_idx += 1
                                    chunks.append(Chunk(
                                        chunk_id=f"pg{page_id:05d}_ch{chunk_idx:03d}",
                                        page_id=page_id,
                                        doc_id=doc_id,
                                        page_number=page_number,
                                        text=word_buffer,
                                        chunk_type="text",
                            doc_ref=doc_ref,
                                    section_path=self.current_section_path(),
                                    ))
                                word_buffer = word
                        if word_buffer:
                            chunk_idx += 1
                            chunks.append(Chunk(
                                chunk_id=f"pg{page_id:05d}_ch{chunk_idx:03d}",
                                page_id=page_id,
                                doc_id=doc_id,
                                page_number=page_number,
                                text=word_buffer,
                                chunk_type="text",
                            doc_ref=doc_ref,
                            section_path=self.current_section_path(),
                            ))
                    elif len(buffer) + len(sent) + 1 <= self.max_chars:
                        buffer = (buffer + " " + sent).strip()
                    else:
                        if buffer:
                            chunk_idx += 1
                            chunks.append(Chunk(
                                chunk_id=f"pg{page_id:05d}_ch{chunk_idx:03d}",
                                page_id=page_id,
                                doc_id=doc_id,
                                page_number=page_number,
                                text=buffer,
                                chunk_type="text",
                            doc_ref=doc_ref,
                            section_path=self.current_section_path(),
                            ))
                        buffer = sent
                if buffer:
                    chunk_idx += 1
                    chunks.append(Chunk(
                        chunk_id=f"pg{page_id:05d}_ch{chunk_idx:03d}",
                        page_id=page_id,
                        doc_id=doc_id,
                        page_number=page_number,
                        text=buffer,
                        chunk_type="text",
                            doc_ref=doc_ref,
                    section_path=self.current_section_path(),
                    ))

        self.link_neighbors(chunks)
        return chunks

    @staticmethod
    def _merge_table_blocks(paragraphs: List[str]) -> List[str]:
        """把被空行拆开的相邻表格段落重新拼回一体。"""
        merged: List[str] = []
        for para in paragraphs:
            if (merged and TextChunker._looks_like_table(merged[-1])
                    and TextChunker._looks_like_table(para)):
                merged[-1] = merged[-1] + "\n" + para
            else:
                merged.append(para)
        return merged

    def _split_table(
        self, table_md: str, page_id: int, doc_id: str,
        page_number: int, doc_ref: str,
    ) -> List[Chunk]:
        """按行切分超长 markdown 表格，每段保留表头（列名 + 分隔行）。

        避免原逻辑按词切碎导致 `|---|---|` 与行对齐丢失。
        """
        lines = table_md.split("\n")
        # 定位分隔行（形如 |---|---| 或 :--: | :--:）
        sep_idx = next(
            (i for i, ln in enumerate(lines)
             if TextChunker._SEP_RE.match(ln) and "|" in ln),
            None,
        )
        if sep_idx is not None:
            header = lines[: sep_idx + 1]          # 列名行 + 分隔行
            body = lines[sep_idx + 1:]
        else:
            header = lines[:1]
            body = lines[1:]

        chunks: List[Chunk] = []
        buf: List[str] = []
        buf_len = 0
        for row in body:
            row_len = len(row)
            if buf and buf_len + row_len + 1 > self.max_chars:
                chunks.append(self._make_table_chunk(
                    "\n".join(header + buf), page_id, doc_id, page_number, doc_ref))
                buf, buf_len = [], 0
            buf.append(row)
            buf_len += row_len + 1
        if buf:
            chunks.append(self._make_table_chunk(
                "\n".join(header + buf), page_id, doc_id, page_number, doc_ref))
        # 兜底：极端情况（无 body）至少保留原表
        if not chunks:
            chunks.append(self._make_table_chunk(
                table_md, page_id, doc_id, page_number, doc_ref))
        return chunks

    @staticmethod
    def _make_table_chunk(
        text: str, page_id: int, doc_id: str,
        page_number: int, doc_ref: str,
    ) -> Chunk:
        return Chunk(
            chunk_id="",  # 由调用方按序填充
            page_id=page_id, doc_id=doc_id, page_number=page_number,
            text=text, chunk_type="table", doc_ref=doc_ref,
        )

    @staticmethod
    def _looks_like_table(text: str) -> bool:
        """启发式判断是否为表格文本（含管道符或明显的列对齐）"""
        lines = text.split("\n")
        pipe_count = sum(line.count("|") for line in lines[:5])
        return pipe_count >= 3

    @staticmethod
    def _normalize_separator_row(table_md: str) -> str:
        """表格分隔行归一化：把"表头后是数据行但缺 |---|---| 分隔行"的 markdown
        表格修复为合法 GFM（在表头行后注入分隔行）。

        能正确处理两种形态：
        - 纯表格（首行即表头）：在首行后注入分隔行；
        - 表头前有 caption 行（如 "Table 7. Pressure limits" 直接贴在表格上，
          中间无空行）：定位到第一个真正的管道表头行，在它之后注入分隔行，
          caption 作为前缀保留，且 _split_table 不会把 caption 误当表头。

        收益：
        - 让 _split_table 能复用完整表头（列名行 + 分隔行），而非把首行当伪表头；
        - 长表切分后的每个子块都是合法 GFM，保留列结构；
        - 单块小表也更利于 Dense embedding 与 LLM 按表格渲染。

        以下情况原样返回（不误改）：
        - 非表格 / 不足两行 / 没有管道表头行；
        - 表头行后已是分隔行；
        - 表头行后非表格数据行（caption 续行 / 正文 / 空行）。
        """
        lines = table_md.split("\n")
        if len(lines) < 2:
            return table_md
        # 定位第一个真正的管道表头行（跳过前置 caption 等非表格行）
        h = next((i for i, ln in enumerate(lines) if ln.count("|") >= 2), None)
        if h is None:
            return table_md  # 没有任何表格行，原样返回
        if h + 1 >= len(lines):
            return table_md  # 只有表头没有后续行
        first = lines[h].strip()
        second = lines[h + 1].strip()
        # 表头行后已是分隔行 → 无需处理
        if TextChunker._SEP_RE.match(second) and "|" in second:
            return table_md
        # 表头行后不是数据行（caption 续行 / 正文 / 空）→ 不注入，避免破坏结构
        if second.count("|") < 2:
            return table_md
        # 依据表头列数构造分隔行
        core = first.strip()
        if core.startswith("|"):
            core = core[1:]
        if core.endswith("|"):
            core = core[:-1]
        n_cols = len(core.split("|"))
        sep = "|" + "|".join("---" for _ in range(n_cols)) + "|"
        # 在表头行(h)之后注入分隔行，前置 caption 行保持原样
        return "\n".join(lines[: h + 1] + [sep] + lines[h + 1:])
