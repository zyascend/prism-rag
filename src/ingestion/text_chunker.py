"""ViDoRe 语料的文本分块策略

策略：
0. 预处理：正则清洗 TO 军事手册特有的噪音（表格碎片、文档编号、断词等）
1. 按双换行切段落
2. 段落 ≤ 512 tokens → 直接作为一块
3. 段落 > 512 tokens → 按句号/换行边界切到 ≤ 512
4. 空段落跳过
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List


@dataclass
class Chunk:
    chunk_id: str
    page_id: int
    doc_id: str
    page_number: int
    text: str
    chunk_type: str = "text"  # text | table
    doc_ref: str = ""         # TO 文档编号，用于 LLM grounding（不进 CtxRel 评估）
    table_summary: str = ""   # 表格自然语言摘要（仅 chunk_type=table 时由 TableSummarizer 填充）

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

    def __init__(self, max_tokens: int = MAX_TOKENS):
        self.max_tokens = max_tokens
        self.max_chars = max_tokens * self.TOKEN_EST_RATIO

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

            if self._looks_like_table(para):
                # 表格：按"行"切分（保留表头），绝不按词切碎
                for t in self._split_table(para, page_id, doc_id, page_number, doc_ref):
                    chunk_idx += 1
                    t.chunk_id = f"pg{page_id:05d}_ch{chunk_idx:03d}"
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
                    ))

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
             if re.match(r"^\s*\|?[\s:|-]*-[\s:|-]*\|?\s*$", ln) and "|" in ln),
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
