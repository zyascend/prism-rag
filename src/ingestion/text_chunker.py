"""ViDoRe 语料的文本分块策略

策略：
1. 按双换行切段落
2. 段落 ≤ 512 tokens → 直接作为一块
3. 段落 > 512 tokens → 按句号/换行边界切到 ≤ 512
4. 空段落跳过
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List


@dataclass
class Chunk:
    chunk_id: str
    page_id: int
    doc_id: str
    page_number: int
    text: str
    chunk_type: str = "text"  # text | table

    def __repr__(self) -> str:
        return f"Chunk(id={self.chunk_id}, page={self.page_id}, type={self.chunk_type})"


class TextChunker:
    """ViDoRe 文本分块器"""

    MAX_TOKENS = 512
    # 简单 token 估算：英文约 4 chars/token
    TOKEN_EST_RATIO = 4

    def __init__(self, max_tokens: int = MAX_TOKENS):
        self.max_tokens = max_tokens
        self.max_chars = max_tokens * self.TOKEN_EST_RATIO

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

        paragraphs = re.split(r"\n\s*\n", markdown_text.strip())
        chunks: List[Chunk] = []
        chunk_idx = 0

        for para in paragraphs:
            para = para.strip()
            if not para:
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
                    chunk_type="table" if self._looks_like_table(para) else "text",
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
                    ))

        return chunks

    @staticmethod
    def _looks_like_table(text: str) -> bool:
        """启发式判断是否为表格文本（含管道符或明显的列对齐）"""
        lines = text.split("\n")
        pipe_count = sum(line.count("|") for line in lines[:5])
        return pipe_count >= 3
