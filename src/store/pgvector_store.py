"""pgvector 存储封装

Schema:
  CREATE TABLE chunks (
    chunk_id TEXT PRIMARY KEY,
    page_id INTEGER NOT NULL,
    doc_id TEXT NOT NULL,
    page_number INTEGER NOT NULL,
    chunk_type TEXT NOT NULL DEFAULT 'text',
    text TEXT NOT NULL,
    bge_vector vector(1024) NOT NULL
  );
  CREATE INDEX idx_chunks_page_id ON chunks(page_id);
  CREATE INDEX idx_chunks_doc_id ON chunks(doc_id);
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import psycopg2
import psycopg2.extras
from pgvector.psycopg2 import register_vector

from src.config import cfg


class PgVectorStore:
    """pgvector 存储客户端"""

    def __init__(self, connection_string: str | None = None):
        self.conn_string = connection_string or self._default_conn_string()
        self._conn: Optional[psycopg2.extensions.connection] = None

    def _default_conn_string(self) -> str:
        pg_host = cfg.get("storage.pgvector.host", "localhost")
        pg_port = cfg.get("storage.pgvector.port", "5432")
        pg_db = cfg.get("storage.pgvector.dbname", "prismrag")
        pg_user = cfg.get("storage.pgvector.user", "prismrag")
        pg_pass = cfg.get("storage.pgvector.password", "prismrag")
        return f"host={pg_host} port={pg_port} dbname={pg_db} user={pg_user} password={pg_pass}"

    @property
    def conn(self):
        if self._conn is None or self._conn.closed:
            self._conn = psycopg2.connect(self.conn_string)
            register_vector(self._conn)
        return self._conn

    def create_schema(self):
        """创建表和索引（幂等）"""
        with self.conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cur.execute("""
                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id TEXT PRIMARY KEY,
                    page_id INTEGER NOT NULL,
                    doc_id TEXT NOT NULL,
                    page_number INTEGER NOT NULL,
                    chunk_type TEXT NOT NULL DEFAULT 'text',
                    text TEXT NOT NULL,
                    bge_vector vector(1024) NOT NULL,
                    doc_ref TEXT NOT NULL DEFAULT ''
                )
            """)
            cur.execute("CREATE INDEX IF NOT EXISTS idx_chunks_page_id ON chunks(page_id)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_chunks_doc_id ON chunks(doc_id)")
            # HNSW 索引
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_chunks_bge_hnsw
                ON chunks USING hnsw (bge_vector vector_cosine_ops)
                WITH (m = 16, ef_construction = 200)
            """)
        self.conn.commit()

    def insert_chunks(self, chunks: List[Tuple]):
        """批量插入 chunk

        Args:
            chunks: [(chunk_id, page_id, doc_id, page_number, chunk_type, text, bge_vector, doc_ref), ...]
        """
        with self.conn.cursor() as cur:
            psycopg2.extras.execute_values(
                cur,
                """
                INSERT INTO chunks (chunk_id, page_id, doc_id, page_number, chunk_type, text, bge_vector, doc_ref)
                VALUES %s
                ON CONFLICT (chunk_id) DO NOTHING
                """,
                chunks,
                template="(%s, %s, %s, %s, %s, %s, %s::vector, %s)",
            )
        self.conn.commit()

    def search_by_vector(self, query_vector: np.ndarray, k: int = 20) -> List[dict]:
        """余弦相似度搜索，返回 Top-k chunk"""
        with self.conn.cursor() as cur:
            cur.execute(
                """
                SELECT chunk_id, page_id, doc_id, page_number, chunk_type, text, doc_ref,
                       1 - (bge_vector <=> %s::vector) AS score
                FROM chunks
                ORDER BY bge_vector <=> %s::vector
                LIMIT %s
                """,
                (query_vector.tolist(), query_vector.tolist(), k),
            )
            rows = cur.fetchall()
            return [
                {
                    "chunk_id": r[0],
                    "page_id": r[1],
                    "doc_id": r[2],
                    "page_number": r[3],
                    "chunk_type": r[4],
                    "text": r[5],
                    "doc_ref": r[6],
                    "score": float(r[7]),
                }
                for r in rows
            ]

    def get_chunks_by_page_ids(self, page_ids: List[int]) -> List[dict]:
        """按 page_id 列表查询所有 chunk（Visual 路 grounding 反查用）"""
        if not page_ids:
            return []
        with self.conn.cursor() as cur:
            cur.execute(
                """
                SELECT chunk_id, page_id, doc_id, page_number, chunk_type, text, doc_ref
                FROM chunks
                WHERE page_id = ANY(%s)
                """,
                (page_ids,),
            )
            rows = cur.fetchall()
            return [
                {
                    "chunk_id": r[0],
                    "page_id": r[1],
                    "doc_id": r[2],
                    "page_number": r[3],
                    "chunk_type": r[4],
                    "text": r[5],
                    "doc_ref": r[6],
                }
                for r in rows
            ]

    def count(self) -> int:
        with self.conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM chunks")
            return cur.fetchone()[0]

    def delete_by_doc_id(self, doc_id: str) -> int:
        """删除某 doc_id 的全部 chunk，返回删除行数（失败清理用）"""
        with self.conn.cursor() as cur:
            cur.execute("DELETE FROM chunks WHERE doc_id = %s", (doc_id,))
            deleted = cur.rowcount
        self.conn.commit()
        return deleted

    def close(self):
        if self._conn and not self._conn.closed:
            self._conn.close()
