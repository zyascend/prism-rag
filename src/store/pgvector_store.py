"""pgvector 存储封装

Schema:
  CREATE TABLE chunks (
    chunk_id TEXT PRIMARY KEY,
    page_id INTEGER NOT NULL,
    doc_id TEXT NOT NULL,
    page_number INTEGER NOT NULL,
    chunk_type TEXT NOT NULL DEFAULT 'text',
    text TEXT NOT NULL,
    bge_vector vector(1024) NOT NULL,
    doc_ref TEXT NOT NULL DEFAULT '',
    table_summary TEXT NOT NULL DEFAULT '',
    page_hash TEXT NOT NULL DEFAULT ''   -- 页面内容哈希（P2 page diff 用，未变页复用）
  );
  CREATE INDEX idx_chunks_page_id ON chunks(page_id);
  CREATE INDEX idx_chunks_doc_id ON chunks(doc_id);
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

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
            # 关键：先开 autocommit，再 register_vector（其内部会发查询，
            # 若 autocommit 仍为 False 会开启事务导致后续无法切换）。
            # 不设 autocommit 的危害：本 store 是 API 单例常驻连接，
            # 每次 search 的 SELECT 都会让连接进入 idle-in-transaction 且永不提交，
            # 长期持有 chunks 共享锁，挡住 DDL（ALTER TABLE 需排他锁），
            # 表现为 API 运行数小时后拖住所有 ingest/迁移（已验证的 23h 泄漏）。
            self._conn.autocommit = True
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
                    doc_ref TEXT NOT NULL DEFAULT '',
                    table_summary TEXT NOT NULL DEFAULT '',
                    page_hash TEXT NOT NULL DEFAULT ''
                )
            """)
            # 兼容旧库：新增列（已在 CREATE 中的新库此句为 no-op）
            cur.execute(
                "ALTER TABLE chunks ADD COLUMN IF NOT EXISTS table_summary TEXT NOT NULL DEFAULT ''"
            )
            cur.execute(
                "ALTER TABLE chunks ADD COLUMN IF NOT EXISTS page_hash TEXT NOT NULL DEFAULT ''"
            )
            cur.execute("CREATE INDEX IF NOT EXISTS idx_chunks_page_id ON chunks(page_id)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_chunks_doc_id ON chunks(doc_id)")
            # HNSW 索引
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_chunks_bge_hnsw
                ON chunks USING hnsw (bge_vector vector_cosine_ops)
                WITH (m = 16, ef_construction = 200)
            """)
            # P1: 文档级内容寻址表（content_hash 幂等，消除 U2 重入库副本）
            cur.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    doc_id TEXT PRIMARY KEY,
                    content_hash TEXT NOT NULL UNIQUE,
                    source_path TEXT NOT NULL DEFAULT '',
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
            """)
            cur.execute("CREATE INDEX IF NOT EXISTS idx_documents_hash ON documents(content_hash)")
        self.conn.commit()

    def insert_chunks(self, chunks: List[Tuple]):
        """批量插入 chunk

        Args:
            chunks: [(chunk_id, page_id, doc_id, page_number, chunk_type, text,
                      bge_vector, doc_ref, table_summary, page_hash), ...]
        """
        with self.conn.cursor() as cur:
            psycopg2.extras.execute_values(
                cur,
                """
                INSERT INTO chunks (chunk_id, page_id, doc_id, page_number, chunk_type, text, bge_vector, doc_ref, table_summary, page_hash)
                VALUES %s
                ON CONFLICT (chunk_id) DO NOTHING
                """,
                chunks,
                template="(%s, %s, %s, %s, %s, %s, %s::vector, %s, %s, %s)",
            )
        self.conn.commit()

    def search_by_vector(self, query_vector: np.ndarray, k: int = 20) -> List[dict]:
        """余弦相似度搜索，返回 Top-k chunk"""
        with self.conn.cursor() as cur:
            cur.execute(
                """
                SELECT chunk_id, page_id, doc_id, page_number, chunk_type, text, doc_ref, table_summary,
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
                    "table_summary": r[7],
                    "score": float(r[8]),
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
                SELECT chunk_id, page_id, doc_id, page_number, chunk_type, text, doc_ref, table_summary
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
                    "table_summary": r[7],
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

    def get_chunk_ids_by_doc_id(self, doc_id: str) -> List[str]:
        """删除前先取该 doc 的全部 chunk_id（避免先删行后丢引用，修复 D4）"""
        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT chunk_id FROM chunks WHERE doc_id = %s ORDER BY chunk_id",
                (doc_id,),
            )
            return [r[0] for r in cur.fetchall()]

    def get_page_ids_by_doc_id(self, doc_id: str) -> List[int]:
        """删除前先取该 doc 的全部 page_id（供 FAISS 墓碑用，P1 接入）"""
        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT DISTINCT page_id FROM chunks WHERE doc_id = %s ORDER BY page_id",
                (doc_id,),
            )
            return [r[0] for r in cur.fetchall()]

    # ── P2: page diff / reconcile 辅助 ─────────────────────

    def get_all_chunk_ids(self) -> List[str]:
        """返回全部 chunk_id（BM25 启动对账用，判断需增量 fit 或 remove 的差额）"""
        with self.conn.cursor() as cur:
            cur.execute("SELECT chunk_id FROM chunks ORDER BY chunk_id")
            return [r[0] for r in cur.fetchall()]

    def get_chunks_by_ids(self, chunk_ids: List[str]) -> List[dict]:
        """按 chunk_id 批量取文档（reconcile 增量 fit 用）。空列表返回 []。"""
        if not chunk_ids:
            return []
        with self.conn.cursor() as cur:
            cur.execute(
                """
                SELECT chunk_id, page_id, doc_id, page_number, chunk_type, text
                FROM chunks WHERE chunk_id = ANY(%s)
                """,
                (chunk_ids,),
            )
            return [
                {
                    "chunk_id": r[0], "page_id": r[1], "doc_id": r[2],
                    "page_number": r[3], "chunk_type": r[4], "text": r[5],
                }
                for r in cur.fetchall()
            ]

    def get_pages_by_doc_id(self, doc_id: str) -> List[tuple]:
        """返回该 doc 的 (page_id, page_number) 列表（page diff 删除/变更对齐用）"""
        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT DISTINCT page_id, page_number FROM chunks WHERE doc_id = %s ORDER BY page_number",
                (doc_id,),
            )
            return [(r[0], r[1]) for r in cur.fetchall()]

    def get_page_hashes_by_doc_id(self, doc_id: str) -> Dict[int, str]:
        """返回该 doc 的 page_number -> page_hash 映射（page diff 未变页复用判定）"""
        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT page_number, page_hash FROM chunks WHERE doc_id = %s",
                (doc_id,),
            )
            result: Dict[int, str] = {}
            for page_number, page_hash in cur.fetchall():
                if page_number not in result:  # 同页多 chunk 共享 page_hash，取首个
                    result[page_number] = page_hash
            return result

    def get_chunk_ids_by_page_ids(self, page_ids: List[int]) -> List[str]:
        """按 page_id 列表取 chunk_id（page diff 变更/删除页的 BM25 清理用）"""
        if not page_ids:
            return []
        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT chunk_id FROM chunks WHERE page_id = ANY(%s)",
                (page_ids,),
            )
            return [r[0] for r in cur.fetchall()]

    def delete_chunks_by_page_ids(self, page_ids: List[int]) -> int:
        """按 page_id 列表删除 chunk 行，返回删除行数（page diff 三路清理之一）"""
        if not page_ids:
            return 0
        with self.conn.cursor() as cur:
            cur.execute("DELETE FROM chunks WHERE page_id = ANY(%s)", (page_ids,))
            deleted = cur.rowcount
        self.conn.commit()
        return deleted

    # ── P1: 文档级内容寻址（幂等重入库）───────────────────

    def upsert_document(self, doc_id: str, content_hash: str, source_path: str = "") -> None:
        """写入/复用文档记录；content_hash 唯一 → 重入库同一内容自动定位到同一 doc_id。"""
        with self.conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO documents (doc_id, content_hash, source_path)
                VALUES (%s, %s, %s)
                ON CONFLICT (content_hash) DO NOTHING
                """,
                (doc_id, content_hash, source_path),
            )
        self.conn.commit()

    def get_doc_id_by_content_hash(self, content_hash: str) -> Optional[str]:
        """按内容哈希查已有 doc_id；无则返回 None。供 ingest 幂等覆盖用。"""
        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT doc_id FROM documents WHERE content_hash = %s",
                (content_hash,),
            )
            row = cur.fetchone()
            return row[0] if row else None

    def document_exists(self, doc_id: str) -> bool:
        """判断 doc_id 是否已入库（page diff UPDATE 路径判定：同 doc_id 修改版）"""
        with self.conn.cursor() as cur:
            cur.execute("SELECT 1 FROM documents WHERE doc_id = %s", (doc_id,))
            return cur.fetchone() is not None

    def update_document(self, doc_id: str, content_hash: str, source_path: str = "") -> None:
        """更新已存在 doc 的内容哈希（page diff：同 doc_id 修改版重入库后刷新版本指纹）"""
        with self.conn.cursor() as cur:
            cur.execute(
                """
                UPDATE documents SET content_hash = %s, source_path = %s
                WHERE doc_id = %s
                """,
                (content_hash, source_path, doc_id),
            )
        self.conn.commit()

    def close(self):
        if self._conn and not self._conn.closed:
            self._conn.close()

    # ── P2-C: 大批量原子快照切换（零停机刷新，§4.6 pg 部分）──

    def _new_connection(self):
        """新建独立连接（autocommit=False），用于需要事务框定的操作（快照 swap）。"""
        conn = psycopg2.connect(self.conn_string)
        conn.autocommit = False
        register_vector(conn)
        return conn

    def _insert_chunks_on(self, cur, chunks: List[Tuple]):
        """在指定 cursor 上批量插入 chunk（供原子 swap 复用当前事务连接）。"""
        psycopg2.extras.execute_values(
            cur,
            """
            INSERT INTO chunks (chunk_id, page_id, doc_id, page_number, chunk_type, text, bge_vector, doc_ref, table_summary, page_hash)
            VALUES %s
            """,
            chunks,
            template="(%s, %s, %s, %s, %s, %s, %s::vector, %s, %s, %s)",
        )

    def atomic_swap_chunks(
        self,
        insert_rows: List[Tuple],
        staging: str = "chunks_staging",
        live: str = "chunks",
        old: str = "chunks_old",
    ):
        """原子切换 chunks 表（零停机批量刷新）。

        在独立非 autocommit 事务内完成：建暂存表 → 插新全量数据 →
        RENAME live→old、RENAME staging→live、DROP old。整个过程在单个事务内，
        检索经 `live` 表，无感知。中途异常自动 ROLLBACK，不会出现「半切换」脏状态。
        """
        conn = self._new_connection()
        try:
            cur = conn.cursor()
            cur.execute(f"CREATE TABLE {staging} (LIKE {live} INCLUDING ALL)")
            if insert_rows:
                self._insert_chunks_on(cur, insert_rows)
            cur.execute(f"ALTER TABLE {live} RENAME TO {old}")
            cur.execute(f"ALTER TABLE {staging} RENAME TO {live}")
            cur.execute(f"DROP TABLE {old}")
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
