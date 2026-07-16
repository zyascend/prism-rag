"""大批量原子快照切换工具（Spec §4.6 零停机刷新）

- 文件级：`atomic_replace` 用 `os.replace`（POSIX 原子 rename），
  FAISS / BM25 语料原子覆盖，检索进程无感知。
- pg 级：`build_chunk_swap_sql` 生成 chunks 表 RENAME swap 的 SQL，
  由 `PgVectorStore.atomic_swap_chunks` 在单事务内执行。
"""
from __future__ import annotations

import os
from typing import List


def atomic_replace(src: str, dst: str) -> None:
    """POSIX 原子 rename：dst 原地替换，读取方不会读到半截文件。

    检索进程若已持有旧数据在内存中（如 FAISS 的 numpy 数组），不受影响；
    新查询在 load() 时读到新文件。整体切换无需下线。
    """
    os.replace(src, dst)


def build_chunk_swap_sql(
    staging: str = "chunks_staging",
    live: str = "chunks",
    old: str = "chunks_old",
) -> List[str]:
    """生成 chunks 表原子切换的 SQL 语句序列（单事务内依次执行）。

    调用方先在 `staging` 表 INSERT 新全量数据，再执行本序列完成切换：
    RENAME live→old、RENAME staging→live、DROP old。检索经 `live` 表，
    整个过程在事务内完成，无感知。
    """
    return [
        f"CREATE TABLE {staging} (LIKE {live} INCLUDING ALL)",
        f"ALTER TABLE {live} RENAME TO {old}",
        f"ALTER TABLE {staging} RENAME TO {live}",
        f"DROP TABLE {old}",
    ]
