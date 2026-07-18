"""检索缓存后端抽象与内存实现。

设计要点：
- CacheStore 为后端抽象，未来可扩展 RedisCache 等共享后端（多 worker 场景）。
- InMemoryLRUCache 为默认后端：进程内 LRU 淘汰 + 可选 TTL 兜底。
- 缓存正确性不依赖 TTL，而依赖调用方在语料变更时调用 invalidate()
  （index_version 盐机制），TTL 仅作跨进程异常残留的安全网。
"""
from __future__ import annotations

import threading
import time
from collections import OrderedDict
from typing import Any, Optional


class CacheStore:
    """检索缓存后端抽象接口。"""

    def get(self, key: str) -> Optional[Any]:
        raise NotImplementedError

    def put(self, key: str, value: Any) -> None:
        raise NotImplementedError

    def clear(self) -> None:
        raise NotImplementedError


class InMemoryLRUCache(CacheStore):
    """进程内 LRU 缓存，支持可选 TTL 兜底。

    Args:
        max_size: 最大条目数，超过则淘汰最久未使用项。
        ttl_seconds: 条目存活秒数。0 或负数表示不启用 TTL（仅依赖调用方失效）。
    """

    def __init__(self, max_size: int = 2048, ttl_seconds: int = 0):
        self._max = max(1, int(max_size))
        self._ttl = ttl_seconds if ttl_seconds and ttl_seconds > 0 else 0
        self._lock = threading.RLock()
        # key -> (value, expire_at|None)
        self._store: "OrderedDict[str, tuple[Any, Optional[float]]]" = OrderedDict()

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            if key not in self._store:
                return None
            value, expire_at = self._store[key]
            if self._ttl and expire_at is not None and time.monotonic() > expire_at:
                # 已过期：惰性淘汰
                self._store.pop(key, None)
                return None
            self._store.move_to_end(key)
            return value

    def put(self, key: str, value: Any) -> None:
        with self._lock:
            expire_at = (time.monotonic() + self._ttl) if self._ttl else None
            self._store[key] = (value, expire_at)
            self._store.move_to_end(key)
            while len(self._store) > self._max:
                self._store.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._store.clear()
