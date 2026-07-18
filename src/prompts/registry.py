"""全局 Prompt 注册表。

进程启动后惰性扫描 ``src/prompts/prompts/*.yaml``，建立 ``id -> Prompt`` 索引。
调用点用 ``get_active(id)`` 取生效版本。运行时单版本生效，不做 A/B。
"""
from __future__ import annotations

import logging
from pathlib import Path
from threading import Lock
from typing import Dict, Optional

from .loader import PromptConfigError, load_prompt_file
from .models import Prompt, PromptVersion

logger = logging.getLogger(__name__)

_DEFAULT_DIR = Path(__file__).parent / "prompts"


class PromptNotFound(KeyError):
    """请求的 prompt id 不存在于注册表。"""


class PromptRegistry:
    def __init__(self) -> None:
        self._prompts: Dict[str, Prompt] = {}
        self._loaded = False
        self._lock = Lock()

    def init(self, prompts_dir: Optional[str] = None, force: bool = False) -> None:
        """扫描目录并构建索引。已加载则跳过（除非 force）。"""
        with self._lock:
            if self._loaded and not force:
                return
            directory = Path(prompts_dir) if prompts_dir else _DEFAULT_DIR
            if not directory.is_dir():
                raise PromptConfigError(f"prompts dir not found: {directory}")

            prompts: Dict[str, Prompt] = {}
            for f in sorted(directory.glob("*.yaml")):
                p = load_prompt_file(f)
                if p.id in prompts:
                    raise PromptConfigError(
                        f"duplicate prompt id '{p.id}' (file {f.name})"
                    )
                prompts[p.id] = p

            if not prompts:
                raise PromptConfigError(f"no prompt yaml files found in {directory}")

            self._prompts = prompts
            self._loaded = True
            logger.info(
                "PromptRegistry loaded %d prompts from %s", len(prompts), directory
            )

    def _ensure(self) -> None:
        if not self._loaded:
            self.init()

    def get_active(self, prompt_id: str) -> PromptVersion:
        """返回指定 prompt 的生效版本。"""
        self._ensure()
        if prompt_id not in self._prompts:
            raise PromptNotFound(prompt_id)
        return self._prompts[prompt_id].active_version

    def get_prompt(self, prompt_id: str) -> Prompt:
        """返回完整 Prompt（含全部版本历史）。"""
        self._ensure()
        if prompt_id not in self._prompts:
            raise PromptNotFound(prompt_id)
        return self._prompts[prompt_id]

    def list_prompts(self) -> dict:
        """返回所有 prompt 的当前生效版本摘要（供只读排查端点）。"""
        self._ensure()
        return {
            pid: {
                "active_version": p.active_version.version,
                "description": p.description,
                "versions": [v.version for v in p.versions],
                "created_at": p.active_version.created_at,
                "changelog": p.active_version.changelog,
            }
            for pid, p in self._prompts.items()
        }


# ─── 模块级单例 + 便捷函数 ──────────────────────────────────────
_registry = PromptRegistry()


def init(prompts_dir: Optional[str] = None, force: bool = False) -> None:
    _registry.init(prompts_dir=prompts_dir, force=force)


def get_active(prompt_id: str) -> PromptVersion:
    return _registry.get_active(prompt_id)


def get_prompt(prompt_id: str) -> Prompt:
    return _registry.get_prompt(prompt_id)


def list_prompts() -> dict:
    return _registry.list_prompts()
