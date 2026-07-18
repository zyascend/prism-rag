"""Prompt 版本管理数据模型。

方案 A（集中管理 + 版本记录）：每个 prompt 一个 YAML 文件，内含多个版本，
其中恰好一个标记 ``active: true`` 为运行时生效版本。不做 A/B、不做在线切换。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

# 版本条目里的元数据字段（非模板正文）。其余键都视为模板正文（system/user/template 等）。
META_KEYS = frozenset({"version", "created_at", "author", "changelog", "active"})


@dataclass(frozen=True)
class PromptVersion:
    """一个 prompt 的单个版本。

    ``body`` 保存模板正文字段（如 ``template`` 或 ``system``/``user``），
    键名即 YAML 里的字段名，值为模板字符串（保留 ``{var}`` 占位符）。
    """

    prompt_id: str
    version: int
    active: bool
    created_at: str = ""
    changelog: str = ""
    author: Optional[str] = None
    body: Dict[str, str] = field(default_factory=dict)

    def text(self, name: str = "template") -> str:
        """取某个模板正文字段（默认 ``template``）。"""
        if name not in self.body:
            raise KeyError(
                f"prompt '{self.prompt_id}' v{self.version} has no field '{name}'; "
                f"available: {sorted(self.body)}"
            )
        return self.body[name]

    def render(self, field: str = "template", /, **kwargs) -> str:
        """取字段并以 ``str.format(**kwargs)`` 注入变量。

        ``field`` 为位置限定参数（positional-only），避免与模板变量同名的
        关键字（如 ``{name}``/``{field}``）冲突。
        """
        return self.text(field).format(**kwargs)

    @property
    def template(self) -> str:
        return self.text("template")

    @property
    def system(self) -> str:
        return self.text("system")

    @property
    def user(self) -> str:
        return self.text("user")


@dataclass
class Prompt:
    """一个 prompt（一个 YAML 文件），含多个版本。"""

    id: str
    versions: List[PromptVersion]
    description: Optional[str] = None

    @property
    def active_version(self) -> PromptVersion:
        """返回生效版本（加载期已校验有且仅有一个 active）。"""
        for v in self.versions:
            if v.active:
                return v
        # 正常路径下 loader 已保证不会到这里
        raise ValueError(f"prompt '{self.id}' has no active version")
