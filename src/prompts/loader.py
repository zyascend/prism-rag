"""从 YAML 文件加载并校验 Prompt。"""
from __future__ import annotations

from pathlib import Path
from typing import Union

import yaml

from .models import META_KEYS, Prompt, PromptVersion


class PromptConfigError(Exception):
    """Prompt YAML 结构非法（缺字段、无/多 active、版本号重复等）。"""


def load_prompt_file(path: Union[str, Path]) -> Prompt:
    """解析单个 prompt YAML 文件为 ``Prompt``。

    校验规则：
    - 根为映射且含非空 ``id``；
    - ``versions`` 为非空列表；
    - 每个版本含正整数 ``version`` 且文件内唯一；
    - 每个版本至少含一个模板正文字段（非元数据键）；
    - 全文件恰好一个 ``active: true``。
    """
    path = Path(path)
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as e:
        raise PromptConfigError(f"{path}: invalid YAML: {e}") from e

    if not isinstance(data, dict):
        raise PromptConfigError(f"{path}: root must be a mapping")

    pid = data.get("id")
    if not pid or not isinstance(pid, str):
        raise PromptConfigError(f"{path}: missing or invalid 'id'")

    raw_versions = data.get("versions")
    if not isinstance(raw_versions, list) or not raw_versions:
        raise PromptConfigError(f"{pid}: 'versions' must be a non-empty list")

    versions = []
    seen: set = set()
    for rv in raw_versions:
        if not isinstance(rv, dict):
            raise PromptConfigError(f"{pid}: each version must be a mapping")
        v = rv.get("version")
        if not isinstance(v, int):
            raise PromptConfigError(f"{pid}: version entry missing integer 'version'")
        if v in seen:
            raise PromptConfigError(f"{pid}: duplicate version {v}")
        seen.add(v)

        body = {k: val for k, val in rv.items() if k not in META_KEYS}
        if not body:
            raise PromptConfigError(f"{pid} v{v}: no template body fields")
        # 正文字段必须是字符串
        for k, val in body.items():
            if not isinstance(val, str):
                raise PromptConfigError(f"{pid} v{v}: field '{k}' must be a string")

        versions.append(
            PromptVersion(
                prompt_id=pid,
                version=v,
                active=bool(rv.get("active", False)),
                created_at=str(rv.get("created_at", "")),
                changelog=str(rv.get("changelog", "")),
                author=rv.get("author"),
                body=body,
            )
        )

    actives = [v for v in versions if v.active]
    if len(actives) != 1:
        raise PromptConfigError(
            f"{pid}: exactly one active version required, got {len(actives)}"
        )

    return Prompt(id=pid, versions=versions, description=data.get("description"))
