"""集中式 Prompt 版本管理（方案 A：集中管理 + 版本记录）。

用法::

    from src.prompts import get_active

    pv = get_active("answer_generation")
    system_msg = pv.system
    user_msg = pv.render("user", context=ctx, query=q)

    tmpl = get_active("hyde").template
    prompt = tmpl.format(query=q)
"""
from .loader import PromptConfigError
from .models import Prompt, PromptVersion
from .registry import (
    PromptNotFound,
    get_active,
    get_prompt,
    init,
    list_prompts,
)

__all__ = [
    "Prompt",
    "PromptVersion",
    "PromptConfigError",
    "PromptNotFound",
    "get_active",
    "get_prompt",
    "list_prompts",
    "init",
]
