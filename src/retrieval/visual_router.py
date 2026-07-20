"""Visual 路按需路由：非图表/表格意图 query 可跳过 Visual，省延迟与 GPU。

Boot-B 对照：mode=always（基线）vs heuristic。
配置：retrieval.visual_routing.enabled / .mode
"""

from __future__ import annotations

import re

_VISUAL_CUES = re.compile(
    r"\b("
    r"table|tables|figure|figures|fig\.?|diagram|diagrams|chart|charts|"
    r"graph|graphs|schematic|drawing|image|images|plot|plots|"
    r"illustration|screenshot|"
    r"page\s+\d+|"
    r"see\s+(the\s+)?(figure|table|diagram|chart)|"
    r"according\s+to\s+(the\s+)?(figure|table|diagram|chart)|"
    r"in\s+the\s+(figure|table|diagram|chart|graph)"
    r")\b",
    re.I,
)


class VisualRouter:
    """Decide whether the Visual retrieval route should run for a query."""

    def __init__(self, mode: str = "heuristic") -> None:
        if mode not in ("heuristic", "always", "never"):
            raise ValueError(
                f"visual routing mode must be heuristic|always|never, got {mode!r}"
            )
        self.mode = mode

    def should_use_visual(self, query: str) -> bool:
        if self.mode == "always":
            return True
        if self.mode == "never":
            return False
        return bool(_VISUAL_CUES.search(query or ""))


def build_visual_router_from_config(get_cfg) -> VisualRouter | None:
    """Return VisualRouter if retrieval.visual_routing.enabled, else None.

    get_cfg: callable like cfg.get (dotted path, default).
    """
    enabled = bool(get_cfg("retrieval.visual_routing.enabled", False))
    if not enabled:
        return None
    mode = str(get_cfg("retrieval.visual_routing.mode", "heuristic"))
    return VisualRouter(mode=mode)
