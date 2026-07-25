"""规则自适应 Search Planning（P1-A）。

动作空间（MVP）:
  - a_full   : 三路全开（planning 关 / always_full）
  - a_text   : bm25+dense，visual=false
  - a_visual : bm25+dense+visual
  - a_none   : skip_retrieval（allow_skip_retrieval 且闲聊；工业默认关）

默认 enabled=false → 调用方 use_* 原样（兼容现状）。
cue 源统一自 query_intent.detect_query_intent。
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, Optional

from src.retrieval.query_intent import QueryIntent, detect_query_intent

# 明显无文档需求的闲聊（仅 allow_skip_retrieval=true 时生效）
_CHITCHAT = re.compile(
    r"^\s*("
    r"hi|hello|hey|thanks|thank\s+you|bye|goodbye|"
    r"你好|谢谢|再见"
    r")[\s!.?？！]*$",
    re.I,
)


@dataclass(frozen=True)
class SearchPlan:
    use_bm25: bool
    use_dense: bool
    use_visual: bool
    skip_retrieval: bool
    intent_label: str
    reason: str

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


def search_planning_config(get_cfg: Optional[Callable] = None) -> Dict[str, Any]:
    """读取 retrieval.search_planning；测试可注入 get_cfg。"""
    if get_cfg is None:
        from src.config import cfg

        get_cfg = cfg.get
    try:
        raw = get_cfg("retrieval.search_planning", {}) or {}
    except Exception:
        raw = {}
    if not isinstance(raw, dict):
        raw = {}
    visual = raw.get("visual") if isinstance(raw.get("visual"), dict) else {}
    return {
        "enabled": bool(raw.get("enabled", False)),
        "mode": str(raw.get("mode") or "heuristic"),
        "allow_skip_retrieval": bool(raw.get("allow_skip_retrieval", False)),
        "table_prefers_text": bool(raw.get("table_prefers_text", True)),
        "visual": {
            "on_cues": bool(visual.get("on_cues", True)),
            "default_visual": bool(visual.get("default_visual", False)),
        },
    }


def plan_search(
    query: str,
    *,
    use_bm25: bool = True,
    use_dense: bool = True,
    use_visual: bool = True,
    cfg: Optional[Dict[str, Any]] = None,
    intent: Optional[QueryIntent] = None,
) -> SearchPlan:
    """根据规则生成 SearchPlan。

    planning disabled → 原样透传 requested flags。
    mode=always_full → 强制三路（在 requested 允许的范围内：若请求 use_bm25=False 仍尊重？）
      消融要求：always_full 在 use_* True 时全开。这里对 requested=False 仍尊重，
      以便 AblationConfig 关路。
    """
    c = cfg or search_planning_config()
    intent = intent if intent is not None else detect_query_intent(query)
    label = intent.label

    if not c.get("enabled"):
        return SearchPlan(
            use_bm25=use_bm25,
            use_dense=use_dense,
            use_visual=use_visual,
            skip_retrieval=False,
            intent_label=label,
            reason="planning_disabled",
        )

    mode = str(c.get("mode") or "heuristic")
    if mode == "always_full":
        return SearchPlan(
            use_bm25=use_bm25,
            use_dense=use_dense,
            use_visual=use_visual,
            skip_retrieval=False,
            intent_label=label,
            reason="always_full",
        )

    if mode == "text_only":
        return SearchPlan(
            use_bm25=use_bm25,
            use_dense=use_dense,
            use_visual=False,
            skip_retrieval=False,
            intent_label=label,
            reason="text_only_mode",
        )

    # heuristic
    if c.get("allow_skip_retrieval") and _CHITCHAT.match(query or ""):
        return SearchPlan(
            use_bm25=False,
            use_dense=False,
            use_visual=False,
            skip_retrieval=True,
            intent_label=label,
            reason="chitchat_skip",
        )

    vis_cfg = c.get("visual") or {}
    on_cues = bool(vis_cfg.get("on_cues", True))
    default_visual = bool(vis_cfg.get("default_visual", False))

    effective_visual = False
    reason = "text_default"
    if not use_visual:
        effective_visual = False
        reason = "request_visual_off"
    elif on_cues and intent.visual:
        effective_visual = True
        reason = "visual_cue"
    elif default_visual:
        effective_visual = True
        reason = "default_visual_on"
    else:
        effective_visual = False
        if intent.table and c.get("table_prefers_text", True):
            reason = "table_prefers_text"
        else:
            reason = "no_visual_cue"

    return SearchPlan(
        use_bm25=use_bm25,
        use_dense=use_dense,
        use_visual=bool(use_visual and effective_visual),
        skip_retrieval=False,
        intent_label=label,
        reason=reason,
    )


def build_planner_enabled(get_cfg: Optional[Callable] = None) -> bool:
    return bool(search_planning_config(get_cfg).get("enabled"))
