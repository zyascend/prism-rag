"""查询意图启发式 + 模态轻 boost（Phase B2）。

纯规则、无 LLM；默认关闭，由 retrieval.modality_boost.enabled 控制。
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Sequence

# 表/规格/数值查找
_TABLE_CUES = re.compile(
    r"\b("
    r"table|tables|column|columns|row|rows|spec|specification|parameter|parameters|"
    r"limit|limits|range|maximum|minimum|allowed|threshold|psi|pressure|rpm|"
    r"voltage|current|frequency|temperature|composition|percent|rating|"
    r"schedule|form\s+\d+|afto"
    r")\b|"
    r"(最大值|最小值|参数|规格|表格|压力|频率|阈值)",
    re.I,
)

# 图/示意/版式
_VISUAL_CUES = re.compile(
    r"\b("
    r"figure|figures|fig\.?|diagram|diagrams|chart|charts|graph|graphs|"
    r"schematic|drawing|wiring|layout|illustration|image|images|plot|"
    r"see\s+(the\s+)?(figure|diagram|chart)|"
    r"in\s+the\s+(figure|diagram|chart|graph)"
    r")\b|"
    r"(示意图|接线图|图\s*\d+|见图|版图)",
    re.I,
)


@dataclass(frozen=True)
class QueryIntent:
    table: bool = False
    visual: bool = False

    @property
    def label(self) -> str:
        tags = []
        if self.table:
            tags.append("table")
        if self.visual:
            tags.append("visual")
        return "+".join(tags) if tags else "none"


def detect_query_intent(query: str) -> QueryIntent:
    q = query or ""
    return QueryIntent(
        table=bool(_TABLE_CUES.search(q)),
        visual=bool(_VISUAL_CUES.search(q)),
    )


def apply_modality_boost(
    results: Sequence[dict],
    intent: QueryIntent,
    *,
    table_bonus: float = 0.02,
    image_bonus: float = 0.02,
) -> List[dict]:
    """对 fused 结果按 chunk_type 加分并重排。

    bonus 加在 score 上（RRF 量级通常 < 0.1，0.02 为轻推）。
    intent 无对应标志时原样返回（不改序）。
    """
    if not results:
        return list(results)
    if not intent.table and not intent.visual:
        return list(results)
    if table_bonus == 0 and image_bonus == 0:
        return list(results)

    out: List[dict] = []
    for r in results:
        item = dict(r)
        ctype = (item.get("chunk_type") or "text").lower()
        bonus = 0.0
        if intent.table and ctype == "table":
            bonus += table_bonus
        if intent.visual and ctype in ("image", "table"):
            # visual 意图也轻推 table（图注表常见）；image 锚点优先
            if ctype == "image":
                bonus += image_bonus
            elif ctype == "table":
                bonus += image_bonus * 0.5
        if bonus:
            item["score"] = float(item.get("score") or 0.0) + bonus
            item["modality_boost"] = bonus
        out.append(item)

    out.sort(key=lambda x: float(x.get("score") or 0.0), reverse=True)
    return out
