"""Neighbor / parent–child expand（Phase B1）+ Cross-ref expand（P1-B）。

对检索 top 命中按 page 或 prev/next 扩邻居块；可选解析「见表/见图」扩引用块。
默认均关闭。
"""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

# 内置交叉引用正则：(kind, pattern) — group 最后一组为编号
_DEFAULT_CROSSREF_PATTERNS: List[Tuple[str, re.Pattern]] = [
    (
        "table",
        re.compile(
            r"(?i)\b(?:see\s+)?(?:table|tbl\.?)\s*([A-Z]?\d+(?:[\-.]\d+)*)"
        ),
    ),
    (
        "figure",
        re.compile(
            r"(?i)\b(?:see\s+)?(?:figure|fig\.?)\s*([A-Z]?\d+(?:[\-.]\d+)*)"
        ),
    ),
    (
        "table",
        re.compile(r"(见|参见|参照)\s*表\s*([\d.\-]+)"),
    ),
    (
        "figure",
        re.compile(r"(见|参见|参照)\s*图\s*([\d.\-]+)"),
    ),
]


def extract_crossrefs(text: str, patterns: Optional[Sequence] = None) -> List[Tuple[str, str]]:
    """从文本抽出 (kind, id_str) 列表，去重保序。"""
    if not text:
        return []
    pats = patterns or _DEFAULT_CROSSREF_PATTERNS
    seen = set()
    out: List[Tuple[str, str]] = []
    for item in pats:
        if isinstance(item, tuple) and len(item) == 2 and hasattr(item[1], "finditer"):
            kind, pat = item
        else:
            # 配置传入的字符串 pattern
            kind, raw = ("ref", item) if isinstance(item, str) else item
            pat = re.compile(raw) if isinstance(raw, str) else raw
            if not isinstance(kind, str):
                kind = "ref"
        for m in pat.finditer(text):
            # 取最后一个捕获组作为编号
            if not m.groups():
                continue
            id_str = (m.group(m.lastindex) or "").strip()
            if not id_str:
                continue
            key = (kind, id_str.lower())
            if key in seen:
                continue
            seen.add(key)
            out.append((kind, id_str))
    return out


def _needles_for_ref(kind: str, id_str: str) -> List[str]:
    """生成 ILIKE/包含匹配用的 needle 列表。"""
    needles = []
    if kind == "table":
        needles.extend([
            f"Table {id_str}",
            f"TABLE {id_str}",
            f"Tbl. {id_str}",
            f"表 {id_str}",
            f"表{id_str}",
        ])
    elif kind == "figure":
        needles.extend([
            f"Figure {id_str}",
            f"Fig. {id_str}",
            f"Fig {id_str}",
            f"图 {id_str}",
            f"图{id_str}",
        ])
    else:
        needles.append(id_str)
    return needles


def expand_neighbors(
    results: Sequence[dict],
    pg_store: Any,
    *,
    mode: str = "page",
    max_extra: int = 2,
    cap: Optional[int] = None,
) -> Tuple[List[dict], Dict[str, Any]]:
    """扩展检索结果邻居。

    Args:
        results: 当前 hit 列表（dict 含 chunk_id, page_id 等）
        pg_store: 需提供 get_chunks_by_page_ids / get_chunks_by_ids
        mode: ``page`` | ``prev_next``
        max_extra: 每个 hit 最多新加几块
        cap: 结果总上限；None 则不截断（由调用方控制）

    Returns:
        (expanded_results, trace_info)
    """
    if not results or max_extra <= 0:
        return list(results), {"enabled": True, "added": 0, "mode": mode}

    seen = {r.get("chunk_id") for r in results if r.get("chunk_id")}
    out: List[dict] = list(results)
    added_ids: List[str] = []
    mode = (mode or "page").lower()

    if mode == "prev_next":
        added_ids = _expand_prev_next(results, pg_store, out, seen, max_extra)
    else:
        added_ids = _expand_page(results, pg_store, out, seen, max_extra)

    if cap is not None and cap > 0 and len(out) > cap:
        # 保序：原 hits 优先，再扩块
        primary = [r for r in out if r.get("chunk_id") not in set(added_ids)]
        extras = [r for r in out if r.get("chunk_id") in set(added_ids)]
        out = (primary + extras)[:cap]

    trace = {
        "enabled": True,
        "mode": mode,
        "max_extra": max_extra,
        "added": len(added_ids),
        "added_ids": added_ids[:20],
    }
    return out, trace


def _expand_page(
    results: Sequence[dict],
    pg_store: Any,
    out: List[dict],
    seen: set,
    max_extra: int,
) -> List[str]:
    page_ids = []
    for r in results:
        pid = r.get("page_id")
        if pid is not None and pid not in page_ids:
            page_ids.append(pid)
    if not page_ids:
        return []
    try:
        page_chunks = pg_store.get_chunks_by_page_ids(list(page_ids))
    except Exception as e:
        logger.warning("neighbor expand page fetch failed: %s", e)
        return []

    by_page: Dict[Any, List[dict]] = {}
    for ch in page_chunks:
        by_page.setdefault(ch.get("page_id"), []).append(ch)

    added: List[str] = []
    for r in results:
        pid = r.get("page_id")
        parent_score = float(r.get("rerank_score") or r.get("score") or 0.0)
        extras_for_hit = 0
        for ch in by_page.get(pid, []):
            if extras_for_hit >= max_extra:
                break
            cid = ch.get("chunk_id")
            if not cid or cid in seen:
                continue
            seen.add(cid)
            out.append(_as_result(ch, parent_score, r.get("chunk_id")))
            added.append(cid)
            extras_for_hit += 1
    return added


def _expand_prev_next(
    results: Sequence[dict],
    pg_store: Any,
    out: List[dict],
    seen: set,
    max_extra: int,
) -> List[str]:
    want: List[str] = []
    parent_of: Dict[str, str] = {}
    parent_score: Dict[str, float] = {}
    for r in results:
        pid = r.get("chunk_id")
        score = float(r.get("rerank_score") or r.get("score") or 0.0)
        for key in ("prev_chunk_id", "next_chunk_id"):
            nid = r.get(key) or ""
            if nid and nid not in seen:
                want.append(nid)
                parent_of[nid] = pid
                parent_score[nid] = score
        # 也允许从结果字段缺失时跳过
    if not want:
        # 结果可能未带 prev/next：尝试按 id 回表
        ids = [r.get("chunk_id") for r in results if r.get("chunk_id")]
        try:
            rows = pg_store.get_chunks_by_ids(ids) if ids else []
        except Exception as e:
            logger.warning("neighbor expand id fetch failed: %s", e)
            return []
        by_id = {x["chunk_id"]: x for x in rows}
        for r in results:
            row = by_id.get(r.get("chunk_id") or "")
            if not row:
                continue
            score = float(r.get("rerank_score") or r.get("score") or 0.0)
            for key in ("prev_chunk_id", "next_chunk_id"):
                nid = row.get(key) or ""
                if nid and nid not in seen:
                    want.append(nid)
                    parent_of[nid] = r.get("chunk_id")
                    parent_score[nid] = score

    # 每个 parent 限 max_extra
    per_parent: Dict[str, int] = {}
    filtered: List[str] = []
    for nid in want:
        p = parent_of.get(nid, "")
        if per_parent.get(p, 0) >= max_extra:
            continue
        per_parent[p] = per_parent.get(p, 0) + 1
        filtered.append(nid)

    if not filtered:
        return []
    try:
        neigh = pg_store.get_chunks_by_ids(filtered)
    except Exception as e:
        logger.warning("neighbor expand neighbor fetch failed: %s", e)
        return []

    added: List[str] = []
    for ch in neigh:
        cid = ch.get("chunk_id")
        if not cid or cid in seen:
            continue
        seen.add(cid)
        out.append(
            _as_result(
                ch,
                parent_score.get(cid, 0.0),
                parent_of.get(cid),
            )
        )
        added.append(cid)
    return added


def _as_result(
    ch: dict,
    parent_score: float,
    expanded_from: Optional[str],
    *,
    retrieval_type: str = "neighbor_expand",
    score_scale: float = 0.5,
) -> dict:
    return {
        "chunk_id": ch.get("chunk_id"),
        "page_id": ch.get("page_id"),
        "doc_id": ch.get("doc_id"),
        "page_number": ch.get("page_number"),
        "chunk_type": ch.get("chunk_type", "text"),
        "text": ch.get("text", ""),
        "doc_ref": ch.get("doc_ref", ""),
        "table_summary": ch.get("table_summary", ""),
        "section_path": ch.get("section_path", ""),
        "caption": ch.get("caption", ""),
        "prev_chunk_id": ch.get("prev_chunk_id", ""),
        "next_chunk_id": ch.get("next_chunk_id", ""),
        "score": parent_score * score_scale,
        "rerank_score": parent_score * score_scale,
        "retrieval_type": retrieval_type,
        "expanded_from": expanded_from or "",
    }


def expand_crossrefs(
    results: Sequence[dict],
    pg_store: Any,
    *,
    max_extra: int = 3,
    max_per_hit: int = 1,
    same_doc_only: bool = True,
    patterns: Optional[Sequence] = None,
    match_fields: Optional[Sequence[str]] = None,
    cap: Optional[int] = None,
) -> Tuple[List[dict], Dict[str, Any]]:
    """解析 hit 文本中的表/图引用并扩入目标 chunk。

    pg_store 需提供 ``find_chunks_by_ref(doc_id, needles, limit=...)``；
    若无该方法则 no-op。
    """
    if not results or max_extra <= 0:
        return list(results), {"enabled": True, "added": 0, "refs": []}

    if not hasattr(pg_store, "find_chunks_by_ref"):
        logger.warning("crossref expand: pg_store has no find_chunks_by_ref; skip")
        return list(results), {"enabled": True, "added": 0, "refs": [], "skipped": "no_api"}

    fields = list(match_fields or ("text", "caption", "table_summary", "section_path"))
    seen = {r.get("chunk_id") for r in results if r.get("chunk_id")}
    out: List[dict] = list(results)
    added_ids: List[str] = []
    found_refs: List[str] = []
    budget = max_extra

    for r in results:
        if budget <= 0:
            break
        blob = " ".join(str(r.get(f) or "") for f in fields)
        refs = extract_crossrefs(blob, patterns=patterns)
        if not refs:
            continue
        parent_score = float(r.get("rerank_score") or r.get("score") or 0.0)
        parent_id = r.get("chunk_id") or ""
        doc_id = r.get("doc_id") or ""
        if same_doc_only and not doc_id:
            continue

        per_hit = 0
        for kind, id_str in refs:
            if budget <= 0 or per_hit >= max_per_hit:
                break
            needles = _needles_for_ref(kind, id_str)
            found_refs.append(f"{kind}:{id_str}")
            try:
                matches = pg_store.find_chunks_by_ref(
                    doc_id if same_doc_only else None,
                    needles,
                    limit=3,
                )
            except Exception as e:
                logger.warning("crossref find failed: %s", e)
                continue
            for ch in matches or []:
                if budget <= 0 or per_hit >= max_per_hit:
                    break
                cid = ch.get("chunk_id")
                if not cid or cid in seen:
                    continue
                # 避免把引用句自身当目标
                if cid == parent_id:
                    continue
                seen.add(cid)
                out.append(
                    _as_result(
                        ch,
                        parent_score,
                        parent_id,
                        retrieval_type="crossref_expand",
                        score_scale=0.95,
                    )
                )
                added_ids.append(cid)
                per_hit += 1
                budget -= 1

    if cap is not None and cap > 0 and len(out) > cap:
        primary = [r for r in out if r.get("chunk_id") not in set(added_ids)]
        extras = [r for r in out if r.get("chunk_id") in set(added_ids)]
        out = (primary + extras)[:cap]

    trace = {
        "enabled": True,
        "added": len(added_ids),
        "added_ids": added_ids[:20],
        "refs": found_refs[:20],
        "max_extra": max_extra,
    }
    return out, trace
