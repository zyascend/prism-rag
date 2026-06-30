"""编码进度状态管理 — 支持中断恢复与进度条"""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import torch

logger = logging.getLogger(__name__)

STATE_DIR = Path("indexes")
STATE_FILE = STATE_DIR / "ingest_state.json"
EMBEDDING_CACHE = STATE_DIR / "page_embeddings_cache.pkl"


def _ensure_dir():
    STATE_DIR.mkdir(parents=True, exist_ok=True)


def save_state(text_phase_done: bool = False, completed_page_ids: Optional[Set[int]] = None):
    """保存编码进度到 JSON"""
    _ensure_dir()
    state = {"text_phase_done": text_phase_done}
    if completed_page_ids is not None:
        state["completed_page_ids"] = sorted(completed_page_ids)
    with open(STATE_FILE, "w") as f:
        json.dump(state, f)
    logger.info(f"💾 进度已保存: {len(completed_page_ids or [])}/{state.get('completed_page_ids', [])} 页完成")


def load_state() -> dict:
    """加载编码进度"""
    if not STATE_FILE.exists():
        return {"text_phase_done": False, "completed_page_ids": []}
    with open(STATE_FILE) as f:
        return json.load(f)


def save_page_embeddings(embeddings: Dict[int, torch.Tensor]):
    """增量保存页面嵌入到缓存文件"""
    _ensure_dir()
    existing = {}
    if EMBEDDING_CACHE.exists():
        with open(EMBEDDING_CACHE, "rb") as f:
            existing = pickle.load(f)
    existing.update(embeddings)
    with open(EMBEDDING_CACHE, "wb") as f:
        pickle.dump(existing, f)
    logger.debug(f"  💾 {len(embeddings)} 页嵌入已缓存 ({len(existing)} 页总计)")


def load_page_embeddings() -> Dict[int, torch.Tensor]:
    """加载缓存的所有页面嵌入"""
    if not EMBEDDING_CACHE.exists():
        return {}
    with open(EMBEDDING_CACHE, "rb") as f:
        return pickle.load(f)


def clear_progress():
    """清除进度状态（全新开始的导入使用）"""
    for p in [STATE_FILE, EMBEDDING_CACHE]:
        if p.exists():
            p.unlink()
    logger.info("🧹 编码进度已清除")
