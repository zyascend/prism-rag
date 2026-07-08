"""编码进度状态管理 — 支持中断恢复与进度条

设计要点：
  - save_state 只存计数不存完整 page_ids 列表（避免 JSON 膨胀）
  - page_embeddings_cache.pkl 增量追加（不每次全量重写）
  - load_state 时从 cache 文件推导实际完成的 page_ids
"""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from typing import Dict

import torch

logger = logging.getLogger(__name__)

STATE_DIR = Path("indexes")
STATE_FILE = STATE_DIR / "ingest_state.json"
EMBEDDING_CACHE = STATE_DIR / "page_embeddings_cache.pkl"


def _ensure_dir():
    STATE_DIR.mkdir(parents=True, exist_ok=True)


def save_state(text_phase_done: bool = False, completed_count: int = 0):
    """保存编码进度到 JSON（只存计数，不存完整列表）"""
    _ensure_dir()
    state = {
        "text_phase_done": text_phase_done,
        "completed_count": completed_count,
    }
    with open(STATE_FILE, "w") as f:
        json.dump(state, f)
    logger.info(f"💾 进度已保存: {completed_count}/5244 页")


def load_state() -> dict:
    """加载编码进度"""
    if not STATE_FILE.exists():
        return {"text_phase_done": False, "completed_count": 0}
    with open(STATE_FILE) as f:
        return json.load(f)


def save_page_embeddings(embeddings: Dict[int, torch.Tensor]):
    """增量追加页面嵌入到缓存文件（不重写）"""
    _ensure_dir()
    mode = "ab" if EMBEDDING_CACHE.exists() else "wb"
    with open(EMBEDDING_CACHE, mode) as f:
        pickle.dump(embeddings, f)
    logger.debug(f"  💾 {len(embeddings)} 页嵌入已追加")


def load_page_embeddings() -> Dict[int, torch.Tensor]:
    """从增量缓存加载全部页面嵌入"""
    if not EMBEDDING_CACHE.exists():
        return {}
    all_embeddings: Dict[int, torch.Tensor] = {}
    with open(EMBEDDING_CACHE, "rb") as f:
        while True:
            try:
                batch = pickle.load(f)
                all_embeddings.update(batch)
            except EOFError:
                break
    logger.info(f"  📂 从缓存加载了 {len(all_embeddings)} 页嵌入")
    return all_embeddings


def clear_progress():
    """清除进度状态"""
    for p in [STATE_FILE, EMBEDDING_CACHE]:
        if p.exists():
            p.unlink()
    logger.info("🧹 编码进度已清除")
