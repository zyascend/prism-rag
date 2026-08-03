"""In-process MemorySaver singleton for agent checkpoint / HITL resume.

Same checkpointer instance is required across interrupt and resume within a process.
Multi-worker durable checkpoints are out of scope (MemorySaver only).
"""
from __future__ import annotations

from typing import Any, Optional

from langgraph.checkpoint.memory import MemorySaver

__all__ = ["get_memory_saver", "reset_memory_saver"]

_SAVER: Optional[MemorySaver] = None


def get_memory_saver() -> MemorySaver:
    """Return process-wide MemorySaver (create on first use)."""
    global _SAVER
    if _SAVER is None:
        _SAVER = MemorySaver()
    return _SAVER


def reset_memory_saver() -> None:
    """Drop singleton — for tests that need a clean store."""
    global _SAVER
    _SAVER = None
