"""PrismRAG diagnostics — failure pattern catalog + clinic helpers."""

from src.diagnostics.failure_patterns import (
    FAILURE_PATTERNS,
    FailurePattern,
    get_pattern,
    list_patterns,
)
from src.diagnostics.failure_clinic import (
    diagnose_failure,
    format_diagnosis_markdown,
    parse_diagnosis_json,
)

__all__ = [
    "FAILURE_PATTERNS",
    "FailurePattern",
    "get_pattern",
    "list_patterns",
    "diagnose_failure",
    "format_diagnosis_markdown",
    "parse_diagnosis_json",
]
