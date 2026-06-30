"""Compatibility shim for tests expecting top-level threat_analyzer.

Re-exports selected symbols from models.threat_analyzer.
"""

from models.threat_analyzer import (
    THREAT_CATEGORIES,
    classify_threat_type,
    get_threat_specific_advice,
)

__all__ = [
    "THREAT_CATEGORIES",
    "classify_threat_type",
    "get_threat_specific_advice",
]
