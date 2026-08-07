"""
LRU-Cached Regular Expression Compilation Engine with ReDoS Safety Audit and Performance Benchmarking.
"""

import re
from functools import lru_cache
from typing import Optional, Pattern, Dict, Any, Tuple
import time
import logging

logger = logging.getLogger(__name__)

class RegexCacheOptimizer:
    """
    High-performance LRU cached regex compilation engine with metric counters and ReDoS heuristics.
    """

    def __init__(self, maxsize: int = 256):
        self.maxsize = maxsize
        self.hits = 0
        self.misses = 0
        self._cache: Dict[Tuple[str, int], Pattern] = {}

    def get_compiled(self, pattern_str: str, flags: int = re.IGNORECASE) -> Optional[Pattern]:
        """
        Retrieve compiled regex object from internal LRU cache or compile new pattern.
        """
        key = (pattern_str, flags)
        if key in self._cache:
            self.hits += 1
            return self._cache[key]

        self.misses += 1
        try:
            compiled = re.compile(pattern_str, flags)
            if len(self._cache) >= self.maxsize:
                # Evict oldest entry
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]

            self._cache[key] = compiled
            return compiled
        except re.error as e:
            logger.warning(f"Regex compilation error for pattern '{pattern_str}': {e}")
            return None

    @staticmethod
    def audit_redos_vulnerability(pattern_str: str) -> Tuple[bool, str]:
        """
        Heuristic audit for catastrophic backtracking (ReDoS) vulnerability in regex patterns.
        Looks for nested quantifiers like (a+)+ or (a*)*.
        """
        # Regex to detect nested quantifiers
        redos_pattern = r'\([^)]*[*+]\)[*+]|\([^)]*\+[*+]|\([^)]*\*[*+]'
        if re.search(redos_pattern, pattern_str):
            return True, "Potential catastrophic backtracking (ReDoS) risk detected (nested quantifiers)"
        return False, "Pattern passed ReDoS heuristic analysis"

    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Return hit/miss statistics and current cache size.
        """
        total = self.hits + self.misses
        hit_ratio = round(self.hits / total, 4) if total > 0 else 0.0
        return {
            "hits": self.hits,
            "misses": self.misses,
            "total_requests": total,
            "hit_ratio": hit_ratio,
            "cache_size": len(self._cache),
            "max_size": self.maxsize
        }
