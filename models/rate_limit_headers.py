"""
API Rate Limit Headers Middleware for Spamlyser
Generates standard HTTP RFC rate-limiting headers for web & API responses.
"""

import time
from typing import Dict, Tuple


class RateLimitHeaderGenerator:
    """Generates standard X-RateLimit-* and Retry-After response headers."""

    @staticmethod
    def get_headers(
        max_limit: int, remaining: int, reset_window_sec: int
    ) -> dict[str, str]:
        """Builds standard rate limit header dictionary."""
        reset_timestamp = int(time.time()) + reset_window_sec
        return {
            "X-RateLimit-Limit": str(max_limit),
            "X-RateLimit-Remaining": str(max(0, remaining)),
            "X-RateLimit-Reset": str(reset_timestamp),
        }

    @staticmethod
    def evaluate_request(
        current_count: int, max_limit: int, window_sec: int
    ) -> tuple[bool, dict[str, str]]:
        """Evaluates whether request exceeds limit and generates headers/Retry-After."""
        remaining = max_limit - current_count
        headers = RateLimitHeaderGenerator.get_headers(max_limit, remaining, window_sec)

        if current_count > max_limit:
            headers["Retry-After"] = str(window_sec)
            return False, headers
        return True, headers
