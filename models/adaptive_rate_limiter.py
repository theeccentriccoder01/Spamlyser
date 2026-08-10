"""
Thread-safe Token Bucket Rate Limiter with Adaptive Refill and Client Throttling Middleware.
"""

import time
import threading
from typing import Dict, Tuple, Optional, Any

class TokenBucket:
    """Individual client token bucket state."""

    def __init__(self, capacity: int, refill_rate: float):
        self.capacity = float(capacity)
        self.refill_rate = float(refill_rate)  # Tokens per second
        self.tokens = float(capacity)
        self.last_refill = time.time()
        self.lock = threading.Lock()

    def _refill(self):
        now = time.time()
        delta = now - self.last_refill
        self.tokens = min(self.capacity, self.tokens + delta * self.refill_rate)
        self.last_refill = now

    def consume(self, tokens: int = 1) -> Tuple[bool, int, float]:
        """
        Attempt to consume requested tokens.
        Returns: (allowed, remaining_tokens, retry_after_seconds)
        """
        with self.lock:
            self._refill()
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True, int(self.tokens), 0.0
            else:
                needed = tokens - self.tokens
                retry_after = needed / self.refill_rate if self.refill_rate > 0 else 1.0
                return False, int(self.tokens), round(retry_after, 2)

class AdaptiveRateLimiter:
    """
    Manages rate limits across multiple API clients / IP addresses with adaptive throttling.
    """

    def __init__(self, default_capacity: int = 10, default_refill_rate: float = 2.0):
        self.default_capacity = default_capacity
        self.default_refill_rate = default_refill_rate
        self.buckets: Dict[str, TokenBucket] = {}
        self.lock = threading.Lock()

    def get_bucket(self, client_id: str) -> TokenBucket:
        with self.lock:
            if client_id not in self.buckets:
                self.buckets[client_id] = TokenBucket(
                    capacity=self.default_capacity,
                    refill_rate=self.default_refill_rate
                )
            return self.buckets[client_id]

    def allow_request(self, client_id: str, cost: int = 1) -> Dict[str, Any]:
        """
        Check rate limit status for given client.
        """
        bucket = self.get_bucket(client_id)
        allowed, remaining, retry_after = bucket.consume(cost)

        return {
            "client_id": client_id,
            "allowed": allowed,
            "remaining_tokens": remaining,
            "retry_after_seconds": retry_after,
            "capacity": int(bucket.capacity)
        }
