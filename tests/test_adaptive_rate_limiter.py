import pytest
import time
from models.adaptive_rate_limiter import AdaptiveRateLimiter, TokenBucket

def test_token_bucket_refill_and_consume():
    bucket = TokenBucket(capacity=5, refill_rate=10.0)
    
    allowed, rem, _ = bucket.consume(3)
    assert allowed is True
    assert rem == 2

    allowed, rem, _ = bucket.consume(3)
    assert allowed is False

def test_adaptive_rate_limiter_client_isolation():
    limiter = AdaptiveRateLimiter(default_capacity=2, default_refill_rate=1.0)
    
    r1 = limiter.allow_request("client_A")
    r2 = limiter.allow_request("client_A")
    r3 = limiter.allow_request("client_A")
    
    assert r1["allowed"] is True
    assert r2["allowed"] is True
    assert r3["allowed"] is False
    assert r3["retry_after_seconds"] > 0

    # Client B should still have full quota
    r_b = limiter.allow_request("client_B")
    assert r_b["allowed"] is True
