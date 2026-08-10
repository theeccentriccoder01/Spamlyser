from models.rate_limit_headers import RateLimitHeaderGenerator


def test_rate_limit_headers_allow():
    allowed, headers = RateLimitHeaderGenerator.evaluate_request(
        current_count=5, max_limit=10, window_sec=60
    )
    assert allowed is True
    assert headers["X-RateLimit-Limit"] == "10"
    assert headers["X-RateLimit-Remaining"] == "5"
    assert "Retry-After" not in headers


def test_rate_limit_headers_exceeded():
    allowed, headers = RateLimitHeaderGenerator.evaluate_request(
        current_count=11, max_limit=10, window_sec=60
    )
    assert allowed is False
    assert headers["X-RateLimit-Remaining"] == "0"
    assert headers["Retry-After"] == "60"
