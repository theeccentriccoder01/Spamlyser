import time

class RateLimiter:
    def __init__(self, max_requests: int, window_seconds: float):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = []
        
    def allow_request(self) -> bool:
        now = time.time()
        self.requests = [r for r in self.requests if now - r < self.window_seconds]
        if len(self.requests) < self.max_requests:
            self.requests.append(now)
            return True
        return False
