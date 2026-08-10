import pytest
import re
from models.regex_cache_optimizer import RegexCacheOptimizer

def test_regex_cache_hits_and_misses():
    optimizer = RegexCacheOptimizer(maxsize=10)
    p1 = "free.*money"
    
    comp1 = optimizer.get_compiled(p1)
    assert comp1 is not None
    assert optimizer.misses == 1
    assert optimizer.hits == 0

    comp2 = optimizer.get_compiled(p1)
    assert comp2 is comp1
    assert optimizer.hits == 1

    stats = optimizer.get_cache_stats()
    assert stats["hit_ratio"] == 0.5
    assert stats["cache_size"] == 1

def test_regex_redos_audit():
    optimizer = RegexCacheOptimizer()
    
    safe_pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
    is_vulnerable, msg = optimizer.audit_redos_vulnerability(safe_pattern)
    assert is_vulnerable is False

    unsafe_pattern = r"(a+)+"
    is_vulnerable_unsafe, _ = optimizer.audit_redos_vulnerability(unsafe_pattern)
    assert is_vulnerable_unsafe is True
