import re

def is_safe_regex_input(text: str, max_len: int = 1000) -> bool:
    """Checks input lengths and prevents catastrophic backtracking patterns."""
    if len(text) > max_len:
        return False
    # Avoid massive repetitive nested groups
    if any(pattern in text for pattern in ["(a+)+", "([a-zA-Z]+)*"]):
        return False
    return True
