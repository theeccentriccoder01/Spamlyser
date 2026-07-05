"""
Validator for custom rules schema and regular expression patterns.
Ensures that all loaded rules are syntactically and semantically valid before usage.
"""

import re
from typing import Any, Dict, List, Tuple


def validate_rule_structure(rule: dict[str, Any]) -> tuple[bool, str]:
    """Validate standard structure of a single rule."""
    if not isinstance(rule, dict):
        return False, "Rule must be a dictionary"

    required_keys = {"name", "pattern", "score", "description"}
    missing = required_keys - rule.keys()
    if missing:
        return False, f"Missing required keys: {', '.join(missing)}"

    if not isinstance(rule["name"], str) or not rule["name"].strip():
        return False, "Rule name must be a non-empty string"

    if not isinstance(rule["pattern"], str) or not rule["pattern"].strip():
        return False, "Rule pattern must be a non-empty string"

    try:
        re.compile(rule["pattern"])
    except re.error as e:
        return False, f"Invalid regex pattern: {e}"

    try:
        float(rule["score"])
    except (ValueError, TypeError):
        return False, "Rule score must be a number"

    return True, ""
