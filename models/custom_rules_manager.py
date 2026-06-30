"""
Custom rules manager for Spamlyser Pro.
Handles allowlist (domains) and blocklist (regexes) checking and persistence.
"""

import os
import re

from .storage_manager import StorageManager

RULES_FILE = "custom_rules.json"

_storage = StorageManager()


def load_custom_rules():
    if not os.path.exists(RULES_FILE):
        return {"allowlist": [], "blocklist": []}
    return _storage.load_json(RULES_FILE, default={"allowlist": [], "blocklist": []})


def save_custom_rules(rules):
    return _storage.save_json(RULES_FILE, rules, backup=True)


def check_custom_rules(text: str) -> str:
    """
    Check if text matches any blocklist (regex) or allowlist (domain name) rules.
    Returns:
        "SPAM" if it matches a blocklist rule.
        "HAM" if it matches an allowlist rule.
        None if no match.
    """
    rules = load_custom_rules()

    # 1. Check Allowlist (Domains/Keywords)
    for domain in rules.get("allowlist", []):
        if domain.strip() and domain.lower() in text.lower():
            return "HAM"

    # 2. Check Blocklist (Regexes/Keywords)
    for pattern_str in rules.get("blocklist", []):
        if pattern_str.strip():
            try:
                # Compile and check regex
                pattern = re.compile(pattern_str, re.IGNORECASE)
                if pattern.search(text):
                    return "SPAM"
            except Exception:
                continue

    return None
