"""
Custom rules manager for Spamlyser Pro.
Handles allowlist (domains) and blocklist (regexes) checking and persistence.
"""

import json
import os
import re

RULES_FILE = "custom_rules.json"

def load_custom_rules():
    if not os.path.exists(RULES_FILE):
        # Default empty structure if file doesn't exist
        return {"allowlist": [], "blocklist": []}
    try:
        with open(RULES_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"allowlist": [], "blocklist": []}

def save_custom_rules(rules):
    try:
        with open(RULES_FILE, "w", encoding="utf-8") as f:
            json.dump(rules, f, indent=2)
        return True
    except Exception:
        return False

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
