import os

import pytest

from models.custom_rules_manager import (
    check_custom_rules,
    load_custom_rules,
    save_custom_rules,
)

RULES_FILE = "custom_rules.json"


@pytest.fixture(autouse=True)
def setup_and_teardown_rules_file():
    # Save original content if it exists
    original_exists = os.path.exists(RULES_FILE)
    original_content = None
    if original_exists:
        with open(RULES_FILE, encoding="utf-8") as f:
            original_content = f.read()

    # Yield control to test
    yield

    # Restore original content
    if original_exists:
        with open(RULES_FILE, "w", encoding="utf-8") as f:
            f.write(original_content)
    elif os.path.exists(RULES_FILE):
        os.remove(RULES_FILE)


def test_load_default_rules():
    if os.path.exists(RULES_FILE):
        os.remove(RULES_FILE)
    rules = load_custom_rules()
    assert isinstance(rules, dict)
    assert "allowlist" in rules
    assert "blocklist" in rules
    assert len(rules["allowlist"]) == 0
    assert len(rules["blocklist"]) == 0


def test_save_and_load_rules():
    test_rules = {
        "allowlist": ["safe-sender.com", "internal-net.org"],
        "blocklist": ["\\bfree-claim\\b", "urgent-prize"],
    }
    assert save_custom_rules(test_rules) is True
    loaded = load_custom_rules()
    assert loaded == test_rules


def test_check_custom_rules_allowlist():
    test_rules = {"allowlist": ["trusted-partner.com"], "blocklist": ["\\bblock-me\\b"]}
    save_custom_rules(test_rules)

    # Message containing allowlisted domain
    res = check_custom_rules("Hello from client@trusted-partner.com, please reply.")
    assert res == "HAM"

    # Normal message
    res = check_custom_rules("Hello, how are you?")
    assert res is None


def test_check_custom_rules_blocklist():
    test_rules = {
        "allowlist": ["trusted-partner.com"],
        "blocklist": ["\\bwin-free-100k\\b", "click-now-scam"],
    }
    save_custom_rules(test_rules)

    # Message containing blocklist regex
    res = check_custom_rules("Congrats! You can win-free-100k today!")
    assert res == "SPAM"

    # Message containing blocklist keyword
    res = check_custom_rules("Urgent notification: click-now-scam link!")
    assert res == "SPAM"

    # Safe message
    res = check_custom_rules("Hello, this is a clean text.")
    assert res is None


def test_check_custom_rules_priority():
    # If a message matches both, allowlist is evaluated first
    test_rules = {
        "allowlist": ["safe-sender.com"],
        "blocklist": ["\\bwin-free-100k\\b"],
    }
    save_custom_rules(test_rules)

    res = check_custom_rules(
        "Safe email from safe-sender.com containing win-free-100k!"
    )
    assert res == "HAM"
