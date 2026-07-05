"""Tests for models/custom_rules_manager.py."""

import os
from unittest.mock import patch

import pytest

from models.custom_rules_manager import (
    _validate_rules,
    check_custom_rules,
    load_custom_rules,
    save_custom_rules,
)

# We redirect the rules file to a temp path so tests never touch production data.
_TEMP_RULES_KEY = "SPAMLYSER_CUSTOM_RULES"


@pytest.fixture(autouse=True)
def isolated_rules_file(tmp_path):
    """Redirect all custom-rules I/O to a temporary directory."""
    rules_path = str(tmp_path / "custom_rules.json")
    with patch.dict(os.environ, {_TEMP_RULES_KEY: rules_path}):
        # Reload the path-resolution function so it picks up the env change
        import importlib

        import models.custom_rules_manager as crm

        importlib.reload(crm)
        yield rules_path
    # Reload once more after the test to restore the module state
    import importlib

    import models.custom_rules_manager as crm

    importlib.reload(crm)


# ── _validate_rules ──────────────────────────────────────────────────────────


def test_validate_rules_accepts_valid_structure():
    assert _validate_rules({"allowlist": [], "blocklist": []}) is True
    assert _validate_rules({"allowlist": ["a.com"], "blocklist": [r"\bspam\b"]}) is True


def test_validate_rules_rejects_missing_keys():
    assert _validate_rules({"allowlist": []}) is False
    assert _validate_rules({"blocklist": []}) is False


def test_validate_rules_rejects_non_dict():
    assert _validate_rules("string") is False
    assert _validate_rules(None) is False
    assert _validate_rules([]) is False


def test_validate_rules_rejects_non_string_items():
    assert _validate_rules({"allowlist": [123], "blocklist": []}) is False


# ── load_custom_rules ────────────────────────────────────────────────────────


def test_load_default_rules_when_file_missing():
    rules = load_custom_rules()
    assert isinstance(rules, dict)
    assert rules == {"allowlist": [], "blocklist": []}


def test_save_and_load_round_trip():
    test_rules = {
        "allowlist": ["safe-sender.com", "internal-net.org"],
        "blocklist": [r"\bfree-claim\b", "urgent-prize"],
    }
    assert save_custom_rules(test_rules) is True
    loaded = load_custom_rules()
    assert loaded == test_rules


# ── save_custom_rules ────────────────────────────────────────────────────────


def test_save_rejects_invalid_schema():
    result = save_custom_rules({"allowlist": []})  # missing blocklist
    assert result is False


def test_save_rejects_non_dict():
    assert save_custom_rules("bad") is False  # type: ignore[arg-type]


def test_save_warns_on_invalid_regex_but_still_saves(caplog):
    """Invalid regex patterns should produce a warning but not abort the save."""
    import logging

    rules = {"allowlist": [], "blocklist": ["[invalid("]}
    with caplog.at_level(logging.WARNING, logger="models.custom_rules_manager"):
        result = save_custom_rules(rules)
    assert result is True
    assert "invalid" in caplog.text.lower() or "Skipping" in caplog.text


# ── check_custom_rules ───────────────────────────────────────────────────────


def test_allowlist_matched_first():
    save_custom_rules(
        {"allowlist": ["trusted-partner.com"], "blocklist": [r"\bblock-me\b"]}
    )
    result = check_custom_rules("Hello from client@trusted-partner.com, please reply.")
    assert result == "HAM"


def test_allowlist_no_match_returns_none():
    save_custom_rules({"allowlist": ["trusted-partner.com"], "blocklist": []})
    assert check_custom_rules("Hello, how are you?") is None


def test_blocklist_regex_matched():
    save_custom_rules(
        {"allowlist": [], "blocklist": [r"\bwin-free-100k\b", "click-now-scam"]}
    )
    assert check_custom_rules("Congrats! You can win-free-100k today!") == "SPAM"


def test_blocklist_keyword_matched():
    save_custom_rules({"allowlist": [], "blocklist": ["click-now-scam"]})
    assert check_custom_rules("Urgent: click-now-scam link!") == "SPAM"


def test_allowlist_takes_priority_over_blocklist():
    save_custom_rules(
        {"allowlist": ["safe-sender.com"], "blocklist": [r"\bwin-free-100k\b"]}
    )
    result = check_custom_rules(
        "Safe email from safe-sender.com containing win-free-100k!"
    )
    assert result == "HAM"


def test_clean_message_returns_none():
    save_custom_rules({"allowlist": [], "blocklist": [r"\bspam\b"]})
    assert check_custom_rules("Hello, this is a clean message.") is None


def test_invalid_regex_in_blocklist_skipped_gracefully():
    """A broken regex should not crash check_custom_rules — it is skipped."""
    save_custom_rules({"allowlist": [], "blocklist": ["[invalid(", "spam-keyword"]})
    # The valid pattern should still match
    assert check_custom_rules("This is a spam-keyword message") == "SPAM"
    # And a clean message is still None
    assert check_custom_rules("A totally harmless message") is None


def test_rules_validator_invalid_regex():
    from models.rules_validator import validate_rule_structure

    invalid_rule = {
        "name": "Invalid Regex Rule",
        "pattern": "[a-z",
        "score": 1.5,
        "description": "Unclosed character class",
    }
    ok, err = validate_rule_structure(invalid_rule)
    assert not ok
    assert "Invalid regex pattern" in err
