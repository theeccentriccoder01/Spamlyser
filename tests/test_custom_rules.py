"""Tests for models/custom_rules_manager.py."""

import importlib
import os
from unittest.mock import patch

import pytest

import models.custom_rules_manager as crm

# Redirect the rules file to a temp path so tests never touch production data.
_TEMP_RULES_KEY = "SPAMLYSER_CUSTOM_RULES"


@pytest.fixture(autouse=True)
def isolated_rules_file(tmp_path):
    """Reload the rules module against a per-test rules file path."""
    global crm
    rules_path = str(tmp_path / "custom_rules.json")
    with patch.dict(os.environ, {_TEMP_RULES_KEY: rules_path}):
        crm = importlib.reload(crm)
        crm._rules_file_path = lambda: rules_path
        yield rules_path
    crm = importlib.reload(crm)


def test_validate_rules_accepts_valid_structure():
    assert crm._validate_rules({"allowlist": [], "blocklist": []}) is True
    assert (
        crm._validate_rules({"allowlist": ["a.com"], "blocklist": [r"\bspam\b"]})
        is True
    )


def test_validate_rules_rejects_missing_keys():
    assert crm._validate_rules({"allowlist": []}) is False
    assert crm._validate_rules({"blocklist": []}) is False


def test_validate_rules_rejects_non_dict():
    assert crm._validate_rules("string") is False
    assert crm._validate_rules(None) is False
    assert crm._validate_rules([]) is False


def test_validate_rules_rejects_non_string_items():
    assert crm._validate_rules({"allowlist": [123], "blocklist": []}) is False


def test_load_default_rules_when_file_missing():
    rules = crm.load_custom_rules()
    assert isinstance(rules, dict)
    assert rules == {"allowlist": [], "blocklist": []}


def test_save_and_load_round_trip():
    test_rules = {
        "allowlist": ["safe-sender.com", "internal-net.org"],
        "blocklist": [r"\bfree-claim\b", "urgent-prize"],
    }
    assert crm.save_custom_rules(test_rules) is True
    loaded = crm.load_custom_rules()
    assert loaded == test_rules


def test_save_rejects_invalid_schema():
    result = crm.save_custom_rules({"allowlist": []})
    assert result is False


def test_save_rejects_non_dict():
    assert crm.save_custom_rules("bad") is False  # type: ignore[arg-type]


def test_save_warns_on_invalid_regex_but_still_saves(caplog):
    """Invalid regex patterns should warn without aborting the save."""
    import logging

    rules = {"allowlist": [], "blocklist": ["[invalid("]}
    with caplog.at_level(logging.WARNING, logger="models.custom_rules_manager"):
        result = crm.save_custom_rules(rules)
    assert result is True
    assert "invalid" in caplog.text.lower() or "Skipping" in caplog.text


def test_allowlist_matched_first():
    crm.save_custom_rules(
        {"allowlist": ["trusted-partner.com"], "blocklist": [r"\bblock-me\b"]}
    )
    result = crm.check_custom_rules(
        "Hello from client@trusted-partner.com, please reply."
    )
    assert result == "HAM"


def test_allowlist_no_match_returns_none():
    crm.save_custom_rules({"allowlist": ["trusted-partner.com"], "blocklist": []})
    assert crm.check_custom_rules("Hello, how are you?") is None


def test_blocklist_regex_matched():
    crm.save_custom_rules(
        {"allowlist": [], "blocklist": [r"\bwin-free-100k\b", "click-now-scam"]}
    )
    assert crm.check_custom_rules("Congrats! You can win-free-100k today!") == "SPAM"


def test_blocklist_keyword_matched():
    crm.save_custom_rules({"allowlist": [], "blocklist": ["click-now-scam"]})
    assert crm.check_custom_rules("Urgent: click-now-scam link!") == "SPAM"


def test_allowlist_takes_priority_over_blocklist():
    crm.save_custom_rules(
        {"allowlist": ["safe-sender.com"], "blocklist": [r"\bwin-free-100k\b"]}
    )
    result = crm.check_custom_rules(
        "Safe email from safe-sender.com containing win-free-100k!"
    )
    assert result == "HAM"


def test_clean_message_returns_none():
    crm.save_custom_rules({"allowlist": [], "blocklist": [r"\bspam\b"]})
    assert crm.check_custom_rules("Hello, this is a clean message.") is None


def test_invalid_regex_in_blocklist_skipped_gracefully():
    """A broken regex should not crash check_custom_rules; it is skipped."""
    crm.save_custom_rules({"allowlist": [], "blocklist": ["[invalid(", "spam-keyword"]})
    assert crm.check_custom_rules("This is a spam-keyword message") == "SPAM"
    assert crm.check_custom_rules("A totally harmless message") is None
