"""Tests for the CSV injection sanitizer module."""

import pytest

from models.csv_sanitizer import is_formula_payload, sanitize_cell, sanitize_row


class TestSanitizeCell:
    """Verify that formula-triggering prefixes are neutralized."""

    def test_plain_text_unchanged(self):
        assert sanitize_cell("Hello world") == "Hello world"

    def test_numeric_value_unchanged(self):
        assert sanitize_cell(42) == 42
        assert sanitize_cell(3.14) == 3.14

    def test_none_unchanged(self):
        assert sanitize_cell(None) is None

    def test_equals_prefix_escaped(self):
        result = sanitize_cell("=CMD('calc')")
        assert result.startswith("'")
        assert "CMD" in result

    def test_plus_prefix_escaped(self):
        result = sanitize_cell("+1-555-1234")
        assert result.startswith("'")

    def test_at_prefix_escaped(self):
        result = sanitize_cell("@SUM(A1:A10)")
        assert result.startswith("'")

    def test_pipe_prefix_escaped(self):
        result = sanitize_cell("|cmd /c calc")
        assert result.startswith("'")

    def test_tab_prefix_escaped(self):
        result = sanitize_cell("\t=CMD('calc')")
        assert result.startswith("'")

    def test_newline_collapsed(self):
        result = sanitize_cell("line1\nline2")
        assert "\n" not in result
        assert "line1 line2" == result

    def test_carriage_return_collapsed(self):
        result = sanitize_cell("line1\r\nline2")
        assert "\r" not in result
        assert "\n" not in result

    def test_null_byte_stripped(self):
        result = sanitize_cell("hello\x00world")
        assert "\x00" not in result
        assert result == "helloworld"

    def test_hyperlink_formula_caught(self):
        payload = '=HYPERLINK("http://evil.com", "click me")'
        result = sanitize_cell(payload)
        assert result.startswith("'")

    def test_importxml_formula_caught(self):
        payload = "=IMPORTXML(\"http://evil.com/data\", \"//name\")"
        result = sanitize_cell(payload)
        assert result.startswith("'")

    def test_whitespace_padded_formula(self):
        """Formula triggers hidden behind leading spaces."""
        result = sanitize_cell("   =CMD('calc')")
        assert result.startswith("'")

    def test_empty_string(self):
        assert sanitize_cell("") == ""

    def test_minus_prefix_on_negative_looking_text(self):
        result = sanitize_cell("-negative sentiment detected")
        assert result.startswith("'")


class TestSanitizeRow:
    """Verify row-level sanitization applies to all values."""

    def test_all_fields_sanitized(self):
        row = {
            "message": "=CMD('calc')",
            "label": "SPAM",
            "score": 0.95,
        }
        result = sanitize_row(row)
        assert result["message"].startswith("'")
        assert result["label"] == "SPAM"  # no trigger prefix
        assert result["score"] == 0.95  # numeric unchanged

    def test_empty_row(self):
        assert sanitize_row({}) == {}


class TestIsFormulaPayload:
    """Verify detection of formula injection payloads."""

    def test_plain_text_not_flagged(self):
        assert is_formula_payload("Hello world") is False

    def test_formula_prefix_flagged(self):
        assert is_formula_payload("=SUM(A1:A10)") is True

    def test_cmd_pattern_flagged(self):
        assert is_formula_payload("=cmd('calc')") is True

    def test_empty_string_not_flagged(self):
        assert is_formula_payload("") is False

    def test_non_string_not_flagged(self):
        assert is_formula_payload(42) is False

    def test_hyperlink_pattern_flagged(self):
        assert is_formula_payload('=HYPERLINK("http://x.com","click")') is True

    def test_pipe_cmd_flagged(self):
        assert is_formula_payload("|cmd /c calc") is True
