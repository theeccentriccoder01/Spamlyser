"""
CSV injection prevention utilities for Spamlyser Pro export pipeline.

Spreadsheet applications (Excel, Google Sheets, LibreOffice Calc) interpret
cell values starting with certain characters as formulas. An attacker who
controls message content could craft payloads like ``=CMD("calc")`` or
``=HYPERLINK("http://evil.com", "click")`` that execute when a user opens
the exported CSV.

This module provides field-level and row-level sanitization to neutralize
such payloads before they reach the export layer. It works alongside the
existing ``_csv_safe_cell`` in ``export_feature.py`` but offers a more
thorough defense-in-depth approach.

References:
    - OWASP CSV Injection: https://owasp.org/www-community/attacks/CSV_Injection
    - CWE-1236: Improper Neutralization of Formula Elements
"""

import re
from typing import Any

# Characters that spreadsheet applications interpret as formula indicators.
# This is intentionally broader than the set in export_feature.py to catch
# edge cases where payloads hide behind whitespace or encoding tricks.
_FORMULA_TRIGGERS = frozenset({"=", "+", "-", "@", "\t", "\r", "\n", "|"})

# Patterns that indicate embedded formula commands even when they appear
# after innocuous-looking prefixes. These are case-insensitive because
# some spreadsheets accept uppercase and lowercase function names.
_DANGEROUS_PATTERNS = [
    re.compile(r"=\s*cmd\s*\(", re.IGNORECASE),
    re.compile(r"=\s*hyperlink\s*\(", re.IGNORECASE),
    re.compile(r"=\s*importxml\s*\(", re.IGNORECASE),
    re.compile(r"=\s*importdata\s*\(", re.IGNORECASE),
    re.compile(r"=\s*importfeed\s*\(", re.IGNORECASE),
    re.compile(r"=\s*importhtml\s*\(", re.IGNORECASE),
    re.compile(r"\|.*cmd", re.IGNORECASE),
]


def sanitize_cell(value: Any) -> Any:
    """Neutralize formula injection in a single cell value.

    Non-string values are returned unchanged. For strings, any leading
    character that could be interpreted as a formula trigger is escaped
    by prepending a single-quote character. Additionally, embedded
    newlines and tabs are stripped because they can be used to smuggle
    payloads across cell boundaries.

    Parameters
    ----------
    value : Any
        The cell value to sanitize.

    Returns
    -------
    Any
        The sanitized value, safe for CSV export.
    """
    if not isinstance(value, str):
        return value

    # Strip characters that can break cell boundaries in CSV
    cleaned = value.replace("\r\n", " ").replace("\r", " ").replace("\n", " ")

    # Remove null bytes that some parsers mishandle
    cleaned = cleaned.replace("\x00", "")

    # Check if the stripped content starts with a formula trigger
    stripped = cleaned.lstrip()
    if stripped and stripped[0] in _FORMULA_TRIGGERS:
        cleaned = "'" + cleaned

    # Even if the prefix looks safe, check for dangerous embedded patterns
    # that could be triggered by certain spreadsheet parsers
    for pattern in _DANGEROUS_PATTERNS:
        if pattern.search(cleaned):
            # Wrap the entire value so it cannot be interpreted as a formula
            cleaned = "'" + cleaned.lstrip("'")
            break

    return cleaned


def sanitize_row(row: dict[str, Any]) -> dict[str, Any]:
    """Apply ``sanitize_cell`` to every value in a row dictionary.

    Parameters
    ----------
    row : dict
        A mapping of column names to cell values.

    Returns
    -------
    dict
        A new dictionary with all string values sanitized.
    """
    return {key: sanitize_cell(val) for key, val in row.items()}


def is_formula_payload(text: str) -> bool:
    """Return ``True`` if *text* looks like a spreadsheet formula injection.

    This is useful for logging or flagging suspicious content without
    necessarily modifying it (e.g., in audit trails).

    Parameters
    ----------
    text : str
        The text to check.

    Returns
    -------
    bool
        ``True`` when the text matches known formula injection patterns.
    """
    if not isinstance(text, str) or not text.strip():
        return False

    stripped = text.lstrip()
    if stripped and stripped[0] in _FORMULA_TRIGGERS:
        return True

    for pattern in _DANGEROUS_PATTERNS:
        if pattern.search(text):
            return True

    return False
