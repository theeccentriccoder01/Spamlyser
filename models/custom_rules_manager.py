import models.rules_simulator
from models.rules_validator import validate_rule_structure

"""
Custom rules manager for Spamlyser Pro.

Handles allowlist (trusted domains/keywords) and blocklist (regexes) checking
and persistence.

Path resolution
---------------
The rules file path is sourced from ``config.CUSTOM_RULES_PATH`` (which in
turn reads the ``SPAMLYSER_CUSTOM_RULES`` environment variable).  This means
the file lands inside the ``data/`` directory by default — not next to
``custom_rules_manager.py`` — so rules survive deployments and are kept out
of the source tree.

Rule validation
---------------
Both ``save_custom_rules`` and ``check_custom_rules`` validate inputs before
writing or compiling.  Malformed regex patterns on the blocklist are silently
skipped during matching (to preserve the existing behaviour) but are caught
and reported at save time so users get early feedback.

Data integrity
--------------
All writes go through ``StorageManager.save_json``, which writes atomically
to a temporary file first and creates a timestamped backup of the previous
version before overwriting.
"""

import logging
import re
from typing import Any

from .storage_manager import StorageManager, default_json_validator

_logger = logging.getLogger(__name__)
_storage = StorageManager()

_EMPTY_RULES: dict[str, list] = {"allowlist": [], "blocklist": []}


def _rules_file_path() -> str:
    """Return the configured path for the custom rules JSON file.

    Reads from ``config.CUSTOM_RULES_PATH`` so the location is controlled
    by the ``SPAMLYSER_CUSTOM_RULES`` environment variable without hard-coding
    a relative path.
    """
    try:
        from config import CUSTOM_RULES_PATH, ensure_data_dir

        ensure_data_dir()
        return CUSTOM_RULES_PATH
    except ImportError:
        # Fallback for environments where config.py is not importable (e.g.
        # isolated unit tests that patch the module directly).
        return "custom_rules.json"


def _validate_rules(rules: Any) -> bool:
    """Return ``True`` only when *rules* has the expected schema.

    Expected structure::

        {
            "allowlist": ["domain.com", ...],
            "blocklist": ["regex_pattern", ...],
            "compounds": [{...}, ...]         # optional
        }

    Both ``allowlist`` and ``blocklist`` are required and must be lists of
    strings.  ``compounds`` (if present) is validated via
    :func:`~models.rule_engine.validate_compound_rules`.
    """
    if not isinstance(rules, dict):
        return False
    for key in ("allowlist", "blocklist"):
        if key not in rules or not isinstance(rules[key], list):
            return False
        if not all(isinstance(item, str) for item in rules[key]):
            return False
    if "compounds" in rules:
        from .rule_engine import validate_compound_rules as _vcr

        if not _vcr(rules["compounds"]):
            return False
    return True


def _compile_blocklist_pattern(pattern: str) -> re.Pattern | None:
    """Compile *pattern* with IGNORECASE; return ``None`` on error."""
    try:
        return re.compile(pattern, re.IGNORECASE)
    except re.error as exc:
        _logger.warning("Skipping invalid blocklist regex %r: %s", pattern, exc)
        return None


def load_custom_rules() -> dict[str, list]:
    """Load rules from the configured rules file.

    Returns the default empty-rules structure when the file does not exist,
    cannot be parsed, or fails schema validation.
    """
    path = _rules_file_path()
    rules = _storage.load_json_safe(path, default=None, validate=_validate_rules)
    if rules is None:
        _logger.info("No valid custom rules found at %s — using defaults.", path)
        return dict(_EMPTY_RULES)
    return rules


def save_custom_rules(rules: dict[str, list]) -> bool:
    """Atomically persist *rules* to the configured rules file.

    Parameters
    ----------
    rules:
        A dict with ``"allowlist"``, ``"blocklist"``, and optionally
        ``"compounds"`` list-of-dict keys.

    Returns
    -------
    bool
        ``True`` on success, ``False`` when validation or I/O fails.
    """
    if not _validate_rules(rules):
        _logger.error(
            "save_custom_rules: rules must be a dict with 'allowlist' and "
            "'blocklist' list-of-string keys."
        )
        return False

    # Warn about invalid blocklist patterns at save time so users get early
    # feedback rather than silent failures during analysis.
    invalid = [
        p for p in rules.get("blocklist", []) if _compile_blocklist_pattern(p) is None
    ]
    if invalid:
        _logger.warning(
            "The following blocklist patterns are invalid regexes and will be "
            "skipped during matching: %s",
            invalid,
        )

    path = _rules_file_path()
    return _storage.save_json(path, rules, backup=True, validate=_validate_rules)


def check_custom_rules(text: str) -> str | None:
    """Check whether *text* matches any allowlist, blocklist, or compound rule.

    Returns
    -------
    ``"HAM"``
        Text matches an allowlist entry (evaluated first).
    ``"SPAM"``
        Text matches a blocklist pattern or compound rule.
    ``None``
        No rule matched.
    """
    rules = load_custom_rules()
    text_lower = text.lower()

    # 1. Allowlist takes precedence (evaluated first)
    for domain in rules.get("allowlist", []):
        if domain.strip() and domain.lower() in text_lower:
            return "HAM"

    # 2. Compound rules — evaluated before the simple blocklist
    compounds = rules.get("compounds", [])
    if compounds:
        from .rule_engine import check_compound_rules as _check_compounds

        compound_result = _check_compounds(text, compounds)
        if compound_result is not None:
            return compound_result

    # 3. Blocklist — invalid regexes are silently skipped
    for pattern_str in rules.get("blocklist", []):
        if not pattern_str.strip():
            continue
        compiled = _compile_blocklist_pattern(pattern_str)
        if compiled is not None and compiled.search(text):
            return "SPAM"

    return None
