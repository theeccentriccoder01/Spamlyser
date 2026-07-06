"""Boolean rule engine for smart filter expressions (AND / OR / NOT).

Extends the basic allowlist/blocklist with compound conditions so users
can write rules like::

    (keyword: "free" AND keyword: "win") OR (regex: r"\\d{6,}" AND NOT domain: "bank.com")
"""

import logging
import re
from typing import Any

_logger = logging.getLogger(__name__)

# --- Condition types -------------------------------------------------------

_EMPTY_RULES: dict[str, list] = {"allowlist": [], "blocklist": [], "compounds": []}


def _match_keyword(text: str, keyword: str) -> bool:
    return keyword.lower() in text.lower()


def _match_regex(text: str, pattern: str) -> bool:
    try:
        return re.search(pattern, text, re.IGNORECASE) is not None
    except re.error:
        _logger.warning("Invalid regex in compound rule: %s", pattern)
        return False


def _match_domain(text: str, domain: str) -> bool:
    return domain.lower() in text.lower() or (
        "://" not in text and domain.lower() in text.lower()
    )


_CONDITION_DISPATCH = {
    "keyword": _match_keyword,
    "regex": _match_regex,
    "domain": _match_domain,
}
_VALID_FIELDS = set(_CONDITION_DISPATCH)


def evaluate_condition(condition: dict[str, Any], text: str) -> bool:
    """Evaluate a single condition dict against *text*.

    A condition has the form::

        {"field": "keyword"|"regex"|"domain", "value": "...", "negate": false}

    Returns ``True`` when the condition matches (or does not match if
    *negate* is ``True``).
    """
    field = condition.get("field", "keyword")
    value = condition.get("value", "")
    negate = condition.get("negate", False)

    if field not in _CONDITION_DISPATCH or not value:
        return False

    matcher = _CONDITION_DISPATCH[field]
    result = matcher(text, value)
    return not result if negate else result


def evaluate_compound_rule(rule: dict[str, Any], text: str) -> bool | None:
    """Evaluate a single compound rule and return ``True`` / ``False`` / ``None``.

    A compound rule dict looks like::

        {
            "label": "sweepstakes or phishing",
            "logic": "AND",              # "AND" | "OR" | "NOT"
            "rules": [                    # list of condition dicts
                {"field": "keyword", "value": "free", "negate": false},
                {"field": "keyword", "value": "win",   "negate": false},
            ],
            "enabled": true,
        }

    ``None`` is returned when the rule is disabled or empty (meaning "no
    decision") — the caller can then fall back to the basic allowlist /
    blocklist.
    """
    if not rule.get("enabled", True):
        return None
    conditions = rule.get("rules", [])
    if not conditions:
        return None

    logic = rule.get("logic", "AND").upper()

    if logic == "NOT":
        return not evaluate_condition(conditions[0], text)

    results = [evaluate_condition(c, text) for c in conditions]
    if logic == "AND":
        return all(results)
    # OR
    return any(results)


def check_compound_rules(text: str, compounds: list[dict[str, Any]]) -> str | None:
    """Run *text* through all compound rules.

    Returns ``"SPAM"``, ``"HAM"``, or ``None``.
    """
    for rule in compounds:
        result = evaluate_compound_rule(rule, text)
        if result is True:
            return rule.get("action", "SPAM")
        if result is False:
            return rule.get("else_action", "HAM")
    return None


def validate_compound_rules(rules: Any) -> bool:
    """Schema checker for the compounds list."""
    if not isinstance(rules, list):
        return False
    for item in rules:
        if not isinstance(item, dict):
            return False
        if "rules" not in item or not isinstance(item["rules"], list):
            return False
        logic = item.get("logic", "AND").upper()
        if logic not in ("AND", "OR", "NOT"):
            return False
        for cond in item["rules"]:
            if not isinstance(cond, dict):
                return False
            if cond.get("field", "keyword") not in _VALID_FIELDS:
                return False
    return True
