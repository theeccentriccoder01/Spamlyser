"""
Regex Rule Dry-Run Validator for Spamlyser
Validates custom regex threat rules against sample message corpora without modifying active production rules.
"""

import re
from typing import Any, Dict, List, Tuple


class RuleDryRunner:
    """Simulates regex rule evaluation across sample message datasets to measure hit rates and false positives."""

    @staticmethod
    def validate_pattern(pattern: str) -> tuple[bool, str]:
        """Validates if a regex pattern compiles cleanly."""
        try:
            re.compile(pattern)
            return True, "Valid Regex"
        except re.error as e:
            return False, f"Invalid Regex: {e!s}"

    @classmethod
    def dry_run(cls, pattern: str, corpus: list[dict[str, str]]) -> dict[str, Any]:
        """Runs dry-run evaluation on a list of dicts [{'text': ..., 'label': 'SPAM'/'HAM'}]"""
        is_valid, msg = cls.validate_pattern(pattern)
        if not is_valid:
            return {"valid": False, "error": msg, "total": 0, "matches": 0}

        compiled = re.compile(pattern, re.IGNORECASE)
        matches = []
        true_positives = 0
        false_positives = 0

        for item in corpus:
            text = item.get("text", "")
            expected_label = item.get("label", "UNKNOWN").upper()
            if compiled.search(text):
                matches.append(text)
                if expected_label == "SPAM":
                    true_positives += 1
                elif expected_label == "HAM":
                    false_positives += 1

        total = len(corpus)
        hit_rate = round((len(matches) / total * 100), 2) if total > 0 else 0.0

        return {
            "valid": True,
            "total_evaluated": total,
            "total_matches": len(matches),
            "hit_rate_pct": hit_rate,
            "true_positives": true_positives,
            "false_positives": false_positives,
            "matched_samples": matches[:5],
        }
