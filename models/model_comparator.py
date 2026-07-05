"""Utility for comparing model predictions across all available classifiers."""

from typing import Any

import numpy as np


def compare_predictions(
    cleaned_sms: str, models: dict[str, Any], fallback_label: str
) -> list[dict[str, Any]]:
    """Run *cleaned_sms* through every loaded model and return results.

    Each result dict contains:
        - model (str): model name
        - label (str): SPAM or HAM
        - confidence (float): prediction confidence
        - is_rule_override (bool): whether a custom rule was applied
    """
    from models.custom_rules_manager import check_custom_rules

    results = []
    for name, clf in models.items():
        if clf is None:
            continue
        rule_match = check_custom_rules(cleaned_sms)
        if rule_match is not None:
            results.append(
                dict(
                    model=name,
                    label=rule_match,
                    confidence=1.0,
                    is_rule_override=True,
                )
            )
        else:
            pred = clf([cleaned_sms])[0]
            lbl = pred["label"].upper()
            if lbl not in ("SPAM", "HAM"):
                lbl = "SPAM" if pred.get("score", 0.5) > 0.5 else "HAM"
            results.append(
                dict(
                    model=name,
                    label=lbl,
                    confidence=pred["score"],
                    is_rule_override=False,
                )
            )
    return results


def agreement_score(results: list[dict[str, Any]]) -> tuple[bool, float]:
    """Return ``(all_agree, agreement_ratio)`` for the given *results*."""
    if not results:
        return True, 1.0
    labels = [r["label"] for r in results]
    n = len(labels)
    majority = max(set(labels), key=labels.count)
    agreement = sum(1 for l in labels if l == majority) / n
    return len(set(labels)) == 1, agreement
