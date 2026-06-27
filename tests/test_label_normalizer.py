"""
Tests for issue #15: model labels other than exactly "SPAM"/"HAM".

A HuggingFace pipeline may emit LABEL_0 / LABEL_1 (or other labels). The
single-prediction path in app.py used the raw label as a model_stats key
(only 'spam'/'ham'/'total' exist) -> KeyError crash, and compared it with
== "SPAM" -> silent misclassification. These tests cover the normalisation
helper that fixes both, and explicitly demonstrate the original crash.
"""

import pytest

from models.label_normalizer import normalize_label


def test_canonical_spam_ham_passthrough():
    assert normalize_label("SPAM") == "SPAM"
    assert normalize_label("HAM") == "HAM"


def test_mixed_and_lower_case_labels():
    assert normalize_label("spam") == "SPAM"
    assert normalize_label("Ham") == "HAM"


def test_huggingface_label_0_and_label_1_mapping():
    # core issue #15 case
    assert normalize_label("LABEL_1") == "SPAM"
    assert normalize_label("LABEL_0") == "HAM"
    assert normalize_label("label_1") == "SPAM"
    assert normalize_label("label_0") == "HAM"


def test_whitespace_padded_label():
    assert normalize_label("  SPAM  ") == "SPAM"


def test_unknown_label_falls_back_to_probability():
    assert normalize_label("WEIRD", spam_probability=0.9) == "SPAM"
    assert normalize_label("WEIRD", spam_probability=0.1) == "HAM"


def test_unknown_label_without_probability_defaults_ham():
    assert normalize_label("WEIRD") == "HAM"


def test_unknown_label_with_non_numeric_probability_defaults_ham():
    assert normalize_label("WEIRD", spam_probability="oops") == "HAM"


def test_raw_label_as_stats_key_crashes_but_normalized_does_not():
    """
    Demonstrates the issue #15 bug and its fix.

    model_stats only has 'spam'/'ham'/'total' keys, so using the RAW label
    'LABEL_1' as a key raises KeyError. Normalising first yields 'spam',
    which is always a valid key.
    """
    model_stats = {"spam": 0, "ham": 0, "total": 0}

    with pytest.raises(KeyError):
        model_stats["LABEL_1".lower()] += 1  # 'label_1' -> KeyError (the bug)

    # the fix: normalise before using as a key
    key = normalize_label("LABEL_1").lower()
    assert key in model_stats
    model_stats[key] += 1
    assert model_stats["spam"] == 1


def test_normalized_key_always_valid_for_all_label_kinds():
    model_stats = {"spam": 0, "ham": 0, "total": 0}
    for raw in ["SPAM", "HAM", "LABEL_0", "LABEL_1", "weird"]:
        key = normalize_label(raw, 0.6).lower()
        assert key in model_stats
        model_stats[key] += 1
