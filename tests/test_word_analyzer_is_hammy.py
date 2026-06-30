"""
Regression tests for issue #42:
Duplicate dictionary key "is_hammy" in WordAnalyzer.analyze_text() caused
silent data corruption — the intended is_hammy value (line 342) was always
overwritten by a stricter expression (line 343), making neutral words appear
incorrectly in UI highlighting and explanations.
"""

import sys
from unittest.mock import MagicMock

# Stub heavy dependencies before importing models
sys.modules.setdefault("torch", MagicMock())
sys.modules.setdefault("transformers", MagicMock())


from models.word_analyzer import WordAnalyzer


class TestIsHammyNoDuplicateKey:
    """Ensure the correct is_hammy logic is stored in every word entry."""

    def setup_method(self):
        self.analyzer = WordAnalyzer()

    def _analyze(self, text: str):
        result = self.analyzer.analyze_text(text)
        return result["words"]

    # ------------------------------------------------------------------
    # Core regression: neutral words must be treated as hammy
    # ------------------------------------------------------------------
    def test_neutral_word_is_hammy(self):
        """A plain neutral word must have is_hammy=True.

        Before the fix, the duplicate key used
        `ham_weight > spam_weight and ham_weight > 0`.
        Neutral words start with ham_weight == 0 (before the 0.15 boost),
        so that expression evaluated to False and the word was wrongly
        treated as neither spam nor ham.
        """
        words = self._analyze("hello")
        assert len(words) > 0
        for w in words:
            if w["word_type"] in ("neutral", "ham"):
                assert w["is_hammy"] is True, (
                    f"Word '{w['word']}' (type={w['word_type']}) "
                    f"should be is_hammy=True but got {w['is_hammy']}. "
                    "Duplicate key regression?"
                )

    def test_spammy_word_not_hammy(self):
        """A high-weight spam word must not be flagged as hammy."""
        words = self._analyze("FREE prize winner lottery")
        for w in words:
            if w["is_spammy"]:
                assert not w["is_hammy"], (
                    f"Word '{w['word']}' is both spammy and hammy — "
                    "is_spammy and is_hammy should be mutually exclusive "
                    "for clearly spammy words."
                )

    def test_is_hammy_consistent_with_is_spammy(self):
        """is_hammy must equal `not is_spammy` for words where word_type is
        not explicitly neutral/ham — verifies the correct branch is kept."""
        words = self._analyze(
            "Congratulations! You won a FREE lottery prize worth $1,000,000!"
        )
        for w in words:
            # is_hammy = not is_spammy OR word_type in (neutral, ham)
            expected_at_minimum = not w["is_spammy"]
            if expected_at_minimum:
                assert w["is_hammy"], (
                    f"Word '{w['word']}': is_spammy={w['is_spammy']} so "
                    f"is_hammy should be True, got {w['is_hammy']}."
                )

    # ------------------------------------------------------------------
    # No duplicate key: confirm only ONE is_hammy value in the dict
    # ------------------------------------------------------------------
    def test_no_duplicate_key_in_word_dict(self):
        """Each word entry must be a plain dict; Python dicts cannot hold
        duplicate keys, so if the duplicate is back, the wrong value wins.
        This test explicitly checks that the stored value matches the
        intended formula, not the overwriting one."""
        words = self._analyze("hello world")
        for w in words:
            # The intended formula: not is_spammy OR word_type neutral/ham
            intended = (
                not w["is_spammy"]
                or w["word_type"] == "neutral"
                or w["word_type"] == "ham"
            )
            assert w["is_hammy"] == intended, (
                f"Word '{w['word']}': stored is_hammy={w['is_hammy']} but "
                f"intended formula gives {intended}. "
                "Likely the duplicate key is back."
            )

    def test_ham_message_words_all_have_is_hammy_true(self):
        """In a clearly hammy message, all non-spam words must be is_hammy=True
        so the UI can highlight them green."""
        words = self._analyze("Thank you for your help today")
        non_spam = [w for w in words if not w["is_spammy"]]
        assert len(non_spam) > 0, "Expected at least one non-spam word"
        for w in non_spam:
            assert w["is_hammy"] is True, (
                f"Non-spam word '{w['word']}' should be is_hammy=True "
                f"but got {w['is_hammy']}."
            )
