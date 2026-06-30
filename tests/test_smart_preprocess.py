"""
Regression tests for issue #43:
correct_leetspeak() in models/smart_preprocess.py used a bare pattern
rf"{k}" which matched leet digits anywhere in the string — including
inside legitimate numbers like "100", "2024", "3.14".

Fix: wrap each key with (?<!\\d) and (?!\\d) so substitutions only apply
to digits that are not adjacent to another digit (i.e., standalone
letter-like tokens in obfuscated words).
"""

from models.smart_preprocess import correct_leetspeak, preprocess_message


class TestCorrectLeetspeak:
    """Unit tests for correct_leetspeak() — issue #43 regression."""

    # ------------------------------------------------------------------
    # Numeric preservation (the core regression from the bug)
    # ------------------------------------------------------------------
    def test_integer_not_corrupted(self):
        """'100' must stay '100', not become 'ioo'."""
        assert correct_leetspeak("100") == "100", (
            "'100' was corrupted — digit lookaround not working."
        )

    def test_year_not_corrupted(self):
        """'2024' must stay '2024', not become 'zo24' or 'zoze'."""
        result = correct_leetspeak("2024")
        assert result == "2024", f"'2024' was corrupted to '{result}'"

    def test_float_not_corrupted(self):
        """'3.14' must stay '3.14', not become 'e.ie'."""
        result = correct_leetspeak("3.14")
        assert result == "3.14", f"'3.14' was corrupted to '{result}'"

    def test_phone_number_not_corrupted(self):
        """A phone-like number must not have its digits replaced."""
        result = correct_leetspeak("9876543210")
        assert result == "9876543210", f"Phone number was corrupted to '{result}'"

    def test_price_not_corrupted(self):
        """'$100' must stay '$100'."""
        result = correct_leetspeak("$100")
        assert result == "$100", f"'$100' was corrupted to '{result}'"

    def test_mixed_sentence_numbers_preserved(self):
        """Numbers embedded in a sentence must survive unchanged."""
        result = correct_leetspeak("Win $1000 in 2024!")
        assert "1000" in result, f"'1000' was lost in: '{result}'"
        assert "2024" in result, f"'2024' was lost in: '{result}'"

    # ------------------------------------------------------------------
    # Leet substitution still works for obfuscated words
    # ------------------------------------------------------------------
    def test_leet_zero_in_word_replaced(self):
        """'fr33' → 'free' (leet digits in letter context must still work)."""
        result = correct_leetspeak("fr33")
        assert result == "free", f"Expected 'free', got '{result}'"

    def test_leet_one_in_word_replaced(self):
        """'h1' → 'hi' (standalone leet digit still replaced)."""
        result = correct_leetspeak("h1")
        assert result == "hi", f"Expected 'hi', got '{result}'"

    def test_leet_four_in_word_replaced(self):
        """'h4ck' → 'hack'."""
        result = correct_leetspeak("h4ck")
        assert result == "hack", f"Expected 'hack', got '{result}'"

    def test_full_leet_phrase_decoded(self):
        """'Fr33 M0n3y' → 'Free Money' (classic spam obfuscation)."""
        result = correct_leetspeak("Fr33 M0n3y")
        assert result == "Free Money", f"Expected 'Free Money', got '{result}'"

    # ------------------------------------------------------------------
    # End-to-end: preprocess_message must not corrupt real numbers
    # ------------------------------------------------------------------
    def test_preprocess_preserves_number_in_message(self):
        """Full pipeline: numbers in a message must not be altered."""
        msg = "Call us at 100 to get your $50 discount before 2024!"
        result = preprocess_message(msg)
        cleaned = result["cleaned"]
        assert "100" in cleaned, f"'100' was lost in cleaned: '{cleaned}'"
        assert "50" in cleaned, f"'50' was lost in cleaned: '{cleaned}'"
        assert "2024" in cleaned, f"'2024' was lost in cleaned: '{cleaned}'"

    def test_preprocess_still_decodes_leet_spam(self):
        """Full pipeline: leet-encoded spam words must still be decoded."""
        msg = "Fr33 M0n3y!!! u r winner!"
        result = preprocess_message(msg)
        cleaned = result["cleaned"]
        # "Fr33" → "Free", "M0n3y" → "Money"
        assert "Free" in cleaned or "free" in cleaned, (
            f"Leet 'Fr33' not decoded in: '{cleaned}'"
        )
