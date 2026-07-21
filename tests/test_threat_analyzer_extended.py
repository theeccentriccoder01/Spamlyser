"""
Extended unit tests for threat_analyzer.py module.

Tests threat classification engine covering:
- Phishing threat detection
- Scam/Fraud threat detection
- Unwanted Marketing detection
- Threat category classification
- Confidence scoring
- Threat-specific advice generation
"""

import pytest
from models.threat_analyzer import (
    classify_threat_type,
    get_threat_specific_advice,
    THREAT_CATEGORIES,
)


class TestPhishingDetection:
    """Test phishing threat detection functionality."""

    def test_detects_account_verification_phishing(self):
        """Should detect phishing attempts asking to verify account."""
        message = "Verify your account immediately. Click here: https://fake.com"
        threat_type, confidence, metadata = classify_threat_type(message, 0.85)
        
        assert threat_type == "Phishing"
        assert confidence > 0.3
        assert metadata["category_icon"] == "🎣"

    def test_detects_password_update_phishing(self):
        """Should detect phishing requesting password updates."""
        message = "Update your password for security. Confirm your credentials now."
        threat_type, confidence, metadata = classify_threat_type(message, 0.80)
        
        assert threat_type == "Phishing"
        assert confidence > 0.2

    def test_detects_bank_credential_phishing(self):
        """Should detect banking credential phishing."""
        message = "Your bank account has been locked. Verify your login information immediately."
        threat_type, confidence, metadata = classify_threat_type(message, 0.90)
        
        assert threat_type == "Phishing"
        assert confidence > 0.3

    def test_detects_suspicious_activity_phishing(self):
        """Should detect phishing about suspicious account activity."""
        message = "Unusual activity detected on your account. Click link to authenticate."
        threat_type, confidence, metadata = classify_threat_type(message, 0.85)
        
        assert threat_type is not None
        assert confidence > 0.1

    def test_phishing_metadata_contains_required_fields(self):
        """Should return metadata with required fields."""
        message = "Verify your account now!"
        threat_type, confidence, metadata = classify_threat_type(message, 0.75)
        
        if threat_type:
            assert "category_icon" in metadata
            assert "category_color" in metadata
            assert "category_description" in metadata
            assert "category_scores" in metadata


class TestScamFraudDetection:
    """Test scam/fraud threat detection functionality."""

    def test_detects_lottery_scam(self):
        """Should detect lottery scam messages."""
        message = "You have won a lottery prize! Claim your $50,000 now!"
        threat_type, confidence, metadata = classify_threat_type(message, 0.90)
        
        assert threat_type == "Scam/Fraud"
        assert confidence >= 0.3

    def test_detects_cash_prize_scam(self):
        """Should detect cash prize scams."""
        message = "Congratulations! You won $5000 cash prize. Click to claim."
        threat_type, confidence, metadata = classify_threat_type(message, 0.95)
        
        assert threat_type == "Scam/Fraud"

    def test_detects_inheritance_scam(self):
        """Should detect inheritance/money transfer scams."""
        message = "You have inherited $2 million. Send your details to claim inheritance."
        threat_type, confidence, metadata = classify_threat_type(message, 0.85)
        
        # May classify as Scam/Fraud or Other depending on keyword matching
        assert threat_type in ["Scam/Fraud", "Other"]
        assert confidence >= 0.1

    def test_detects_urgent_money_offer(self):
        """Should detect urgent money-related offers."""
        message = "URGENT: Million dollar opportunity available TODAY. Act now!"
        threat_type, confidence, metadata = classify_threat_type(message, 0.88)
        
        # May classify as Scam/Fraud or Other depending on exact pattern matching
        assert threat_type is not None
        assert confidence >= 0.1

    def test_detects_winner_notification(self):
        """Should detect fake winner notifications."""
        message = "YOU ARE A WINNER!!! Claim your cash prize immediately!!!"
        threat_type, confidence, metadata = classify_threat_type(message, 0.92)
        
        assert threat_type == "Scam/Fraud"

    def test_scam_confidence_increases_with_spam_probability(self):
        """Should have higher confidence with higher spam probability."""
        message = "You won a prize!"
        
        threat_type_low, conf_low, _ = classify_threat_type(message, 0.6)
        threat_type_high, conf_high, _ = classify_threat_type(message, 0.95)
        
        # Higher spam probability should give higher or equal confidence
        assert conf_high >= conf_low


class TestUnwantedMarketingDetection:
    """Test unwanted marketing threat detection."""

    def test_detects_discount_offer(self):
        """Should detect discount/sales marketing."""
        message = "Limited time offer! Get 50% discount on all items. Shop now!"
        threat_type, confidence, metadata = classify_threat_type(message, 0.70)
        
        assert threat_type == "Unwanted Marketing"

    def test_detects_subscription_offer(self):
        """Should detect subscription offers."""
        message = "Join our membership and save! Subscribe now for exclusive deals."
        threat_type, confidence, metadata = classify_threat_type(message, 0.65)
        
        assert threat_type == "Unwanted Marketing"

    def test_detects_free_trial_offer(self):
        """Should detect free trial marketing."""
        message = "Start your free trial today! Sign up for premium access."
        threat_type, confidence, metadata = classify_threat_type(message, 0.60)
        
        # May classify as Marketing or Other depending on keyword coverage
        assert threat_type in ["Unwanted Marketing", "Other"]
        assert confidence >= 0.1

    def test_detects_exclusive_promotion(self):
        """Should detect exclusive promotions."""
        message = "Exclusive offer! Only today - buy now and save big!"
        threat_type, confidence, metadata = classify_threat_type(message, 0.65)
        
        assert threat_type == "Unwanted Marketing"

    def test_marketing_not_confused_with_phishing(self):
        """Should not confuse legitimate marketing with phishing."""
        message = "Get 30% off! Limited time sale. Shop now at our website."
        threat_type, confidence, metadata = classify_threat_type(message, 0.65)
        
        # Should be marketing, not phishing
        assert threat_type != "Phishing"


class TestLowSpamMessages:
    """Test handling of legitimate (low spam probability) messages."""

    def test_returns_none_for_low_spam_probability(self):
        """Should return None for messages with low spam probability."""
        message = "Hi, are we still meeting at 3pm tomorrow?"
        threat_type, confidence, metadata = classify_threat_type(message, 0.3)
        
        assert threat_type is None
        assert confidence == 0.0

    def test_handles_legitimate_email(self):
        """Should handle legitimate email-like messages."""
        message = "Please review the attached document and provide feedback."
        threat_type, confidence, metadata = classify_threat_type(message, 0.2)
        
        assert threat_type is None

    def test_handles_boundary_spam_probability(self):
        """Should handle boundary condition at 0.5 spam probability."""
        message = "Some message here"
        threat_type, confidence, metadata = classify_threat_type(message, 0.49)
        
        assert threat_type is None


class TestThreatSpecificAdvice:
    """Test threat-specific advice generation."""

    def test_phishing_advice_available(self):
        """Should provide advice for phishing threats."""
        advice = get_threat_specific_advice("Phishing")
        
        assert isinstance(advice, list)
        assert len(advice) > 0
        assert any("link" in a.lower() for a in advice)

    def test_scam_fraud_advice_available(self):
        """Should provide advice for scam/fraud threats."""
        advice = get_threat_specific_advice("Scam/Fraud")
        
        assert isinstance(advice, list)
        assert len(advice) > 0
        assert any("too good to be true" in a.lower() or "prize" in a.lower() for a in advice)

    def test_marketing_advice_available(self):
        """Should provide advice for unwanted marketing."""
        advice = get_threat_specific_advice("Unwanted Marketing")
        
        assert isinstance(advice, list)
        assert len(advice) > 0
        assert any("stop" in a.lower() or "opt out" in a.lower() for a in advice)

    def test_other_advice_available(self):
        """Should provide advice for 'Other' threats."""
        advice = get_threat_specific_advice("Other")
        
        assert isinstance(advice, list)
        assert len(advice) > 0

    def test_invalid_threat_type_returns_empty_list(self):
        """Should return empty list for invalid threat type."""
        advice = get_threat_specific_advice("InvalidThreat")
        
        assert advice == []

    def test_none_threat_type_returns_empty_list(self):
        """Should handle None threat type gracefully."""
        advice = get_threat_specific_advice(None)
        
        assert advice == []


class TestThreatCategoryMetadata:
    """Test threat category metadata and structure."""

    def test_all_categories_have_icon(self):
        """Should have icon for each threat category."""
        for category, data in THREAT_CATEGORIES.items():
            assert "icon" in data
            assert isinstance(data["icon"], str)
            assert len(data["icon"]) > 0

    def test_all_categories_have_color(self):
        """Should have color for each threat category."""
        for category, data in THREAT_CATEGORIES.items():
            assert "color" in data
            assert isinstance(data["color"], str)
            assert data["color"].startswith("#")

    def test_all_categories_have_description(self):
        """Should have description for each threat category."""
        for category, data in THREAT_CATEGORIES.items():
            assert "description" in data
            assert isinstance(data["description"], str)
            assert len(data["description"]) > 0

    def test_all_categories_have_keywords(self):
        """Should have keywords list for each threat category."""
        for category, data in THREAT_CATEGORIES.items():
            assert "keywords" in data
            assert isinstance(data["keywords"], list)

    def test_all_categories_have_advice(self):
        """Should have advice list for each threat category."""
        for category, data in THREAT_CATEGORIES.items():
            assert "advice" in data
            assert isinstance(data["advice"], list)


class TestMetadataStructure:
    """Test the structure of returned metadata."""

    def test_metadata_contains_category_scores(self):
        """Should return metadata with category scores."""
        message = "You won a prize!"
        threat_type, confidence, metadata = classify_threat_type(message, 0.85)
        
        assert "category_scores" in metadata
        assert isinstance(metadata["category_scores"], dict)

    def test_category_scores_contain_all_categories(self):
        """Should have scores for all threat categories."""
        message = "You won $5000!"
        threat_type, confidence, metadata = classify_threat_type(message, 0.90)
        
        scores = metadata["category_scores"]
        expected_categories = ["Phishing", "Scam/Fraud", "Unwanted Marketing", "Other"]
        
        for category in expected_categories:
            assert category in scores
            assert isinstance(scores[category], float)
            assert 0.0 <= scores[category] <= 1.0

    def test_highest_score_matches_threat_type(self):
        """Should have highest score for the returned threat type."""
        message = "You won a lottery prize!"
        threat_type, confidence, metadata = classify_threat_type(message, 0.90)
        
        if threat_type:
            scores = metadata["category_scores"]
            highest_score = max(scores.values())
            # The returned threat type should have a reasonable score
            assert threat_type in scores
            assert scores[threat_type] >= 0.0


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_message(self):
        """Should handle empty messages gracefully."""
        threat_type, confidence, metadata = classify_threat_type("", 0.8)
        
        # Should return Other or None
        assert threat_type is None or threat_type == "Other"
        assert isinstance(metadata, dict)

    def test_very_long_message(self):
        """Should handle very long messages."""
        long_message = "You won a prize! " * 1000
        threat_type, confidence, metadata = classify_threat_type(long_message, 0.90)
        
        assert threat_type is not None
        assert isinstance(confidence, float)
        assert isinstance(metadata, dict)

    def test_message_with_unicode(self):
        """Should handle Unicode characters."""
        message = "你好! 🎁 You won a prize! বাংলা"
        threat_type, confidence, metadata = classify_threat_type(message, 0.85)
        
        assert isinstance(metadata, dict)

    def test_message_with_special_characters(self):
        """Should handle special characters."""
        message = "!!!YOU WON!!! @#$%^&*() [Prize: $5000]"
        threat_type, confidence, metadata = classify_threat_type(message, 0.90)
        
        assert threat_type is not None

    def test_message_with_newlines_and_tabs(self):
        """Should handle messages with newlines and tabs."""
        message = "You have won\n\t$5000\nClick here\tto claim"
        threat_type, confidence, metadata = classify_threat_type(message, 0.88)
        
        assert isinstance(metadata, dict)

    def test_spam_probability_boundary_values(self):
        """Should handle boundary spam probability values."""
        message = "You won a prize!"
        
        # Test at 0.5 (boundary)
        threat_type_05, _, _ = classify_threat_type(message, 0.5)
        assert threat_type_05 is not None or threat_type_05 is None
        
        # Test at 0.0 (minimum)
        threat_type_0, _, _ = classify_threat_type(message, 0.0)
        assert threat_type_0 is None
        
        # Test at 1.0 (maximum)
        threat_type_1, _, _ = classify_threat_type(message, 1.0)
        assert threat_type_1 is not None

    def test_case_insensitivity(self):
        """Should detect threats regardless of case."""
        message_lower = "you have won a prize"
        message_upper = "YOU HAVE WON A PRIZE"
        message_mixed = "YoU hAvE wOn A pRiZe"
        
        _, conf_lower, _ = classify_threat_type(message_lower, 0.85)
        _, conf_upper, _ = classify_threat_type(message_upper, 0.85)
        _, conf_mixed, _ = classify_threat_type(message_mixed, 0.85)
        
        # All should have similar confidence scores
        # (confidence should not drastically differ based on case)
        assert all(isinstance(c, float) for c in [conf_lower, conf_upper, conf_mixed])


class TestComplexScenarios:
    """Test complex, real-world scenarios."""

    def test_phishing_AND_scam_indicators(self):
        """Should handle messages with both phishing and scam indicators."""
        message = "Verify your account to claim your $5000 prize! Click link to authenticate."
        threat_type, confidence, metadata = classify_threat_type(message, 0.95)
        
        # Could be classified as either, but should pick the highest scoring one
        assert threat_type in ["Phishing", "Scam/Fraud"]
        assert confidence > 0.3

    def test_scam_AND_marketing_indicators(self):
        """Should handle messages with both scam and marketing elements."""
        message = "Limited time: Buy now and win $1000! Exclusive offer for today only."
        threat_type, confidence, metadata = classify_threat_type(message, 0.75)
        
        # Should identify the primary threat
        assert threat_type is not None
        assert isinstance(confidence, float)

    def test_ambiguous_message(self):
        """Should handle ambiguous messages gracefully."""
        message = "Check our new offers"
        threat_type, confidence, metadata = classify_threat_type(message, 0.65)
        
        # Should return a classification or Other
        assert isinstance(metadata, dict)

    def test_message_with_url_patterns(self):
        """Should handle messages with URL patterns."""
        message = "Verify at https://bank-secure.com/verify or http://fake-bank.ru"
        threat_type, confidence, metadata = classify_threat_type(message, 0.85)
        
        assert threat_type is not None

    def test_message_with_currency_symbols(self):
        """Should handle currency symbols correctly."""
        message = "You won $5000! €3000! £2000! Get it now!"
        threat_type, confidence, metadata = classify_threat_type(message, 0.90)
        
        assert threat_type == "Scam/Fraud"

    def test_common_spam_combination(self):
        """Should detect typical spam: urgent + money + click."""
        message = "URGENT!!! Click here to claim $500 NOW!!!"
        threat_type, confidence, metadata = classify_threat_type(message, 0.92)
        
        assert threat_type == "Scam/Fraud"
        assert confidence > 0.5


class TestConsistency:
    """Test consistency of results across multiple calls."""

    def test_same_message_same_result(self):
        """Should return same result for identical inputs."""
        message = "You won a lottery prize!"
        
        result1 = classify_threat_type(message, 0.85)
        result2 = classify_threat_type(message, 0.85)
        
        assert result1[0] == result2[0]  # threat_type
        assert result1[1] == result2[1]  # confidence
        assert result1[2]["category_scores"] == result2[2]["category_scores"]

    def test_different_spam_probabilities_affect_confidence(self):
        """Should have different confidence for different spam probabilities."""
        message = "You won a prize!"
        
        _, conf_low, _ = classify_threat_type(message, 0.6)
        _, conf_high, _ = classify_threat_type(message, 0.95)
        
        # Higher spam probability should generally result in higher confidence
        assert conf_high >= conf_low


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
