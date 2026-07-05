"""Tests for threat analyzer."""

from models.threat_analyzer import (
    THREAT_CATEGORIES,
    classify_threat_type,
    get_threat_specific_advice,
)


def test_threat_classification():
    """Test the threat classification function"""

    # Test phishing classification
    phishing_message = "URGENT: Your bank account has been locked. Click here to verify your information: http://fakebank.com"
    threat_type, confidence, metadata = classify_threat_type(phishing_message, 0.9)
    assert threat_type == "Phishing"
    assert confidence > 0.5

    # Test scam classification
    scam_message = "Congratulations! You've won £1,000,000 in the lottery! Call now to claim your prize."
    threat_type, confidence, metadata = classify_threat_type(scam_message, 0.9)
    assert threat_type == "Scam/Fraud"
    assert confidence > 0.5

    # Test marketing classification
    marketing_message = (
        "Limited time offer! 50% off all products. Shop now at our online store."
    )
    threat_type, confidence, metadata = classify_threat_type(marketing_message, 0.9)
    assert threat_type == "Unwanted Marketing"
    assert confidence > 0.3

    # Test other classification (ambiguous)
    other_message = "Your package delivery failed. Contact us."
    threat_type, confidence, metadata = classify_threat_type(other_message, 0.9)
    assert threat_type in THREAT_CATEGORIES.keys()

    # Test non-spam message
    ham_message = "Hey mom, I'll be home for dinner at 6pm."
    threat_type, confidence, _metadata = classify_threat_type(ham_message, 0.1)
    assert threat_type is None


def test_threat_advice():
    """Test getting threat-specific advice"""

    # Test advice for phishing
    phishing_advice = get_threat_specific_advice("Phishing")
    assert len(phishing_advice) > 0
    assert "passwords" in " ".join(phishing_advice).lower()

    # Test advice for scam
    scam_advice = get_threat_specific_advice("Scam/Fraud")
    assert len(scam_advice) > 0
    assert "money" in " ".join(scam_advice).lower()

    # Test advice for marketing
    marketing_advice = get_threat_specific_advice("Unwanted Marketing")
    assert len(marketing_advice) > 0
    assert "opt out" in " ".join(marketing_advice).lower() or "STOP" in " ".join(
        marketing_advice
    )

    # Test invalid category
    invalid_advice = get_threat_specific_advice("InvalidCategory")
    assert invalid_advice == []

    # Test None category
    none_advice = get_threat_specific_advice(None)
    assert none_advice == []


def test_load_custom_categories():
    from models.threat_analyzer import load_custom_categories

    cats = load_custom_categories()
    assert isinstance(cats, dict)
