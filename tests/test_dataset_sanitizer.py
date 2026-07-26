from models.dataset_sanitizer import DatasetSanitizer


def test_dataset_sanitizer_pii_masking():
    raw_text = "Call me at 555-123-4567 or email john.doe@example.com with SSN 123-45-6789"
    sanitized, counts = DatasetSanitizer.sanitize(raw_text)

    assert "[PHONE]" in sanitized
    assert "[EMAIL]" in sanitized
    assert "[SSN]" in sanitized
    assert "john.doe@example.com" not in sanitized
    assert counts["emails"] == 1
    assert counts["phones"] == 1
    assert counts["ssns"] == 1
