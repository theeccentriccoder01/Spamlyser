"""
Dataset Sanitization Preprocessor for Spamlyser
Redacts personally identifiable information (PII) like email addresses, phone numbers, and SSNs from message text.
"""

import re
from typing import Tuple, Dict


class DatasetSanitizer:
    """Sanitizes text by masking sensitive PII tokens before model training or logging."""

    EMAIL_REGEX = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")
    PHONE_REGEX = re.compile(r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b")
    SSN_REGEX = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")

    @classmethod
    def sanitize(cls, text: str) -> Tuple[str, Dict[str, int]]:
        """Replaces PII patterns with standard placeholders [EMAIL], [PHONE], [SSN]."""
        counts = {"emails": 0, "phones": 0, "ssns": 0}

        text, counts["emails"] = cls.EMAIL_REGEX.subn("[EMAIL]", text)
        text, counts["phones"] = cls.PHONE_REGEX.subn("[PHONE]", text)
        text, counts["ssns"] = cls.SSN_REGEX.subn("[SSN]", text)

        return text, counts
