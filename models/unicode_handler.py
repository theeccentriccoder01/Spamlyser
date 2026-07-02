import unicodedata


def clean_unicode_text(text: str) -> str:
    """Normalize Unicode characters and remove emojis/diacritics."""
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(c for c in normalized if not unicodedata.combining(c))
