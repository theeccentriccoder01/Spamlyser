import re

from models.unicode_handler import clean_unicode_text

from .text_sanitizer import safe_regex_findall, safe_regex_sub

ABBREVIATIONS = {
    "u": "you",
    "r": "are",
    "ur": "your",
    "pls": "please",
    "msg": "message",
    "b4": "before",
    "gr8": "great",
    "l8r": "later",
    "thx": "thanks",
    "plz": "please",
    "im": "I am",
    "btw": "by the way",
    "idk": "I don't know",
    "omg": "oh my god",
    "ttyl": "talk to you later",
    "asap": "as soon as possible",
}
LEETSPEAK = {"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t"}


def expand_abbreviations(text: str) -> str:
    words = text.split()
    expanded = [ABBREVIATIONS.get(w.lower(), w) for w in words]
    return " ".join(expanded)


def correct_leetspeak(text: str) -> str:
    # Apply leet substitutions only to tokens that contain at least one
    # ASCII letter.  Pure numeric tokens (integers, floats, phone numbers,
    # prices like "$100") are passed through unchanged, preventing corruption
    # of legitimate numeric content while still decoding obfuscated words
    # such as "Fr33" → "Free" or "M0n3y" → "Money".
    def _decode_token(token: str) -> str:
        if not any(c.isalpha() for c in token):
            return token  # nothing to decode in a purely numeric token
        for k, v in LEETSPEAK.items():
            token = safe_regex_sub(rf"{k}", v, token, default=token)
        return token

    return " ".join(_decode_token(tok) for tok in text.split(" "))


def count_suspicious_elements(text: str) -> dict:
    suspicious = {
        "all_caps": sum(1 for w in text.split() if w.isupper() and len(w) > 2),
        "excessive_punct": len(
            safe_regex_findall(r"!{2,}|\?{2,}|\${2,}", text, default=[])
        ),
        "phone_numbers": len(safe_regex_findall(r"\b\d{10,}\b", text, default=[])),
        "urls": len(safe_regex_findall(r"https?://|www\.", text, default=[])),
    }
    return suspicious


def preprocess_message(text: str) -> dict:
    cleaned = expand_abbreviations(text)
    cleaned = correct_leetspeak(cleaned)
    suspicious = count_suspicious_elements(cleaned)
    return {"cleaned": cleaned, "suspicious": suspicious}


# Example usage
if __name__ == "__main__":
    msg = "Fr33 M0n3y!!! u r winner! Call 9876543210 now!"
    result = preprocess_message(msg)
    print(result)
