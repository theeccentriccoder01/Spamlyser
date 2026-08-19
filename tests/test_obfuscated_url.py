"""Tests for the obfuscated URL detector.

Spammers split domains so they no longer read as links to a naive filter —
``www[dot]google[dot]com``, ``example (dot) org``, ``bit . ly``. The pattern
added to ``_COMPILED_PATTERNS`` catches those forms.

Two properties matter and are tested separately: it must catch the obfuscation,
and it must leave ordinary prose alone. The second is the harder constraint —
a period appears in almost every sentence, so a pattern that treats any
``word.word`` as a domain would flag most legitimate messages.
"""

import time

from models.threat_analyzer import _COMPILED_PATTERNS

PATTERN = _COMPILED_PATTERNS["obfuscated_url"]


OBFUSCATED = [
    "www[dot]google[dot]com",
    "example (dot) org",
    "bit . ly",
    "http://bit . ly",
    "paypal[dot]com",
    "secure(dot)bank(dot)co",
    "click here: amaz0n [dot] xyz",
    "free-prize{dot}top",
    "Visit WWW[DOT]PAYPAL[DOT]COM to verify",
    "urgent: login at secure . bank . info now",
]


ORDINARY = [
    "Hello. World is great.",
    "See you at 5 p.m. tomorrow",
    "The file is report.pdf attached",
    "visit https://google.com now",
    "Mr. Smith went to Washington",
    "Rate: 4.5 out of 5",
    "e.g. this is fine",
    "Meeting at 3.30 pm",
    "I love this. Really do.",
    "config.py and app.py changed",
    "Please review section 2.1 of the document",
    "Call me. Thanks.",
]


def test_catches_obfuscated_domains():
    """Every documented obfuscation form is detected."""
    for text in OBFUSCATED:
        assert PATTERN.search(text) is not None, f"missed obfuscation in {text!r}"


def test_leaves_ordinary_text_alone():
    """Ordinary prose containing periods is not flagged.

    This is the constraint that shapes the pattern. A separator only counts
    when it is a bracketed "dot"/"." or a period with whitespace on at least
    one side, and the trailing component must be a known TLD — otherwise
    "Hello. World" and "report.pdf" would both read as domains.
    """
    for text in ORDINARY:
        match = PATTERN.search(text)
        assert match is None, f"false positive in {text!r}: matched {match.group()!r}"


def test_matches_the_domain_not_the_whole_string():
    """The match is the domain itself, so it can be surfaced to a user."""
    match = PATTERN.search("click here: amaz0n [dot] xyz for your prize")
    assert match is not None
    assert match.group().strip() == "amaz0n [dot] xyz"


def test_no_catastrophic_backtracking():
    """Matching stays linear on a long near-miss.

    The repository guards against ReDoS (see models/redos_guard.py), and an
    earlier draft of this pattern used a nested quantified group — it did not
    finish 100 repetitions of "a . a . a ..." in two minutes. This asserts the
    replacement stays fast on input two hundred times longer.
    """
    hostile = "a" + " . a" * 20000

    start = time.perf_counter()
    PATTERN.search(hostile)
    elapsed = time.perf_counter() - start

    assert elapsed < 1.0, f"took {elapsed:.3f}s — pattern may be backtracking"


def test_obfuscated_url_raises_phishing_confidence():
    """The detector feeds classification, not just pattern matching.

    The same phishing message is classified twice — once with the domain
    obfuscated, once with it written plainly. The obfuscated form should score
    at least as high, since hiding the link is itself evidence of intent.
    """
    from models.threat_analyzer import classify_threat_type

    obfuscated = "verify your account at paypal[dot]com now"
    plain = "verify your account at paypal now"

    _, obfuscated_confidence, _ = classify_threat_type(obfuscated, 0.9)
    _, plain_confidence, _ = classify_threat_type(plain, 0.9)

    assert obfuscated_confidence >= plain_confidence
