"""Lightweight language detection for SMS preprocessing.

Uses Unicode script ranges and stop-word frequency heuristics instead of
external ML models so it stays dependency-free and fast enough for
real-time analysis.
"""

import re
from typing import Literal

LanguageCode = Literal[
    "en",
    "es",
    "fr",
    "de",
    "it",
    "pt",
    "nl",
    "ru",
    "ar",
    "zh",
    "ja",
    "ko",
    "hi",
    "bn",
    "ta",
    "te",
    "mr",
    "gu",
    "pa",
    "ur",
    "vi",
    "th",
    "tr",
    "pl",
    "ro",
    "cs",
    "hu",
    "el",
    "he",
    "id",
    "ms",
    "tl",
    "sv",
    "da",
    "fi",
    "no",
    "uk",
    "bg",
    "sr",
    "hr",
    "sk",
    "sl",
    "lt",
    "lv",
    "et",
    "unknown",
]

# Maps Unicode script ranges to language codes
_SCRIPT_MAP: dict[str, list[LanguageCode]] = {
    "Cyrillic": ["ru", "uk", "bg", "sr"],
    "Devanagari": ["hi", "mr", "ne"],
    "Arabic": ["ar", "ur"],
    "Hebrew": ["he"],
    "Bengali": ["bn"],
    "Gurmukhi": ["pa"],
    "Gujarati": ["gu"],
    "Tamil": ["ta"],
    "Telugu": ["te"],
    "Kannada": ["kn"],
    "Malayalam": ["ml"],
    "Thai": ["th"],
    "Hangul": ["ko"],
    "Hiragana": ["ja"],
    "Katakana": ["ja"],
    "CJK Ideographs": ["zh", "ja"],
    "Greek": ["el"],
}

# Latin-script stop-word fingerprints for disambiguation
_STOP_SETS: dict[LanguageCode, set[str]] = {
    "en": {
        "the",
        "is",
        "and",
        "to",
        "of",
        "in",
        "it",
        "you",
        "that",
        "he",
        "was",
        "for",
        "on",
        "are",
        "with",
    },
    "es": {
        "el",
        "la",
        "de",
        "que",
        "y",
        "en",
        "un",
        "ser",
        "por",
        "con",
        "no",
        "su",
        "para",
        "como",
        "más",
    },
    "fr": {
        "le",
        "la",
        "de",
        "et",
        "un",
        "être",
        "dans",
        "pour",
        "sur",
        "pas",
        "avec",
        "ce",
        "il",
        "elle",
        "nous",
    },
    "de": {
        "der",
        "die",
        "das",
        "und",
        "ist",
        "ein",
        "eine",
        "auf",
        "für",
        "mit",
        "nicht",
        "sich",
        "auch",
        "bei",
        "von",
    },
    "it": {
        "il",
        "lo",
        "la",
        "di",
        "che",
        "e",
        "un",
        "una",
        "per",
        "con",
        "su",
        "non",
        "si",
        "sono",
        "come",
    },
    "pt": {
        "o",
        "a",
        "de",
        "que",
        "e",
        "um",
        "para",
        "com",
        "não",
        "uma",
        "os",
        "as",
        "se",
        "é",
        "por",
    },
    "nl": {
        "de",
        "het",
        "een",
        "van",
        "en",
        "is",
        "te",
        "met",
        "voor",
        "niet",
        "op",
        "zijn",
        "aan",
        "dat",
        "die",
    },
    "pl": {
        "i",
        "w",
        "na",
        "z",
        "się",
        "do",
        "a",
        "to",
        "nie",
        "że",
        "jak",
        "od",
        "ma",
        "dla",
        "jest",
    },
    "ro": {
        "și",
        "de",
        "a",
        "în",
        "un",
        "o",
        "cu",
        "pe",
        "la",
        "că",
        "este",
        "nu",
        "sunt",
        "din",
        "mai",
    },
    "cs": {
        "a",
        "v",
        "na",
        "se",
        "je",
        "že",
        "do",
        "to",
        "s",
        "z",
        "o",
        "pro",
        "za",
        "si",
        "ale",
    },
    "hu": {
        "a",
        "az",
        "és",
        "hogy",
        "nem",
        "egy",
        "meg",
        "van",
        "is",
        "azt",
        "csak",
        "volt",
        "lesz",
        "mint",
        "már",
    },
    "sv": {
        "och",
        "i",
        "är",
        "det",
        "som",
        "en",
        "på",
        "att",
        "ett",
        "till",
        "med",
        "för",
        "inte",
        "har",
        "om",
    },
    "da": {
        "og",
        "i",
        "er",
        "det",
        "en",
        "at",
        "til",
        "på",
        "med",
        "for",
        "han",
        "som",
        "ikke",
        "har",
        "vi",
    },
    "fi": {
        "ja",
        "on",
        "ei",
        "se",
        "en",
        "oli",
        "hän",
        "mutta",
        "myös",
        "yksi",
        "voi",
        "että",
        "siitä",
        "ole",
        "tai",
    },
    "id": {
        "dan",
        "di",
        "ke",
        "dengan",
        "yang",
        "ini",
        "untuk",
        "tidak",
        "ada",
        "dari",
        "akan",
        "dalam",
        "saya",
        "ia",
        "juga",
    },
    "ms": {
        "dan",
        "di",
        "ke",
        "yang",
        "ini",
        "untuk",
        "tidak",
        "ada",
        "dari",
        "akan",
        "dalam",
        "saya",
        "ia",
        "juga",
        "itu",
    },
    "tl": {
        "ang",
        "ay",
        "ng",
        "sa",
        "at",
        "na",
        "ngunit",
        "ito",
        "iyon",
        "ko",
        "mo",
        "kanya",
        "aking",
        "atin",
        "aming",
    },
    "tr": {
        "bir",
        "ve",
        "bu",
        "ile",
        "için",
        "daha",
        "ben",
        "sen",
        "o",
        "biz",
        "siz",
        "onlar",
        "değil",
        "çok",
        "gibi",
    },
    "vi": {
        "và",
        "là",
        "của",
        "có",
        "trong",
        "cho",
        "không",
        "người",
        "này",
        "một",
        "tại",
        "nhưng",
        "hoặc",
        "được",
        "sẽ",
    },
    "el": {
        "και",
        "το",
        "η",
        "να",
        "για",
        "με",
        "του",
        "τη",
        "τα",
        "σε",
        "στο",
        "τις",
        "από",
        "που",
    },
}

# European languages that share the Latin script
_LATIN_LANGS: list[LanguageCode] = [
    "en",
    "es",
    "fr",
    "de",
    "it",
    "pt",
    "nl",
    "pl",
    "ro",
    "cs",
    "hu",
    "sv",
    "da",
    "fi",
    "id",
    "ms",
    "tl",
    "tr",
    "vi",
]


def _detect_script(text: str) -> str | None:
    """Identify the dominant Unicode script in *text*."""
    counts: dict[str, int] = {}
    for ch in text:
        cp = ord(ch)
        if 0x0400 <= cp <= 0x04FF or 0x0500 <= cp <= 0x052F:
            counts["Cyrillic"] = counts.get("Cyrillic", 0) + 1
        elif 0x0900 <= cp <= 0x097F:
            counts["Devanagari"] = counts.get("Devanagari", 0) + 1
        elif 0x0600 <= cp <= 0x06FF:
            counts["Arabic"] = counts.get("Arabic", 0) + 1
        elif 0x0590 <= cp <= 0x05FF:
            counts["Hebrew"] = counts.get("Hebrew", 0) + 1
        elif 0x0980 <= cp <= 0x09FF:
            counts["Bengali"] = counts.get("Bengali", 0) + 1
        elif 0x0A00 <= cp <= 0x0A7F:
            counts["Gurmukhi"] = counts.get("Gurmukhi", 0) + 1
        elif 0x0A80 <= cp <= 0x0AFF:
            counts["Gujarati"] = counts.get("Gujarati", 0) + 1
        elif 0x0B80 <= cp <= 0x0BFF:
            counts["Tamil"] = counts.get("Tamil", 0) + 1
        elif 0x0C00 <= cp <= 0x0C7F:
            counts["Telugu"] = counts.get("Telugu", 0) + 1
        elif 0x0D00 <= cp <= 0x0D7F:
            counts["Malayalam"] = counts.get("Malayalam", 0) + 1
        elif 0x0E00 <= cp <= 0x0E7F:
            counts["Thai"] = counts.get("Thai", 0) + 1
        elif 0xAC00 <= cp <= 0xD7AF:
            counts["Hangul"] = counts.get("Hangul", 0) + 1
        elif 0x3040 <= cp <= 0x309F:
            counts["Hiragana"] = counts.get("Hiragana", 0) + 1
        elif 0x30A0 <= cp <= 0x30FF:
            counts["Katakana"] = counts.get("Katakana", 0) + 1
        elif 0x4E00 <= cp <= 0x9FFF:
            counts["CJK Ideographs"] = counts.get("CJK Ideographs", 0) + 1
        elif 0x0370 <= cp <= 0x03FF:
            counts["Greek"] = counts.get("Greek", 0) + 1

    if not counts:
        return None  # Latin or ASCII
    return max(counts, key=counts.get)  # type: ignore[type-var]


def _disambiguate_latin(text: str) -> LanguageCode:
    """Score *text* against Latin-script stop-word sets."""
    words = set(re.findall(r"[a-zA-Zà-ÿÀ-ß]+", text.lower()))
    if not words:
        return "en"

    best_lang: LanguageCode = "en"
    best_score = 0
    for lang, stops in _STOP_SETS.items():
        score = len(words & stops)
        if score > best_score:
            best_score = score
            best_lang = lang
    return best_lang


def detect_language(text: str) -> LanguageCode:
    """Detect the language of *text*.

    Returns a two-letter ISO 639-1 code or ``"unknown"``.
    """
    if not text.strip():
        return "unknown"

    script = _detect_script(text)
    if script is None:
        return _disambiguate_latin(text)

    candidates = _SCRIPT_MAP.get(script, [])
    if not candidates:
        return "unknown"
    if len(candidates) == 1:
        return candidates[0]
    return candidates[0]


def is_language_supported(lang: LanguageCode) -> bool:
    """Return ``True`` if the preprocessor handles *lang* natively."""
    return lang in _LATIN_LANGS or lang in _SCRIPT_MAP
