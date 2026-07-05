from models.ham_explainer_viz import format_ham_explanation

"""
Lightweight, dependency-free alternative to :class:`ModelExplainer`.

Provides keyword-based spam explanations without requiring LIME.  Useful when
``lime`` is not installed or when a faster, simpler explanation is sufficient.
"""

import re
from collections.abc import Callable
from typing import Any, Dict, List, Optional

# Default spam-indicative keywords grouped by category.
SPAM_KEYWORDS: dict[str, list[str]] = {
    "Urgency / Pressure": [
        "urgent",
        "immediately",
        "act now",
        "don't miss",
        "expires",
        "limited time",
        "last chance",
        "hurry",
        "today only",
        "deadline",
    ],
    "Financial / Prize": [
        "winner",
        "won",
        "prize",
        "cash",
        "lottery",
        "million",
        "pounds",
        "dollars",
        "inheritance",
        "claim",
        "reward",
        "money",
        "fortune",
    ],
    "Security / Account": [
        "account",
        "verify",
        "bank",
        "login",
        "password",
        "security",
        "confirm",
        "update your",
        "suspicious",
        "unauthorized",
        "blocked",
    ],
    "Marketing": [
        "free",
        "discount",
        "offer",
        "trial",
        "subscribe",
        "exclusive",
        "limited offer",
        "buy now",
        "click here",
        "sale",
        "deal",
    ],
    "Personal Info": [
        "ssn",
        "social security",
        "credit card",
        "bank account",
        "paypal",
        "wire transfer",
        "western union",
        "money gram",
    ],
}

# HAM (legitimate message) indicator patterns.  These are common in everyday
# conversation or professional communication and are notably absent from spam.
HAM_KEYWORDS: dict[str, list[str]] = {
    "Greeting / Casual": [
        "hi",
        "hello",
        "hey",
        "good morning",
        "good afternoon",
        "good evening",
        "hope you",
        "how are you",
        "thanks",
        "thank you",
        "cheers",
    ],
    "Personal / Relational": [
        "see you",
        "talk later",
        "catch up",
        "miss you",
        "love you",
        "take care",
        "thinking of you",
        "happy birthday",
        "congratulations on",
        "well done",
    ],
    "Scheduling / Meeting": [
        "meeting",
        "appointment",
        "schedule",
        "calendar",
        "at noon",
        "tomorrow",
        "next week",
        "reminder",
        "due date",
        "deadline for",
        "rsvp",
    ],
    "Professional / Work": [
        "attached",
        "please find",
        "as discussed",
        "follow up",
        "let me know",
        "kind regards",
        "best regards",
        "sincerely",
        "invoice",
        "report",
        "ticket",
    ],
    "Negation of Spam Tactics": [
        "no cost",
        "no spam",
        "unsubscribe",
        "opt out",
        "privacy policy",
        "terms of service",
        "do not reply",
    ],
}


class SimpleExplainer:
    """Keyword-based explainer that identifies spam signals in text.

    Does not require LIME or any external ML explainability library.  Useful
    as a fallback when ``ModelExplainer`` (which depends on ``lime``) cannot
    be imported.

    HAM indicator detection is now implemented via :attr:`ham_keywords`.
    ``_find_ham_indicators`` returns words/phrases from ``HAM_KEYWORDS`` that
    are present in the text with a negative weight (indicating they *reduce*
    the spam probability).
    """

    def __init__(
        self,
        predict_fn: Callable | None = None,
        class_names: list[str] | None = None,
        keywords: dict[str, list[str]] | None = None,
        ham_keywords: dict[str, list[str]] | None = None,
    ):
        self.predict_fn = predict_fn
        self.class_names = class_names or ["HAM", "SPAM"]
        self.keywords = keywords or SPAM_KEYWORDS
        self.ham_keywords = ham_keywords or HAM_KEYWORDS

    def explain_prediction(
        self, text: str, num_features: int = 10, num_samples: int = 5000
    ) -> dict[str, Any]:
        """Explain a prediction by matching spam and ham keywords.

        The ``num_features`` and ``num_samples`` parameters are accepted for
        interface compatibility with :class:`ModelExplainer` but are not used
        by this keyword-based implementation.

        Returns a dict with the same structure as ``ModelExplainer``:
        ``text``, ``class_names``, ``features`` (each with ``class``,
        ``important_words``), and optionally an ``explanation`` placeholder.
        """
        text_lower = text.lower()
        features: list[dict[str, Any]] = []

        for class_name in self.class_names:
            if class_name == "SPAM":
                important_words = self._find_important_words(text_lower)
            else:
                important_words = self._find_ham_indicators(text_lower)

            important_words = important_words[:num_features]
            features.append({"class": class_name, "important_words": important_words})

        return {
            "text": text,
            "class_names": self.class_names,
            "features": features,
            "explanation": None,
        }

    def visualize_explanation(self, explanation_data: dict[str, Any]) -> dict[str, Any]:
        """Produce a visualisation dict compatible with ``ModelExplainer``.

        Returns a dict with ``feature_importance`` per class (each entry
        contains ``feature``, ``importance``, ``effect``, ``weight``) and
        an optional ``summary`` field.
        """
        visualization: dict[str, Any] = {
            "highlighted_text": {},
            "feature_importance": {},
            "summary": "",
        }

        for class_data in explanation_data.get("features", []):
            class_name = class_data["class"]
            feature_importance = []
            for item in class_data.get("important_words", []):
                feature_importance.append(
                    {
                        "feature": item["word"],
                        "importance": abs(item["weight"]),
                        "effect": "positive" if item["is_positive"] else "negative",
                        "weight": item["weight"],
                    }
                )
            feature_importance.sort(key=lambda x: abs(x["weight"]), reverse=True)
            visualization["feature_importance"][class_name] = feature_importance

        spam_features = visualization["feature_importance"].get("SPAM", [])
        if spam_features:
            top = spam_features[0]["feature"]
            visualization["summary"] = (
                f"Top spam signal: '{top}' (weight: {spam_features[0]['weight']:.2f})"
            )

        return visualization

    def get_threat_explanation(
        self, text: str, threat_type: str | None = None
    ) -> dict[str, Any]:
        """Return threat-specific keyword matches without calling LIME."""
        if not threat_type:
            return {
                "threat_type": None,
                "matching_keywords": [],
                "threat_features": [],
                "explanation": "No specific threat type identified.",
            }

        text_lower = text.lower()
        threat_category_keywords = SPAM_KEYWORDS.get(threat_type, [])
        matches = [kw for kw in threat_category_keywords if kw in text_lower]

        threat_features = [
            {"word": kw, "weight": 1.0, "class": "SPAM", "is_positive": True}
            for kw in matches
        ]

        return {
            "threat_type": threat_type,
            "matching_keywords": matches,
            "threat_features": threat_features,
        }

    # ── Internal helpers ──────────────────────────────────────────────

    def _find_important_words(self, text_lower: str) -> list[dict[str, Any]]:
        """Return words/phrases that match known spam patterns, with weights."""
        matches: list[dict[str, Any]] = []
        seen: set = set()

        for category, keywords in self.keywords.items():
            for keyword in keywords:
                occurrences = _count_keyword_occurrences(keyword, text_lower)
                if occurrences and keyword not in seen:
                    seen.add(keyword)
                    matches.append(
                        {
                            "word": keyword,
                            "weight": round(0.5 * occurrences, 3),
                            "is_positive": True,
                            "category": category,
                        }
                    )

        matches.sort(key=lambda x: x["weight"], reverse=True)
        return matches

    def _find_ham_indicators(self, text_lower: str) -> list[dict[str, Any]]:
        """Return words/phrases that signal legitimate (HAM) content.

        Weights are **negative** to indicate that these features *reduce* the
        probability of a message being spam.  The magnitude reflects how many
        times the keyword appears in the text (same formula as spam weights).
        """
        matches: list[dict[str, Any]] = []
        seen: set = set()

        for category, keywords in self.ham_keywords.items():
            for keyword in keywords:
                occurrences = _count_keyword_occurrences(keyword, text_lower)
                if occurrences and keyword not in seen:
                    seen.add(keyword)
                    matches.append(
                        {
                            "word": keyword,
                            "weight": round(-0.5 * occurrences, 3),
                            "is_positive": False,
                            "category": category,
                        }
                    )

        # Sort by descending absolute weight (strongest ham signals first)
        matches.sort(key=lambda x: abs(x["weight"]), reverse=True)
        return matches


def _count_keyword_occurrences(keyword: str, text_lower: str) -> int:
    """Count full keyword or phrase occurrences in already-lowercased text."""
    escaped = re.escape(keyword.lower())
    pattern = rf"(?<!\w){escaped}(?!\w)"
    return len(re.findall(pattern, text_lower))
