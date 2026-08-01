"""
Spam Category Keyword Cloud & Frequency Generator for Spamlyser
Extracts top indicative spam keywords and weighted frequencies from message sets.
"""

import re
from collections import Counter
from typing import List, Dict, Tuple


class KeywordCloudGenerator:
    """Extracts top word frequencies for generating spam keyword clouds."""

    STOPWORDS = {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "has",
        "he",
        "in",
        "is",
        "it",
        "its",
        "of",
        "on",
        "that",
        "the",
        "to",
        "was",
        "were",
        "will",
        "with",
        "you",
        "your",
        "this",
        "or",
    }

    @classmethod
    def generate(
        cls, messages: List[str], top_n: int = 20
    ) -> List[Tuple[str, int]]:
        """Extracts top_n non-stopword token frequencies across messages."""
        token_counter = Counter()
        for msg in messages:
            tokens = re.findall(r"\b[a-zA-Z]{3,}\b", msg.lower())
            filtered = [t for t in tokens if t not in cls.STOPWORDS]
            token_counter.update(filtered)
        return token_counter.most_common(top_n)

    @classmethod
    def generate_dict(
        cls, messages: List[str], top_n: int = 20
    ) -> Dict[str, int]:
        """Returns top_n keywords as a dictionary {word: count} for UI wordcloud components."""
        return dict(cls.generate(messages, top_n=top_n))
