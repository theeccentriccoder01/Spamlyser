"""Message categorization and tagging system for Spamlyser Pro.

Allows users to define custom categories, assign tags to messages
based on content patterns, and filter analysis history by category.
"""

import json
import os
import re
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any

DEFAULT_CATEGORIES = {
    "personal": {
        "label": "Personal",
        "icon": "👤",
        "color": "#4ecdc4",
        "keywords": ["hi", "hello", "friend", "family", "mom", "dad", "how are you"],
        "description": "Personal messages from friends and family",
    },
    "financial": {
        "label": "Financial",
        "icon": "💰",
        "color": "#feca57",
        "keywords": [
            "bank",
            "account",
            "payment",
            "transaction",
            "credit",
            "invoice",
        ],
        "description": "Banking, payment, and financial notifications",
    },
    "promotional": {
        "label": "Promotional",
        "icon": "📢",
        "color": "#ff9ff3",
        "keywords": ["offer", "discount", "sale", "deal", "free", "promotion"],
        "description": "Marketing and promotional messages",
    },
    "security": {
        "label": "Security Alert",
        "icon": "🔒",
        "color": "#ff6b6b",
        "keywords": [
            "security",
            "alert",
            "unauthorized",
            "login",
            "password",
            "verify",
        ],
        "description": "Security notifications and alerts",
    },
    "otp": {
        "label": "OTP / Verification",
        "icon": "🔑",
        "color": "#667eea",
        "keywords": ["otp", "verification", "code", "pin", "one-time"],
        "description": "One-time passwords and verification codes",
    },
}


class MessageCategorizer:
    """Assigns categories and tags to messages based on content analysis."""

    def __init__(self, config_path: str | None = None):
        if config_path is None:
            config_path = os.getenv(
                "SPAMLYSER_CATEGORIES_CONFIG",
                str(
                    Path(__file__).resolve().parent.parent / "data" / "categories.json"
                ),
            )
        self._config_path = Path(config_path)
        self._lock = Lock()
        self._categories: dict[str, Any] = {}
        self._compiled_patterns: dict[str, re.Pattern] = {}
        self._load_config()

    def _load_config(self):
        if self._config_path.exists():
            try:
                raw = self._config_path.read_text(encoding="utf-8")
                self._categories = json.loads(raw)
            except (json.JSONDecodeError, OSError):
                self._categories = dict(DEFAULT_CATEGORIES)
        else:
            self._categories = dict(DEFAULT_CATEGORIES)
            self._save_config()
        self._compile_patterns()

    def _save_config(self):
        self._config_path.parent.mkdir(parents=True, exist_ok=True)
        self._config_path.write_text(
            json.dumps(self._categories, indent=2, default=str),
            encoding="utf-8",
        )

    def _compile_patterns(self):
        self._compiled_patterns = {}
        for cat_id, cat_data in self._categories.items():
            keywords = cat_data.get("keywords", [])
            if keywords:
                pattern = r"\b(?:" + "|".join(re.escape(kw) for kw in keywords) + r")\b"
                try:
                    self._compiled_patterns[cat_id] = re.compile(pattern, re.IGNORECASE)
                except re.error:
                    pass

    def categorize(self, message: str) -> list[dict[str, Any]]:
        """Assign categories to a message based on keyword matching."""
        matches = []
        for cat_id, pattern in self._compiled_patterns.items():
            if pattern.search(message):
                cat_data = self._categories.get(cat_id, {})
                matches.append(
                    {
                        "category_id": cat_id,
                        "label": cat_data.get("label", cat_id),
                        "icon": cat_data.get("icon", "📁"),
                        "color": cat_data.get("color", "#888888"),
                        "confidence": min(1.0, len(pattern.findall(message)) * 0.25),
                    }
                )
        return sorted(matches, key=lambda x: x["confidence"], reverse=True)

    def get_all_categories(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._categories)

    def add_category(
        self,
        cat_id: str,
        label: str,
        keywords: list[str],
        icon: str = "📁",
        color: str = "#888888",
    ) -> bool:
        with self._lock:
            if cat_id in self._categories:
                return False
            self._categories[cat_id] = {
                "label": label,
                "icon": icon,
                "color": color,
                "keywords": keywords,
                "description": "",
            }
            self._compile_patterns()
            self._save_config()
            return True

    def remove_category(self, cat_id: str) -> bool:
        with self._lock:
            if cat_id not in self._categories or cat_id in DEFAULT_CATEGORIES:
                return False
            del self._categories[cat_id]
            self._compiled_patterns.pop(cat_id, None)
            self._save_config()
            return True
