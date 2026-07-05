"""Sender reputation tracking for SMS threat analysis.

Tracks and scores phone numbers/senders based on analysis history,
allowing the system to flag messages from known spam sources faster.
"""

import json
import os
import time
from datetime import datetime
from pathlib import Path
from threading import Lock


class SenderReputation:
    """Tracks sender reputation scores using persistent JSON storage."""

    def __init__(self, db_path: str | None = None):
        if db_path is None:
            db_path = os.getenv(
                "SPAMLYSER_SENDER_REPUTATION_DB",
                str(
                    Path(__file__).resolve().parent.parent
                    / "data"
                    / "sender_reputation.json"
                ),
            )
        self._db_path = Path(db_path)
        self._lock = Lock()
        self._cache: dict[str, dict] = {}
        self._dirty = False
        self._load()

    def _load(self):
        if self._db_path.exists():
            try:
                raw = self._db_path.read_text(encoding="utf-8")
                self._cache = json.loads(raw)
            except (json.JSONDecodeError, OSError):
                self._cache = {}

    def _save(self):
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db_path.write_text(
            json.dumps(self._cache, indent=2, default=str),
            encoding="utf-8",
        )
        self._dirty = False

    def _ensure_sender(self, sender: str) -> dict:
        if sender not in self._cache:
            self._cache[sender] = {
                "first_seen": datetime.now().isoformat(),
                "last_seen": datetime.now().isoformat(),
                "total_messages": 0,
                "spam_count": 0,
                "ham_count": 0,
                "reputation_score": 0.5,
                "threat_types": {},
                "confidence_scores": [],
            }
        return self._cache[sender]

    def record_analysis(
        self,
        sender: str,
        is_spam: bool,
        confidence: float,
        threat_type: str | None = None,
    ) -> dict:
        with self._lock:
            entry = self._ensure_sender(sender)
            entry["last_seen"] = datetime.now().isoformat()
            entry["total_messages"] += 1
            if is_spam:
                entry["spam_count"] += 1
            else:
                entry["ham_count"] += 1
            entry["confidence_scores"].append(confidence)
            if len(entry["confidence_scores"]) > 100:
                entry["confidence_scores"] = entry["confidence_scores"][-100:]

            if threat_type:
                entry["threat_types"][threat_type] = (
                    entry["threat_types"].get(threat_type, 0) + 1
                )

            total = entry["total_messages"]
            spam_ratio = entry["spam_count"] / total if total > 0 else 0
            avg_conf = (
                sum(entry["confidence_scores"]) / len(entry["confidence_scores"])
                if entry["confidence_scores"]
                else 0.5
            )
            entry["reputation_score"] = 1.0 - (spam_ratio * 0.7 + avg_conf * 0.3)

            self._dirty = True
            return entry

    def get_reputation(self, sender: str) -> dict:
        with self._lock:
            entry = self._ensure_sender(sender)
            return {
                "sender": sender,
                "reputation_score": entry["reputation_score"],
                "total_messages": entry["total_messages"],
                "spam_count": entry["spam_count"],
                "ham_count": entry["ham_count"],
                "threat_types": entry["threat_types"],
                "first_seen": entry["first_seen"],
                "last_seen": entry["last_seen"],
            }

    def get_top_spam_senders(self, limit: int = 20) -> list[dict]:
        with self._lock:
            entries = []
            for sender, data in self._cache.items():
                if data["spam_count"] > 0:
                    entries.append(
                        {
                            "sender": sender,
                            "reputation_score": data["reputation_score"],
                            "spam_count": data["spam_count"],
                            "ham_count": data["ham_count"],
                            "total_messages": data["total_messages"],
                            "last_seen": data["last_seen"],
                        }
                    )
            entries.sort(key=lambda x: x["reputation_score"])
            return entries[:limit]

    def flush(self):
        if self._dirty:
            self._save()
