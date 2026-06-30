"""
Feedback handler for Spamlyser Pro.
Handles storing and retrieving user feedback with SQLite for concurrent write safety.
"""

import json
import os
import sqlite3
import threading
from datetime import datetime
from typing import Any, Dict, List

import streamlit as st

_local = threading.local()


def _get_connection(db_path: str) -> sqlite3.Connection:
    """Get a thread-local SQLite connection with WAL mode enabled."""
    if not hasattr(_local, "conn") or _local.conn is None:
        _local.conn = sqlite3.connect(db_path, check_same_thread=False)
        _local.conn.row_factory = sqlite3.Row
        _local.conn.execute("PRAGMA journal_mode=WAL")
        _local.conn.execute("PRAGMA busy_timeout=5000")
    return _local.conn


class FeedbackHandler:
    """Handles user feedback operations using SQLite for concurrent write safety."""

    def __init__(self, feedback_file: str = "feedback_data.json"):
        self.db_path = os.path.splitext(feedback_file)[0] + ".db"
        self._init_db()
        self._migrate_from_json(feedback_file)

    def _init_db(self) -> None:
        conn = _get_connection(self.db_path)
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                data TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.commit()

    def _migrate_from_json(self, json_path: str) -> None:
        if not os.path.exists(json_path):
            return
        conn = _get_connection(self.db_path)
        existing = conn.execute("SELECT COUNT(*) FROM feedback").fetchone()[0]
        if existing > 0:
            return
        try:
            with open(json_path) as f:
                entries = json.load(f)
            if not isinstance(entries, list):
                return
            for entry in entries:
                conn.execute(
                    "INSERT INTO feedback (data) VALUES (?)",
                    (json.dumps(entry),),
                )
            conn.commit()
            os.rename(json_path, json_path + ".bak")
        except Exception as e:
            st.warning(f"Could not migrate feedback from {json_path}: {e}")

    def save_feedback(self, feedback_data: dict[str, Any]) -> bool:
        try:
            if "timestamp" not in feedback_data:
                feedback_data["timestamp"] = datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
            conn = _get_connection(self.db_path)
            conn.execute(
                "INSERT INTO feedback (data) VALUES (?)",
                (json.dumps(feedback_data),),
            )
            conn.commit()
            return True
        except Exception as e:
            st.error(f"Error saving feedback: {e!s}")
            return False

    def get_all_feedback(self) -> list[dict[str, Any]]:
        try:
            conn = _get_connection(self.db_path)
            rows = conn.execute("SELECT id, data FROM feedback ORDER BY id").fetchall()
            return [json.loads(row["data"]) for row in rows]
        except Exception:
            return []

    def get_feedback_by_type(self, feedback_type: str) -> list[dict[str, Any]]:
        all_feedback = self.get_all_feedback()
        return [f for f in all_feedback if f.get("feedback_type") == feedback_type]

    def get_word_analysis_feedback(self) -> list[dict[str, Any]]:
        all_feedback = self.get_all_feedback()
        return [f for f in all_feedback if f.get("context") == "Word Analysis"]

    def get_feedback_stats(self) -> dict[str, Any]:
        feedbacks = self.get_all_feedback()

        if not feedbacks:
            return {"total": 0, "average_rating": 0, "by_type": {}, "has_email": 0}

        stats: dict[str, Any] = {
            "total": len(feedbacks),
            "by_type": {},
            "has_email": sum(1 for f in feedbacks if f.get("email")),
        }

        ratings = [f.get("rating", 0) for f in feedbacks if f.get("rating") is not None]
        stats["average_rating"] = sum(ratings) / len(ratings) if ratings else 0

        for feedback in feedbacks:
            feedback_type = feedback.get("feedback_type", "unknown")
            stats["by_type"][feedback_type] = stats["by_type"].get(feedback_type, 0) + 1

        return stats

    def export_to_github_issue(self, feedback_id: int) -> str:
        feedbacks = self.get_all_feedback()

        if feedback_id < 0 or feedback_id >= len(feedbacks):
            return "Invalid feedback ID"

        feedback = feedbacks[feedback_id]

        issue_body = f"""
## User Feedback

**Type:** {feedback.get("feedback_type", "Not specified")}
**Rating:** {"⭐" * feedback.get("rating", 0)}
**Date:** {feedback.get("timestamp", "Not recorded")}

### Message:
{feedback.get("message", "No message provided")}

### Contact:
{feedback.get("email", "No contact information provided")}
"""
        return issue_body
