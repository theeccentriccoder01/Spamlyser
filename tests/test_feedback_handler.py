"""Tests for feedback persistence and statistics."""

from models.feedback_handler import FeedbackHandler, _coerce_rating


def test_coerce_rating_accepts_numeric_values():
    assert _coerce_rating(5) == 5.0
    assert _coerce_rating("4") == 4.0
    assert _coerce_rating(3.5) == 3.5


def test_coerce_rating_rejects_invalid_values():
    assert _coerce_rating(None) is None
    assert _coerce_rating("not-a-rating") is None
    assert _coerce_rating(True) is None


def test_feedback_stats_ignore_malformed_ratings(tmp_path):
    handler = FeedbackHandler(str(tmp_path / "feedback.json"))
    assert handler.save_feedback(
        {"feedback_type": "bug", "rating": "5", "email": "user@example.com"}
    )
    assert handler.save_feedback({"feedback_type": "bug", "rating": "bad"})
    assert handler.save_feedback({"feedback_type": "idea", "rating": 3})

    stats = handler.get_feedback_stats()

    assert stats["total"] == 3
    assert stats["average_rating"] == 4.0
    assert stats["by_type"] == {"bug": 2, "idea": 1}
    assert stats["has_email"] == 1
