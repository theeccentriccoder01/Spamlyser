"""Tests for feedback-to-issue markdown export."""

from models.feedback_handler import FeedbackHandler, _format_rating_stars


def test_format_rating_stars_accepts_numeric_strings():
    assert _format_rating_stars("4") == "⭐⭐⭐⭐"
    assert _format_rating_stars(2.8) == "⭐⭐"


def test_format_rating_stars_ignores_invalid_values():
    assert _format_rating_stars("bad") == ""
    assert _format_rating_stars(None) == ""
    assert _format_rating_stars(True) == ""


def test_export_to_github_issue_handles_string_rating(tmp_path):
    handler = FeedbackHandler(str(tmp_path / "feedback.json"))
    assert handler.save_feedback(
        {
            "feedback_type": "bug",
            "rating": "5",
            "message": "Incorrect classification",
            "email": "user@example.com",
        }
    )

    body = handler.export_to_github_issue(0)

    assert "**Rating:** ⭐⭐⭐⭐⭐" in body
    assert "Incorrect classification" in body
    assert "user@example.com" in body
