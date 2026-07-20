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


def test_concurrent_feedback_writes(tmp_path):
    """Simulate multiple Streamlit sessions writing feedback concurrently."""
    import threading

    handler = FeedbackHandler(str(tmp_path / "concurrent.json"))
    errors = []

    def write_feedback(thread_id):
        try:
            handler.save_feedback(
                {
                    "feedback_type": "test",
                    "rating": str(thread_id % 5 + 1),
                    "comment": f"Feedback from thread {thread_id}",
                }
            )
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=write_feedback, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)

    assert len(errors) == 0, f"Concurrent write errors: {errors}"
    # All 8 feedback entries should have been saved
    stats = handler.get_feedback_stats()
    assert stats["total"] == 8


def test_handler_cleanup_does_not_raise(tmp_path):
    """Verify that __del__ cleanup does not raise even when called multiple times."""
    handler = FeedbackHandler(str(tmp_path / "cleanup.json"))
    handler.save_feedback({"feedback_type": "test", "rating": "3"})
    # Manually calling __del__ should not raise
    handler.__del__()
    handler.__del__()  # Double call should be safe
