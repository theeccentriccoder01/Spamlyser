"""
Tests for BatchProcessor.process_batch ZeroDivisionError guards (issue #13).

These tests use a lightweight fake ensemble classifier so they run headless on
CI without loading any transformer models.
"""

import itertools
from datetime import datetime, timedelta
from unittest import mock

from models.batch_processor import BatchProcessor


class _FakePrediction:
    def __init__(self, label="HAM", score=0.1, spam_probability=0.1):
        self.label = label
        self.score = score
        self.spam_probability = spam_probability


class _FakeEnsemble:
    """Minimal stand-in for EnsembleSpamClassifier so no real models load."""

    def get_model_prediction(self, model_name, message):
        return _FakePrediction()

    def get_all_predictions(self, predictions):
        return {
            "majority_voting": {
                "label": "HAM",
                "confidence": 0.1,
                "spam_probability": 0.1,
            }
        }


def _make_processor():
    return BatchProcessor(ensemble_classifier=_FakeEnsemble())


def test_empty_batch_does_not_crash():
    """Empty input must not raise ZeroDivisionError and must report zeroed stats."""
    bp = _make_processor()
    results, stats = bp.process_batch([])
    assert results == []
    assert stats["total_messages"] == 0
    assert stats["processed_messages"] == 0
    assert stats["processing_time"] == 0.0
    assert stats["messages_per_second"] == 0.0


def test_zero_batch_size_uses_one_worker():
    """Invalid UI/API batch sizes should not crash ThreadPoolExecutor."""
    bp = _make_processor()
    results, stats = bp.process_batch(["hello"], batch_size=0)
    assert len(results) == 1
    assert stats["processed_messages"] == 1


def test_negative_batch_size_uses_one_worker():
    """Negative batch sizes are clamped to one worker."""
    bp = _make_processor()
    results, stats = bp.process_batch(["hello"], batch_size=-5)
    assert len(results) == 1
    assert stats["processed_messages"] == 1


def test_batch_size_is_capped_to_max_workers():
    """Large batch sizes should stay within the worker cap."""
    assert BatchProcessor._normalise_batch_size(500) == 10
    assert BatchProcessor._normalise_batch_size("bad") == 1


def test_instantaneous_batch_does_not_crash():
    """When elapsed time rounds to 0.0, the throughput guard prevents a crash."""
    bp = _make_processor()
    fixed = datetime(2024, 1, 1, 0, 0, 0)
    # Every datetime.now() returns the same instant => processing_time == 0.0
    with mock.patch("models.batch_processor.datetime") as m:
        m.now.return_value = fixed
        results, stats = bp.process_batch(["hello", "win cash now"])
    assert len(results) == 2
    assert stats["processing_time"] == 0.0
    assert stats["messages_per_second"] == 0.0  # guarded, not ZeroDivisionError


def test_normal_batch_still_computes_throughput():
    """A normal (non-instantaneous) batch still reports positive throughput."""
    bp = _make_processor()
    base = datetime(2024, 1, 1, 0, 0, 0)
    clock = itertools.count()  # start is the first now() call, end is the last
    with mock.patch("models.batch_processor.datetime") as m:
        m.now.side_effect = lambda: base + timedelta(seconds=next(clock))
        results, stats = bp.process_batch(["a", "b"])
    assert len(results) == 2
    assert stats["processing_time"] > 0
    assert stats["messages_per_second"] > 0


def test_rate_limiter():
    from models.batch_processor import RateLimiter

    limiter = RateLimiter(2, 60)
    assert limiter.allow_request() is True
    assert limiter.allow_request() is True
    assert limiter.allow_request() is False
