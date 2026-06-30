"""
Regression tests for issue #44:
BatchProcessor.process_message() called self.ensemble_classifier.get_model_prediction()
but EnsembleSpamClassifier defined no such method, causing:

    AttributeError: 'EnsembleSpamClassifier' object has no attribute 'get_model_prediction'

Fix: add get_model_prediction(model_name, message) to EnsembleSpamClassifier
returning a PredictionResult fallback when no real inference pipeline is wired up.
"""

import os
import sys
import types

# Stub heavy dependencies before any models import
sys.modules.setdefault("numpy", types.ModuleType("numpy"))
import unittest.mock as _mock

sys.modules["numpy"].var = _mock.MagicMock(return_value=0.0)
sys.modules["numpy"].mean = _mock.MagicMock(return_value=0.5)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.batch_processor import BatchProcessor
from models.ensemble_classifier_method import (
    EnsembleSpamClassifier,
    ModelPerformanceTracker,
    PredictionResult,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_classifier() -> EnsembleSpamClassifier:
    tracker = ModelPerformanceTracker()
    return EnsembleSpamClassifier(performance_tracker=tracker)


KNOWN_MODELS = ["DistilBERT", "BERT", "RoBERTa", "ALBERT"]


# ---------------------------------------------------------------------------
# Tests for the new get_model_prediction() method
# ---------------------------------------------------------------------------


class TestGetModelPredictionExists:
    """The method must exist and be callable — core regression for issue #44."""

    def test_method_exists_on_class(self):
        clf = _make_classifier()
        assert hasattr(clf, "get_model_prediction"), (
            "EnsembleSpamClassifier is missing get_model_prediction(). "
            "BatchProcessor.process_message() will crash with AttributeError."
        )
        assert callable(clf.get_model_prediction)

    def test_returns_prediction_result(self):
        """Return value must be a PredictionResult (or duck-type compatible)."""
        clf = _make_classifier()
        result = clf.get_model_prediction("DistilBERT", "hello world")
        assert isinstance(result, PredictionResult), (
            f"Expected PredictionResult, got {type(result)}"
        )

    def test_result_has_required_attributes(self):
        """BatchProcessor accesses .label, .score, .spam_probability."""
        clf = _make_classifier()
        result = clf.get_model_prediction("BERT", "win a prize now")
        assert hasattr(result, "label"), "PredictionResult missing .label"
        assert hasattr(result, "score"), "PredictionResult missing .score"
        assert hasattr(result, "spam_probability"), (
            "PredictionResult missing .spam_probability"
        )

    def test_label_is_spam_or_ham(self):
        clf = _make_classifier()
        result = clf.get_model_prediction("RoBERTa", "test message")
        assert result.label in ("SPAM", "HAM"), (
            f"label must be 'SPAM' or 'HAM', got '{result.label}'"
        )

    def test_score_in_unit_interval(self):
        clf = _make_classifier()
        result = clf.get_model_prediction("ALBERT", "click here to claim")
        assert 0.0 <= result.score <= 1.0, (
            f"score must be in [0, 1], got {result.score}"
        )

    def test_spam_probability_in_unit_interval(self):
        clf = _make_classifier()
        result = clf.get_model_prediction("DistilBERT", "call now")
        assert 0.0 <= result.spam_probability <= 1.0, (
            f"spam_probability must be in [0, 1], got {result.spam_probability}"
        )

    def test_all_known_models_accepted(self):
        """Method must not raise for any of the four known model names."""
        clf = _make_classifier()
        for model in KNOWN_MODELS:
            result = clf.get_model_prediction(model, "test")
            assert result is not None, f"Got None for model '{model}'"


# ---------------------------------------------------------------------------
# Integration: BatchProcessor.process_message() must not crash
# ---------------------------------------------------------------------------


class _FakeEnsembleWithRealMethod(EnsembleSpamClassifier):
    """Subclass that wires get_model_prediction to a deterministic result
    so process_message() can run end-to-end without loading real models."""

    def get_model_prediction(self, model_name, message):
        return PredictionResult(
            method=model_name,
            label="HAM",
            confidence=0.9,
            spam_probability=0.1,
            details="test stub",
            metadata={},
        )

    def get_all_predictions(self, predictions):
        return {
            "majority_voting": {
                "label": "HAM",
                "confidence": 0.9,
                "spam_probability": 0.1,
            }
        }


class TestBatchProcessorIntegration:
    """process_message() and process_batch() must work end-to-end after fix."""

    def setup_method(self):
        self.bp = BatchProcessor(ensemble_classifier=_FakeEnsembleWithRealMethod())

    def test_process_message_does_not_raise_attribute_error(self):
        """The core bug: this used to crash with AttributeError."""
        try:
            result = self.bp.process_message("hello world")
        except AttributeError as exc:
            raise AssertionError(
                f"process_message() raised AttributeError — fix not applied: {exc}"
            ) from exc
        assert result is not None

    def test_process_message_returns_expected_keys(self):
        result = self.bp.process_message("win a prize")
        for key in (
            "message",
            "model_predictions",
            "ensemble_predictions",
            "risk_indicators",
            "timestamp",
        ):
            assert key in result, f"Missing key '{key}' in process_message result"

    def test_process_message_has_all_model_predictions(self):
        result = self.bp.process_message("urgent: click now")
        for model in KNOWN_MODELS:
            assert model in result["model_predictions"], (
                f"Model '{model}' missing from model_predictions"
            )

    def test_process_batch_single_message(self):
        results, stats = self.bp.process_batch(["hi there"])
        assert len(results) == 1
        assert stats["processed_messages"] == 1

    def test_process_batch_multiple_messages(self):
        messages = ["hello", "win cash now", "call 9876543210"]
        results, stats = self.bp.process_batch(messages)
        assert len(results) == 3
        assert stats["total_messages"] == 3
