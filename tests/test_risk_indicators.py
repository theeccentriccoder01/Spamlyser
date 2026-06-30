"""
Regression tests for issue #14:

`BatchProcessor._analyze_risk_indicators` divided the count of "suspicious"
characters by `len(message)` with no guard. An empty message (e.g. a blank
row in an uploaded CSV) made `len(message) == 0` and raised
`ZeroDivisionError`, crashing batch analysis.

These tests run headless on CI without loading any transformer models: they
stub `streamlit`/`fpdf` only when those packages are absent (the real packages
are present on CI via requirements.txt) and drive `BatchProcessor` with a tiny
fake ensemble classifier.
"""

import sys
import types


def _ensure_stub(name: str) -> None:
    """Provide a minimal stub for `name` only if it cannot be imported.

    `models/__init__.py` imports `export_feature`, which imports `streamlit`
    and `fpdf`. On CI those are installed, so the real modules are used and
    these stubs are never created.
    """
    if name in sys.modules:
        return
    try:
        __import__(name)
    except Exception:
        module = types.ModuleType(name)
        if name == "fpdf":

            class FPDF:  # minimal placeholder; unused by these tests
                pass

            module.FPDF = FPDF
        sys.modules[name] = module


_ensure_stub("streamlit")
_ensure_stub("fpdf")

from models.batch_processor import BatchProcessor


class _FakePrediction:
    """Stand-in for a single model's prediction result object."""

    def __init__(self, label="HAM", score=0.5, spam_probability=0.5):
        self.label = label
        self.score = score
        self.spam_probability = spam_probability


class _FakeEnsemble:
    """Minimal ensemble classifier so `process_message` runs without models."""

    def get_model_prediction(self, model_name, message):
        return _FakePrediction()

    def get_all_predictions(self, predictions):
        return {"final_prediction": {"label": "HAM", "confidence": 0.5}}


def _make_processor():
    bp = BatchProcessor.__new__(BatchProcessor)  # bypass model-loading __init__
    bp.ensemble_classifier = _FakeEnsemble()
    bp.batch_stats = {}
    return bp


def test_empty_message_risk_indicators_does_not_crash():
    """The exact issue #14 trigger: an empty message must not raise."""
    bp = _make_processor()
    indicators = bp._analyze_risk_indicators("")
    assert indicators["suspicious_chars"] is False


def test_whitespace_only_message_does_not_crash():
    """A whitespace-only message (len > 0) must also be handled safely."""
    bp = _make_processor()
    indicators = bp._analyze_risk_indicators("   ")
    assert indicators["suspicious_chars"] is False


def test_normal_message_still_flags_suspicious_chars():
    """Happy path preserved: a high special-char message still flags True."""
    bp = _make_processor()
    indicators = bp._analyze_risk_indicators("!!!@@@###$$$ %%%^^^&&&")
    assert indicators["suspicious_chars"] is True


def test_normal_clean_message_not_flagged():
    """A clean sentence has a low special-char ratio -> False (not crashing)."""
    bp = _make_processor()
    indicators = bp._analyze_risk_indicators("hello how are you today")
    assert indicators["suspicious_chars"] is False


def test_process_message_with_empty_string_does_not_crash():
    """
    Real call path: process_message -> _analyze_risk_indicators.

    A blank row reaching process_message as "" previously crashed here with
    ZeroDivisionError (this call is not inside the model-prediction try/except).
    """
    bp = _make_processor()
    result = bp.process_message("")
    assert result["risk_indicators"]["suspicious_chars"] is False
    assert result["message"] == ""
