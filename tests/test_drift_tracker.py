"""Tests for ModelDriftTracker — comprehensive coverage.

Covers drift calculation, severity classification, trend detection,
windowed history, summary generation, and edge cases.
"""

import pytest

from models.drift_tracker import (
    DriftSeverity,
    DriftTrend,
    ModelDriftTracker,
)


# ── Backward-compatibility: original calculate_drift API ──────────────


class TestCalculateDrift:
    """Ensure the original public API still works identically."""

    def test_drift_positive_drop(self):
        tracker = ModelDriftTracker(baseline_accuracy=0.90)
        assert tracker.calculate_drift(0.85) == 0.05

    def test_drift_no_drop(self):
        tracker = ModelDriftTracker(baseline_accuracy=0.90)
        assert tracker.calculate_drift(0.95) == 0.0

    def test_drift_exact_match(self):
        tracker = ModelDriftTracker(baseline_accuracy=0.90)
        assert tracker.calculate_drift(0.90) == 0.0

    def test_drift_zero_accuracy(self):
        tracker = ModelDriftTracker(baseline_accuracy=0.95)
        assert tracker.calculate_drift(0.0) == 0.95

    def test_drift_rounding(self):
        """Drift should be rounded to 4 decimal places."""
        tracker = ModelDriftTracker(baseline_accuracy=0.9)
        # 0.9 - 0.8333 = 0.0667
        assert tracker.calculate_drift(0.8333) == 0.0667


# ── Recording and window management ───────────────────────────────────


class TestRecording:
    def test_record_single(self):
        tracker = ModelDriftTracker()
        tracker.record(0.93)
        assert tracker.current_accuracy == 0.93
        assert len(tracker.history) == 1

    def test_record_multiple(self):
        tracker = ModelDriftTracker()
        for acc in [0.93, 0.92, 0.91]:
            tracker.record(acc)
        assert tracker.current_accuracy == 0.91
        assert tracker.history == [0.93, 0.92, 0.91]

    def test_window_eviction(self):
        """Oldest entries are dropped when window_size is exceeded."""
        tracker = ModelDriftTracker(window_size=3)
        for acc in [0.95, 0.94, 0.93, 0.92, 0.91]:
            tracker.record(acc)
        assert len(tracker.history) == 3
        assert tracker.history == [0.93, 0.92, 0.91]

    def test_history_returns_copy(self):
        """Mutating the returned list must not affect internal state."""
        tracker = ModelDriftTracker()
        tracker.record(0.90)
        h = tracker.history
        h.append(0.50)
        assert len(tracker.history) == 1

    def test_current_accuracy_empty(self):
        tracker = ModelDriftTracker()
        assert tracker.current_accuracy is None

    def test_reset_clears_history(self):
        tracker = ModelDriftTracker()
        tracker.record(0.90)
        tracker.record(0.88)
        tracker.reset()
        assert tracker.history == []
        assert tracker.current_accuracy is None


# ── Severity classification ───────────────────────────────────────────


class TestSeverity:
    def test_none_when_no_drift(self):
        tracker = ModelDriftTracker(baseline_accuracy=0.95)
        assert tracker.classify_severity(0.0) == DriftSeverity.NONE

    def test_none_below_minor(self):
        tracker = ModelDriftTracker(baseline_accuracy=0.95)
        assert tracker.classify_severity(0.01) == DriftSeverity.NONE

    def test_minor_at_boundary(self):
        tracker = ModelDriftTracker(baseline_accuracy=0.95)
        assert tracker.classify_severity(0.02) == DriftSeverity.MINOR

    def test_minor_in_range(self):
        tracker = ModelDriftTracker(baseline_accuracy=0.95)
        assert tracker.classify_severity(0.04) == DriftSeverity.MINOR

    def test_moderate_at_boundary(self):
        tracker = ModelDriftTracker(baseline_accuracy=0.95)
        assert tracker.classify_severity(0.05) == DriftSeverity.MODERATE

    def test_moderate_in_range(self):
        tracker = ModelDriftTracker(baseline_accuracy=0.95)
        assert tracker.classify_severity(0.08) == DriftSeverity.MODERATE

    def test_critical_at_boundary(self):
        tracker = ModelDriftTracker(baseline_accuracy=0.95)
        assert tracker.classify_severity(0.10) == DriftSeverity.CRITICAL

    def test_critical_extreme(self):
        tracker = ModelDriftTracker(baseline_accuracy=0.95)
        assert tracker.classify_severity(0.50) == DriftSeverity.CRITICAL

    def test_severity_from_recorded_accuracy(self):
        """classify_severity() without args uses current_accuracy."""
        tracker = ModelDriftTracker(baseline_accuracy=0.95)
        tracker.record(0.84)  # drift = 0.11 → critical
        assert tracker.classify_severity() == DriftSeverity.CRITICAL

    def test_severity_empty_history(self):
        """No history should return NONE."""
        tracker = ModelDriftTracker()
        assert tracker.classify_severity() == DriftSeverity.NONE

    def test_custom_thresholds(self):
        custom = {"minor": 0.01, "moderate": 0.03, "critical": 0.05}
        tracker = ModelDriftTracker(baseline_accuracy=0.95, thresholds=custom)
        assert tracker.classify_severity(0.01) == DriftSeverity.MINOR
        assert tracker.classify_severity(0.03) == DriftSeverity.MODERATE
        assert tracker.classify_severity(0.05) == DriftSeverity.CRITICAL


# ── Trend detection ───────────────────────────────────────────────────


class TestTrend:
    def test_stable_with_empty_history(self):
        tracker = ModelDriftTracker()
        assert tracker.detect_trend() == DriftTrend.STABLE

    def test_stable_with_one_record(self):
        tracker = ModelDriftTracker()
        tracker.record(0.95)
        assert tracker.detect_trend() == DriftTrend.STABLE

    def test_degrading_trend(self):
        tracker = ModelDriftTracker()
        # Clearly declining: 0.95 → 0.80
        for acc in [0.95, 0.94, 0.93, 0.92, 0.91, 0.88, 0.86, 0.84, 0.82, 0.80]:
            tracker.record(acc)
        assert tracker.detect_trend() == DriftTrend.DEGRADING

    def test_improving_trend(self):
        tracker = ModelDriftTracker()
        # Clearly improving: 0.80 → 0.95
        for acc in [0.80, 0.82, 0.84, 0.86, 0.88, 0.90, 0.92, 0.93, 0.94, 0.95]:
            tracker.record(acc)
        assert tracker.detect_trend() == DriftTrend.IMPROVING

    def test_stable_flat_line(self):
        tracker = ModelDriftTracker()
        for _ in range(10):
            tracker.record(0.95)
        assert tracker.detect_trend() == DriftTrend.STABLE

    def test_stable_minor_jitter(self):
        """Very small fluctuations (< 0.005 diff) should be STABLE."""
        tracker = ModelDriftTracker()
        for acc in [0.950, 0.951, 0.949, 0.950, 0.952, 0.950, 0.951, 0.949, 0.950, 0.951]:
            tracker.record(acc)
        assert tracker.detect_trend() == DriftTrend.STABLE

    def test_trend_with_small_history(self):
        """With just 2 records, trend should still work."""
        tracker = ModelDriftTracker()
        tracker.record(0.95)
        tracker.record(0.80)
        assert tracker.detect_trend() == DriftTrend.DEGRADING

    def test_custom_lookback(self):
        tracker = ModelDriftTracker()
        # With lookback=2: recent=[0.80, 0.80], previous=[0.90, 0.90] → degrading
        for acc in [0.90, 0.90, 0.80, 0.80]:
            tracker.record(acc)
        assert tracker.detect_trend(lookback=2) == DriftTrend.DEGRADING


# ── Summary ───────────────────────────────────────────────────────────


class TestSummary:
    def test_summary_with_data(self):
        tracker = ModelDriftTracker(baseline_accuracy=0.95, window_size=50)
        tracker.record(0.93)
        tracker.record(0.90)
        s = tracker.summary()

        assert s["baseline"] == 0.95
        assert s["current_accuracy"] == 0.90
        assert s["drift"] == 0.05
        assert s["severity"] == "moderate"
        assert s["trend"] in ("stable", "degrading")
        assert s["records_count"] == 2
        assert s["window_size"] == 50

    def test_summary_empty(self):
        tracker = ModelDriftTracker()
        s = tracker.summary()
        assert s["current_accuracy"] is None
        assert s["drift"] == 0.0
        assert s["severity"] == "none"
        assert s["trend"] == "stable"
        assert s["records_count"] == 0

    def test_summary_keys(self):
        tracker = ModelDriftTracker()
        expected_keys = {
            "baseline",
            "current_accuracy",
            "drift",
            "severity",
            "trend",
            "records_count",
            "window_size",
        }
        assert set(tracker.summary().keys()) == expected_keys


# ── Enum values ───────────────────────────────────────────────────────


class TestEnums:
    def test_severity_values(self):
        assert DriftSeverity.NONE.value == "none"
        assert DriftSeverity.MINOR.value == "minor"
        assert DriftSeverity.MODERATE.value == "moderate"
        assert DriftSeverity.CRITICAL.value == "critical"

    def test_trend_values(self):
        assert DriftTrend.STABLE.value == "stable"
        assert DriftTrend.IMPROVING.value == "improving"
        assert DriftTrend.DEGRADING.value == "degrading"
