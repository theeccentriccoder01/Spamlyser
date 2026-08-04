"""
Model Drift Tracker for Spamlyser Pro.

Monitors changes in model accuracy over time by maintaining a windowed
history of accuracy snapshots.  Drift is classified into severity levels
and a trend direction is computed so operators can react before model
quality degrades significantly.

Usage::

    tracker = ModelDriftTracker(baseline_accuracy=0.95)
    tracker.record(0.94)
    tracker.record(0.92)
    summary = tracker.summary()
    # {'current_accuracy': 0.92, 'drift': 0.03, 'severity': 'moderate', ...}
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class DriftSeverity(Enum):
    """Classification of how severe the observed drift is."""

    NONE = "none"
    MINOR = "minor"
    MODERATE = "moderate"
    CRITICAL = "critical"


class DriftTrend(Enum):
    """Direction in which the model accuracy is moving."""

    STABLE = "stable"
    IMPROVING = "improving"
    DEGRADING = "degrading"


# Default thresholds — expressed as *absolute* drops from the baseline.
_DEFAULT_THRESHOLDS: dict[str, float] = {
    "minor": 0.02,      # ≥2 pp drop
    "moderate": 0.05,    # ≥5 pp drop
    "critical": 0.10,    # ≥10 pp drop
}


@dataclass
class ModelDriftTracker:
    """Track model accuracy drift against a fixed baseline.

    Parameters
    ----------
    baseline_accuracy:
        The reference accuracy the model was validated at (e.g. 0.95).
    window_size:
        Maximum number of accuracy records to retain.  Older entries are
        evicted automatically so memory stays bounded.
    thresholds:
        Optional custom mapping of severity names to drift-magnitude
        boundaries.  Keys must be ``"minor"``, ``"moderate"``, and
        ``"critical"``; values are absolute drops from the baseline.
    """

    baseline_accuracy: float = 0.95
    window_size: int = 100
    thresholds: dict[str, float] = field(default_factory=lambda: dict(_DEFAULT_THRESHOLDS))
    _history: list[float] = field(default_factory=list, init=False, repr=False)

    # ── Core API ───────────────────────────────────────────────────────

    def calculate_drift(self, current_accuracy: float) -> float:
        """Return the non-negative drift between *baseline* and *current_accuracy*."""
        return round(max(0.0, self.baseline_accuracy - current_accuracy), 4)

    def record(self, accuracy: float) -> None:
        """Append an accuracy snapshot and enforce the window limit."""
        self._history.append(accuracy)
        if len(self._history) > self.window_size:
            self._history = self._history[-self.window_size:]

    @property
    def history(self) -> list[float]:
        """Return a copy of the recorded accuracy history."""
        return list(self._history)

    @property
    def current_accuracy(self) -> float | None:
        """Most recently recorded accuracy, or ``None`` if empty."""
        return self._history[-1] if self._history else None

    # ── Severity ───────────────────────────────────────────────────────

    def classify_severity(self, drift: float | None = None) -> DriftSeverity:
        """Classify drift magnitude into a severity bucket.

        If *drift* is not supplied the latest recorded accuracy is used.
        """
        if drift is None:
            if self.current_accuracy is None:
                return DriftSeverity.NONE
            drift = self.calculate_drift(self.current_accuracy)

        if drift >= self.thresholds.get("critical", 0.10):
            return DriftSeverity.CRITICAL
        if drift >= self.thresholds.get("moderate", 0.05):
            return DriftSeverity.MODERATE
        if drift >= self.thresholds.get("minor", 0.02):
            return DriftSeverity.MINOR
        return DriftSeverity.NONE

    # ── Trend detection ────────────────────────────────────────────────

    def detect_trend(self, lookback: int = 5) -> DriftTrend:
        """Determine if accuracy is improving, degrading, or stable.

        Compares the mean of the last *lookback* entries against the mean
        of the *lookback* entries before them.  If fewer than
        ``2 * lookback`` records exist, falls back to comparing the first
        and last half of the available history.

        Returns :pyattr:`DriftTrend.STABLE` when there are fewer than 2
        records or the difference is negligible (< 0.005).
        """
        if len(self._history) < 2:
            return DriftTrend.STABLE

        if len(self._history) >= 2 * lookback:
            recent = self._history[-lookback:]
            previous = self._history[-2 * lookback:-lookback]
        else:
            mid = len(self._history) // 2
            previous = self._history[:mid]
            recent = self._history[mid:]

        avg_recent = sum(recent) / len(recent)
        avg_previous = sum(previous) / len(previous)
        diff = avg_recent - avg_previous

        if abs(diff) < 0.005:
            return DriftTrend.STABLE
        return DriftTrend.IMPROVING if diff > 0 else DriftTrend.DEGRADING

    # ── Summary ────────────────────────────────────────────────────────

    def summary(self) -> dict:
        """Return a single dict summarising the current drift state.

        Keys
        ----
        baseline, current_accuracy, drift, severity, trend,
        records_count, window_size
        """
        drift = (
            self.calculate_drift(self.current_accuracy)
            if self.current_accuracy is not None
            else 0.0
        )
        return {
            "baseline": self.baseline_accuracy,
            "current_accuracy": self.current_accuracy,
            "drift": drift,
            "severity": self.classify_severity(drift).value,
            "trend": self.detect_trend().value,
            "records_count": len(self._history),
            "window_size": self.window_size,
        }

    def reset(self) -> None:
        """Clear all recorded history."""
        self._history.clear()
