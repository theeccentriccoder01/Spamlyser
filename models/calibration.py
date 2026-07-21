

"""
Model confidence calibration module for Expected Calibration Error (ECE),
Platt Scaling, and Temperature Scaling.
"""

from typing import Any, Dict, Tuple

import numpy as np

try:
    from scipy.optimize import minimize

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


class ConfidenceCalibrator:
    def __init__(
        self, temperature: float = 1.0, platt_a: float = 1.0, platt_b: float = 0.0
    ):
        self.temperature = temperature
        self.platt_a = platt_a
        self.platt_b = platt_b
        self.is_calibrated = False

    @staticmethod
    def logit(p: float) -> float:
        p = np.clip(p, 1e-15, 1 - 1e-15)
        return np.log(p / (1.0 - p))

    @staticmethod
    def sigmoid(x: float) -> float:
        return 1.0 / (1.0 + np.exp(-x))

    def calibrate_probability(self, prob: float, method: str = "temperature") -> float:
        """Calibrate a single probability using temperature scaling or Platt scaling."""
        if method == "temperature":
            l = self.logit(prob)
            return self.sigmoid(l / self.temperature)
        elif method == "platt":
            l = self.logit(prob)
            return self.sigmoid(self.platt_a * l + self.platt_b)
        return prob

    def calculate_ece(
        self, y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
    ) -> float:
        """Calculate Expected Calibration Error (ECE)."""
        y_true = np.array(y_true)
        y_prob = np.array(y_prob)

        y_pred = (y_prob >= 0.5).astype(int)
        confidences = np.where(y_pred == 1, y_prob, 1.0 - y_prob)
        accuracies = (y_pred == y_true).astype(float)

        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0

        for i in range(n_bins):
            bin_lower = bin_boundaries[i]
            bin_upper = bin_boundaries[i + 1]

            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            prop_in_bin = np.mean(in_bin)

            if prop_in_bin > 0:
                accuracy_in_bin = np.mean(accuracies[in_bin])
                avg_confidence_in_bin = np.mean(confidences[in_bin])
                ece += prop_in_bin * np.abs(avg_confidence_in_bin - accuracy_in_bin)

        return ece

    def fit_temperature(self, y_true: np.ndarray, y_prob: np.ndarray) -> float:
        """Find the optimal temperature T using negative log likelihood minimization."""
        if not SCIPY_AVAILABLE:
            import warnings

            warnings.warn("scipy not found, skipping temperature fitting", stacklevel=2)
            self.temperature = 1.0
            return self.temperature

        y_true = np.array(y_true)
        y_prob = np.array(y_prob)

        logits = np.vectorize(self.logit)(y_prob)

        def nll_loss(t):
            temp = max(t[0], 0.01)
            scaled_probs = self.sigmoid(logits / temp)
            scaled_probs = np.clip(scaled_probs, 1e-15, 1.0 - 1e-15)
            loss = -np.mean(
                y_true * np.log(scaled_probs)
                + (1.0 - y_true) * np.log(1.0 - scaled_probs)
            )
            return loss

        res = minimize(nll_loss, x0=[1.0], method="Nelder-Mead")
        self.temperature = max(res.x[0], 0.01)
        self.is_calibrated = True
        return self.temperature

    def fit_platt(self, y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, float]:
        """Find Platt scaling parameters A and B using logistic regression."""
        if not SCIPY_AVAILABLE:
            import warnings

            warnings.warn("scipy not found, skipping Platt scaling", stacklevel=2)
            self.platt_a = 1.0
            self.platt_b = 0.0
            return self.platt_a, self.platt_b

        y_true = np.array(y_true)
        y_prob = np.array(y_prob)

        logits = np.vectorize(self.logit)(y_prob)

        def nll_loss(params):
            a, b = params
            scaled_probs = self.sigmoid(a * logits + b)
            scaled_probs = np.clip(scaled_probs, 1e-15, 1.0 - 1e-15)
            loss = -np.mean(
                y_true * np.log(scaled_probs)
                + (1.0 - y_true) * np.log(1.0 - scaled_probs)
            )
            return loss

        res = minimize(nll_loss, x0=[1.0, 0.0], method="Nelder-Mead")
        self.platt_a, self.platt_b = res.x
        self.is_calibrated = True
        return self.platt_a, self.platt_b

    def generate_calibration_curve(
        self, y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
    ) -> dict[str, Any]:
        """Generate data for reliability diagram / calibration curve."""
        y_true = np.array(y_true)
        y_prob = np.array(y_prob)

        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_centers = []
        bin_accuracies = []
        bin_confidences = []
        bin_counts = []

        y_pred = (y_prob >= 0.5).astype(int)
        confidences = np.where(y_pred == 1, y_prob, 1.0 - y_prob)
        accuracies = (y_pred == y_true).astype(float)

        for i in range(n_bins):
            bin_lower = bin_boundaries[i]
            bin_upper = bin_boundaries[i + 1]
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)

            if np.sum(in_bin) > 0:
                bin_centers.append((bin_lower + bin_upper) / 2.0)
                bin_accuracies.append(np.mean(accuracies[in_bin]))
                bin_confidences.append(np.mean(confidences[in_bin]))
                bin_counts.append(int(np.sum(in_bin)))

        return {
            "bin_centers": bin_centers,
            "accuracies": bin_accuracies,
            "confidences": bin_confidences,
            "counts": bin_counts,
        }


class CalibratorStore:
    def __init__(self):
        self.calibrators = {}

    def get_calibrated_probability(self, raw_score: float) -> float:
        return 1.0 / (1.0 + raw_score)
