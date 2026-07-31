"""
Statistical Data & Concept Drift Engine for Spamlyser ML models.
Computes Population Stability Index (PSI) and score distribution shifts.
"""

import math
from typing import List, Dict, Any, Tuple

class StatisticalDriftEngine:
    """
    Evaluates distribution drift between baseline reference scores and current production scores.
    """

    def __init__(self, num_bins: int = 10, psi_threshold: float = 0.25):
        self.num_bins = num_bins
        self.psi_threshold = psi_threshold

    def calculate_psi(self, reference: List[float], current: List[float]) -> float:
        """
        Calculate Population Stability Index (PSI) between baseline reference and current dataset.
        PSI < 0.1: No significant change.
        0.1 <= PSI < 0.25: Moderate drift.
        PSI >= 0.25: Significant drift detected.
        """
        if not reference or not current:
            return 0.0

        min_val = min(min(reference), min(current))
        max_val = max(max(reference), max(current))
        if min_val == max_val:
            return 0.0

        bin_width = (max_val - min_val) / self.num_bins
        bins = [min_val + i * bin_width for i in range(self.num_bins + 1)]

        ref_counts = [0] * self.num_bins
        cur_counts = [0] * self.num_bins

        for val in reference:
            idx = min(int((val - min_val) / bin_width), self.num_bins - 1)
            ref_counts[idx] += 1

        for val in current:
            idx = min(int((val - min_val) / bin_width), self.num_bins - 1)
            cur_counts[idx] += 1

        psi_total = 0.0
        ref_total = len(reference)
        cur_total = len(current)

        for i in range(self.num_bins):
            ref_pct = max(ref_counts[i] / ref_total, 1e-4)
            cur_pct = max(cur_counts[i] / cur_total, 1e-4)
            psi_total += (cur_pct - ref_pct) * math.log(cur_pct / ref_pct)

        return round(psi_total, 4)

    def evaluate_drift(self, reference_scores: List[float], current_scores: List[float]) -> Dict[str, Any]:
        """
        Evaluate full statistical drift status.
        """
        psi = self.calculate_psi(reference_scores, current_scores)
        drift_detected = psi >= self.psi_threshold

        status = "CRITICAL_DRIFT" if psi >= 0.25 else ("MODERATE_DRIFT" if psi >= 0.1 else "STABLE")

        return {
            "psi_score": psi,
            "drift_detected": drift_detected,
            "status": status,
            "sample_sizes": {
                "reference": len(reference_scores),
                "current": len(current_scores)
            }
        }
