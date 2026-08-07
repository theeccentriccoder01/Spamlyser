"""
Model Drift Tracker for monitoring concept and feature distribution drift.
"""

from typing import List, Dict, Any
from .statistical_drift_engine import StatisticalDriftEngine

class ModelDriftTracker:
    def __init__(self, baseline_accuracy: float = 0.95, baseline_scores: List[float] = None):
        self.baseline_accuracy = baseline_accuracy
        self.baseline_scores = baseline_scores or []
        self.engine = StatisticalDriftEngine()
        
    def calculate_drift(self, current_accuracy: float) -> float:
        return round(max(0.0, self.baseline_accuracy - current_accuracy), 4)

    def analyze_distribution_drift(self, current_scores: List[float]) -> Dict[str, Any]:
        """Analyze score distribution drift using StatisticalDriftEngine."""
        return self.engine.evaluate_drift(self.baseline_scores, current_scores)
