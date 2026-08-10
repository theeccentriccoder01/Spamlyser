import pytest
from models.statistical_drift_engine import StatisticalDriftEngine
from models.drift_tracker import ModelDriftTracker

def test_statistical_drift_engine_stable():
    engine = StatisticalDriftEngine()
    ref = [0.1, 0.2, 0.3, 0.4, 0.5, 0.8, 0.9] * 100
    cur = [0.10, 0.20, 0.30, 0.40, 0.50, 0.80, 0.90] * 100

    res = engine.evaluate_drift(ref, cur)
    assert res["drift_detected"] is False
    assert res["status"] == "STABLE"
    assert res["psi_score"] < 0.1

def test_statistical_drift_engine_critical_drift():
    engine = StatisticalDriftEngine()
    ref = [0.1, 0.12, 0.15, 0.18, 0.2] * 100
    cur = [0.8, 0.85, 0.9, 0.92, 0.95] * 100

    res = engine.evaluate_drift(ref, cur)
    assert res["drift_detected"] is True
    assert res["status"] == "CRITICAL_DRIFT"
    assert res["psi_score"] >= 0.25

def test_model_drift_tracker_integration():
    baseline_scores = [0.1, 0.2, 0.3, 0.4, 0.5] * 50
    tracker = ModelDriftTracker(baseline_accuracy=0.96, baseline_scores=baseline_scores)

    acc_drift = tracker.calculate_drift(0.90)
    assert acc_drift == 0.06

    dist_res = tracker.analyze_distribution_drift([0.8, 0.85, 0.9] * 50)
    assert dist_res["drift_detected"] is True
