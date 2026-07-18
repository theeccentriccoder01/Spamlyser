from models.drift_tracker import ModelDriftTracker

def test_model_drift_tracker():
    tracker = ModelDriftTracker(0.90)
    assert tracker.calculate_drift(0.85) == 0.05
    assert tracker.calculate_drift(0.95) == 0.0
