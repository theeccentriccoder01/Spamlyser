class ModelDriftTracker:
    def __init__(self, baseline_accuracy: float = 0.95):
        self.baseline_accuracy = baseline_accuracy
        
    def calculate_drift(self, current_accuracy: float) -> float:
        return max(0.0, self.baseline_accuracy - current_accuracy)
