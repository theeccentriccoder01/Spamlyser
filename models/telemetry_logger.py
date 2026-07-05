import json
import time
from pathlib import Path
from typing import Any, Dict


class TelemetryLogger:
    def __init__(self, log_path: str = "spamlyser_telemetry.json"):
        self.log_path = Path(log_path)

    def log_inference(self, duration_ms: float, confidence: float, classification: str):
        log_entry = {
            "timestamp": time.time(),
            "duration_ms": duration_ms,
            "confidence": confidence,
            "classification": classification,
        }
        try:
            data = []
            if self.log_path.exists():
                with open(self.log_path) as f:
                    data = json.load(f)
            data.append(log_entry)
            with open(self.log_path, "w") as f:
                json.dump(data[-100:], f, indent=2)
        except Exception:
            pass
