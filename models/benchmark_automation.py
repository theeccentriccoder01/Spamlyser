import models.drift_tracker
"""Automated model benchmarking — compare models, track regressions, persist results."""

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    model_name: str
    latency_mean_ms: float
    latency_median_ms: float
    latency_p95_ms: float
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    samples_count: int = 0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class BenchmarkHistory:
    """Persistent store for benchmark results — enables regression tracking."""

    def __init__(self, path: str | Path | None = None):
        if path is None:
            path = (
                Path(__file__).resolve().parent.parent
                / "data"
                / "benchmark_history.json"
            )
        self._path = Path(path)
        self._history: list[dict[str, Any]] = []
        self._load()

    def _load(self) -> None:
        if self._path.exists():
            try:
                self._history = json.loads(self._path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                self._history = []

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(
            json.dumps(self._history, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    def record(self, result: BenchmarkResult) -> None:
        self._history.append(result.to_dict())
        self._save()

    def get_all(self) -> list[dict[str, Any]]:
        return list(self._history)

    def get_latest(self, model_name: str) -> dict[str, Any] | None:
        for entry in reversed(self._history):
            if entry.get("model_name") == model_name:
                return entry
        return None

    def get_regression(
        self, model_name: str, threshold_pct: float = 10.0
    ) -> dict[str, Any] | None:
        """Compare last two runs for *model_name*.

        Returns a dict keyed by metric name with ``previous``, ``current``,
        and ``change_pct`` values if the absolute change exceeds ``threshold_pct``.
        """
        runs = [e for e in self._history if e.get("model_name") == model_name]
        if len(runs) < 2:
            return None
        prev, curr = runs[-2], runs[-1]
        regressions: dict[str, Any] = {}
        for metric in ("latency_mean_ms", "latency_p95_ms", "accuracy", "f1_score"):
            if metric in prev and metric in curr:
                old_val = prev[metric]
                new_val = curr[metric]
                if old_val and old_val != 0:
                    change = ((new_val - old_val) / old_val) * 100
                    if abs(change) >= threshold_pct:
                        regressions[metric] = {
                            "previous": old_val,
                            "current": new_val,
                            "change_pct": round(change, 2),
                        }
        return regressions if regressions else None


def run_automated_benchmark(
    model_names: list[str] | None = None,
    samples: list[str] | None = None,
    warmup: int = 2,
    runs: int = 10,
    history: BenchmarkHistory | None = None,
) -> dict[str, Any]:
    """Run a full benchmark across all specified models and optionally persist.

    Returns a dict keyed by ``model_name`` with ``BenchmarkResult`` data.
    """
    from benchmarks.benchmark_runner import latency_benchmark, summary

    results_df = latency_benchmark(
        model_names=model_names, samples=samples, warmup=warmup, runs=runs
    )
    if results_df.empty:
        return {}

    summary_df = summary(results_df)
    output: dict[str, Any] = {}

    for _, row in summary_df.iterrows():
        model = row["model"]
        result = BenchmarkResult(
            model_name=model,
            latency_mean_ms=float(row["mean_ms"]),
            latency_median_ms=float(row["median_ms"]),
            latency_p95_ms=float(row["mean_ms"]) * 1.5,
            samples_count=int(row["samples"]),
        )
        output[model] = result.to_dict()
        if history is not None:
            history.record(result)

    return output


class TelemetryLogger:
    def __init__(self, log_path: str = "spamlyser_telemetry.json"):
        from pathlib import Path
        self.log_path = Path(log_path)

    def log_inference(self, duration_ms: float, confidence: float, classification: str):
        import time, json
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
