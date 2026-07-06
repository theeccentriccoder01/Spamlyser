"""Standalone performance benchmarks for model inference speed and accuracy.

Usage
-----
    python -m benchmarks.benchmark_runner

This will run a small set of sample messages through each loaded model and
report average inference time and confidence distribution.
"""

import importlib
import logging
import statistics
import time
from typing import Any

import pandas as pd

logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger("benchmark")


SAMPLE_MESSAGES = [
    "Hey, are we still meeting for lunch tomorrow?",
    "Congratulations! You've won a $1000 gift card. Call now!",
    "Your invoice is ready. Please review and pay at your earliest convenience.",
    "URGENT: Your account has been compromised. Verify immediately.",
    "Reminder: Your dentist appointment is at 10am on Monday.",
    "FREE iPhone! Claim yours now at http://scam-link.com",
    "Don't forget to pick up milk and bread on your way home.",
    "You have been selected for a special offer. Reply YES to claim.",
    "Your package will be delivered tomorrow between 9am and 5pm.",
    "This is your final notice. Please call us immediately at 1-800-xxx-xxxx.",
]


def _load_model(name: str) -> Any:
    """Load a model by name and return the classifier callable."""
    try:
        m = importlib.import_module(".model_loader", package="models")
        return m.load_model_if_needed(name)
    except Exception as e:
        _logger.warning("Cannot load %s: %s", name, e)
        return None


def latency_benchmark(
    model_names: list[str] | None = None,
    samples: list[str] | None = None,
    warmup: int = 1,
    runs: int = 5,
) -> pd.DataFrame:
    """Benchmark per-model inference latency on *samples*.

    Returns a DataFrame with columns:
        model, run, sample_idx, latency_ms, label, confidence
    """
    from models.smart_preprocess import preprocess_message

    if model_names is None:
        model_names = ["DistilBERT", "BERT", "RoBERTa", "ALBERT"]
    if samples is None:
        samples = SAMPLE_MESSAGES

    import sys
    rows: list[dict[str, Any]] = []

    for name in model_names:
        clf = _load_model(name)
        if clf is None:
            _logger.info("Skipping %s (not available)", name)
            continue

        # Estimate model memory footprint dynamically
        mem_bytes = sys.getsizeof(clf)
        try:
            if hasattr(clf, "model"):
                mem_bytes += sys.getsizeof(clf.model)
        except Exception:
            pass

        for sample_idx, raw in enumerate(samples):
            cleaned = preprocess_message(raw)["cleaned"]
            for run in range(runs):
                if run < warmup:
                    clf([cleaned])
                    continue
                t0 = time.perf_counter()
                pred = clf([cleaned])[0]
                elapsed = (time.perf_counter() - t0) * 1000
                rows.append(
                    {
                        "model": name,
                        "run": run - warmup,
                        "sample_idx": sample_idx,
                        "latency_ms": round(elapsed, 2),
                        "memory_bytes": mem_bytes,
                        "label": pred["label"],
                        "confidence": round(pred["score"], 4),
                    }
                )

    return pd.DataFrame(rows)


def summary(results: pd.DataFrame) -> pd.DataFrame:
    """Aggregate *results* to per-model summary statistics."""
    if results.empty:
        return pd.DataFrame()

    grouped = results.groupby("model")["latency_ms"]
    return pd.DataFrame(
        {
            "mean_ms": grouped.mean().round(2),
            "median_ms": grouped.median().round(2),
            "std_ms": grouped.std().round(2),
            "min_ms": grouped.min().round(2),
            "max_ms": grouped.max().round(2),
            "samples": grouped.count(),
        }
    ).reset_index()


def confidence_distribution(results: pd.DataFrame) -> pd.DataFrame:
    """Count SPAM/HAM predictions per model."""
    if results.empty:
        return pd.DataFrame()
    return results.groupby(["model", "label"]).size().reset_index(name="count")


def run_all(output_csv: str | None = None) -> pd.DataFrame:
    """Convenience: run latency benchmark and optionally save CSV."""
    _logger.info("Starting benchmark...")
    results = latency_benchmark(runs=3)
    _logger.info("Benchmark complete — %d rows", len(results))
    summ = summary(results)
    print(summ.to_string(index=False))
    if output_csv:
        results.to_csv(output_csv, index=False)
        _logger.info("Saved raw results to %s", output_csv)
    return results


def run_full_benchmark(
    model_names: list[str] | None = None,
    samples: list[str] | None = None,
    warmup: int = 2,
    runs: int = 10,
    save_history: bool = True,
) -> dict[str, Any]:
    """Run a comprehensive benchmark and persist results for regression tracking.

    Returns a dict keyed by model name with latency percentiles and
    optional regression warnings.
    """
    from models.benchmark_automation import BenchmarkHistory, run_automated_benchmark

    history = BenchmarkHistory() if save_history else None
    results = run_automated_benchmark(
        model_names=model_names,
        samples=samples,
        warmup=warmup,
        runs=runs,
        history=history,
    )
    if history and results:
        for model_name in results:
            reg = history.get_regression(model_name)
            if reg:
                _logger.warning(
                    "Regression detected for %s: %s", model_name, reg
                )
    return results


if __name__ == "__main__":
    run_all()
