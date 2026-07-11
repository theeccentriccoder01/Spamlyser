"""Tests for benchmark automation history and dashboard components."""

import json
import os
import tempfile

import pytest

from models.benchmark_automation import BenchmarkHistory, BenchmarkResult


@pytest.fixture
def temp_history():
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", encoding="utf-8", delete=False
    ) as f:
        json.dump([], f)
        tmp = f.name
    yield tmp
    os.unlink(tmp)


def test_record_and_get_all(temp_history):
    bh = BenchmarkHistory(temp_history)
    result = BenchmarkResult(
        model_name="DistilBERT",
        latency_mean_ms=12.5,
        latency_median_ms=10.2,
        latency_p95_ms=20.1,
        samples_count=10,
    )
    bh.record(result)
    all_records = bh.get_all()
    assert len(all_records) == 1
    assert all_records[0]["model_name"] == "DistilBERT"


def test_get_latest(temp_history):
    bh = BenchmarkHistory(temp_history)
    r1 = BenchmarkResult("ModelA", 10.0, 9.0, 15.0, 5)
    r2 = BenchmarkResult("ModelB", 20.0, 18.0, 30.0, 5)
    bh.record(r1)
    bh.record(r2)
    latest = bh.get_latest("ModelA")
    assert latest is not None
    assert latest["model_name"] == "ModelA"


def test_clear_history(temp_history):
    bh = BenchmarkHistory(temp_history)
    result = BenchmarkResult("Test", 10.0, 9.0, 15.0, 5)
    bh.record(result)
    assert len(bh.get_all()) == 1
    bh.clear()
    assert len(bh.get_all()) == 0


def test_benchmark_result_to_dict():
    result = BenchmarkResult("Test", 10.5, 9.2, 18.0, 20)
    d = result.to_dict()
    assert d["model_name"] == "Test"
    assert d["latency_mean_ms"] == 10.5
    assert d["samples_count"] == 20
    assert "timestamp" in d


def test_persistence_across_reload(temp_history):
    bh1 = BenchmarkHistory(temp_history)
    bh1.record(BenchmarkResult("Persist", 5.0, 4.5, 8.0, 3))
    bh2 = BenchmarkHistory(temp_history)
    assert len(bh2.get_all()) == 1
    assert bh2.get_all()[0]["model_name"] == "Persist"


def test_get_regression_detects_change(temp_history):
    bh = BenchmarkHistory(temp_history)
    bh.record(BenchmarkResult("Model", 10.0, 9.0, 15.0, 5))
    bh.record(BenchmarkResult("Model", 20.0, 18.0, 30.0, 5))
    reg = bh.get_regression("Model", threshold_pct=5.0)
    assert reg is not None
    assert "latency_mean_ms" in reg
    assert reg["latency_mean_ms"]["change_pct"] > 5.0


def test_get_regression_insufficient_data(temp_history):
    bh = BenchmarkHistory(temp_history)
    bh.record(BenchmarkResult("Model", 10.0, 9.0, 15.0, 5))
    reg = bh.get_regression("Model")
    assert reg is None
