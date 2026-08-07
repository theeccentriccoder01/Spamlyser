import pytest
import asyncio
from models.async_batch_pipeline import AsyncBatchPipeline, run_async_batch

def dummy_classifier(text: str):
    if "error" in text:
        raise ValueError("Simulated classification error")
    return {
        "is_spam": "free" in text.lower() or "win" in text.lower(),
        "score": 0.95 if "free" in text.lower() else 0.10
    }

def test_async_batch_pipeline_success():
    payload = ["Hello friend, meet for coffee?", "WIN a FREE iPhone now!", "Project report attached."]
    res = run_async_batch(dummy_classifier, payload, max_concurrency=2)
    
    assert res["total_processed"] == 3
    assert res["successful"] == 3
    assert res["failed"] == 0
    assert res["results"][1]["result"]["is_spam"] is True
    assert res["results"][0]["result"]["is_spam"] is False

def test_async_batch_pipeline_with_retries_and_failure():
    payload = ["Valid message", "Trigger error message", "Another valid message"]
    res = run_async_batch(dummy_classifier, payload, max_concurrency=2)
    
    assert res["total_processed"] == 3
    assert res["successful"] == 2
    assert res["failed"] == 1
    assert res["results"][1]["status"] == "failed"
