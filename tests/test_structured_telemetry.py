import pytest
import json
from models.structured_telemetry import StructuredTelemetryLogger, mask_pii_text

def test_mask_pii_text():
    raw = "User email is john.doe@example.com in message body"
    masked = mask_pii_text(raw)
    assert "john.doe@example.com" not in masked
    assert "j***@example.com" in masked

def test_structured_telemetry_event_generation():
    telemetry = StructuredTelemetryLogger()
    cid = "test-corr-12345"
    payload = {"user": "alice@company.com", "action": "classification", "latency_ms": 12.5}
    
    event_str = telemetry.log_event("EMAIL_CLASSIFIED", correlation_id=cid, payload=payload)
    parsed = json.loads(event_str)

    assert parsed["event_name"] == "EMAIL_CLASSIFIED"
    assert parsed["correlation_id"] == cid
    assert parsed["payload"]["user"] == "a***@company.com"
    assert parsed["payload"]["latency_ms"] == 12.5
