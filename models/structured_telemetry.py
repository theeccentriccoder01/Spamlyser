"""
Structured JSON Telemetry & Security Audit Logging Engine.
Formats backend events as structured JSON logs with correlation IDs and automated PII masking.
"""

import json
import logging
import time
import uuid
import re
from typing import Dict, Any, Optional

PII_EMAIL_REGEX = re.compile(r'([\w\.-]+)@([\w\.-]+\.\w+)')

def mask_pii_text(text: str) -> str:
    """Mask email addresses in string (e.g. 'john@domain.com' -> 'j***@domain.com')."""
    def _repl(match):
        user, domain = match.group(1), match.group(2)
        masked_user = user[0] + "***" if len(user) > 1 else "*"
        return f"{masked_user}@{domain}"
    return PII_EMAIL_REGEX.sub(_repl, text)

class StructuredTelemetryLogger:
    """
    Structured JSON logger emitting standardized audit and operational metrics.
    """

    def __init__(self, service_name: str = "Spamlyser-Backend"):
        self.service_name = service_name
        self.logger = logging.getLogger(service_name)

    def build_event(
        self,
        event_name: str,
        correlation_id: Optional[str] = None,
        payload: Optional[Dict[str, Any]] = None,
        level: str = "INFO"
    ) -> Dict[str, Any]:
        cid = correlation_id or str(uuid.uuid4())
        payload_data = payload or {}

        # Apply PII masking to string values in payload
        sanitized_payload = {}
        for k, v in payload_data.items():
            if isinstance(v, str):
                sanitized_payload[k] = mask_pii_text(v)
            else:
                sanitized_payload[k] = v

        event_record = {
            "timestamp": time.time(),
            "iso_time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "service": self.service_name,
            "event_name": event_name,
            "correlation_id": cid,
            "level": level,
            "payload": sanitized_payload
        }
        return event_record

    def log_event(
        self,
        event_name: str,
        correlation_id: Optional[str] = None,
        payload: Optional[Dict[str, Any]] = None,
        level: str = "INFO"
    ) -> str:
        record = self.build_event(event_name, correlation_id, payload, level)
        json_output = json.dumps(record, ensure_ascii=False)
        
        if level.upper() == "ERROR":
            self.logger.error(json_output)
        elif level.upper() == "WARNING":
            self.logger.warning(json_output)
        else:
            self.logger.info(json_output)

        return json_output
