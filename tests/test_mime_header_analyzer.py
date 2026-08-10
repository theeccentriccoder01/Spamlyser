import pytest
from models.mime_header_analyzer import MIMEHeaderAnalyzer

def test_mime_header_analyzer_legitimate():
    raw_headers = """From: Security Team <security@company.com>
Reply-To: support <security@company.com>
Return-Path: <bounce@company.com>
Authentication-Results: mx.google.com; spf=pass; dkim=pass; dmarc=pass
"""
    analyzer = MIMEHeaderAnalyzer()
    res = analyzer.analyze_headers(raw_headers)

    assert res["from_domain"] == "company.com"
    assert res["authentication_status"]["spf"] == "PASS"
    assert res["authentication_status"]["dkim"] == "PASS"
    assert res["is_suspicious_header"] is False
    assert res["spoof_risk_score"] < 0.40

def test_mime_header_analyzer_spoofed():
    raw_headers = """From: Bank Official <service@paypal.com>
Reply-To: Hacker <phisher@badsite.net>
Return-Path: <spammer@malicious.org>
Authentication-Results: mx.google.com; spf=fail; dkim=fail; dmarc=fail
"""
    analyzer = MIMEHeaderAnalyzer()
    res = analyzer.analyze_headers(raw_headers)

    assert res["is_suspicious_header"] is True
    assert res["spoof_risk_score"] > 0.70
    assert len(res["header_anomalies"]) >= 3
    assert res["authentication_status"]["spf"] == "FAIL"
