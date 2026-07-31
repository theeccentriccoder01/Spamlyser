"""
MIME Email Header Parsing and Authentication Protocol Analyzer (SPF, DKIM, DMARC Emulation).
Detects header spoofing, Return-Path domain mismatches, and email origin anomalies.
"""

import email
from email.message import Message
import re
from typing import Dict, Any, List, Optional

class MIMEHeaderAnalyzer:
    """
    Parses raw email header content to inspect SPF/DKIM/DMARC authentication status and header spoofing.
    """

    def __init__(self):
        pass

    def parse_raw_headers(self, raw_headers: str) -> Message:
        """
        Parse raw header text into an email Message object.
        """
        return email.message_from_string(raw_headers)

    def extract_domain(self, email_str: str) -> str:
        """
        Extract domain part from email address string (e.g., 'John <john@domain.com>' -> 'domain.com').
        """
        match = re.search(r'[\w\.-]+@([\w\.-]+\.\w+)', email_str)
        if match:
            return match.group(1).lower()
        return ""

    def analyze_headers(self, raw_headers: str) -> Dict[str, Any]:
        """
        Analyze MIME email headers for security anomalies and domain spoofing.
        """
        msg = self.parse_raw_headers(raw_headers)

        from_header = msg.get("From", "")
        reply_to_header = msg.get("Reply-To", "")
        return_path = msg.get("Return-Path", "")
        auth_results = msg.get("Authentication-Results", "")

        from_domain = self.extract_domain(from_header)
        reply_to_domain = self.extract_domain(reply_to_header)
        return_path_domain = self.extract_domain(return_path)

        warnings: List[str] = []
        risk_score = 0.0

        # Check Return-Path mismatch (common in phishing/spoofing)
        if return_path_domain and from_domain and return_path_domain != from_domain:
            warnings.append(f"Return-Path domain ({return_path_domain}) mismatches From domain ({from_domain})")
            risk_score += 0.40

        # Check Reply-To domain mismatch
        if reply_to_domain and from_domain and reply_to_domain != from_domain:
            warnings.append(f"Reply-To domain ({reply_to_domain}) mismatches From domain ({from_domain})")
            risk_score += 0.35

        # Inspect Authentication-Results header for SPF/DKIM failures
        spf_pass = "spf=pass" in auth_results.lower() if auth_results else None
        dkim_pass = "dkim=pass" in auth_results.lower() if auth_results else None
        dmarc_pass = "dmarc=pass" in auth_results.lower() if auth_results else None

        if auth_results:
            if spf_pass is False or "spf=fail" in auth_results.lower() or "spf=softfail" in auth_results.lower():
                warnings.append("SPF verification failed or soft-failed")
                risk_score += 0.30
            if dkim_pass is False or "dkim=fail" in auth_results.lower():
                warnings.append("DKIM signature verification failed")
                risk_score += 0.30
            if dmarc_pass is False or "dmarc=fail" in auth_results.lower():
                warnings.append("DMARC policy check failed")
                risk_score += 0.35

        final_risk = min(1.0, round(risk_score, 2))

        return {
            "from_domain": from_domain,
            "reply_to_domain": reply_to_domain,
            "return_path_domain": return_path_domain,
            "authentication_status": {
                "spf": "PASS" if spf_pass else ("FAIL" if spf_pass is False else "UNKNOWN"),
                "dkim": "PASS" if dkim_pass else ("FAIL" if dkim_pass is False else "UNKNOWN"),
                "dmarc": "PASS" if dmarc_pass else ("FAIL" if dmarc_pass is False else "UNKNOWN")
            },
            "header_anomalies": warnings,
            "spoof_risk_score": final_risk,
            "is_suspicious_header": final_risk >= 0.40
        }
