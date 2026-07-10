"""Cross-session trend analytics — segments analysis history into sessions."""

from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import entropy


class SessionAnalytics:
    """Segments analysis history into temporal sessions and computes KPIs."""

    def __init__(self, gap_minutes: int = 30):
        self.gap_minutes = gap_minutes
        self.gap = timedelta(minutes=gap_minutes)

    def segment_sessions(self, analyses: list[dict]) -> list[dict[str, Any]]:
        """Split analyses into sessions based on time gaps."""
        if not analyses:
            return []

        sorted_analyses = sorted(analyses, key=lambda a: a.get("timestamp", ""))
        sessions = []
        current = {"id": 0, "start": None, "end": None, "analyses": [], "messages": []}

        for a in sorted_analyses:
            ts = a.get("timestamp")
            if not ts:
                continue
            try:
                dt = datetime.fromisoformat(ts) if isinstance(ts, str) else ts
            except (ValueError, TypeError):
                continue

            if current["start"] is None:
                current["start"] = dt
                current["end"] = dt
                current["analyses"].append(a)
                current["messages"].append(a.get("message", ""))
            elif (dt - current["end"]) > self.gap:
                current = self._finalize_session(current)
                sessions.append(current)
                current = {
                    "id": len(sessions), "start": dt, "end": dt,
                    "analyses": [a], "messages": [a.get("message", "")],
                }
            else:
                current["end"] = dt
                current["analyses"].append(a)
                current["messages"].append(a.get("message", ""))

        if current["analyses"]:
            current = self._finalize_session(current)
            sessions.append(current)

        return sessions

    def _finalize_session(self, sess: dict) -> dict:
        analyses = sess["analyses"]
        total = len(analyses)
        spams = sum(1 for a in analyses if self._get_label(a) == "SPAM")
        hams = total - spams
        confs = [self._get_confidence(a) for a in analyses if self._get_confidence(a) > 0]
        avg_conf = np.mean(confs) if confs else 0.0

        models_used = defaultdict(int)
        threat_types = defaultdict(int)
        for a in analyses:
            mp = a.get("model_predictions", {})
            for m in mp:
                models_used[m] += 1
            threat = self._get_threat(a)
            if threat:
                threat_types[threat] += 1

        duration = (sess["end"] - sess["start"]).total_seconds()
        throughput = total / duration * 3600 if duration > 0 else 0

        sess["total"] = total
        sess["spams"] = spams
        sess["hams"] = hams
        sess["spam_rate"] = spams / total if total > 0 else 0.0
        sess["avg_confidence"] = avg_conf
        sess["duration_seconds"] = duration
        sess["throughput_per_hour"] = throughput
        sess["models_used"] = dict(models_used)
        sess["threat_types"] = dict(threat_types)
        sess["model_count"] = len(models_used)
        return sess

    def compute_comparison(self, sessions: list[dict]) -> pd.DataFrame:
        """Build a comparison DataFrame from session KPI dicts."""
        rows = []
        for s in sessions:
            rows.append({
                "session_id": s["id"],
                "start": s["start"].isoformat() if hasattr(s["start"], "isoformat") else str(s["start"]),
                "duration_min": s["duration_seconds"] / 60,
                "total": s["total"],
                "spam_rate": s["spam_rate"],
                "avg_confidence": s["avg_confidence"],
                "throughput": s["throughput_per_hour"],
                "model_count": s["model_count"],
            })
        return pd.DataFrame(rows)

    def compute_drift(self, sessions: list[dict]) -> dict[str, Any]:
        """Compute drift metrics between consecutive sessions."""
        drifts = []
        for i in range(1, len(sessions)):
            prev, curr = sessions[i - 1], sessions[i]
            spam_drift = curr["spam_rate"] - prev["spam_rate"]
            conf_drift = curr["avg_confidence"] - prev["avg_confidence"]
            direction = "up" if spam_drift > 0.05 else ("down" if spam_drift < -0.05 else "stable")
            drift = {
                "from_session": prev["id"],
                "to_session": curr["id"],
                "spam_rate_delta": spam_drift,
                "confidence_delta": conf_drift,
                "volume_change": curr["total"] - prev["total"],
                "direction": direction,
            }

            prev_labels = [self._get_label(a) for a in prev["analyses"]]
            curr_labels = [self._get_label(a) for a in curr["analyses"]]
            if prev_labels and curr_labels:
                p_spam = prev_labels.count("SPAM") / len(prev_labels)
                c_spam = curr_labels.count("SPAM") / len(curr_labels)
                eps = 1e-10
                kl = p_spam * np.log((p_spam + eps) / (c_spam + eps)) + \
                     (1 - p_spam) * np.log((1 - p_spam + eps) / (1 - c_spam + eps))
                drift["kl_divergence"] = float(kl)
                drift["drift_severity"] = "high" if kl > 0.5 else ("medium" if kl > 0.1 else "low")

            drifts.append(drift)
        return {"deltas": drifts, "total_drift_sessions": len(drifts)}

    def _get_label(self, a: dict) -> str:
        return a.get("ensemble_predictions", {}).get("majority_voting", {}).get("label", "?")

    def _get_confidence(self, a: dict) -> float:
        return a.get("ensemble_predictions", {}).get("majority_voting", {}).get("confidence", 0.0)

    def _get_threat(self, a: dict) -> str | None:
        return a.get("risk_indicators", {})
