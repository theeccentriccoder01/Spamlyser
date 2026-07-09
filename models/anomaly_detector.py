"""Anomaly detection for SMS classification — identifies outlier messages."""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.stats import zscore


class AnomalyDetector:
    """Scores messages on multiple anomaly dimensions."""

    def __init__(self, contamination: float = 0.1):
        self.contamination = contamination
        self._iforest: IsolationForest | None = None
        self._vectorizer: TfidfVectorizer | None = None

    def score_messages(self, analyses: list[dict]) -> pd.DataFrame:
        """Compute anomaly scores for a list of analysis results."""
        if not analyses:
            return pd.DataFrame()

        df = pd.DataFrame(analyses)
        scores = {}

        # 1. Content anomaly (TF-IDF + Isolation Forest)
        texts = [a.get("message", "") for a in analyses]
        self._vectorizer = TfidfVectorizer(max_features=100, stop_words="english", sublinear_tf=True)
        X = self._vectorizer.fit_transform(texts).toarray()
        self._iforest = IsolationForest(
            contamination=self.contamination,
            random_state=42,
            n_estimators=100,
        )
        preds = self._iforest.fit_predict(X)
        scores["content_anomaly"] = (preds == -1).astype(float)
        scores["content_score"] = -self._iforest.score_samples(X)

        # 2. Confidence anomaly (z-score of confidence values)
        confs = [a.get("ensemble_predictions", {}).get("majority_voting", {}).get("confidence", 0.5)
                 for a in analyses]
        if len(set(confs)) > 1:
            z_conf = np.abs(zscore(confs))
            scores["confidence_anomaly"] = (z_conf > 2.0).astype(float)
            scores["confidence_zscore"] = z_conf
        else:
            scores["confidence_anomaly"] = np.zeros(len(analyses))
            scores["confidence_zscore"] = np.zeros(len(analyses))

        # 3. Model disagreement anomaly
        disagreements = []
        for a in analyses:
            model_preds = a.get("model_predictions", {})
            labels = [p.get("label", "") for p in model_preds.values()]
            if labels:
                spams = labels.count("SPAM")
                hams = labels.count("HAM")
                disagreements.append(1.0 - max(spams, hams) / len(labels))
            else:
                disagreements.append(0.0)
        scores["disagreement_score"] = disagreements
        scores["disagreement_anomaly"] = (np.array(disagreements) > 0.5).astype(float)

        # 4. Composite anomaly score (weighted average)
        weights = {"content": 0.35, "confidence": 0.35, "disagreement": 0.30}
        composite = (
            weights["content"] * scores["content_score"]
            + weights["confidence"] * scores["confidence_zscore"]
            + weights["disagreement"] * scores["disagreement_score"]
        )
        scores["composite_anomaly"] = composite
        scores["is_anomaly"] = (composite > np.percentile(composite, 90)).astype(int)

        result = df.copy()
        for k, v in scores.items():
            result[k] = v
        return result

    def get_outlier_explanations(self, row: pd.Series) -> list[str]:
        """Generate human-readable explanations for why a row is anomalous."""
        explanations = []
        if row.get("content_anomaly", 0):
            explanations.append(f"Unusual content pattern (score: {row.get('content_score', 0):.2f})")
        if row.get("confidence_anomaly", 0):
            explanations.append(f"Confidence outlier (z={row.get('confidence_zscore', 0):.1f})")
        if row.get("disagreement_anomaly", 0):
            explanations.append(f"Strong model disagreement ({row.get('disagreement_score', 0):.0%})")
        return explanations or ["Within normal range"]

    def top_anomalies(self, df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
        """Return the top-n most anomalous messages."""
        if "composite_anomaly" not in df.columns:
            return df.head(n)
        return df.sort_values("composite_anomaly", ascending=False).head(n)
