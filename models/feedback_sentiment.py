"""
User Feedback Sentiment & Trend Analyzer for Spamlyser
Analyzes user rating trends and sentiment feedback for spam classification predictions.
"""

from typing import Any, Dict, List


class FeedbackSentimentAnalyzer:
    """Aggregates user rating feedback and calculates model satisfaction metrics."""

    @staticmethod
    def calculate_satisfaction(ratings: list[int]) -> dict[str, Any]:
        """Calculates average satisfaction score and rating distribution (1 to 5 stars)."""
        if not ratings:
            return {"total_feedback": 0, "avg_rating": 0.0, "distribution": {}}

        total = len(ratings)
        avg = sum(ratings) / total
        dist = {star: ratings.count(star) for star in range(1, 6)}
        satisfaction_pct = round((sum(1 for r in ratings if r >= 4) / total) * 100, 2)

        return {
            "total_feedback": total,
            "avg_rating": round(avg, 2),
            "satisfaction_pct": satisfaction_pct,
            "distribution": dist,
        }

    @staticmethod
    def group_by_model(feedback_entries: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
        """Groups feedback ratings by individual classifier model."""
        grouped: dict[str, list[int]] = {}
        for entry in feedback_entries:
            model = entry.get("model_name", "unknown")
            rating = entry.get("rating", 3)
            grouped.setdefault(model, []).append(rating)

        return {
            model: FeedbackSentimentAnalyzer.calculate_satisfaction(ratings)
            for model, ratings in grouped.items()
        }
