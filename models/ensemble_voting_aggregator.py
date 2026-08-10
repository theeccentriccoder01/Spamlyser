"""
Multi-Engine Ensemble Classifier Aggregator.
Combines probabilities and decisions from multiple detection engines (ML, Regex, Heuristic, LLM)
using Softmax/Weighted Voting, Majority Rule, or Soft Thresholding.
"""

from typing import List, Dict, Any, Optional
import math

class EnsembleVotingAggregator:
    """
    Ensemble decision engine aggregating predictions from heterogeneous classifiers.
    """

    def __init__(self, weights: Optional[Dict[str, float]] = None):
        # Default equal weighting if custom weight vector is omitted
        self.weights = weights or {}

    def soft_voting(self, predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compute weighted probability average (soft voting).
        Each prediction item must contain: {'engine': str, 'score': float} (0.0 to 1.0)
        """
        if not predictions:
            return {"is_spam": False, "confidence": 0.0, "engines_evaluated": 0}

        total_weight = 0.0
        weighted_score_sum = 0.0

        for pred in predictions:
            engine = pred.get("engine", "default")
            score = float(pred.get("score", 0.0))
            w = self.weights.get(engine, 1.0)

            weighted_score_sum += score * w
            total_weight += w

        final_score = weighted_score_sum / total_weight if total_weight > 0 else 0.0
        final_score = round(final_score, 4)

        return {
            "is_spam": final_score >= 0.5,
            "confidence": final_score,
            "decision": "SPAM" if final_score >= 0.5 else "HAM",
            "engines_evaluated": len(predictions),
            "method": "soft_voting"
        }

    def majority_voting(self, predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compute hard majority vote across classifier predictions.
        """
        if not predictions:
            return {"is_spam": False, "confidence": 0.0, "engines_evaluated": 0}

        spam_votes = 0
        ham_votes = 0

        for pred in predictions:
            score = float(pred.get("score", 0.0))
            is_spam = pred.get("is_spam", score >= 0.5)
            if is_spam:
                spam_votes += 1
            else:
                ham_votes += 1

        is_spam_decision = spam_votes > ham_votes
        vote_ratio = round(max(spam_votes, ham_votes) / len(predictions), 4)

        return {
            "is_spam": is_spam_decision,
            "confidence": vote_ratio,
            "decision": "SPAM" if is_spam_decision else "HAM",
            "spam_votes": spam_votes,
            "ham_votes": ham_votes,
            "engines_evaluated": len(predictions),
            "method": "majority_voting"
        }
