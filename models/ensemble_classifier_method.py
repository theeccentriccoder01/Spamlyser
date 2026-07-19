import models.quantizer
import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List

import numpy as np

from .storage_manager import StorageManager


@dataclass
class PredictionResult:
    """Structured prediction result"""

    method: str
    label: str
    confidence: float
    spam_probability: float
    details: str
    metadata: dict[str, Any]
    threat_type: str = None  # Can be "Phishing", "Scam/Fraud", "Unwanted Marketing", "Other", or None for HAM

    @property
    def score(self) -> float:
        """Alias for ``confidence``.

        ``BatchProcessor.process_message()`` reads ``result.score`` on the
        object returned by ``get_model_prediction()``.  This property keeps
        both attribute names in sync without breaking existing callers that
        already use ``.confidence``.
        """
        return self.confidence


class EnsembleSpamClassifier:
    # Enhanced ensemble classifier that combines predictions from multiple spam detection models

    def __init__(
        self, model_weights: dict[str, float] | None = None, performance_tracker=None
    ):
        self.default_weights = {
            "DistilBERT": 0.20,  # Fast but less accurate
            "BERT": 0.30,  # Balanced performance
            "RoBERTa": 0.30,  # Robust and accurate
            "ALBERT": 0.20,  # Parameter efficient
        }

        self.model_weights = model_weights or self.default_weights.copy()
        self.performance_tracker = performance_tracker
        self._normalize_weights()

        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def _normalize_weights(self):
        """Normalize weights to sum to 1"""
        total_weight = sum(self.model_weights.values())
        if total_weight > 0:
            self.model_weights = {
                k: v / total_weight for k, v in self.model_weights.items()
            }
        else:
            # Fallback to equal weights
            n_models = len(self.model_weights)
            self.model_weights = {k: 1 / n_models for k in self.model_weights.keys()}

    def _validate_predictions(self, predictions: dict[str, dict[str, Any]]) -> bool:
        """Validate input predictions format"""
        if not predictions or not isinstance(predictions, dict):
            self.logger.error("Predictions must be a non-empty dictionary")
            return False

        for model_name, pred in predictions.items():
            if not isinstance(pred, dict):
                self.logger.error(f"Prediction for {model_name} must be a dictionary")
                return False

            if "label" not in pred or "score" not in pred:
                self.logger.error(
                    f"Prediction for {model_name} missing 'label' or 'score'"
                )
                return False

            if (
                not isinstance(pred["score"], (int, float))
                or not 0 <= pred["score"] <= 1
            ):
                self.logger.error(f"Score for {model_name} must be between 0 and 1")
                return False

            if pred["label"].upper() not in ["SPAM", "HAM"]:
                self.logger.error(f"Label for {model_name} must be 'SPAM' or 'HAM'")
                return False

        return True

    def majority_voting(self, predictions: dict[str, dict[str, Any]]) -> dict[str, Any]:

        # Approach 1: Majority Voting
        try:
            if not self._validate_predictions(predictions):
                return self._fallback_prediction(predictions)

            spam_votes = 0
            ham_votes = 0
            total_confidence = 0
            model_predictions = []

            for model_name, pred in predictions.items():
                label = pred["label"].upper()
                confidence = pred["score"]

                model_predictions.append(
                    {"model": model_name, "prediction": label, "confidence": confidence}
                )

                if label == "SPAM":
                    spam_votes += 1
                else:
                    ham_votes += 1

                total_confidence += confidence

            # Determine final prediction
            if spam_votes > ham_votes:
                final_label = "SPAM"
                vote_ratio = spam_votes / (spam_votes + ham_votes)
                spam_prob = vote_ratio
            elif ham_votes > spam_votes:
                final_label = "HAM"
                vote_ratio = ham_votes / (spam_votes + ham_votes)
                spam_prob = 1 - vote_ratio
            else:
                # Tie-breaking: use weighted average of confidences
                spam_conf_sum = sum(
                    pred["score"]
                    for model, pred in predictions.items()
                    if pred["label"].upper() == "SPAM"
                )
                ham_conf_sum = sum(
                    pred["score"]
                    for model, pred in predictions.items()
                    if pred["label"].upper() == "HAM"
                )

                if spam_conf_sum >= ham_conf_sum:
                    final_label = "SPAM"
                    spam_prob = 0.5
                else:
                    final_label = "HAM"
                    spam_prob = 0.5
                vote_ratio = 0.5

            avg_confidence = total_confidence / len(predictions) if predictions else 0.5
            final_confidence = spam_prob if final_label == "SPAM" else (1 - spam_prob)

            return {
                "method": "Majority Voting",
                "label": final_label,
                "confidence": final_confidence,
                "spam_probability": spam_prob,
                "vote_ratio": vote_ratio,
                "spam_votes": spam_votes,
                "ham_votes": ham_votes,
                "individual_predictions": model_predictions,
                "details": f"{spam_votes} models voted SPAM, {ham_votes} voted HAM",
                "metadata": {
                    "avg_model_confidence": avg_confidence,
                    "tie_broken": spam_votes == ham_votes,
                },
            }

        except Exception as e:
            self.logger.error(f"Error in majority voting: {e!s}")
            return self._fallback_prediction(predictions)

    def weighted_average(
        self, predictions: dict[str, dict[str, Any]], threshold: float = 0.5
    ) -> dict[str, Any]:

        # Approach 2: Weighted Average
        try:
            if not self._validate_predictions(predictions):
                return self._fallback_prediction(predictions)

            # Update weights from performance tracker if available
            if self.performance_tracker:
                self.model_weights = self.performance_tracker.get_dynamic_weights()
                self._normalize_weights()

            weighted_spam_prob = 0.0
            total_weight_used = 0.0
            model_contributions = []

            for model_name, pred in predictions.items():
                # Use default weight if model not in weights
                weight = self.model_weights.get(model_name, 1 / len(predictions))

                label = pred["label"].upper()
                confidence = pred["score"]

                # Convert to spam probability
                spam_prob = confidence if label == "SPAM" else (1 - confidence)

                weighted_contribution = weight * spam_prob
                weighted_spam_prob += weighted_contribution
                total_weight_used += weight

                model_contributions.append(
                    {
                        "model": model_name,
                        "weight": weight,
                        "spam_probability": spam_prob,
                        "contribution": weighted_contribution,
                        "original_prediction": label,
                        "original_confidence": confidence,
                    }
                )

            # Normalize by total weight used
            final_spam_prob = (
                weighted_spam_prob / total_weight_used if total_weight_used > 0 else 0.5
            )

            # Make final decision
            if final_spam_prob >= threshold:
                final_label = "SPAM"
                final_confidence = final_spam_prob
            else:
                final_label = "HAM"
                final_confidence = 1 - final_spam_prob

            return {
                "method": "Weighted Average",
                "label": final_label,
                "confidence": final_confidence,
                "spam_probability": final_spam_prob,
                "threshold": threshold,
                "model_contributions": model_contributions,
                "total_weight_used": total_weight_used,
                "details": f"Weighted spam probability: {final_spam_prob:.3f} (threshold: {threshold})",
                "metadata": {
                    "weights_used": dict(self.model_weights),
                    "dynamic_weights": self.performance_tracker is not None,
                },
            }

        except Exception as e:
            self.logger.error(f"Error in weighted average: {e!s}")
            return self._fallback_prediction(predictions)

    def confidence_weighted_voting(
        self, predictions: dict[str, dict[str, Any]]
    ) -> dict[str, Any]:

        # Approach 3: Confidence-Weighted Voting

        try:
            if not self._validate_predictions(predictions):
                return self._fallback_prediction(predictions)

            # Update weights from performance tracker if available
            if self.performance_tracker:
                self.model_weights = self.performance_tracker.get_dynamic_weights()
                self._normalize_weights()

            spam_weight = 0.0
            ham_weight = 0.0
            model_votes = []

            for model_name, pred in predictions.items():
                label = pred["label"].upper()
                confidence = pred["score"]

                # Incorporate per-model reliability score and recent accuracy
                reliability = self.model_weights.get(model_name, 1.0)
                accuracy = 1.0
                if hasattr(self, "performance_tracker") and self.performance_tracker:
                    stats = self.performance_tracker.get_model_stats(model_name)
                    accuracy = stats.get("recent_accuracy", 1.0)

                # Weight the vote by adjusted confidence (reliability * accuracy)
                adjusted_weight = confidence * reliability * accuracy

                if label == "SPAM":
                    spam_weight += adjusted_weight
                    weight_contribution_spam = adjusted_weight
                    weight_contribution_ham = 0.0
                else:
                    ham_weight += adjusted_weight
                    weight_contribution_spam = 0.0
                    weight_contribution_ham = adjusted_weight

                model_votes.append(
                    {
                        "model": model_name,
                        "prediction": label,
                        "confidence": confidence,
                        "reliability_score": reliability,
                        "weight_contribution_spam": weight_contribution_spam,
                        "weight_contribution_ham": weight_contribution_ham,
                    }
                )

            total_weight = spam_weight + ham_weight

            if total_weight > 0:
                spam_prob = spam_weight / total_weight
                if spam_weight > ham_weight:
                    final_label = "SPAM"
                    final_confidence = spam_prob
                else:
                    final_label = "HAM"
                    final_confidence = 1 - spam_prob
            else:
                final_label = "HAM"
                final_confidence = 0.5
                spam_prob = 0.5

            return {
                "method": "Confidence-Weighted Voting",
                "label": final_label,
                "confidence": final_confidence,
                "spam_probability": spam_prob,
                "spam_weight": spam_weight,
                "ham_weight": ham_weight,
                "total_weight": total_weight,
                "model_votes": model_votes,
                "details": f"SPAM weight: {spam_weight:.3f}, HAM weight: {ham_weight:.3f}",
                "metadata": {
                    "weight_difference": abs(spam_weight - ham_weight),
                    "confidence_spread": max(
                        pred["score"] for pred in predictions.values()
                    )
                    - min(pred["score"] for pred in predictions.values()),
                },
            }

        except Exception as e:
            self.logger.error(f"Error in confidence weighted voting: {e!s}")
            return self._fallback_prediction(predictions)

    def adaptive_threshold_ensemble(
        self, predictions: dict[str, dict[str, Any]], base_threshold: float = 0.5
    ) -> dict[str, Any]:

        # Approach 4: Adaptive Threshold Ensemble
        try:
            if not self._validate_predictions(predictions):
                return self._fallback_prediction(predictions)

            # First get weighted average result
            weighted_result = self.weighted_average(predictions, base_threshold)

            # Calculate model agreement
            spam_predictions = sum(
                1 for pred in predictions.values() if pred["label"].upper() == "SPAM"
            )
            total_models = len(predictions)
            agreement_ratio = (
                max(spam_predictions, total_models - spam_predictions) / total_models
            )

            # Calculate confidence variance
            confidences = [pred["score"] for pred in predictions.values()]
            confidence_variance = np.var(confidences) if len(confidences) > 1 else 0
            avg_confidence = np.mean(confidences)

            # Adjust threshold based on agreement and confidence distribution
            threshold_adjustment = 1.0

            if agreement_ratio >= 0.75:  # High agreement
                threshold_adjustment *= 0.8
            elif agreement_ratio <= 0.5:  # Low agreement (tie or close)
                threshold_adjustment *= 1.3

            # If models are very confident but disagree, be more conservative
            if confidence_variance > 0.1 and avg_confidence > 0.8:
                threshold_adjustment *= 1.2

            adjusted_threshold = base_threshold * threshold_adjustment

            # Ensure threshold stays within reasonable bounds
            adjusted_threshold = max(0.25, min(0.75, adjusted_threshold))

            # Re-classify with adjusted threshold
            spam_prob = weighted_result["spam_probability"]
            if spam_prob >= adjusted_threshold:
                final_label = "SPAM"
                final_confidence = spam_prob
            else:
                final_label = "HAM"
                final_confidence = 1 - spam_prob

            return {
                "method": "Adaptive Threshold Ensemble",
                "label": final_label,
                "confidence": final_confidence,
                "spam_probability": spam_prob,
                "base_threshold": base_threshold,
                "adjusted_threshold": adjusted_threshold,
                "threshold_adjustment": threshold_adjustment,
                "agreement_ratio": agreement_ratio,
                "spam_predictions": spam_predictions,
                "total_models": total_models,
                "model_contributions": weighted_result["model_contributions"],
                "details": f"Agreement: {agreement_ratio:.2f}, Adjusted threshold: {adjusted_threshold:.3f}",
                "metadata": {
                    "confidence_variance": confidence_variance,
                    "avg_confidence": avg_confidence,
                    "high_disagreement": confidence_variance > 0.1
                    and avg_confidence > 0.8,
                },
            }

        except Exception as e:
            self.logger.error(f"Error in adaptive threshold ensemble: {e!s}")
            return self._fallback_prediction(predictions)

    def meta_ensemble(self, predictions: dict[str, dict[str, Any]]) -> dict[str, Any]:

        #   Approach 5: Meta-Ensemble
        #   Combines all ensemble methods and chooses the most confident result
        try:
            if not self._validate_predictions(predictions):
                return self._fallback_prediction(predictions)

            methods = {
                "majority_voting": self.majority_voting,
                "weighted_average": self.weighted_average,
                "confidence_weighted": self.confidence_weighted_voting,
                "adaptive_threshold": self.adaptive_threshold_ensemble,
            }

            method_results = {}
            for method_name, method_func in methods.items():
                try:
                    result = method_func(predictions)
                    method_results[method_name] = result
                except Exception as e:
                    self.logger.warning(f"Method {method_name} failed: {e!s}")
                    continue

            if not method_results:
                return self._fallback_prediction(predictions)

            # Find the method with highest confidence
            best_method = max(method_results.items(), key=lambda x: x[1]["confidence"])
            best_result = best_method[1].copy()

            # Update metadata to show it's a meta-ensemble
            best_result["method"] = f"Meta-Ensemble ({best_method[0]})"
            best_result["details"] = (
                f"Selected {best_method[0]} (confidence: {best_result['confidence']:.3f})"
            )
            best_result["metadata"] = best_result.get("metadata", {})
            best_result["metadata"]["all_method_results"] = {
                name: {"label": res["label"], "confidence": res["confidence"]}
                for name, res in method_results.items()
            }

            return best_result

        except Exception as e:
            self.logger.error(f"Error in meta ensemble: {e!s}")
            return self._fallback_prediction(predictions)

    def get_ensemble_prediction(
        self, predictions: dict[str, dict[str, Any]], method: str = "weighted_average"
    ) -> dict[str, Any]:
        method_map = {
            "majority_voting": self.majority_voting,
            "weighted_average": self.weighted_average,
            "confidence_weighted": self.confidence_weighted_voting,
            "adaptive_threshold": self.adaptive_threshold_ensemble,
            "meta_ensemble": self.meta_ensemble,
        }

        if method not in method_map:
            self.logger.warning(f"Unknown method {method}, using weighted_average")
            method = "weighted_average"

        result = method_map[method](predictions)

        # Add timestamp
        result["metadata"] = result.get("metadata", {})
        result["metadata"]["timestamp"] = datetime.now().isoformat()
        result["metadata"]["input_models"] = list(predictions.keys())

        # Add threat type if it's SPAM (to be populated by threat_analyzer later)
        result["threat_type"] = None

        return result

    def get_all_predictions(
        self, predictions: dict[str, dict[str, Any]]
    ) -> dict[str, dict[str, Any]]:

        methods = [
            "majority_voting",
            "weighted_average",
            "confidence_weighted",
            "adaptive_threshold",
            "meta_ensemble",
        ]

        results = {}
        for method in methods:
            try:
                results[method] = self.get_ensemble_prediction(predictions, method)
            except Exception as e:
                self.logger.error(
                    f"Failed to get prediction for method {method}: {e!s}"
                )
                results[method] = self._fallback_prediction(predictions)

        return results

    def get_model_prediction(self, model_name: str, message: str) -> "PredictionResult":
        """Return a per-model prediction for *message*.

        The actual transformer inference lives in ``app.py`` where the models
        are loaded.  This base implementation returns a neutral fallback so
        that ``BatchProcessor.process_message()`` does not crash with
        ``AttributeError`` when no subclass or mock is provided.

        Subclasses or test doubles should override this method to supply real
        inference results.  The returned object must expose ``.label``,
        ``.score``, and ``.spam_probability`` attributes (see
        :class:`PredictionResult`).

        Args:
            model_name: One of the known model names (e.g. ``"DistilBERT"``).
            message:    The raw SMS text to classify.

        Returns:
            A :class:`PredictionResult` with a neutral HAM default.
        """
        self.logger.warning(
            "get_model_prediction() called on base EnsembleSpamClassifier for "
            "model '%s'. Returning neutral fallback — override this method to "
            "supply real inference.",
            model_name,
        )
        return PredictionResult(
            method=model_name,
            label="HAM",
            confidence=0.5,
            spam_probability=0.0,
            details="Fallback prediction — no inference pipeline attached",
            metadata={"model_name": model_name, "fallback": True},
        )

    def update_model_weights(self, new_weights: dict[str, float]):
        """Update model weights"""
        self.model_weights = new_weights.copy()
        self._normalize_weights()

    def get_model_weights(self) -> dict[str, float]:
        """Get current model weights"""
        return self.model_weights.copy()

    def _fallback_prediction(
        self, predictions: dict[str, dict[str, Any]]
    ) -> dict[str, Any]:

        # Fallback prediction when ensemble methods fail

        try:
            if not predictions:
                return {
                    "method": "Emergency Fallback",
                    "label": "HAM",
                    "confidence": 0.5,
                    "spam_probability": 0.0,
                    "details": "No predictions available",
                    "metadata": {"fallback_reason": "no_predictions"},
                }

            best_pred = max(predictions.values(), key=lambda x: x.get("score", 0))
            spam_prob = (
                best_pred["score"]
                if best_pred["label"].upper() == "SPAM"
                else (1 - best_pred["score"])
            )

            return {
                "method": "Fallback (Highest Confidence)",
                "label": best_pred["label"].upper(),
                "confidence": best_pred["score"],
                "spam_probability": spam_prob,
                "details": "Used fallback - selected most confident model",
                "metadata": {
                    "fallback_reason": "ensemble_error",
                    "selected_from": list(predictions.keys()),
                },
            }
        except Exception as e:
            return {
                "method": "Emergency Fallback",
                "label": "HAM",
                "confidence": 0.5,
                "spam_probability": 0.0,
                "details": f"Emergency fallback due to error: {e!s}",
                "metadata": {"fallback_reason": "complete_failure"},
            }


class ModelPerformanceTracker:
    # Enhanced performance tracker with persistence and advanced metrics

    def __init__(self, history_size: int = 100, min_samples: int = 10):
        self.history_size = history_size
        self.min_samples = min_samples

        self.performance_history = defaultdict(list)
        self.model_metrics = defaultdict(
            lambda: {
                "accuracy": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "f1_score": 0.0,
                "total_predictions": 0,
                "correct_predictions": 0,
            }
        )

        self.default_weights = {
            "DistilBERT": 0.20,
            "BERT": 0.30,
            "RoBERTa": 0.30,
            "ALBERT": 0.20,
        }

    def update_performance(
        self,
        model_name: str,
        was_correct: bool,
        predicted_label: str | None = None,
        true_label: str | None = None,
    ):
        """
        Update performance history for a model

        Args:
            model_name: Name of the model
            was_correct: Whether the prediction was correct
            predicted_label: The predicted label ('SPAM' or 'HAM')
            true_label: The true label ('SPAM' or 'HAM')
        """
        # Update binary history
        self.performance_history[model_name].append(was_correct)

        # Keep only recent history
        if len(self.performance_history[model_name]) > self.history_size:
            self.performance_history[model_name] = self.performance_history[model_name][
                -self.history_size :
            ]

        # Update detailed metrics
        metrics = self.model_metrics[model_name]
        metrics["total_predictions"] += 1

        if was_correct:
            metrics["correct_predictions"] += 1

        # Calculate accuracy
        metrics["accuracy"] = (
            metrics["correct_predictions"] / metrics["total_predictions"]
        )

        # Update precision, recall, F1 if labels provided
        if predicted_label and true_label:
            self._update_classification_metrics(
                model_name, predicted_label.upper(), true_label.upper()
            )

    def _update_classification_metrics(
        self, model_name: str, predicted_label: str, true_label: str
    ):
        # Update precision, recall, and F1 score (simplified calculation)
        # This is a simplified version - in practice, you'd want to track
        # true positives, false positives, etc. more carefully
        self.model_metrics[model_name]

        # For spam detection, we typically care about spam detection performance
        if true_label == "SPAM" and predicted_label == "SPAM":
            # True positive for spam
            pass
        elif true_label == "HAM" and predicted_label == "SPAM":
            # False positive for spam
            pass
        # ... implement full precision/recall calculation as needed

    def get_dynamic_weights(self) -> dict[str, float]:
        # ictionary of model weights normalized to sum to 1
        weights = {}

        for model_name in self.default_weights.keys():
            history = self.performance_history.get(model_name, [])

            if len(history) >= self.min_samples:
                # Use recent performance with some smoothing
                recent_accuracy = sum(history) / len(history)

                # Apply smoothing to prevent extreme weights
                smoothed_weight = (
                    0.7 * recent_accuracy + 0.3 * self.default_weights[model_name]
                )
                weights[model_name] = max(0.05, smoothed_weight)  # Minimum weight of 5%
            else:
                # Use default weight if insufficient history
                weights[model_name] = self.default_weights[model_name]

        # Normalize weights
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v / total_weight for k, v in weights.items()}

        return weights

    def get_model_stats(self, model_name: str) -> dict[str, Any]:
        # Get detailed statistics for a specific model
        if model_name not in self.performance_history:
            return {}

        history = self.performance_history[model_name]
        metrics = self.model_metrics[model_name]

        return {
            "model_name": model_name,
            "total_predictions": len(history),
            "accuracy": metrics["accuracy"],
            "recent_accuracy": sum(history) / len(history) if history else 0.0,
            "current_weight": self.get_dynamic_weights().get(model_name, 0.0),
            "performance_trend": self._calculate_trend(history),
            "metrics": dict(metrics),
        }

    def _calculate_trend(self, history: list[bool]) -> str:
        # Calculate performance trend (improving, declining, stable)
        if len(history) < 20:
            return "insufficient_data"

        # Compare first half vs second half
        mid_point = len(history) // 2
        first_half = sum(history[:mid_point]) / mid_point
        second_half = sum(history[mid_point:]) / (len(history) - mid_point)

        diff = second_half - first_half

        if diff > 0.1:
            return "improving"
        elif diff < -0.1:
            return "declining"
        else:
            return "stable"

    def get_all_stats(self) -> dict[str, dict[str, Any]]:
        # Get statistics for all models
        return {
            model: self.get_model_stats(model) for model in self.default_weights.keys()
        }

    def save_to_file(self, filepath: str):
        data = {
            "performance_history": dict(self.performance_history),
            "model_metrics": dict(self.model_metrics),
            "config": {
                "history_size": self.history_size,
                "min_samples": self.min_samples,
            },
            "timestamp": datetime.now().isoformat(),
        }

        mgr = StorageManager()
        mgr.save_json(filepath, data, backup=True)

    def load_from_file(self, filepath: str):
        mgr = StorageManager()
        data = mgr.load_json(filepath)
        if data is None:
            return

        try:
            self.performance_history = defaultdict(
                list, data.get("performance_history", {})
            )
            self.model_metrics = defaultdict(
                lambda: {
                    "accuracy": 0.0,
                    "precision": 0.0,
                    "recall": 0.0,
                    "f1_score": 0.0,
                    "total_predictions": 0,
                    "correct_predictions": 0,
                }
            )

            for model, metrics in data.get("model_metrics", {}).items():
                self.model_metrics[model].update(metrics)

            config = data.get("config", {})
            self.history_size = config.get("history_size", self.history_size)
            self.min_samples = config.get("min_samples", self.min_samples)

        except Exception as e:
            logging.error(f"Error loading performance data: {e!s}")


# Example usage and testing functions
def create_sample_predictions():
    # Create sample predictions for testing
    return {
        "DistilBERT": {"label": "SPAM", "score": 0.85},
        "BERT": {"label": "SPAM", "score": 0.92},
        "RoBERTa": {"label": "HAM", "score": 0.78},
        "ALBERT": {"label": "SPAM", "score": 0.88},
    }


def test_ensemble_classifier():
    """Test function for the ensemble classifier"""
    # Initialize tracker and classifier
    tracker = ModelPerformanceTracker()
    classifier = EnsembleSpamClassifier(performance_tracker=tracker)

    # Sample predictions
    predictions = create_sample_predictions()

    print("Testing Ensemble Spam Classifier")
    print("=" * 50)

    # Test all methods
    methods = [
        "majority_voting",
        "weighted_average",
        "confidence_weighted",
        "adaptive_threshold",
        "meta_ensemble",
    ]

    for method in methods:
        result = classifier.get_ensemble_prediction(predictions, method)
        print(f"\n{method.upper()}:")
        print(f"  Label: {result['label']}")
        print(f"  Confidence: {result['confidence']:.3f}")
        print(f"  Details: {result['details']}")

    # Test getting all predictions at once
    print("\nALL METHODS COMPARISON:")
    print("-" * 30)
    all_results = classifier.get_all_predictions(predictions)
    for method, result in all_results.items():
        print(f"{method:20} | {result['label']:4} | {result['confidence']:.3f}")


if __name__ == "__main__":
    test_ensemble_classifier()


def compare_predictions(cleaned_sms: str, models: dict, fallback_label: str) -> list:
    from models.custom_rules_manager import check_custom_rules
    results = []
    for name, clf in models.items():
        if clf is None:
            continue
        rule_match = check_custom_rules(cleaned_sms)
        if rule_match is not None:
            results.append(dict(model=name, label=rule_match, confidence=1.0, is_rule_override=True))
        else:
            pred = clf([cleaned_sms])[0]
            lbl = pred["label"].upper()
            if lbl not in ("SPAM", "HAM"):
                lbl = "SPAM" if pred.get("score", 0.5) > 0.5 else "HAM"
            results.append(dict(model=name, label=lbl, confidence=pred["score"], is_rule_override=False))
    return results

def agreement_score(results: list) -> tuple:
    if not results:
        return True, 1.0
    labels = [r["label"] for r in results]
    n = len(labels)
    majority = max(set(labels), key=labels.count)
    agreement = sum(1 for l in labels if l == majority) / n
    return len(set(labels)) == 1, agreement
