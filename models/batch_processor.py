"""
Module for handling batch processing of SMS messages using ensemble models.
"""

from collections.abc import Callable, Generator
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from .ensemble_classifier_method import EnsembleSpamClassifier, ModelPerformanceTracker


class BatchProcessor:
    """Handles batch processing of SMS messages using ensemble models."""

    def __init__(self, ensemble_classifier: EnsembleSpamClassifier | None = None):
        """
        Initialize the batch processor.

        Args:
            ensemble_classifier: Optional pre-configured ensemble classifier
        """
        if ensemble_classifier is None:
            performance_tracker = ModelPerformanceTracker()
            ensemble_classifier = EnsembleSpamClassifier(performance_tracker)

        self.ensemble_classifier = ensemble_classifier
        self.batch_stats: dict[str, Any] = {
            "total_messages": 0,
            "processed_messages": 0,
            "spam_detected": 0,
            "ham_detected": 0,
            "avg_confidence": 0.0,
            "start_time": None,
            "end_time": None,
        }

    def process_message(self, message: str) -> dict[str, Any]:
        """
        Process a single message using all models.

        Args:
            message: The SMS message to analyze

        Returns:
            Dict containing analysis results
        """
        # Get predictions from all models
        predictions = {}
        for model_name in ["DistilBERT", "BERT", "RoBERTa", "ALBERT"]:
            try:
                # Get prediction from individual model
                result = self.ensemble_classifier.get_model_prediction(
                    model_name, message
                )
                predictions[model_name] = {
                    "label": result.label,
                    "score": result.score,
                    "spam_probability": result.spam_probability,
                }
            except Exception as e:
                predictions[model_name] = {
                    "label": "ERROR",
                    "score": 0.0,
                    "spam_probability": 0.0,
                    "error": str(e),
                }

        # Get ensemble prediction using all available methods
        ensemble_results = self.ensemble_classifier.get_all_predictions(predictions)

        # Analyze text for risk indicators
        risk_indicators = self._analyze_risk_indicators(message)

        return {
            "message": message,
            "model_predictions": predictions,
            "ensemble_predictions": ensemble_results,
            "risk_indicators": risk_indicators,
            "timestamp": datetime.now().isoformat(),
        }

    def _analyze_risk_indicators(self, message: str) -> dict[str, bool]:
        """
        Analyze message for common spam/threat indicators.

        Args:
            message: The SMS message to analyze

        Returns:
            Dict of risk indicators and their presence (True/False)
        """
        original_message = message
        message = message.lower()

        # Common risk patterns
        patterns = {
            "urls": any(
                x in message for x in ["http://", "https://", ".com", ".net", ".org"]
            ),
            "urgency": any(
                x in message
                for x in ["urgent", "immediately", "act now", "limited time"]
            ),
            "money": any(
                x in message for x in ["$", "€", "£", "win", "cash", "prize", "money"]
            ),
            "personal_info": any(
                x in message
                for x in ["password", "account", "login", "ssn", "credit card"]
            ),
            "all_caps": any(
                word.isupper() and len(word) > 2
                for word in original_message.split()
            ),
            "suspicious_chars": (
                len([c for c in message if not c.isalnum() and not c.isspace()])
                / len(message)
                > 0.1
                if message
                else False
            ),
        }

        return patterns

    def process_batch(
        self,
        messages: list[str],
        batch_size: int = 100,
        progress_callback: Callable | None = None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """
        Process a batch of messages with progress tracking.

        Args:
            messages: List of SMS messages to analyze
            batch_size: Number of messages to process in parallel
            progress_callback: Optional callback function to report progress

        Returns:
            Tuple of (list of results, batch statistics)
        """
        self.batch_stats["total_messages"] = len(messages)
        self.batch_stats["processed_messages"] = 0
        self.batch_stats["start_time"] = datetime.now()

        results = []

        # Nothing to process: return zeroed throughput stats instead of
        # spinning up an executor and dividing by a ~0 elapsed time (issue #13).
        if not messages:
            self.batch_stats["end_time"] = self.batch_stats["start_time"]
            self.batch_stats["processing_time"] = 0.0
            self.batch_stats["messages_per_second"] = 0.0
            return results, self.batch_stats

        # Process messages in parallel batches
        with ThreadPoolExecutor(max_workers=min(batch_size, 10)) as executor:
            future_to_message = {
                executor.submit(self.process_message, msg): msg for msg in messages
            }

            for future in future_to_message:
                result = future.result()
                results.append(result)

                # Update statistics
                self.batch_stats["processed_messages"] += 1
                if result["ensemble_predictions"]["majority_voting"]["label"] == "SPAM":
                    self.batch_stats["spam_detected"] += 1
                else:
                    self.batch_stats["ham_detected"] += 1

                # Calculate running average confidence
                self.batch_stats["avg_confidence"] = (
                    self.batch_stats["avg_confidence"]
                    * (self.batch_stats["processed_messages"] - 1)
                    + result["ensemble_predictions"]["majority_voting"]["confidence"]
                ) / self.batch_stats["processed_messages"]

                # Report progress if callback provided
                if progress_callback:
                    progress = (
                        self.batch_stats["processed_messages"]
                        / self.batch_stats["total_messages"]
                    )
                    progress_callback(progress)

        self.batch_stats["end_time"] = datetime.now()
        processing_time = (
            self.batch_stats["end_time"] - self.batch_stats["start_time"]
        ).total_seconds()
        self.batch_stats["processing_time"] = processing_time
        self.batch_stats["messages_per_second"] = (
            self.batch_stats["total_messages"] / processing_time
            if processing_time > 0
            else 0.0
        )

        return results, self.batch_stats

    def process_batch_generator(
        self,
        messages: list[str],
        cancel_check: Callable | None = None,
    ) -> Generator[dict[str, Any], None, None]:
        """
        Generator that processes messages sequentially and yields real-time
        progress updates.  Supports cancellation via a callable that returns
        ``True`` when processing should stop.

        Args:
            messages: List of SMS messages to analyze
            cancel_check: Optional callable checked before each message;
                          return ``True`` to cancel

        Yields:
            Dict with keys:
                - progress (float): 0.0 to 1.0
                - processed (int)
                - total (int)
                - current_message (str)
                - spam_count (int)
                - ham_count (int)
                - avg_confidence (float)
                - cancelled (bool) — ``True`` on the final yield if cancelled

        Side effects:
            Sets ``self.last_results`` and ``self.batch_stats`` so callers
            can retrieve partial/final results after iteration.
        """
        self.batch_stats["total_messages"] = len(messages)
        self.batch_stats["processed_messages"] = 0
        self.batch_stats["spam_detected"] = 0
        self.batch_stats["ham_detected"] = 0
        self.batch_stats["avg_confidence"] = 0.0
        self.batch_stats["start_time"] = datetime.now()

        self.last_results: list[dict[str, Any]] = []

        if not messages:
            self.batch_stats["end_time"] = self.batch_stats["start_time"]
            self.batch_stats["processing_time"] = 0.0
            self.batch_stats["messages_per_second"] = 0.0
            return

        for message in messages:
            if cancel_check and cancel_check():
                self.batch_stats["cancelled"] = True
                yield {
                    "progress": self.batch_stats["processed_messages"]
                    / self.batch_stats["total_messages"],
                    "processed": self.batch_stats["processed_messages"],
                    "total": self.batch_stats["total_messages"],
                    "current_message": message,
                    "spam_count": self.batch_stats["spam_detected"],
                    "ham_count": self.batch_stats["ham_detected"],
                    "avg_confidence": self.batch_stats["avg_confidence"],
                    "cancelled": True,
                }
                break

            result = self.process_message(message)
            self.last_results.append(result)

            self.batch_stats["processed_messages"] += 1
            if result["ensemble_predictions"]["majority_voting"]["label"] == "SPAM":
                self.batch_stats["spam_detected"] += 1
            else:
                self.batch_stats["ham_detected"] += 1

            prev = self.batch_stats["processed_messages"] - 1
            self.batch_stats["avg_confidence"] = (
                self.batch_stats["avg_confidence"] * prev
                + result["ensemble_predictions"]["majority_voting"]["confidence"]
            ) / self.batch_stats["processed_messages"]

            yield {
                "progress": self.batch_stats["processed_messages"]
                / self.batch_stats["total_messages"],
                "processed": self.batch_stats["processed_messages"],
                "total": self.batch_stats["total_messages"],
                "current_message": message,
                "spam_count": self.batch_stats["spam_detected"],
                "ham_count": self.batch_stats["ham_detected"],
                "avg_confidence": self.batch_stats["avg_confidence"],
                "cancelled": False,
            }

        self.batch_stats["end_time"] = datetime.now()
        processing_time = (
            self.batch_stats["end_time"] - self.batch_stats["start_time"]
        ).total_seconds()
        self.batch_stats["processing_time"] = processing_time
        self.batch_stats["messages_per_second"] = (
            self.batch_stats["total_messages"] / processing_time
            if processing_time > 0
            else 0.0
        )

    def generate_report(
        self, results: list[dict[str, Any]], format: str = "csv"
    ) -> pd.DataFrame:
        """
        Generate a detailed report from batch processing results.

        Args:
            results: List of processing results
            format: Output format ('csv' or 'excel')

        Returns:
            DataFrame containing the report data
        """
        report_data = []

        for result in results:
            # Prepare row data
            row = {"Message": result["message"], "Timestamp": result["timestamp"]}

            # Add individual model predictions
            for model, pred in result["model_predictions"].items():
                row[f"{model}_Classification"] = pred["label"]
                row[f"{model}_Confidence"] = pred["score"]
                row[f"{model}_SpamProbability"] = pred["spam_probability"]

            # Add ensemble predictions
            for method, pred in result["ensemble_predictions"].items():
                row[f"Ensemble_{method}_Classification"] = pred["label"]
                row[f"Ensemble_{method}_Confidence"] = pred["confidence"]
                row[f"Ensemble_{method}_SpamProbability"] = pred["spam_probability"]

            # Add risk indicators
            for indicator, present in result["risk_indicators"].items():
                row[f"Risk_{indicator}"] = present

            report_data.append(row)

        # Create DataFrame
        df = pd.DataFrame(report_data)

        return df
