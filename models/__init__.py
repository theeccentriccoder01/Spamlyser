"""
Spamlyser Pro Models Package
This package contains all the model-related functionality for SMS threat detection.
"""

# Import benchmark helpers from the top-level benchmarks package (optional).
try:
    from benchmarks.benchmark_runner import (
        confidence_distribution,
        latency_benchmark,
        run_all,
        summary,
    )
except ImportError:
    pass

from .batch_processor import BatchProcessor
from .benchmark_automation import BenchmarkHistory, BenchmarkResult, run_automated_benchmark
from .calibration import ConfidenceCalibrator
from .custom_rules_manager import (
    check_custom_rules,
    load_custom_rules,
    save_custom_rules,
)
from .encrypted_report import ReportEncryptor
from .export_feature import export_results_button
from .language_detector import detect_language, is_language_supported
from .message_categorizer import MessageCategorizer
from .model_comparator import agreement_score, compare_predictions
from .rule_engine import (
    check_compound_rules,
    evaluate_compound_rule,
    evaluate_condition,
    validate_compound_rules,
)
from .sender_reputation import SenderReputation
from .simple_explainer import SPAM_KEYWORDS, SimpleExplainer
from .storage_manager import StorageManager, default_json_validator
from .threat_analyzer import (
    THREAT_CATEGORIES,
    classify_threat_type,
    get_threat_specific_advice,
)
from .webhook_notifier import WebhookNotifier
from .word_analyzer import WordAnalyzer

__all__ = [
    "SPAM_KEYWORDS",
    "THREAT_CATEGORIES",
    "BatchProcessor",
    "BenchmarkHistory",
    "BenchmarkResult",
    "ConfidenceCalibrator",
    "MessageCategorizer",
    "ReportEncryptor",
    "SenderReputation",
    "SimpleExplainer",
    "StorageManager",
    "WebhookNotifier",
    "WordAnalyzer",
    "agreement_score",
    "check_compound_rules",
    "check_custom_rules",
    "classify_threat_type",
    "compare_predictions",
    "confidence_distribution",
    "run_automated_benchmark",
    "default_json_validator",
    "detect_language",
    "evaluate_compound_rule",
    "evaluate_condition",
    "export_results_button",
    "get_threat_specific_advice",
    "is_language_supported",
    "latency_benchmark",
    "load_custom_rules",
    "run_all",
    "save_custom_rules",
    "summary",
    "validate_compound_rules",
]
