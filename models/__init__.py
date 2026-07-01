"""
Spamlyser Pro Models Package
This package contains all the model-related functionality for SMS threat detection.
"""

from .batch_processor import BatchProcessor
from .export_feature import export_results_button
from .threat_analyzer import (
    classify_threat_type,
    get_threat_specific_advice,
    THREAT_CATEGORIES,
)
from .word_analyzer import WordAnalyzer
from .calibration import ConfidenceCalibrator
from .custom_rules_manager import (
    load_custom_rules,
    save_custom_rules,
    check_custom_rules,
)
from .feedback_handler import FeedbackHandler

# ✅ Explicit __all__ export for better IDE support and clarity
__all__ = [
    "BatchProcessor",
    "export_results_button",
    "classify_threat_type",
    "get_threat_specific_advice",
    "THREAT_CATEGORIES",
    "WordAnalyzer",
    "ConfidenceCalibrator",
    "load_custom_rules",
    "save_custom_rules",
    "check_custom_rules",
    "FeedbackHandler",
]
