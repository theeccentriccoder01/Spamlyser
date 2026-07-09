"""
Central configuration for Spamlyser Pro.

Loads settings from environment variables (via python-dotenv) with sensible
defaults so the app works out-of-the-box.  In production, override any value
through a ``.env`` file or system environment variables.

All path constants resolve to absolute ``Path`` objects so they work
regardless of the current working directory when the application starts.
"""

import os
from pathlib import Path
from typing import Optional

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


# ── Paths ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent

# Where per-run data (feedback, performance snapshots) is stored.
DATA_DIR = Path(os.getenv("SPAMLYSER_DATA_DIR", str(PROJECT_ROOT / "data")))

# ── Model settings ─────────────────────────────────────────────────────────
MODEL_CACHE_DIR = Path(
    os.getenv(
        "SPAMLYSER_MODEL_CACHE_DIR",
        str(Path.home() / ".cache" / "huggingface" / "transformers"),
    )
)

# Default verification model used by ``model_init`` on startup.
VERIFICATION_MODEL_NAME: str = os.getenv(
    "SPAMLYSER_VERIFICATION_MODEL", "distilbert-base-uncased"
)

# ── Data files ─────────────────────────────────────────────────────────────
FEEDBACK_DB_PATH: str = os.getenv(
    "SPAMLYSER_FEEDBACK_DB", str(DATA_DIR / "feedback.db")
)

FEEDBACK_JSON_PATH: str = os.getenv(
    "SPAMLYSER_FEEDBACK_JSON", str(DATA_DIR / "feedback_data.json")
)

CUSTOM_RULES_PATH: str = os.getenv(
    "SPAMLYSER_CUSTOM_RULES", str(DATA_DIR / "custom_rules.json")
)

SENDER_REPUTATION_DB_PATH: str = os.getenv(
    "SPAMLYSER_SENDER_REPUTATION_DB", str(DATA_DIR / "sender_reputation.json")
)

CATEGORIES_CONFIG_PATH: str = os.getenv(
    "SPAMLYSER_CATEGORIES_CONFIG", str(DATA_DIR / "categories.json")
)

PERFORMANCE_DATA_PATH: str = os.getenv(
    "SPAMLYSER_PERFORMANCE_DATA", str(DATA_DIR / "performance_data.json")
)

WEBHOOK_CONFIG_PATH: str = os.getenv(
    "SPAMLYSER_WEBHOOK_CONFIG", str(DATA_DIR / "webhooks.json")
)

WEBHOOK_RETRY_COUNT: int = int(os.getenv("SPAMLYSER_WEBHOOK_RETRY", "3"))

ENCRYPT_REPORT_BY_DEFAULT: bool = (
    os.getenv("SPAMLYSER_ENCRYPT_REPORT", "false").lower() == "true"
)

# When True, all CSV exports run through the formula injection sanitizer
# to prevent CWE-1236 attacks when users open exports in spreadsheet apps.
CSV_EXPORT_SANITIZE_FORMULAS: bool = (
    os.getenv("SPAMLYSER_CSV_SANITIZE_FORMULAS", "true").lower() == "true"
)

# ── Application ────────────────────────────────────────────────────────────
APP_TITLE: str = os.getenv("SPAMLYSER_APP_TITLE", "Spamlyser Pro - Ensemble Edition")
APP_ICON: str = os.getenv("SPAMLYSER_APP_ICON", "🛡️")

# ── Feature flags ──────────────────────────────────────────────────────────
ENABLE_TELEMETRY: bool = os.getenv("SPAMLYSER_ENABLE_TELEMETRY", "false").lower() in (
    "1",
    "true",
    "yes",
)

MAX_SMS_LENGTH: int = int(os.getenv("SPAMLYSER_MAX_SMS_LENGTH", "1000"))

# Maximum number of entries allowed in each custom-rules list (allowlist or
# blocklist).  Exceeding this limit is a soft warning — the app still works —
# but it is enforced in the UI to keep the rules file manageable.
MAX_CUSTOM_RULES_PER_LIST: int = int(
    os.getenv("SPAMLYSER_MAX_CUSTOM_RULES_PER_LIST", "500")
)

MODEL_COMPARISON_SAMPLE_SIZE: int = int(os.getenv("SPAMLYSER_COMPARE_SAMPLES", "5"))
MAX_COMPOUND_RULES: int = int(os.getenv("SPAMLYSER_MAX_COMPOUND_RULES", "50"))
BENCHMARK_SAMPLE_SIZE: int = int(os.getenv("SPAMLYSER_BENCHMARK_SAMPLES", "10"))

BENCHMARK_WARMUP_RUNS: int = int(os.getenv("SPAMLYSER_BENCHMARK_WARMUP", "1"))

LANGUAGE_DETECTION_ENABLED: bool = (
    os.getenv("SPAMLYSER_LANG_DETECT", "true").lower() == "true"
)

LANGUAGE_DETECTION_CONFIDENCE_THRESHOLD: float = float(
    os.getenv("SPAMLYSER_LANG_CONFIDENCE", "0.3")
)


# ── Helpers ────────────────────────────────────────────────────────────────
def ensure_data_dir() -> Path:
    """Create the data directory (and any parents) if it does not exist.

    Returns the resolved ``Path`` object so callers can use it directly::

        path = ensure_data_dir() / "my_file.json"
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return DATA_DIR


def get_optional(key: str, default: str | None = None) -> str | None:
    """Return the env value for *key* or *default* when unset or empty."""
    val = os.getenv(key, default)
    return val if val else default


ANALYTICS_DB_PATH: str = os.getenv(
    "SPAMLYSER_ANALYTICS_DB", str(DATA_DIR / "analytics.db")
)

ANALYTICS_RETENTION_DAYS: int = int(os.getenv("SPAMLYSER_ANALYTICS_RETENTION", "90"))

BATCH_RATE_LIMIT = 50

# ── Automated benchmarks ──────────────────────────────────────────────────
BENCHMARK_AUTO_RUNS: int = int(os.getenv("SPAMLYSER_BENCH_RUNS", "10"))
BENCHMARK_AUTO_WARMUP: int = int(os.getenv("SPAMLYSER_BENCH_WARMUP", "2"))
BENCHMARK_REGRESSION_THRESHOLD_PCT: float = float(
    os.getenv("SPAMLYSER_BENCH_REGRESSION_PCT", "10.0")
)
BENCHMARK_HISTORY_PATH: str = os.getenv(
    "SPAMLYSER_BENCH_HISTORY", str(DATA_DIR / "benchmark_history.json")
)

# ── Error boundary / resilience ────────────────────────────────────────────
ERROR_BOUNDARY_ENABLED: bool = os.getenv("SPAMLYSER_ERROR_BOUNDARY", "true").lower() == "true"
ERROR_BOUNDARY_SHOW_DETAIL: bool = os.getenv("SPAMLYSER_ERROR_DETAIL", "false").lower() == "true"
