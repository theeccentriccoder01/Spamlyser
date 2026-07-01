"""
Centralized configuration management for Spamlyser Pro.

This module provides a single source of truth for all configuration settings,
supporting environment variables with sensible defaults.

Features:
- Environment variable support via python-dotenv
- Configurable model cache directory
- Configurable feedback database paths
- Streamlit secrets support
- Fallback to defaults for missing environment variables
"""

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Config:
    """Centralized configuration management."""

    # ========================
    # Model Configuration
    # ========================

    @staticmethod
    def get_verification_model_name() -> str:
        """
        Get the verification model name.

        Default: "distilbert-base-uncased"
        Environment Variable: VERIFICATION_MODEL_NAME
        """
        return os.getenv("VERIFICATION_MODEL_NAME", "distilbert-base-uncased")

    @staticmethod
    def get_model_cache_dir() -> Path:
        """
        Get the model cache directory path.

        Default: ~/.cache/huggingface
        Environment Variable: MODEL_CACHE_DIR
        """
        cache_dir = os.getenv(
            "MODEL_CACHE_DIR",
            str(Path.home() / ".cache" / "huggingface")
        )
        return Path(cache_dir)

    @staticmethod
    def ensure_model_cache_dir() -> Path:
        """Create and return the model cache directory."""
        cache_dir = Config.get_model_cache_dir()
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir

    # ========================
    # Feedback & Data Configuration
    # ========================

    @staticmethod
    def get_feedback_json_path() -> Path:
        """
        Get the feedback JSON file path (legacy, for migration).

        Default: ./feedback_data.json
        Environment Variable: FEEDBACK_JSON_PATH
        """
        path = os.getenv("FEEDBACK_JSON_PATH", "feedback_data.json")
        return Path(path)

    @staticmethod
    def get_feedback_db_path() -> Path:
        """
        Get the feedback SQLite database path.

        Default: ./feedback_data.db
        Environment Variable: FEEDBACK_DB_PATH
        """
        path = os.getenv("FEEDBACK_DB_PATH", "feedback_data.db")
        return Path(path)

    @staticmethod
    def ensure_feedback_dir() -> Path:
        """Create and return the feedback data directory."""
        db_path = Config.get_feedback_db_path()
        db_path.parent.mkdir(parents=True, exist_ok=True)
        return db_path.parent

    # ========================
    # CSV & Data Paths
    # ========================

    @staticmethod
    def get_csv_dir() -> Path:
        """
        Get the CSV data directory.

        Default: ./csv
        Environment Variable: CSV_DIR
        """
        path = os.getenv("CSV_DIR", "csv")
        return Path(path)

    @staticmethod
    def ensure_csv_dir() -> Path:
        """Create and return the CSV directory."""
        csv_dir = Config.get_csv_dir()
        csv_dir.mkdir(parents=True, exist_ok=True)
        return csv_dir

    # ========================
    # Assets & Images
    # ========================

    @staticmethod
    def get_assets_dir() -> Path:
        """
        Get the assets directory.

        Default: ./assets
        Environment Variable: ASSETS_DIR
        """
        path = os.getenv("ASSETS_DIR", "assets")
        return Path(path)

    @staticmethod
    def get_imgs_dir() -> Path:
        """
        Get the images directory.

        Default: ./imgs
        Environment Variable: IMGS_DIR
        """
        path = os.getenv("IMGS_DIR", "imgs")
        return Path(path)

    # ========================
    # API & External Services
    # ========================

    @staticmethod
    def get_huggingface_token() -> Optional[str]:
        """
        Get HuggingFace API token from environment.

        Environment Variable: HF_TOKEN
        Returns: Token or None if not set
        """
        return os.getenv("HF_TOKEN")

    @staticmethod
    def get_github_token() -> Optional[str]:
        """
        Get GitHub API token from environment.

        Environment Variable: GITHUB_TOKEN
        Returns: Token or None if not set
        """
        return os.getenv("GITHUB_TOKEN")

    # ========================
    # Streamlit Configuration
    # ========================

    @staticmethod
    def get_streamlit_theme() -> str:
        """
        Get Streamlit theme preference.

        Default: "dark"
        Environment Variable: STREAMLIT_THEME
        Options: "light", "dark"
        """
        return os.getenv("STREAMLIT_THEME", "dark")

    @staticmethod
    def get_streamlit_port() -> int:
        """
        Get Streamlit server port.

        Default: 8501
        Environment Variable: STREAMLIT_PORT
        """
        try:
            return int(os.getenv("STREAMLIT_PORT", "8501"))
        except ValueError:
            return 8501

    @staticmethod
    def get_streamlit_logger_level() -> str:
        """
        Get Streamlit logger level.

        Default: "info"
        Environment Variable: STREAMLIT_LOGGER_LEVEL
        Options: "debug", "info", "warning", "error"
        """
        return os.getenv("STREAMLIT_LOGGER_LEVEL", "info")

    # ========================
    # Application Configuration
    # ========================

    @staticmethod
    def get_app_name() -> str:
        """
        Get application name.

        Default: "Spamlyser Pro"
        Environment Variable: APP_NAME
        """
        return os.getenv("APP_NAME", "Spamlyser Pro")

    @staticmethod
    def get_app_version() -> str:
        """
        Get application version.

        Default: "1.0.0"
        Environment Variable: APP_VERSION
        """
        return os.getenv("APP_VERSION", "1.0.0")

    @staticmethod
    def get_debug_mode() -> bool:
        """
        Get debug mode flag.

        Default: False
        Environment Variable: DEBUG_MODE
        """
        return os.getenv("DEBUG_MODE", "false").lower() in ("true", "1", "yes")

    @staticmethod
    def get_log_level() -> str:
        """
        Get application log level.

        Default: "info"
        Environment Variable: LOG_LEVEL
        Options: "debug", "info", "warning", "error", "critical"
        """
        return os.getenv("LOG_LEVEL", "info")

    # ========================
    # Batch Processing
    # ========================

    @staticmethod
    def get_batch_size() -> int:
        """
        Get batch processing size.

        Default: 32
        Environment Variable: BATCH_SIZE
        """
        try:
            return int(os.getenv("BATCH_SIZE", "32"))
        except ValueError:
            return 32

    @staticmethod
    def get_max_workers() -> int:
        """
        Get maximum number of worker threads.

        Default: 4
        Environment Variable: MAX_WORKERS
        """
        try:
            return int(os.getenv("MAX_WORKERS", "4"))
        except ValueError:
            return 4

    # ========================
    # Model Performance & Caching
    # ========================

    @staticmethod
    def get_use_quantization() -> bool:
        """
        Get whether to use model quantization (int8).

        Default: True
        Environment Variable: USE_QUANTIZATION
        """
        return os.getenv("USE_QUANTIZATION", "true").lower() in ("true", "1", "yes")

    @staticmethod
    def get_use_dynamic_quantization() -> bool:
        """
        Get whether to use dynamic quantization when CUDA unavailable.

        Default: True
        Environment Variable: USE_DYNAMIC_QUANTIZATION
        """
        return os.getenv("USE_DYNAMIC_QUANTIZATION", "true").lower() in ("true", "1", "yes")

    @staticmethod
    def get_cache_models() -> bool:
        """
        Get whether to cache loaded models in memory.

        Default: True
        Environment Variable: CACHE_MODELS
        """
        return os.getenv("CACHE_MODELS", "true").lower() in ("true", "1", "yes")

    # ========================
    # Feature Flags
    # ========================

    @staticmethod
    def is_feature_enabled(feature_name: str) -> bool:
        """
        Check if a feature is enabled.

        Environment Variable: FEATURE_{FEATURE_NAME_UPPER}
        Returns: True if enabled, False otherwise
        """
        env_key = f"FEATURE_{feature_name.upper()}"
        return os.getenv(env_key, "false").lower() in ("true", "1", "yes")

    @staticmethod
    def get_enabled_features() -> list:
        """Get list of all enabled features."""
        features = []
        common_features = [
            "word_analysis",
            "threat_analysis",
            "feedback_system",
            "model_explainability",
            "batch_processing",
            "pdf_export",
            "dark_mode",
        ]
        for feature in common_features:
            if Config.is_feature_enabled(feature):
                features.append(feature)
        return features

    # ========================
    # Validation
    # ========================

    @staticmethod
    def validate_configuration() -> tuple:
        """
        Validate configuration and return (is_valid, errors, warnings).

        Returns:
            tuple: (is_valid: bool, errors: list[str], warnings: list[str])
        """
        errors = []
        warnings = []

        # Check if model cache directory exists or is writable
        try:
            cache_dir = Config.ensure_model_cache_dir()
            if not cache_dir.exists():
                errors.append(f"Model cache directory not accessible: {cache_dir}")
        except PermissionError:
            errors.append(f"Permission denied for model cache directory: {cache_dir}")

        # Check feedback paths
        try:
            Config.ensure_feedback_dir()
        except PermissionError:
            errors.append(f"Permission denied for feedback directory")

        # Check CSV directory
        try:
            Config.ensure_csv_dir()
        except PermissionError:
            errors.append(f"Permission denied for CSV directory")

        # Warnings for missing optional credentials
        if not Config.get_huggingface_token():
            warnings.append("HF_TOKEN not set - some features may be limited")

        if not Config.get_github_token():
            warnings.append("GITHUB_TOKEN not set - GitHub integration limited")

        is_valid = len(errors) == 0
        return is_valid, errors, warnings

    @staticmethod
    def print_configuration_summary():
        """Print a summary of current configuration."""
        print("\n" + "=" * 60)
        print("SPAMLYSER CONFIGURATION SUMMARY")
        print("=" * 60)
        print(f"\n📱 Application:")
        print(f"  App Name: {Config.get_app_name()}")
        print(f"  App Version: {Config.get_app_version()}")
        print(f"  Debug Mode: {Config.get_debug_mode()}")
        print(f"\n🤖 Model Configuration:")
        print(f"  Verification Model: {Config.get_verification_model_name()}")
        print(f"  Model Cache Dir: {Config.get_model_cache_dir()}")
        print(f"  Use Quantization: {Config.get_use_quantization()}")
        print(f"\n💾 Data Paths:")
        print(f"  Feedback DB: {Config.get_feedback_db_path()}")
        print(f"  CSV Dir: {Config.get_csv_dir()}")
        print(f"\n🎨 Streamlit:")
        print(f"  Theme: {Config.get_streamlit_theme()}")
        print(f"  Port: {Config.get_streamlit_port()}")
        print(f"\n⚙️  Processing:")
        print(f"  Batch Size: {Config.get_batch_size()}")
        print(f"  Max Workers: {Config.get_max_workers()}")
        print(f"\n✨ Enabled Features: {', '.join(Config.get_enabled_features()) or 'None'}")

        is_valid, errors, warnings = Config.validate_configuration()
        if errors:
            print(f"\n❌ Configuration Errors:")
            for error in errors:
                print(f"  - {error}")
        if warnings:
            print(f"\n⚠️  Warnings:")
            for warning in warnings:
                print(f"  - {warning}")

        print("=" * 60 + "\n")

        return is_valid
