"""
Unit tests for the config module.

Tests centralized configuration management, environment variable loading,
and path validation.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import Config


class TestModelConfiguration:
    """Tests for model-related configuration."""

    def test_get_verification_model_name_default(self):
        """Test default verification model name."""
        with patch.dict(os.environ, {}, clear=False):
            if "VERIFICATION_MODEL_NAME" in os.environ:
                del os.environ["VERIFICATION_MODEL_NAME"]
            assert Config.get_verification_model_name() == "distilbert-base-uncased"

    def test_get_verification_model_name_custom(self):
        """Test custom verification model name from environment."""
        with patch.dict(os.environ, {"VERIFICATION_MODEL_NAME": "bert-base-uncased"}):
            assert Config.get_verification_model_name() == "bert-base-uncased"

    def test_get_model_cache_dir_default(self):
        """Test default model cache directory."""
        with patch.dict(os.environ, {}, clear=False):
            if "MODEL_CACHE_DIR" in os.environ:
                del os.environ["MODEL_CACHE_DIR"]
            cache_dir = Config.get_model_cache_dir()
            expected = Path.home() / ".cache" / "huggingface"
            assert cache_dir == expected

    def test_get_model_cache_dir_custom(self):
        """Test custom model cache directory from environment."""
        with patch.dict(os.environ, {"MODEL_CACHE_DIR": "/tmp/models"}):
            cache_dir = Config.get_model_cache_dir()
            assert cache_dir == Path("/tmp/models")

    def test_ensure_model_cache_dir_creates_directory(self):
        """Test that ensure_model_cache_dir creates the directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_path = Path(tmpdir) / "test_cache"
            assert not test_path.exists()
            with patch.dict(os.environ, {"MODEL_CACHE_DIR": str(test_path)}):
                result = Config.ensure_model_cache_dir()
                assert result.exists()
                assert result == test_path


class TestFeedbackConfiguration:
    """Tests for feedback-related configuration."""

    def test_get_feedback_db_path_default(self):
        """Test default feedback database path."""
        with patch.dict(os.environ, {}, clear=False):
            if "FEEDBACK_DB_PATH" in os.environ:
                del os.environ["FEEDBACK_DB_PATH"]
            path = Config.get_feedback_db_path()
            assert path == Path("feedback_data.db")

    def test_get_feedback_db_path_custom(self):
        """Test custom feedback database path from environment."""
        with patch.dict(os.environ, {"FEEDBACK_DB_PATH": "/data/feedback.db"}):
            path = Config.get_feedback_db_path()
            assert path == Path("/data/feedback.db")

    def test_ensure_feedback_dir_creates_directory(self):
        """Test that ensure_feedback_dir creates the directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_path = Path(tmpdir) / "feedback" / "data.db"
            assert not test_path.parent.exists()
            with patch.dict(os.environ, {"FEEDBACK_DB_PATH": str(test_path)}):
                result = Config.ensure_feedback_dir()
                assert result.exists()
                assert result == test_path.parent


class TestDataDirectoryConfiguration:
    """Tests for data directory configuration."""

    def test_get_csv_dir_default(self):
        """Test default CSV directory."""
        with patch.dict(os.environ, {}, clear=False):
            if "CSV_DIR" in os.environ:
                del os.environ["CSV_DIR"]
            csv_dir = Config.get_csv_dir()
            assert csv_dir == Path("csv")

    def test_get_csv_dir_custom(self):
        """Test custom CSV directory from environment."""
        with patch.dict(os.environ, {"CSV_DIR": "/data/csv_files"}):
            csv_dir = Config.get_csv_dir()
            assert csv_dir == Path("/data/csv_files")

    def test_get_assets_dir(self):
        """Test assets directory configuration."""
        with patch.dict(os.environ, {"ASSETS_DIR": "/app/assets"}):
            assets_dir = Config.get_assets_dir()
            assert assets_dir == Path("/app/assets")

    def test_get_imgs_dir(self):
        """Test images directory configuration."""
        with patch.dict(os.environ, {"IMGS_DIR": "/app/images"}):
            imgs_dir = Config.get_imgs_dir()
            assert imgs_dir == Path("/app/images")


class TestApplicationConfiguration:
    """Tests for general application configuration."""

    def test_get_app_name_default(self):
        """Test default application name."""
        with patch.dict(os.environ, {}, clear=False):
            if "APP_NAME" in os.environ:
                del os.environ["APP_NAME"]
            assert Config.get_app_name() == "Spamlyser Pro"

    def test_get_app_name_custom(self):
        """Test custom application name."""
        with patch.dict(os.environ, {"APP_NAME": "Custom Spamlyser"}):
            assert Config.get_app_name() == "Custom Spamlyser"

    def test_get_debug_mode_default(self):
        """Test debug mode defaults to False."""
        with patch.dict(os.environ, {}, clear=False):
            if "DEBUG_MODE" in os.environ:
                del os.environ["DEBUG_MODE"]
            assert Config.get_debug_mode() is False

    def test_get_debug_mode_enabled(self):
        """Test debug mode can be enabled."""
        with patch.dict(os.environ, {"DEBUG_MODE": "true"}):
            assert Config.get_debug_mode() is True

    def test_get_debug_mode_disabled_variations(self):
        """Test various false values for debug mode."""
        false_values = ["false", "0", "no", "FALSE", "No"]
        for value in false_values:
            with patch.dict(os.environ, {"DEBUG_MODE": value}):
                assert Config.get_debug_mode() is False

    def test_get_batch_size_default(self):
        """Test default batch size."""
        with patch.dict(os.environ, {}, clear=False):
            if "BATCH_SIZE" in os.environ:
                del os.environ["BATCH_SIZE"]
            assert Config.get_batch_size() == 32

    def test_get_batch_size_custom(self):
        """Test custom batch size."""
        with patch.dict(os.environ, {"BATCH_SIZE": "64"}):
            assert Config.get_batch_size() == 64

    def test_get_batch_size_invalid_defaults(self):
        """Test invalid batch size falls back to default."""
        with patch.dict(os.environ, {"BATCH_SIZE": "invalid"}):
            assert Config.get_batch_size() == 32


class TestBooleanConfiguration:
    """Tests for boolean configuration values."""

    def test_get_use_quantization_default(self):
        """Test quantization defaults to True."""
        with patch.dict(os.environ, {}, clear=False):
            if "USE_QUANTIZATION" in os.environ:
                del os.environ["USE_QUANTIZATION"]
            assert Config.get_use_quantization() is True

    def test_get_use_quantization_variations(self):
        """Test various true/false values for quantization."""
        true_values = ["true", "1", "yes", "TRUE", "Yes"]
        for value in true_values:
            with patch.dict(os.environ, {"USE_QUANTIZATION": value}):
                assert Config.get_use_quantization() is True

    def test_get_cache_models_default(self):
        """Test model caching defaults to True."""
        with patch.dict(os.environ, {}, clear=False):
            if "CACHE_MODELS" in os.environ:
                del os.environ["CACHE_MODELS"]
            assert Config.get_cache_models() is True


class TestFeatureFlags:
    """Tests for feature flag configuration."""

    def test_is_feature_enabled_default_false(self):
        """Test feature defaults to disabled."""
        with patch.dict(os.environ, {}, clear=False):
            if "FEATURE_TEST_FEATURE" in os.environ:
                del os.environ["FEATURE_TEST_FEATURE"]
            assert Config.is_feature_enabled("test_feature") is False

    def test_is_feature_enabled_custom(self):
        """Test custom feature enablement."""
        with patch.dict(os.environ, {"FEATURE_TEST_FEATURE": "true"}):
            assert Config.is_feature_enabled("test_feature") is True

    def test_get_enabled_features(self):
        """Test retrieving list of enabled features."""
        env_vars = {
            "FEATURE_WORD_ANALYSIS": "true",
            "FEATURE_THREAT_ANALYSIS": "false",
            "FEATURE_FEEDBACK_SYSTEM": "true",
        }
        with patch.dict(os.environ, env_vars):
            features = Config.get_enabled_features()
            assert "word_analysis" in features
            assert "feedback_system" in features
            assert "threat_analysis" not in features


class TestConfigValidation:
    """Tests for configuration validation."""

    def test_validate_configuration_success(self):
        """Test successful configuration validation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            env_vars = {
                "MODEL_CACHE_DIR": str(Path(tmpdir) / "models"),
                "FEEDBACK_DB_PATH": str(Path(tmpdir) / "feedback.db"),
                "CSV_DIR": str(Path(tmpdir) / "csv"),
            }
            with patch.dict(os.environ, env_vars):
                is_valid, errors, warnings = Config.validate_configuration()
                assert is_valid is True
                assert len(errors) == 0

    def test_validate_configuration_warnings_for_missing_tokens(self):
        """Test validation warns about missing API tokens."""
        with patch.dict(os.environ, {}, clear=False):
            # Remove token env vars
            for key in ["HF_TOKEN", "GITHUB_TOKEN"]:
                if key in os.environ:
                    del os.environ[key]
            
            is_valid, errors, warnings = Config.validate_configuration()
            assert is_valid is True
            assert len(warnings) > 0


class TestStreamlitConfiguration:
    """Tests for Streamlit-specific configuration."""

    def test_get_streamlit_theme_default(self):
        """Test default Streamlit theme."""
        with patch.dict(os.environ, {}, clear=False):
            if "STREAMLIT_THEME" in os.environ:
                del os.environ["STREAMLIT_THEME"]
            assert Config.get_streamlit_theme() == "dark"

    def test_get_streamlit_port_default(self):
        """Test default Streamlit port."""
        with patch.dict(os.environ, {}, clear=False):
            if "STREAMLIT_PORT" in os.environ:
                del os.environ["STREAMLIT_PORT"]
            assert Config.get_streamlit_port() == 8501

    def test_get_streamlit_port_custom(self):
        """Test custom Streamlit port."""
        with patch.dict(os.environ, {"STREAMLIT_PORT": "9000"}):
            assert Config.get_streamlit_port() == 9000


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
