"""Tests for the central configuration module."""

from pathlib import Path

import config


def test_defaults_are_sane():
    assert config.MAX_SMS_LENGTH == 1000
    assert config.APP_TITLE == "Spamlyser Pro - Ensemble Edition"
    assert config.VERIFICATION_MODEL_NAME == "distilbert-base-uncased"


def test_data_dir_defaults_to_project_root():
    expected = Path(__file__).resolve().parents[1] / "data"
    assert config.DATA_DIR == expected


def test_ensure_data_dir_creates_path(tmp_path):
    original = config.DATA_DIR
    try:
        config.DATA_DIR = tmp_path / "custom_data"
        result = config.ensure_data_dir()
        assert result.exists()
        assert result.is_dir()
        result.rmdir()
    finally:
        config.DATA_DIR = original


def test_get_optional_returns_default_when_unset(monkeypatch):
    monkeypatch.delenv("SPAMLYSER_TEST_VAR", raising=False)
    assert config.get_optional("SPAMLYSER_TEST_VAR", "fallback") == "fallback"


def test_get_optional_returns_env_value_when_set(monkeypatch):
    monkeypatch.setenv("SPAMLYSER_TEST_VAR", "hello")
    assert config.get_optional("SPAMLYSER_TEST_VAR") == "hello"


def test_feedback_paths_resolve_to_data_dir():
    assert config.FEEDBACK_DB_PATH
    assert config.FEEDBACK_JSON_PATH
    assert config.CUSTOM_RULES_PATH
    assert config.PERFORMANCE_DATA_PATH


def test_telemetry_disabled_by_default():
    assert config.ENABLE_TELEMETRY is False


def test_ensure_data_dir_is_idempotent(tmp_path):
    original = config.DATA_DIR
    try:
        config.DATA_DIR = tmp_path / "custom_data"
        first = config.ensure_data_dir()
        second = config.ensure_data_dir()
        assert first == second
        assert first.exists()
    finally:
        config.DATA_DIR = original
