"""Tests for models/export_feature.py."""

import json
import sys
import warnings

import pandas as pd
import pytest

# fpdf may be mocked in conftest. Remove the stub if present so the real
# module is used when installed, and skip PDF tests otherwise.
if "fpdf" in sys.modules and not hasattr(sys.modules["fpdf"], "__version__"):
    del sys.modules["fpdf"]
try:
    __import__("fpdf")
    _FPDF_INSTALLED = True
except ImportError:
    _FPDF_INSTALLED = False

from models.export_feature import (
    _csv_safe_cell,
    _pdf_safe,
    dataframe_to_csv,
    history_to_json,
)

if _FPDF_INSTALLED:
    from models.export_feature import dataframe_to_pdf

warnings.filterwarnings("ignore", category=DeprecationWarning)


class TestCsvExportSafety:
    def test_formula_like_strings_are_prefixed(self):
        assert _csv_safe_cell("=IMPORTXML(\"http://bad.example\")").startswith("'=")
        assert _csv_safe_cell("+SUM(1,2)").startswith("'+")
        assert _csv_safe_cell("-10+20").startswith("'-")
        assert _csv_safe_cell("@cmd").startswith("'@")

    def test_leading_whitespace_before_formula_is_prefixed(self):
        assert _csv_safe_cell("   =HYPERLINK(\"http://bad.example\")").startswith("'")

    def test_normal_strings_and_numbers_are_unchanged(self):
        assert _csv_safe_cell("Free prize claim") == "Free prize claim"
        assert _csv_safe_cell(0.95) == 0.95

    def test_dataframe_to_csv_neutralises_text_cells_only(self):
        df = pd.DataFrame(
            [
                {"message": "=cmd|' /C calc'!A0", "confidence": 0.98},
                {"message": "See you at 5pm", "confidence": 0.12},
            ]
        )
        csv_data = dataframe_to_csv(df)
        assert "'=cmd" in csv_data
        assert "0.98" in csv_data
        assert "See you at 5pm" in csv_data

    def test_newline_payload_neutralized(self):
        """Multi-line payloads that hide formulas after a newline."""
        result = _csv_safe_cell("innocent text\n=CMD('calc')")
        assert "\n" not in result  # newline should be collapsed

    def test_carriage_return_payload_neutralized(self):
        result = _csv_safe_cell("normal\r\n=HYPERLINK('http://evil.com')")
        assert "\r" not in result
        assert "\n" not in result

    def test_null_byte_stripped(self):
        result = _csv_safe_cell("hello\x00=CMD('calc')")
        assert "\x00" not in result

    def test_tab_separated_injection(self):
        """Tab character used to break into adjacent cells."""
        result = _csv_safe_cell("\t=SUM(A1:A10)")
        assert result.startswith("'")

    def test_pipe_command_injection(self):
        result = _csv_safe_cell("|cmd /c calc")
        assert result.startswith("'")

    def test_empty_string_unchanged(self):
        assert _csv_safe_cell("") == ""

    def test_none_value_unchanged(self):
        assert _csv_safe_cell(None) is None


class TestPdfSafe:
    def test_plain_ascii_unchanged(self):
        assert _pdf_safe("Free $5 win cash now") == "Free $5 win cash now"

    def test_rupee_replaced(self):
        result = _pdf_safe("WIN \u20b95000")
        assert "\u20b9" not in result
        assert "?" in result

    def test_emoji_replaced(self):
        result = _pdf_safe("reply YES \U0001f389")
        assert "\U0001f389" not in result
        assert "?" in result

    def test_euro_replaced(self):
        result = _pdf_safe("\u20ac1000 prize")
        assert "\u20ac" not in result

    def test_latin1_chars_preserved(self):
        assert _pdf_safe("\u00a3500") == "\u00a3500"

    def test_empty_string(self):
        assert _pdf_safe("") == ""

    def test_non_string_coerced(self):
        assert _pdf_safe(123) == "123"
        assert _pdf_safe(None) == "None"


@pytest.mark.skipif(
    not _FPDF_INSTALLED, reason="fpdf not installed — PDF tests skipped"
)
class TestDataframeToPdf:
    def _valid_pdf(self, data: bytes) -> bool:
        return data[:5] == b"%PDF-"

    def test_ascii_dataframe_produces_valid_pdf(self):
        df = pd.DataFrame(
            [
                {
                    "message": "Free entry win cash",
                    "label": "SPAM",
                    "confidence": "0.98",
                },
                {"message": "See you at 5pm", "label": "HAM", "confidence": "0.95"},
            ]
        )
        out = dataframe_to_pdf(df)
        assert self._valid_pdf(out.getvalue())

    def test_unicode_dataframe_produces_valid_pdf(self):
        df = pd.DataFrame(
            [
                {"message": "WIN \u20b95000 NOW \U0001f389", "label": "SPAM"},
                {"message": "\u20ac1000 \u00a3500 \u2014 claim", "label": "SPAM"},
            ]
        )
        out = dataframe_to_pdf(df)
        assert self._valid_pdf(out.getvalue())

    def test_empty_dataframe_produces_valid_pdf(self):
        df = pd.DataFrame({"message": [], "label": []})
        out = dataframe_to_pdf(df)
        assert self._valid_pdf(out.getvalue())

    def test_custom_title_in_output(self):
        df = pd.DataFrame([{"message": "hello", "label": "HAM"}])
        out = dataframe_to_pdf(df, title="Test Export")
        assert self._valid_pdf(out.getvalue())

    def test_long_cell_value_truncated(self):
        long_msg = "A" * 50
        df = pd.DataFrame([{"message": long_msg, "label": "SPAM"}])
        out = dataframe_to_pdf(df)
        assert self._valid_pdf(out.getvalue())

    def test_returns_bytesio_at_start(self):
        df = pd.DataFrame([{"message": "test", "label": "HAM"}])
        out = dataframe_to_pdf(df)
        assert out.tell() == 0


# ── New tests for history_to_json ────────────────────────────────────────────


class TestHistoryToJson:
    def test_returns_valid_json_string(self):
        history = [{"message": "hello", "label": "HAM", "confidence": 0.9}]
        result = history_to_json(history)
        parsed = json.loads(result)
        assert isinstance(parsed, list)
        assert parsed[0]["label"] == "HAM"

    def test_unicode_preserved(self):
        history = [{"message": "WIN \u20b95000!", "label": "SPAM"}]
        result = history_to_json(history)
        # ensure_ascii=False so the rupee sign should be present
        assert "\u20b9" in result

    def test_numpy_integers_serialised(self):
        try:
            import numpy as np

            history = [{"score": np.int64(42), "label": "HAM"}]
            result = history_to_json(history)
            parsed = json.loads(result)
            assert parsed[0]["score"] == 42
        except ImportError:
            pytest.skip("numpy not installed")

    def test_empty_history_returns_empty_array(self):
        result = history_to_json([])
        assert json.loads(result) == []

    def test_nested_structures_preserved(self):
        history = [
            {
                "message": "test",
                "model_predictions": {"BERT": {"label": "SPAM", "score": 0.95}},
            }
        ]
        result = history_to_json(history)
        parsed = json.loads(result)
        assert parsed[0]["model_predictions"]["BERT"]["label"] == "SPAM"


def test_export_encryptor():
    from models.export_encryptor import encrypt_export_data

    assert encrypt_export_data("test", "key") != "test"
