"""
Tests for models/export_feature.py — dataframe_to_pdf

These tests run without a Streamlit server context; we mock the `streamlit`
module so the import succeeds on CI (where no browser/server is present).
"""

import sys
import types
import warnings

import pandas as pd

# ---------------------------------------------------------------------------
# Mock streamlit before importing the module under test
# ---------------------------------------------------------------------------
_st = types.ModuleType("streamlit")
for _name in ("info", "selectbox", "download_button", "error", "warning"):
    setattr(_st, _name, lambda *a, **kw: None)
sys.modules.setdefault("streamlit", _st)

# Suppress fpdf DeprecationWarnings (dest="S" legacy param)
warnings.filterwarnings("ignore")

from models.export_feature import _pdf_safe, dataframe_to_pdf

# ---------------------------------------------------------------------------
# _pdf_safe helper
# ---------------------------------------------------------------------------


class TestPdfSafe:
    def test_plain_ascii_unchanged(self):
        assert _pdf_safe("Free $5 win cash now") == "Free $5 win cash now"

    def test_rupee_replaced(self):
        result = _pdf_safe("WIN \u20b95000")  # ₹ is U+20B9
        assert "\u20b9" not in result
        assert "?" in result

    def test_emoji_replaced(self):
        result = _pdf_safe("reply YES \U0001f389")  # 🎉
        assert "\U0001f389" not in result
        assert "?" in result

    def test_euro_replaced(self):
        result = _pdf_safe("\u20ac1000 prize")  # €
        assert "\u20ac" not in result

    def test_latin1_chars_preserved(self):
        # £ (U+00A3) IS in latin-1 — must not be replaced
        assert _pdf_safe("\u00a3500") == "\u00a3500"

    def test_empty_string(self):
        assert _pdf_safe("") == ""

    def test_non_string_coerced(self):
        assert _pdf_safe(123) == "123"
        assert _pdf_safe(None) == "None"


# ---------------------------------------------------------------------------
# dataframe_to_pdf
# ---------------------------------------------------------------------------


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
        """The original code crashed with UnicodeEncodeError on ₹/emoji rows."""
        df = pd.DataFrame(
            [
                {"message": "WIN \u20b95000 NOW \U0001f389", "label": "SPAM"},
                {"message": "\u20ac1000 \u00a3500 \u2014 claim", "label": "SPAM"},
            ]
        )
        out = dataframe_to_pdf(df)
        assert self._valid_pdf(out.getvalue())

    def test_empty_dataframe_produces_valid_pdf(self):
        """Header-only PDF — no rows."""
        df = pd.DataFrame({"message": [], "label": []})
        out = dataframe_to_pdf(df)
        assert self._valid_pdf(out.getvalue())

    def test_custom_title_in_output(self):
        df = pd.DataFrame([{"message": "hello", "label": "HAM"}])
        out = dataframe_to_pdf(df, title="Test Export")
        assert self._valid_pdf(out.getvalue())

    def test_long_cell_value_truncated(self):
        """Cells longer than 25 chars should be truncated to 22 + '...'"""
        long_msg = "A" * 50
        df = pd.DataFrame([{"message": long_msg, "label": "SPAM"}])
        out = dataframe_to_pdf(df)
        # If truncation is broken the cell write would raise; valid PDF = truncation worked
        assert self._valid_pdf(out.getvalue())

    def test_returns_bytesio_at_start(self):
        """Caller (st.download_button) needs a seeked-to-start bytes buffer."""
        df = pd.DataFrame([{"message": "test", "label": "HAM"}])
        out = dataframe_to_pdf(df)
        pos = out.tell()
        assert pos == 0, f"Expected BytesIO position 0, got {pos}"
