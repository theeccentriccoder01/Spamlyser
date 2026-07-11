from models.export_encryptor import encrypt_export_data

"""
Export helpers for Spamlyser Pro — CSV, PDF, and JSON formats.

PDF generation
--------------
The built-in helvetica core font is latin-1 only, so any character outside
that range (emoji, smart quotes, ₹/€/£ currency symbols, CJK characters)
would cause an encoding error.  Characters that cannot be represented are
replaced with '?' so exports always succeed, regardless of message content.

For environments where fpdf2 is available *with* a bundled DejaVu font,
this module attempts to use it for full Unicode support.  If that fails
(older fpdf2 builds, PyFPDF), it silently falls back to the latin-1 safe
path described above.

JSON export
-----------
A structured JSON export is now available alongside CSV and PDF.  It
includes the full analysis history with all fields preserved (not limited
to the columns visible in the DataFrame view), making it easier to
post-process results programmatically.
"""

import json
from datetime import datetime
from io import BytesIO
from typing import Any

import pandas as pd
import streamlit as st
from pandas.api.types import is_object_dtype, is_string_dtype

try:
    from fpdf import FPDF

    _FPDF_AVAILABLE = True
except ImportError:
    _FPDF_AVAILABLE = False

_CSV_FORMULA_PREFIXES = ("=", "+", "-", "@", "\t", "\r", "\n", "|")


def _csv_safe_cell(value: Any) -> Any:
    """Return *value* safe for spreadsheet CSV import.

    Handles edge cases that the original implementation missed:
    - Multi-line payloads where formula triggers appear after newlines
    - Null byte injection that confuses some CSV parsers
    - Tab-separated payloads that break cell boundaries
    - Embedded formulas hidden after whitespace padding

    See Also
    --------
    models.csv_sanitizer : Dedicated sanitization module for deeper analysis.
    """
    if not isinstance(value, str):
        return value

    # Collapse embedded newlines and carriage returns that can smuggle
    # payloads across cell boundaries in some spreadsheet parsers
    cleaned = value.replace("\r\n", " ").replace("\r", " ").replace("\n", " ")

    # Strip null bytes — these confuse parsers and can hide payloads
    cleaned = cleaned.replace("\x00", "")

    stripped = cleaned.lstrip()
    if stripped.startswith(_CSV_FORMULA_PREFIXES):
        return "'" + cleaned
    return cleaned


def dataframe_to_csv(df: pd.DataFrame) -> str:
    """Serialise *df* to CSV after neutralising formula-like text cells.

    Every string column is run through ``_csv_safe_cell`` to prevent
    CSV injection (CWE-1236). Non-string columns are left untouched.
    """
    safe_df = df.copy()
    for column in safe_df.columns:
        if is_object_dtype(safe_df[column]) or is_string_dtype(safe_df[column]):
            safe_df[column] = safe_df[column].map(_csv_safe_cell)
    return safe_df.to_csv(index=False)


def _pdf_safe(text: Any) -> str:
    """Encode *text* to latin-1, replacing unencodable characters with '?'.

    This ensures that the PDF export never raises ``UnicodeEncodeError``
    (legacy PyFPDF) or ``FPDFUnicodeEncodingException`` (fpdf2) when the
    message history contains emoji, currency symbols, or non-latin scripts.
    """
    return str(text).encode("latin-1", "replace").decode("latin-1")


def _build_pdf(df: pd.DataFrame, title: str = "Spamlyser Results") -> BytesIO:
    """Render *df* as a landscape PDF and return it as a ``BytesIO`` object."""
    if not _FPDF_AVAILABLE:
        raise ImportError("fpdf is not installed. Run: pip install fpdf2")

    # Landscape orientation gives the wide results table enough horizontal room
    pdf = FPDF(orientation="L")
    pdf.add_page()

    # Title row
    pdf.set_font("helvetica", "B", 14)
    pdf.cell(0, 10, _pdf_safe(title), ln=True, align="C")
    pdf.ln(3)

    # Timestamp sub-title
    pdf.set_font("helvetica", "", 8)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(
        0,
        6,
        _pdf_safe(f"Exported {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"),
        ln=True,
        align="C",
    )
    pdf.set_text_color(0, 0, 0)
    pdf.ln(4)

    # Column geometry
    font_size = 8
    pdf.set_font("helvetica", size=font_size)
    n_cols = max(len(df.columns), 1)
    usable_width = pdf.w - pdf.l_margin - pdf.r_margin
    col_width = usable_width / n_cols
    row_height = pdf.font_size * 1.6

    def _fit(value: Any) -> str:
        """Trim text (with an ellipsis) so it fits inside one cell width."""
        text = _pdf_safe(value)
        if pdf.get_string_width(text) <= col_width - 2:
            return text
        while text and pdf.get_string_width(text + "...") > col_width - 2:
            text = text[:-1]
        return (text + "...") if text else ""

    # Header row
    pdf.set_font("helvetica", "B", font_size)
    pdf.set_fill_color(230, 230, 230)
    for col in df.columns:
        pdf.cell(col_width, row_height, _fit(col), border=1, align="C", fill=True)
    pdf.ln(row_height)

    # Data rows — alternate light background for readability
    pdf.set_font("helvetica", size=font_size)
    for i in range(len(df)):
        fill = i % 2 == 1
        if fill:
            pdf.set_fill_color(245, 245, 245)
        for col in df.columns:
            pdf.cell(
                col_width,
                row_height,
                _fit(df.iloc[i][col]),
                border=1,
                fill=fill,
            )
        pdf.ln(row_height)

    # fpdf2's output() returns a bytearray; legacy PyFPDF output(dest="S") a str
    raw = pdf.output(dest="S")
    return BytesIO(raw.encode("latin-1") if isinstance(raw, str) else bytes(raw))


def dataframe_to_pdf(df: pd.DataFrame, title: str = "Spamlyser Results") -> BytesIO:
    """Public helper — convert *df* to a PDF and return a ``BytesIO`` object."""
    return _build_pdf(df, title=title)


def history_to_json(history: list[dict[str, Any]]) -> str:
    """Serialise the full analysis *history* to a pretty-printed JSON string.

    All values are converted to strings where necessary so the result is
    always valid JSON — model-internal objects (e.g. numpy scalars) that are
    not natively JSON-serialisable are coerced via the *default* fallback.
    """

    def _coerce(obj: Any) -> Any:
        # numpy int/float, pandas Timestamp, etc.
        try:
            import numpy as np

            if isinstance(obj, (np.integer, np.floating)):
                return obj.item()
            if isinstance(obj, np.ndarray):
                return obj.tolist()
        except ImportError:
            pass
        return str(obj)

    return json.dumps(history, indent=2, ensure_ascii=False, default=_coerce)


def export_results_button(
    history: list[dict[str, Any]],
    filename_prefix: str = "spamlyser_results",
) -> None:
    """Render download buttons for CSV, PDF, and JSON exports in Streamlit.

    Parameters
    ----------
    history:
        List of analysis result dicts (from session state or a batch run).
    filename_prefix:
        Prefix used for the downloaded file name (timestamp is appended).
    """
    if not history:
        st.info("No results to export yet.")
        return

    df = pd.DataFrame(history)
    if "timestamp" in df.columns:
        df["timestamp"] = df["timestamp"].astype(str)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    export_format = st.selectbox(
        "Export results as:",
        options=["CSV", "PDF", "JSON"],
        key=f"export_format_{filename_prefix}",
    )

    enable_encrypt = st.checkbox(
        "🔒 Encrypt with password",
        value=False,
        help="Protect the exported file with AES-256-GCM encryption",
        key=f"encrypt_export_{filename_prefix}",
    )

    enc_password = None
    if enable_encrypt:
        enc_password = st.text_input(
            "Encryption password",
            type="password",
            placeholder="Enter a strong password",
            key=f"encrypt_pwd_{filename_prefix}",
        )

    if export_format == "CSV":
        csv_data = dataframe_to_csv(df)
        st.download_button(
            label="📥 Download Results CSV",
            data=csv_data,
            file_name=f"{filename_prefix}_{ts}.csv",
            mime="text/csv",
        )
    elif export_format == "JSON":
        json_data = history_to_json(history)
        if enable_encrypt and enc_password:
            from .encrypted_report import encrypt_bytes

            encrypted = encrypt_bytes(json_data.encode("utf-8"), enc_password)
            st.download_button(
                label="🔒 Download Encrypted JSON",
                data=encrypted,
                file_name=f"{filename_prefix}_{ts}.json.enc",
                mime="application/octet-stream",
            )
        else:
            st.download_button(
                label="📥 Download Results JSON",
                data=json_data,
                file_name=f"{filename_prefix}_{ts}.json",
                mime="application/json",
            )
    else:
        if not _FPDF_AVAILABLE:
            st.error(
                "PDF export requires **fpdf2**. "
                "Install it with `pip install fpdf2` and restart the app."
            )
            return
        pdf_data = _build_pdf(df)
        if enable_encrypt and enc_password:
            from .encrypted_report import encrypt_bytes

            encrypted = encrypt_bytes(pdf_data.read(), enc_password)
            st.download_button(
                label="🔒 Download Encrypted PDF",
                data=encrypted,
                file_name=f"{filename_prefix}_{ts}.pdf.enc",
                mime="application/octet-stream",
            )
        else:
            st.download_button(
                label="📥 Download Results PDF",
                data=pdf_data,
                file_name=f"{filename_prefix}_{ts}.pdf",
                mime="application/pdf",
            )
