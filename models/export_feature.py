from datetime import datetime
from io import BytesIO

import pandas as pd
import streamlit as st
from fpdf import FPDF


def _pdf_safe(text):
    """Make text safe for the core PDF font (helvetica), which only supports
    latin-1. Characters outside that range — ₹, €, emoji, smart quotes, etc.,
    all common in SMS spam — are replaced with "?" so the export always
    produces a valid PDF instead of raising UnicodeEncodeError (legacy PyFPDF)
    or FPDFUnicodeEncodingException (fpdf2)."""
    return str(text).encode("latin-1", "replace").decode("latin-1")


def dataframe_to_pdf(df, title="Spamlyser Results"):
    # Landscape orientation gives the wide results table (which can have many
    # columns) enough horizontal room to stay readable.
    pdf = FPDF(orientation="L")
    pdf.add_page()
    pdf.set_font("helvetica", "B", 14)
    pdf.cell(0, 10, _pdf_safe(title), ln=True, align="C")
    pdf.ln(5)

    font_size = 8
    pdf.set_font("helvetica", size=font_size)
    n_cols = max(len(df.columns), 1)
    usable_width = pdf.w - pdf.l_margin - pdf.r_margin
    col_width = usable_width / n_cols
    row_height = pdf.font_size * 1.6

    def _fit(value):
        """Trim text (with an ellipsis) until it fits inside one cell, so
        adjacent columns never overlap no matter how many columns there are."""
        text = _pdf_safe(value)
        if pdf.get_string_width(text) <= col_width - 2:
            return text
        while text and pdf.get_string_width(text + "...") > col_width - 2:
            text = text[:-1]
        return (text + "...") if text else ""

    # Header
    pdf.set_font("helvetica", "B", font_size)
    for col in df.columns:
        pdf.cell(col_width, row_height, _fit(col), border=1, align="C")
    pdf.ln(row_height)

    # Rows
    pdf.set_font("helvetica", size=font_size)
    for i in range(len(df)):
        for col in df.columns:
            pdf.cell(col_width, row_height, _fit(df.iloc[i][col]), border=1)
        pdf.ln(row_height)

    # fpdf2's output() returns a bytearray; legacy PyFPDF's output(dest="S")
    # returns a str. Handle both so the export works regardless of which
    # fpdf fork the environment resolves "fpdf" to.
    raw = pdf.output(dest="S")
    pdf_bytes = BytesIO(raw.encode("latin-1") if isinstance(raw, str) else bytes(raw))
    return pdf_bytes


def export_results_button(history, filename_prefix="spamlyser_results"):
    if not history:
        st.info("No results to export yet.")
        return
    df = pd.DataFrame(history)
    if "timestamp" in df.columns:
        df["timestamp"] = df["timestamp"].astype(str)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    export_format = st.selectbox(
        "Export results as:",
        options=["CSV", "PDF"],
        key=f"export_format_{filename_prefix}",
    )
    if export_format == "CSV":
        csv_data = df.to_csv(index=False)
        st.download_button(
            label="📥 Download Results CSV",
            data=csv_data,
            file_name=f"{filename_prefix}_{ts}.csv",
            mime="text/csv",
        )
    else:
        pdf_data = dataframe_to_pdf(df)
        st.download_button(
            label="📥 Download Results PDF",
            data=pdf_data,
            file_name=f"{filename_prefix}_{ts}.pdf",
            mime="application/pdf",
        )
