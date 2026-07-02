# Export Formats

Spamlyser Pro supports three export formats for analysis results.  All formats
are available from the **Batch Processor** results panel via the
"Export results as:" dropdown.

## CSV

The default format.  Every column visible in the results table is included.
Timestamps are serialised as strings.

**Use when:** you want to open results in Excel, Google Sheets, or process them
with pandas.

```
message,label,confidence,spam_probability,threat_type
"Free entry win cash now",SPAM,0.98,0.98,Scam/Fraud
"See you at 5pm",HAM,0.97,0.03,
```

## PDF

Landscape A4 PDF generated with fpdf2.  Includes:

- Centred title and export timestamp
- Grey header row, alternating-row shading for readability
- Text is clipped with a trailing ellipsis when it would overflow a cell

**Character encoding:** fpdf2's built-in helvetica font covers latin-1 only.
Characters outside that range (₹, €, emoji, CJK) are silently replaced with
`?` so the export always succeeds without raising an encoding error.

**Use when:** you need a print-ready summary or want to share results without
requiring the recipient to have a spreadsheet application.

> **Dependency:** requires `fpdf2` (`pip install fpdf2`).  If fpdf2 is not
> installed the PDF option is hidden and an informational message is shown.

## JSON

A pretty-printed JSON array where every element is the full analysis result
dict — including nested sub-objects like `model_predictions` and
`ensemble_predictions` that are not visible in the CSV/PDF table view.

Non-JSON-serialisable values (e.g. numpy integers, floats) are automatically
coerced to their Python equivalents so the export never fails silently.

**Use when:** you want to process results programmatically, feed them into
another tool, or archive the complete analysis output.

```json
[
  {
    "message": "WIN ₹5000 lottery prize",
    "label": "SPAM",
    "confidence": 0.97,
    "spam_probability": 0.97,
    "threat_type": "Scam/Fraud",
    "model_predictions": {
      "BERT": { "label": "SPAM", "score": 0.96 },
      "RoBERTa": { "label": "SPAM", "score": 0.98 }
    }
  }
]
```

## Implementation

| Symbol | Module | Purpose |
|---|---|---|
| `_pdf_safe(text)` | `models/export_feature.py` | latin-1 safe encoding with `?` replacement |
| `_build_pdf(df, title)` | `models/export_feature.py` | internal PDF renderer |
| `dataframe_to_pdf(df, title)` | `models/export_feature.py` | public helper used by tests |
| `history_to_json(history)` | `models/export_feature.py` | numpy-safe JSON serialiser |
| `export_results_button(history)` | `models/export_feature.py` | Streamlit download widget |

Batch exports can also include risk indicator columns produced by
`BatchProcessor._analyze_risk_indicators`. See
[`BATCH_RISK_INDICATORS.md`](BATCH_RISK_INDICATORS.md) for the current signal
definitions and regression coverage.

## Running export tests

```bash
pytest tests/test_export_feature.py -v
```

PDF tests are skipped automatically when fpdf2 is not installed.
