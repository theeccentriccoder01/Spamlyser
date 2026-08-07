"""
Structured Feature Importance & Attribution Exporter for Explainable AI (XAI).
Serializes token score contributions into structured formats (JSON, CSV, HTML audit report).
"""

import json
import csv
import io
from typing import Dict, Any, List

class ExplainabilityExporter:
    """
    Exports model explainer results into structured formats for audit and compliance.
    """

    @staticmethod
    def export_json(explanation: Dict[str, Any]) -> str:
        """
        Serialize explanation dict to JSON string.
        """
        return json.dumps(explanation, indent=2, ensure_ascii=False)

    @staticmethod
    def export_csv(explanation: Dict[str, Any]) -> str:
        """
        Serialize feature attributions to CSV format.
        """
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["token", "weight", "direction", "category"])

        top_features = explanation.get("top_features", [])
        for item in top_features:
            writer.writerow([
                item.get("word", item.get("token", "")),
                item.get("weight", item.get("score", 0.0)),
                item.get("direction", "spam" if item.get("weight", 0.0) > 0 else "ham"),
                item.get("category", "N/A")
            ])

        return output.getvalue()

    @staticmethod
    def export_html_report(explanation: Dict[str, Any]) -> str:
        """
        Generate standalone HTML report for visual model inspection.
        """
        title = explanation.get("title", "Spamlyser Feature Explanation Report")
        score = explanation.get("spam_score", 0.0)
        label = "SPAM" if score > 0.5 else "HAM"

        rows = ""
        for feat in explanation.get("top_features", []):
            word = feat.get("word", feat.get("token", ""))
            weight = feat.get("weight", feat.get("score", 0.0))
            direction = "Spam Indicator" if weight > 0 else "Ham Indicator"
            color = "#ef4444" if weight > 0 else "#10b981"
            rows += f"<tr><td><b>{word}</b></td><td style='color: {color}'>{weight:+.4f}</td><td>{direction}</td></tr>"

        html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>{title}</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #0f172a; color: #f8fafc; padding: 2rem; }}
        .card {{ background: #1e293b; border-radius: 8px; padding: 1.5rem; max-width: 700px; margin: 0 auto; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.3); }}
        h2 {{ color: #38bdf8; margin-top: 0; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 1rem; }}
        th, td {{ padding: 0.75rem; text-align: left; border-bottom: 1px solid #334155; }}
        th {{ background: #0f172a; color: #94a3b8; }}
        .badge {{ padding: 0.25rem 0.75rem; border-radius: 9999px; font-weight: bold; display: inline-block; }}
        .spam {{ background: #7f1d1d; color: #fca5a5; }}
        .ham {{ background: #064e3b; color: #6ee7b7; }}
    </style>
</head>
<body>
    <div class="card">
        <h2>Spamlyser Model Explanation Report</h2>
        <p>Prediction: <span class="badge {'spam' if label == 'SPAM' else 'ham'}">{label}</span> (Score: {score:.2f})</p>
        <table>
            <thead>
                <tr><th>Token / Keyword</th><th>Attribution Weight</th><th>Impact Direction</th></tr>
            </thead>
            <tbody>
                {rows}
            </tbody>
        </table>
    </div>
</body>
</html>"""
        return html
