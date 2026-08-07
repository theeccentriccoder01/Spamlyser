import pytest
import json
from models.explainability_exporter import ExplainabilityExporter
from models.simple_explainer import export_explanation

@pytest.fixture
def sample_explanation():
    return {
        "spam_score": 0.85,
        "top_features": [
            {"word": "winner", "weight": 0.8, "category": "Financial / Prize"},
            {"word": "urgent", "weight": 0.5, "category": "Urgency / Pressure"},
            {"word": "meeting", "weight": -0.4, "category": "Work"}
        ]
    }

def test_explainability_exporter_json(sample_explanation):
    json_out = ExplainabilityExporter.export_json(sample_explanation)
    parsed = json.loads(json_out)
    assert parsed["spam_score"] == 0.85
    assert len(parsed["top_features"]) == 3

def test_explainability_exporter_csv(sample_explanation):
    csv_out = ExplainabilityExporter.export_csv(sample_explanation)
    assert "token,weight,direction,category" in csv_out
    assert "winner,0.8,spam,Financial / Prize" in csv_out
    assert "meeting,-0.4,ham,Work" in csv_out

def test_explainability_exporter_html(sample_explanation):
    html_out = ExplainabilityExporter.export_html_report(sample_explanation)
    assert "<!DOCTYPE html>" in html_out
    assert "SPAM" in html_out
    assert "winner" in html_out

def test_export_explanation_wrapper(sample_explanation):
    json_res = export_explanation(sample_explanation, fmt="json")
    csv_res = export_explanation(sample_explanation, fmt="csv")
    html_res = export_explanation(sample_explanation, fmt="html")

    assert "spam_score" in json_res
    assert "token,weight" in csv_res
    assert "<html>" in html_res
