import pytest

from models.word_analyzer import WordAnalyzer


def test_word_analyzer_xss_prevention():
    analyzer = WordAnalyzer()
    malicious_text = "<script>alert('XSS')</script> and some \"quotes\""
    
    # Analyze text
    analysis = analyzer.analyze_text(malicious_text)
    
    # Generate HTML
    html_output = analyzer.create_highlighted_html(analysis)
    
    # Check that raw tags and quotes are escaped
    assert "<script>" not in html_output
    assert "&lt;script&gt;" in html_output
    assert "alert(&#x27;XSS&#x27;)" in html_output or "alert(&#39;XSS&#39;)" in html_output
    assert "&quot;quotes&quot;" in html_output
