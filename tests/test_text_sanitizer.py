"""Tests for text sanitization helpers."""

from models.text_sanitizer import sanitize_text, strip_html_unsafe, validate_sms_message


def test_strip_html_removes_script_blocks_and_tags():
    text = "<p>Hello</p><script>alert('x')</script><strong>world</strong>"
    assert strip_html_unsafe(text) == "Helloworld"


def test_strip_html_preserves_plain_text():
    assert strip_html_unsafe("hello there") == "hello there"


def test_sanitize_text_escapes_html_for_display():
    assert sanitize_text("<b>hello</b>") == "&lt;b&gt;hello&lt;/b&gt;"


def test_validate_sms_message_strips_html_before_model_use():
    valid, error, sanitized = validate_sms_message("<b>hello</b>")
    assert valid is True
    assert error == ""
    assert sanitized == "hello"
