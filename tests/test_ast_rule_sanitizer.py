import pytest
from models.ast_rule_sanitizer import sanitize_rule_expression
from models.rules_validator import validate_rule_expression_security

def test_ast_rule_sanitizer_valid_expressions():
    valid_exprs = [
        "score > 0.8 and 'urgent' in text",
        "len(text) < 100 or is_whitelisted",
        "has_attachment and (score >= 0.5)"
    ]
    for expr in valid_exprs:
        is_safe, msg = sanitize_rule_expression(expr)
        assert is_safe is True, f"Failed for valid expression: {expr} ({msg})"

def test_ast_rule_sanitizer_blocked_expressions():
    dangerous = [
        "__import__('os').system('dir')",
        "eval('1 + 1')",
        "text.__class__.__subclasses__()",
        "open('/etc/passwd').read()"
    ]
    for expr in dangerous:
        is_safe, msg = sanitize_rule_expression(expr)
        assert is_safe is False
        assert len(msg) > 0

def test_validate_rule_expression_security_integration():
    ok, _ = validate_rule_expression_security("score > 0.5")
    assert ok is True

    bad, msg = validate_rule_expression_security("exec('import os')")
    assert bad is False
