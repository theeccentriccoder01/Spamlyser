from models.custom_rules_manager import validate_rule_schema

def test_rules_schema_validation():
    valid_rule = {"id": "1", "pattern": "win", "risk_level": "high"}
    invalid_rule = {"id": "1", "pattern": "win"}
    assert validate_rule_schema(valid_rule) is True
    assert validate_rule_schema(invalid_rule) is False
