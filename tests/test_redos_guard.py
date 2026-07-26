from models.redos_guard import is_safe_regex_input

def test_redos_guard():
    assert is_safe_regex_input("Hello, normal text.") is True
    assert is_safe_regex_input("a" * 1005) is False
    assert is_safe_regex_input("(a+)+") is False
