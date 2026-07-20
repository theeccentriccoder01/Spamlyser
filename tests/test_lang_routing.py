from models.lang_routing import get_language_rule_engine

def test_lang_routing():
    assert get_language_rule_engine("en") == "English Rules Engine"
    assert get_language_rule_engine("es") == "Spanish Rules Engine"
    assert get_language_rule_engine("de") == "Default Rules Engine"
