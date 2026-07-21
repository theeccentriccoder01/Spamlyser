def get_language_rule_engine(lang_code: str):
    """Returns specific rule configurations based on identified language."""
    engines = {
        "en": "English Rules Engine",
        "es": "Spanish Rules Engine",
        "fr": "French Rules Engine"
    }
    return engines.get(lang_code, "Default Rules Engine")
