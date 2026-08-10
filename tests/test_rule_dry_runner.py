from models.rule_dry_runner import RuleDryRunner


def test_rule_dry_runner_valid():
    corpus = [
        {"text": "Win $1000 cash prize now!", "label": "SPAM"},
        {"text": "Hey are we still meeting?", "label": "HAM"},
        {"text": "Claim your free gift card", "label": "SPAM"},
    ]

    res = RuleDryRunner.dry_run(r"win|claim|prize", corpus)
    assert res["valid"] is True
    assert res["total_matches"] == 2
    assert res["true_positives"] == 2
    assert res["false_positives"] == 0


def test_rule_dry_runner_invalid_regex():
    res = RuleDryRunner.dry_run(r"[unclosed_bracket", [])
    assert res["valid"] is False
    assert "Invalid Regex" in res["error"]
