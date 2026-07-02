"""Tests for the lightweight SimpleExplainer module."""

from models.simple_explainer import HAM_KEYWORDS, SPAM_KEYWORDS, SimpleExplainer


def test_default_spam_keywords_are_populated():
    assert isinstance(SPAM_KEYWORDS, dict)
    assert len(SPAM_KEYWORDS) >= 4


def test_default_ham_keywords_are_populated():
    assert isinstance(HAM_KEYWORDS, dict)
    assert len(HAM_KEYWORDS) >= 4


def test_explain_prediction_returns_expected_structure():
    explainer = SimpleExplainer()
    result = explainer.explain_prediction("Free money! Click here to claim your prize.")
    assert "text" in result
    assert "features" in result
    assert len(result["features"]) == 2
    assert result["features"][0]["class"] == "HAM"
    assert result["features"][1]["class"] == "SPAM"


def test_explain_prediction_finds_spam_keywords():
    explainer = SimpleExplainer()
    result = explainer.explain_prediction(
        "URGENT: Your account has been locked. Verify now!"
    )
    spam_features = result["features"][1]["important_words"]
    words = [f["word"] for f in spam_features]
    assert "urgent" in words or "account" in words or "verify" in words


def test_explain_prediction_ham_finds_indicators():
    explainer = SimpleExplainer()
    result = explainer.explain_prediction(
        "Hey, hello! Good morning, let me know when we can meet."
    )
    ham_features = result["features"][0]["important_words"]
    words = [f["word"] for f in ham_features]
    assert len(ham_features) > 0
    assert (
        "hello" in words
        or "let me know" in words
        or "meeting" in words
        or "hi" in words
        or "hey" in words
    )


def test_visualize_explanation_returns_feature_importance():
    explainer = SimpleExplainer()
    explanation = explainer.explain_prediction("Free money now!")
    viz = explainer.visualize_explanation(explanation)
    assert "feature_importance" in viz
    assert "SPAM" in viz["feature_importance"]
    assert "summary" in viz


def test_get_threat_explanation_no_threat():
    explainer = SimpleExplainer()
    result = explainer.get_threat_explanation("Hello", None)
    assert result["threat_type"] is None
    assert result["matching_keywords"] == []


def test_get_threat_explanation_with_phishing():
    explainer = SimpleExplainer()
    result = explainer.get_threat_explanation(
        "Urgent! Act now to verify your account.", "Urgency / Pressure"
    )
    assert result["threat_type"] == "Urgency / Pressure"
    assert len(result["matching_keywords"]) > 0


def test_visualize_explanation_summary_contains_top_signal():
    explainer = SimpleExplainer()
    explanation = explainer.explain_prediction("Free prize money! Claim now!")
    viz = explainer.visualize_explanation(explanation)
    spam_features = viz["feature_importance"].get("SPAM", [])
    if spam_features:
        top_word = spam_features[0]["feature"]
        assert top_word in viz["summary"]


def test_respects_num_features_limit():
    explainer = SimpleExplainer()
    result = explainer.explain_prediction(
        "Free money! Urgent account verification required. Click here.", num_features=2
    )
    spam_features = result["features"][1]["important_words"]
    assert len(spam_features) <= 2


def test_construct_with_custom_keywords():
    custom = {"Test": ["hello"]}
    explainer = SimpleExplainer(keywords=custom)
    result = explainer.explain_prediction("hello world")
    spam_words = [f["word"] for f in result["features"][1]["important_words"]]
    assert "hello" in spam_words


def test_explain_prediction_honors_predict_fn():
    def dummy_predict_fn(texts):
        import numpy as np

        return np.array([[0.1, 0.9]])

    explainer = SimpleExplainer(predict_fn=dummy_predict_fn)
    result = explainer.explain_prediction("test")
    assert result["text"] == "test"


def test_visualize_explanation_empty_features():
    explainer = SimpleExplainer()
    explanation = {
        "features": [{"class": "SPAM", "important_words": []}],
        "class_names": ["HAM", "SPAM"],
        "text": "hello",
    }
    viz = explainer.visualize_explanation(explanation)
    assert viz["feature_importance"]["SPAM"] == []
    assert viz["summary"] == ""


def test_get_threat_explanation_unknown_threat():
    explainer = SimpleExplainer()
    result = explainer.get_threat_explanation("hello", "UnknownType")
    assert result["matching_keywords"] == []
    assert result["threat_features"] == []


def test_spam_keyword_does_not_match_inside_larger_word():
    explainer = SimpleExplainer()
    result = explainer.explain_prediction("The window is open")
    spam_words = result["features"][1]["important_words"]
    assert all(item["word"] != "win" for item in spam_words)


def test_ham_keyword_does_not_match_inside_larger_word():
    explainer = SimpleExplainer()
    result = explainer.explain_prediction("The meetinghouse is old")
    ham_words = result["features"][0]["important_words"]
    assert all(item["word"] != "meeting" for item in ham_words)


def test_keyword_phrases_still_match_with_boundaries():
    explainer = SimpleExplainer()
    result = explainer.explain_prediction("This is a limited time offer")
    spam_words = result["features"][1]["important_words"]
    assert {item["word"] for item in spam_words} >= {"limited time", "offer"}
