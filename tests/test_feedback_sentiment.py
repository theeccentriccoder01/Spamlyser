from models.feedback_sentiment import FeedbackSentimentAnalyzer


def test_feedback_satisfaction():
    ratings = [5, 4, 5, 2, 1]
    res = FeedbackSentimentAnalyzer.calculate_satisfaction(ratings)
    assert res["total_feedback"] == 5
    assert res["avg_rating"] == 3.4
    assert res["satisfaction_pct"] == 60.0
    assert res["distribution"][5] == 2


def test_feedback_group_by_model():
    entries = [
        {"model_name": "distilbert", "rating": 5},
        {"model_name": "distilbert", "rating": 4},
        {"model_name": "naive_bayes", "rating": 2},
    ]
    res = FeedbackSentimentAnalyzer.group_by_model(entries)
    assert "distilbert" in res
    assert res["distilbert"]["avg_rating"] == 4.5
    assert res["naive_bayes"]["avg_rating"] == 2.0
