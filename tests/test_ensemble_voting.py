import pytest
from models.ensemble_voting_aggregator import EnsembleVotingAggregator

def test_ensemble_soft_voting_weighted():
    weights = {"ml_model": 2.0, "regex_engine": 1.0, "heuristic": 1.0}
    aggregator = EnsembleVotingAggregator(weights=weights)

    preds = [
        {"engine": "ml_model", "score": 0.90},
        {"engine": "regex_engine", "score": 0.20},
        {"engine": "heuristic", "score": 0.30}
    ]

    # Weighted: (0.9*2 + 0.2*1 + 0.3*1) / (2+1+1) = 2.3 / 4 = 0.575
    res = aggregator.soft_voting(preds)
    assert res["is_spam"] is True
    assert res["confidence"] == 0.575
    assert res["decision"] == "SPAM"

def test_ensemble_majority_voting():
    aggregator = EnsembleVotingAggregator()
    preds = [
        {"engine": "e1", "is_spam": False},
        {"engine": "e2", "is_spam": True},
        {"engine": "e3", "is_spam": True}
    ]

    res = aggregator.majority_voting(preds)
    assert res["is_spam"] is True
    assert res["spam_votes"] == 2
    assert res["ham_votes"] == 1
    assert res["confidence"] == round(2/3, 4)
