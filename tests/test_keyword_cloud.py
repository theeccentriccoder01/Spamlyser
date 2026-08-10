from models.keyword_cloud import KeywordCloudGenerator


def test_keyword_cloud_generator():
    messages = [
        "Urgent! Claim your cash prize now!",
        "Urgent cash offer available today",
        "Hello friend how are you",
    ]
    res_dict = KeywordCloudGenerator.generate_dict(messages, top_n=5)
    assert "urgent" in res_dict
    assert "cash" in res_dict
    assert res_dict["urgent"] == 2
    assert "the" not in res_dict
