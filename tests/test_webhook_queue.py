from models.webhook_queue import WebhookRetryQueue

def test_webhook_queue():
    queue = WebhookRetryQueue()
    queue.enqueue("http://example.com", "{}")
    pending = queue.get_pending()
    assert len(pending) == 1
    assert pending[0][1] == "http://example.com"
