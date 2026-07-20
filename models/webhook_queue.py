import sqlite3

class WebhookRetryQueue:
    def __init__(self, db_path=":memory:"):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.execute("CREATE TABLE IF NOT EXISTS queue (id INTEGER PRIMARY KEY, url TEXT, payload TEXT, retries INTEGER)")
        self.conn.commit()
        
    def enqueue(self, url, payload):
        self.conn.execute("INSERT INTO queue (url, payload, retries) VALUES (?, ?, 0)", (url, payload))
        self.conn.commit()
        
    def get_pending(self):
        return self.conn.execute("SELECT id, url, payload FROM queue").fetchall()
