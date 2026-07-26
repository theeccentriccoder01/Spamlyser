"""Tests for the SQLite connection pool module."""

import os
import sqlite3
import tempfile
import threading

import pytest

from models.db_connection_pool import ConnectionPool


@pytest.fixture
def db_path(tmp_path):
    """Provide a temporary SQLite database path."""
    return str(tmp_path / "test_pool.db")


class TestConnectionPoolLifecycle:
    """Verify pool creation, checkout, release, and shutdown."""

    def test_get_connection_returns_usable_connection(self, db_path):
        pool = ConnectionPool(db_path, max_connections=2)
        conn = pool.get_connection()
        assert conn is not None
        conn.execute("SELECT 1")
        pool.release_connection(conn)
        pool.close_all()

    def test_released_connection_is_reused(self, db_path):
        pool = ConnectionPool(db_path, max_connections=2)
        conn1 = pool.get_connection()
        pool.release_connection(conn1)
        conn2 = pool.get_connection()
        # The same connection object should be reused from the pool
        assert conn2 is conn1
        pool.release_connection(conn2)
        pool.close_all()

    def test_pool_respects_max_connections(self, db_path):
        pool = ConnectionPool(db_path, max_connections=1)
        conn1 = pool.get_connection()
        # Pool is now at capacity — next call should raise after retries
        with pytest.raises(RuntimeError, match="Could not obtain"):
            pool.get_connection()
        pool.release_connection(conn1)
        pool.close_all()

    def test_close_all_releases_everything(self, db_path):
        pool = ConnectionPool(db_path, max_connections=3)
        conns = [pool.get_connection() for _ in range(3)]
        for c in conns:
            pool.release_connection(c)
        pool.close_all()
        stats = pool.stats
        assert stats["in_use"] == 0
        assert stats["idle"] == 0

    def test_stats_reflect_pool_state(self, db_path):
        pool = ConnectionPool(db_path, max_connections=5)
        conn = pool.get_connection()
        stats = pool.stats
        assert stats["in_use"] == 1
        assert stats["idle"] == 0
        pool.release_connection(conn)
        stats = pool.stats
        assert stats["in_use"] == 0
        assert stats["idle"] == 1
        pool.close_all()


class TestHealthCheck:
    """Verify that unhealthy connections are discarded."""

    def test_closed_connection_is_replaced(self, db_path):
        pool = ConnectionPool(db_path, max_connections=2)
        conn1 = pool.get_connection()
        conn1.close()  # Simulate a broken connection
        pool.release_connection(conn1)
        # The pool should detect the dead connection and create a new one
        conn2 = pool.get_connection()
        assert conn2 is not conn1
        conn2.execute("SELECT 1")  # Should work fine
        pool.release_connection(conn2)
        pool.close_all()


class TestConcurrentAccess:
    """Verify thread safety under concurrent load."""

    def test_concurrent_checkout_and_release(self, db_path):
        pool = ConnectionPool(db_path, max_connections=5)
        errors = []

        def worker():
            try:
                conn = pool.get_connection()
                conn.execute("SELECT 1")
                pool.release_connection(conn)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        pool.close_all()
        assert len(errors) == 0, f"Concurrent errors: {errors}"
