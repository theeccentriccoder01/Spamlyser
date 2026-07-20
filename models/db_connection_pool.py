"""
Lightweight SQLite connection pool for Spamlyser Pro.

Streamlit re-runs the entire script on every interaction, which means naive
connection handling can leave stale file descriptors when the database is
rotated or when the app reloads. This module provides a simple pool that:

- Validates connections with a health-check ping before handing them out
- Automatically closes connections that have been idle too long
- Limits the total number of open connections to avoid file descriptor leaks
- Uses a threading lock for safe concurrent access across Streamlit threads

This replaces the bare thread-local pattern that was previously used in
``feedback_handler.py``.
"""

import logging
import sqlite3
import threading
import time
from typing import Optional

_logger = logging.getLogger(__name__)


class ConnectionPool:
    """A bounded connection pool for SQLite databases.

    Parameters
    ----------
    db_path : str
        Path to the SQLite database file.
    max_connections : int
        Maximum number of simultaneous connections (default 5).
    idle_timeout : float
        Seconds after which an idle connection is considered stale
        and will be closed on the next checkout (default 300).
    """

    def __init__(
        self,
        db_path: str,
        max_connections: int = 5,
        idle_timeout: float = 300.0,
    ):
        self._db_path = db_path
        self._max_connections = max_connections
        self._idle_timeout = idle_timeout
        self._lock = threading.Lock()

        # Pool stores tuples of (connection, last_used_timestamp)
        self._available: list[tuple[sqlite3.Connection, float]] = []
        self._in_use: int = 0

    def _create_connection(self) -> sqlite3.Connection:
        """Open a fresh SQLite connection with recommended pragmas."""
        conn = sqlite3.connect(self._db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def _is_healthy(self, conn: sqlite3.Connection) -> bool:
        """Verify a connection is still usable with a lightweight ping."""
        try:
            conn.execute("SELECT 1")
            return True
        except Exception:
            return False

    def _prune_stale(self) -> None:
        """Close connections that have been idle longer than the timeout.

        Must be called while holding ``self._lock``.
        """
        now = time.monotonic()
        still_good = []
        for conn, last_used in self._available:
            if now - last_used > self._idle_timeout:
                try:
                    conn.close()
                except Exception:
                    pass
                _logger.debug("Pruned stale connection (idle %.0fs)", now - last_used)
            else:
                still_good.append((conn, last_used))
        self._available = still_good

    def get_connection(self) -> sqlite3.Connection:
        """Retrieve a healthy connection from the pool.

        If no idle connections are available and the pool has capacity, a
        new connection is created. If the pool is at capacity, this method
        waits briefly and retries.

        Returns
        -------
        sqlite3.Connection
            A validated, ready-to-use database connection.

        Raises
        ------
        RuntimeError
            If a connection cannot be obtained after several retries.
        """
        retries = 3
        for attempt in range(retries):
            with self._lock:
                self._prune_stale()

                # Try to reuse an idle connection
                while self._available:
                    conn, _ts = self._available.pop()
                    if self._is_healthy(conn):
                        self._in_use += 1
                        return conn
                    # Connection is dead, discard it
                    try:
                        conn.close()
                    except Exception:
                        pass

                # Create a new connection if under the limit
                if self._in_use < self._max_connections:
                    conn = self._create_connection()
                    self._in_use += 1
                    return conn

            # Pool is full — wait a bit and retry
            _logger.warning(
                "Connection pool exhausted (%d/%d), retry %d/%d",
                self._in_use,
                self._max_connections,
                attempt + 1,
                retries,
            )
            time.sleep(0.1 * (attempt + 1))

        raise RuntimeError(
            f"Could not obtain a database connection after {retries} attempts. "
            f"Pool: {self._in_use}/{self._max_connections} in use."
        )

    def release_connection(self, conn: sqlite3.Connection) -> None:
        """Return a connection to the pool for reuse.

        Parameters
        ----------
        conn : sqlite3.Connection
            The connection to return. If it fails a health check it is
            closed instead of being returned to the pool.
        """
        with self._lock:
            self._in_use = max(0, self._in_use - 1)
            if self._is_healthy(conn):
                self._available.append((conn, time.monotonic()))
            else:
                try:
                    conn.close()
                except Exception:
                    pass

    def close_all(self) -> None:
        """Close every connection in the pool (both idle and in-use tracking).

        Call this during application shutdown or when rotating the database
        file to ensure all file descriptors are released.
        """
        with self._lock:
            for conn, _ in self._available:
                try:
                    conn.close()
                except Exception:
                    pass
            self._available.clear()
            self._in_use = 0
            _logger.info("Connection pool closed (all connections released)")

    @property
    def stats(self) -> dict:
        """Return current pool statistics for monitoring."""
        with self._lock:
            return {
                "db_path": self._db_path,
                "max_connections": self._max_connections,
                "in_use": self._in_use,
                "idle": len(self._available),
                "idle_timeout": self._idle_timeout,
            }
