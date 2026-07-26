"""
SQLite Maintenance & WAL Checkpoint Manager for Spamlyser
Automates PRAGMA wal_checkpoint(PASSIVE/FULL/RESTART) execution and database storage optimization.
"""

import sqlite3
from typing import Any, Dict, Tuple


class DatabaseMaintenanceManager:
    """Manages SQLite WAL checkpointing, PRAGMA integrity checks, and space optimization."""

    def __init__(self, db_path: str = ":memory:"):
        self.db_path = db_path

    def run_checkpoint(self, mode: str = "PASSIVE") -> tuple[int, int, int]:
        """Runs PRAGMA wal_checkpoint(mode) and returns (busy, log, checkpointed) page counts."""
        valid_modes = {"PASSIVE": 0, "FULL": 1, "RESTART": 2, "TRUNCATE": 3}
        mode_upper = mode.upper()
        if mode_upper not in valid_modes:
            raise ValueError(f"Invalid checkpoint mode: {mode}")

        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            cursor.execute(f"PRAGMA wal_checkpoint({mode_upper});")
            row = cursor.fetchone()
            return row if row else (0, 0, 0)
        finally:
            conn.close()

    def run_integrity_check(self) -> bool:
        """Executes PRAGMA integrity_check to verify database file health."""
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            cursor.execute("PRAGMA integrity_check;")
            result = cursor.fetchone()
            return result is not None and result[0] == "ok"
        finally:
            conn.close()
