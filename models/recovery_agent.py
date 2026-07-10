import shutil
import sqlite3
from pathlib import Path


def verify_db_integrity(db_path: Path) -> bool:
    """Run SQLite PRAGMA integrity_check to verify file health."""
    if not db_path.exists():
        return False
    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        cursor.execute("PRAGMA integrity_check;")
        result = cursor.fetchone()
        conn.close()
        return result and result[0] == "ok"
    except sqlite3.Error:
        return False


def attempt_self_healing(primary_path: Path, backup_path: Path) -> bool:
    """Restore corrupted primary storage file from backup if integrity check fails."""
    if verify_db_integrity(primary_path):
        return True

    if backup_path.exists() and verify_db_integrity(backup_path):
        try:
            shutil.copy2(str(backup_path), str(primary_path))
            return True
        except OSError:
            pass
    return False

