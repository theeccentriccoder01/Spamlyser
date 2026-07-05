import shutil
from pathlib import Path


def attempt_self_healing(primary_path: Path, backup_path: Path) -> bool:
    """Restore corrupted primary storage file from backup."""
    if backup_path.exists():
        try:
            shutil.copy2(str(backup_path), str(primary_path))
            return True
        except OSError:
            pass
    return False
