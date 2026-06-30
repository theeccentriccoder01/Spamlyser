"""
Safe file storage with atomic writes, automatic backups, and rotation.

Every write goes to a temporary file first, then is renamed atomically over
the target.  Before overwriting an existing file, a timestamped backup is
created in a ``.backups/`` sibling directory.  Old backups are automatically
pruned so only the *N* most recent are kept.
"""

import json
import os
import shutil
import tempfile
from collections.abc import Callable
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional


class StorageManager:
    """Atomic writes with backup rotation for JSON data files.

    Parameters
    ----------
    backup_dir : str or Path, optional
        Directory where backups are kept.  Defaults to ``<parent>/<stem>.backups/``
        next to the managed file.
    keep_backups : int
        Maximum number of timestamped backups to retain (default 5).
    max_backup_age_days : float
        Backups older than this many days are pruned regardless of count.
    """

    def __init__(
        self,
        backup_dir: Path | None = None,
        keep_backups: int = 5,
        max_backup_age_days: float = 30.0,
    ):
        self.backup_dir = Path(backup_dir) if backup_dir else None
        self.keep_backups = keep_backups
        self.max_backup_age_days = max_backup_age_days

    # ── Public API ──────────────────────────────────────────────────────

    def save_json(
        self,
        filepath: str,
        data: Any,
        backup: bool = True,
        validate: Callable[[Any], bool] | None = None,
    ) -> bool:
        """Atomically write *data* as JSON to *filepath*.

        Parameters
        ----------
        filepath : str
            Destination file path.
        data : Any
            JSON-serialisable object.
        backup : bool
            Create a timestamped backup of the existing file before overwriting.
        validate : callable, optional
            A ``callable(data) -> bool`` invoked on the *new* data before
            writing to disk.  If it returns ``False`` the write is aborted.

        Returns
        -------
        bool
            ``True`` on success, ``False`` on failure.
        """
        path = Path(filepath)
        if validate is not None and not validate(data):
            return False

        parent = path.parent
        parent.mkdir(parents=True, exist_ok=True)

        if backup and path.exists():
            self._create_backup(path)

        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=str(parent),
                suffix=".tmp",
                delete=False,
            ) as tmp:
                json.dump(data, tmp, indent=2, ensure_ascii=False)
                tmp_path = tmp.name

            shutil.move(tmp_path, str(path))
            self._prune_backups(path)
            return True
        except (OSError, ValueError, TypeError):
            return False

    def load_json(
        self,
        filepath: str,
        default: Any = None,
    ) -> Any:
        """Load JSON from *filepath* returning *default* on failure.

        If the file is corrupted, attempts to restore from the most recent
        backup automatically.
        """
        path = Path(filepath)
        if not path.exists():
            return default

        try:
            with open(str(path), encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            backup = self._latest_backup(path)
            if backup is not None:
                try:
                    with open(str(backup), encoding="utf-8") as f:
                        data = json.load(f)
                    self.save_json(str(path), data, backup=False)
                    return data
                except (json.JSONDecodeError, OSError):
                    pass
            return default

    def list_backups(self, filepath: str) -> list[Path]:
        """Return all backup paths for *filepath*, sorted newest-first."""
        backup_dir = self._backup_dir_for(Path(filepath))
        if not backup_dir.exists():
            return []
        stem = Path(filepath).stem
        backups = sorted(
            backup_dir.glob(f"{stem}.bak.*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        return backups

    def restore_from_backup(self, filepath: str, index: int = 0) -> bool:
        """Restore *filepath* from its *index*-th backup (0 = most recent)."""
        backups = self.list_backups(filepath)
        if index < 0 or index >= len(backups):
            return False
        try:
            shutil.copy2(str(backups[index]), str(filepath))
            return True
        except OSError:
            return False

    # ── Internal helpers ────────────────────────────────────────────────

    def _backup_dir_for(self, path: Path) -> Path:
        if self.backup_dir:
            return self.backup_dir
        return path.parent / f".{path.stem}.backups"

    def _create_backup(self, path: Path) -> Path | None:
        backup_dir = self._backup_dir_for(path)
        backup_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        backup_path = backup_dir / f"{path.stem}.bak.{ts}.json"
        try:
            shutil.copy2(str(path), str(backup_path))
            return backup_path
        except OSError:
            return None

    def _latest_backup(self, path: Path) -> Path | None:
        backups = self.list_backups(str(path))
        return backups[0] if backups else None

    def _prune_backups(self, path: Path) -> None:
        backup_dir = self._backup_dir_for(path)
        if not backup_dir.exists():
            return

        all_backups = sorted(
            backup_dir.glob(f"{path.stem}.bak.*.json"),
            key=lambda p: p.stat().st_mtime,
        )

        cutoff = datetime.now() - timedelta(days=self.max_backup_age_days)
        remaining: list[Path] = []

        for bp in all_backups:
            mtime = datetime.fromtimestamp(bp.stat().st_mtime)
            if mtime < cutoff:
                try:
                    bp.unlink()
                except OSError:
                    pass
            else:
                remaining.append(bp)

        # Keep only the N most recent
        while len(remaining) > self.keep_backups:
            old = remaining.pop(0)
            try:
                old.unlink()
            except OSError:
                pass


def default_json_validator(data: Any) -> bool:
    """Return ``True`` if *data* is a dict or list (a valid JSON container)."""
    return isinstance(data, (dict, list))
