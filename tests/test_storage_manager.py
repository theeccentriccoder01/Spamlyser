"""Tests for the StorageManager module."""

import json
import os

from models.storage_manager import StorageManager, default_json_validator


def _read_text(path: str) -> str:
    with open(path) as f:
        return f.read()


def test_save_json_creates_file(tmp_path):
    target = str(tmp_path / "data.json")
    mgr = StorageManager()
    result = mgr.save_json(target, {"key": "value"}, backup=False)
    assert result is True
    assert os.path.exists(target)
    assert json.loads(_read_text(target)) == {"key": "value"}


def test_save_json_returns_false_on_invalid_data(tmp_path):
    target = str(tmp_path / "data.json")

    class NotSerializable:
        pass

    mgr = StorageManager()
    result = mgr.save_json(target, NotSerializable(), backup=False)
    assert result is False


def test_save_json_creates_backup(tmp_path):
    target = str(tmp_path / "data.json")
    mgr = StorageManager()

    mgr.save_json(target, {"v": 1}, backup=False)
    mgr.save_json(target, {"v": 2}, backup=True)

    backups = mgr.list_backups(target)
    assert len(backups) >= 1


def test_load_json_returns_default_on_missing(tmp_path):
    target = str(tmp_path / "missing.json")
    mgr = StorageManager()
    assert mgr.load_json(target, default="fallback") == "fallback"


def test_load_json_reads_valid_file(tmp_path):
    target = str(tmp_path / "data.json")
    mgr = StorageManager()
    mgr.save_json(target, {"a": 1}, backup=False)
    assert mgr.load_json(target) == {"a": 1}


def test_load_json_restores_from_backup_on_corruption(tmp_path):
    target = str(tmp_path / "data.json")
    mgr = StorageManager()

    mgr.save_json(target, {"v": "backup_value"}, backup=False)
    mgr.save_json(target, {"v": "corrupted_value"}, backup=True)
    with open(target, "w") as f:
        f.write("{corrupt")

    result = mgr.load_json(target, default="fail")
    assert result == {"v": "backup_value"}


def test_list_backups_sorted_newest_first(tmp_path):
    target = str(tmp_path / "data.json")
    mgr = StorageManager()

    for v in range(3):
        mgr.save_json(target, {"v": v}, backup=True)

    backups = mgr.list_backups(target)
    assert len(backups) >= 2

    timestamps = [p.stat().st_mtime for p in backups]
    assert timestamps == sorted(timestamps, reverse=True)


def test_restore_from_backup(tmp_path):
    target = str(tmp_path / "data.json")
    mgr = StorageManager()

    mgr.save_json(target, {"v": "a"}, backup=False)
    mgr.save_json(target, {"v": "b"}, backup=True)

    assert mgr.restore_from_backup(target, index=0) is True
    assert json.loads(_read_text(target)) == {"v": "a"}


def test_prune_old_backups(tmp_path):
    target = str(tmp_path / "data.json")
    mgr = StorageManager(keep_backups=2)

    for v in range(5):
        mgr.save_json(target, {"v": v}, backup=True)

    backups = mgr.list_backups(target)
    assert len(backups) <= 2


def test_default_json_validator():
    assert default_json_validator({}) is True
    assert default_json_validator([]) is True
    assert default_json_validator("string") is False
    assert default_json_validator(42) is False
    assert default_json_validator(None) is False


def test_save_json_validator_aborts(tmp_path):
    target = str(tmp_path / "data.json")
    mgr = StorageManager()

    result = mgr.save_json(target, "not-a-dict", validate=default_json_validator)
    assert result is False
    assert not os.path.exists(target)


def test_backup_dir_default_is_dot_prefixed(tmp_path):
    from pathlib import Path

    target = Path(str(tmp_path / "data.json"))
    mgr = StorageManager()
    backup_dir = mgr._backup_dir_for(target)
    assert backup_dir.name == ".data.backups"


# ── New tests for load_json_safe ─────────────────────────────────────────────


def test_load_json_safe_returns_valid_data(tmp_path):
    """load_json_safe should return the file contents when validation passes."""
    target = str(tmp_path / "data.json")
    mgr = StorageManager()
    mgr.save_json(target, {"status": "ok"}, backup=False)

    result = mgr.load_json_safe(target, validate=default_json_validator)
    assert result == {"status": "ok"}


def test_load_json_safe_falls_back_to_backup_on_validation_failure(tmp_path):
    """When primary file fails validation, the most recent valid backup is used."""
    target = str(tmp_path / "data.json")
    mgr = StorageManager()

    # Write a valid backup first
    mgr.save_json(target, {"status": "ok"}, backup=False)
    # Overwrite with invalid content (passes JSON parse but fails our validator)
    mgr.save_json(target, {"status": "ok"}, backup=True)  # creates backup of good data
    with open(target, "w") as f:
        f.write('"just-a-string"')  # valid JSON but not a dict/list

    result = mgr.load_json_safe(
        target,
        default="DEFAULT",
        validate=default_json_validator,
    )
    assert result == {"status": "ok"}


def test_load_json_safe_returns_default_when_all_fail(tmp_path):
    """Returns *default* when neither primary file nor backups pass validation."""
    target = str(tmp_path / "data.json")
    mgr = StorageManager()
    # No valid file exists at all
    result = mgr.load_json_safe(
        target, default="SENTINEL", validate=default_json_validator
    )
    assert result == "SENTINEL"


def test_load_json_safe_no_validator_falls_back_on_parse_error(tmp_path):
    """Without a validator, load_json_safe still falls back on JSON parse errors."""
    target = str(tmp_path / "data.json")
    mgr = StorageManager()

    mgr.save_json(target, [1, 2, 3], backup=False)
    mgr.save_json(target, [1, 2, 3], backup=True)
    with open(target, "w") as f:
        f.write("{broken")

    result = mgr.load_json_safe(target, default=None)
    assert result == [1, 2, 3]


def test_load_json_safe_restores_primary_file_from_backup(tmp_path):
    """After falling back to a backup, the primary file should be restored."""
    target = str(tmp_path / "data.json")
    mgr = StorageManager()

    mgr.save_json(target, {"restored": True}, backup=False)
    mgr.save_json(target, {"restored": True}, backup=True)
    with open(target, "w") as f:
        f.write("{corrupt")

    mgr.load_json_safe(target, default=None)

    # Primary file should now be readable again
    with open(target, encoding="utf-8") as f:
        restored = json.load(f)
    assert restored == {"restored": True}


def test_attempt_self_healing(tmp_path):
    from models.recovery_agent import attempt_self_healing

    primary = tmp_path / "primary.json"
    backup = tmp_path / "backup.json"
    backup.write_text("ok")
    assert attempt_self_healing(primary, backup) is True
    assert primary.read_text() == "ok"
