import pytest
from models.db_maintenance import DatabaseMaintenanceManager


def test_db_maintenance_integrity():
    manager = DatabaseMaintenanceManager(":memory:")
    assert manager.run_integrity_check() is True


def test_db_maintenance_checkpoint():
    manager = DatabaseMaintenanceManager(":memory:")
    res = manager.run_checkpoint("PASSIVE")
    assert isinstance(res, tuple)
    assert len(res) == 3


def test_db_maintenance_invalid_mode():
    manager = DatabaseMaintenanceManager(":memory:")
    with pytest.raises(ValueError):
        manager.run_checkpoint("INVALID_MODE")
