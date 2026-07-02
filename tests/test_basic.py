def test_telemetry_logger_init():
    from models.telemetry_logger import TelemetryLogger
    logger = TelemetryLogger()
    assert logger is not None
