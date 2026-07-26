def test_telemetry_logger_init():
    from models.benchmark_automation import TelemetryLogger

    logger = TelemetryLogger()
    assert logger is not None
